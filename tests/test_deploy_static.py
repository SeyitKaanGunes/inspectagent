"""Static validation of the AWS deployment (SPEC section 6) - offline: no AWS, no network, no Docker.

* CloudFormation template: required resources / properties / outputs, least-privilege IAM, cfn-lint clean.
* buildspec + Dockerfiles: parse, reference existing paths that survive ``deploy/.dockerignore``.
* requirements.lock.txt pins every direct requirement.
* deploy.py: ``--help``, Docker-compatible ignore matching, deterministic source zip, pure helpers, and the
  full deploy / redeploy / destroy flow against an in-memory AWS fake that sits behind *real* boto3
  clients (botocore validates every request against the real service models before the fake answers).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"
TEMPLATE = DEPLOY / "aws" / "template.yaml"
BUILDSPEC = DEPLOY / "aws" / "buildspec.yml"
DEPLOY_PY = DEPLOY / "aws" / "deploy.py"
DOCKERIGNORE = DEPLOY / ".dockerignore"


# ------------------------------------------------------------------------------------------------
# loaders
# ------------------------------------------------------------------------------------------------
class CfnLoader(yaml.SafeLoader):
    """SafeLoader that turns CloudFormation short-form intrinsics (!Ref, !Sub, ...) into dicts."""


def _cfn_tag(loader: yaml.SafeLoader, suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.ScalarNode):
        value: Any = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    if suffix == "Ref":
        return {"Ref": value}
    if suffix == "Condition":
        return {"Condition": value}
    if suffix == "GetAtt" and isinstance(value, str):
        value = value.split(".", 1)
    return {f"Fn::{suffix}": value}


CfnLoader.add_multi_constructor("!", _cfn_tag)


@pytest.fixture
def work_dir():
    """Fresh temporary directory (own mkdtemp instead of pytest's shared basetemp, which may be unreadable
    when several sandboxed processes share one Windows account)."""
    d = Path(tempfile.mkdtemp(prefix="inspectagent-test-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def load_template() -> dict:
    return yaml.load(TEMPLATE.read_text(encoding="utf-8"), Loader=CfnLoader)


@pytest.fixture(scope="module")
def tpl() -> dict:
    return load_template()


@pytest.fixture(scope="module")
def dep():
    """deploy/aws/deploy.py imported as a module (it is a script, not a package member)."""
    spec = importlib.util.spec_from_file_location("inspectagent_deploy_script", DEPLOY_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


def resources_of(tpl: dict, rtype: str) -> dict[str, dict]:
    return {k: v for k, v in tpl["Resources"].items() if v["Type"] == rtype}


def one(tpl: dict, rtype: str) -> dict:
    found = resources_of(tpl, rtype)
    assert len(found) == 1, f"expected exactly one {rtype}, got {sorted(found)}"
    return next(iter(found.values()))


def iam_statements(tpl: dict) -> list[tuple[str, dict, str | None]]:
    """(role logical id, statement, condition guarding the policy or None) for every inline statement."""
    out = []
    for rid, role in resources_of(tpl, "AWS::IAM::Role").items():
        for pol in role["Properties"].get("Policies", []):
            cond = None
            if "Fn::If" in pol:
                cond, pol = pol["Fn::If"][0], pol["Fn::If"][1]
            for st in pol["PolicyDocument"]["Statement"]:
                out.append((rid, st, cond))
    return out


def as_list(v: Any) -> list:
    return v if isinstance(v, list) else [v]


# ------------------------------------------------------------------------------------------------
# CloudFormation template
# ------------------------------------------------------------------------------------------------
def test_template_parameters_and_conditions(tpl):
    params = tpl["Parameters"]
    assert params["ImageUri"]["Default"] == ""
    assert params["ImageUriX86"]["Default"] == ""
    assert "BedrockModelId" in params and params["BedrockModelId"]["Default"] == ""
    assert params["StackPrefix"]["Default"] == "inspectagent"
    assert re.fullmatch(params["StackPrefix"]["AllowedPattern"], "inspectagent")
    assert params["ReservedConcurrency"]["Default"] == 5
    conds = tpl["Conditions"]
    assert conds["HasImage"] == {"Fn::Not": [{"Fn::Equals": [{"Ref": "ImageUri"}, ""]}]}
    assert "HasX86Image" in conds and "HasBedrock" in conds
    # every image-dependent resource is behind HasImage / HasX86Image
    for rtype in ("AWS::Lambda::Function", "AWS::Lambda::Url", "AWS::Lambda::Permission"):
        for rid, res in resources_of(tpl, rtype).items():
            assert res.get("Condition") in ("HasImage", "HasX86Image"), rid


def test_template_bucket_private_encrypted_lifecycle(tpl):
    bucket = one(tpl, "AWS::S3::Bucket")["Properties"]
    pab = bucket["PublicAccessBlockConfiguration"]
    assert all(pab[k] is True for k in ("BlockPublicAcls", "BlockPublicPolicy", "IgnorePublicAcls", "RestrictPublicBuckets"))
    sse = bucket["BucketEncryption"]["ServerSideEncryptionConfiguration"][0]["ServerSideEncryptionByDefault"]
    assert sse["SSEAlgorithm"] == "AES256"  # SSE-S3
    rules = {r.get("Prefix"): r for r in bucket["LifecycleConfiguration"]["Rules"]}
    for prefix in ("traces/", "artifacts/"):
        assert rules[prefix]["Status"] == "Enabled" and rules[prefix]["ExpirationInDays"] == 30
    policy = one(tpl, "AWS::S3::BucketPolicy")["Properties"]["PolicyDocument"]["Statement"][0]
    assert policy["Effect"] == "Deny" and policy["Condition"]["Bool"]["aws:SecureTransport"] == "false"


def test_template_table_on_demand_pk_sk(tpl):
    table = one(tpl, "AWS::DynamoDB::Table")["Properties"]
    assert table["BillingMode"] == "PAY_PER_REQUEST"
    assert {(k["AttributeName"], k["KeyType"]) for k in table["KeySchema"]} == {("pk", "HASH"), ("sk", "RANGE")}
    assert {(a["AttributeName"], a["AttributeType"]) for a in table["AttributeDefinitions"]} == {("pk", "S"), ("sk", "S")}


def test_template_ecr_repository(tpl):
    repo = one(tpl, "AWS::ECR::Repository")["Properties"]
    assert repo["ImageScanningConfiguration"]["ScanOnPush"] is True
    assert repo["EmptyOnDelete"] is True
    policy = json.loads(repo["LifecyclePolicy"]["LifecyclePolicyText"])
    sel = policy["rules"][0]["selection"]
    assert sel["countType"] == "imageCountMoreThan" and sel["countNumber"] == 10
    assert policy["rules"][0]["action"]["type"] == "expire"


def test_template_codebuild_projects(tpl):
    projects = resources_of(tpl, "AWS::CodeBuild::Project")
    by_type = {p["Properties"]["Environment"]["Type"]: p["Properties"] for p in projects.values()}
    arm = by_type["ARM_CONTAINER"]
    assert arm["Environment"]["Image"] == "aws/codebuild/amazonlinux-aarch64-standard:3.0"
    x86 = by_type["LINUX_CONTAINER"]
    for proj in (arm, x86):
        env = proj["Environment"]
        assert env["PrivilegedMode"] is True
        assert proj["Source"]["Type"] == "S3"
        assert proj["Source"]["BuildSpec"] == "deploy/aws/buildspec.yml"
        names = {v["Name"] for v in env["EnvironmentVariables"]}
        assert {"ECR_REPO_URI", "ECR_REPO_NAME", "TARGET_ARCH", "DOCKER_PLATFORM", "TAG_SUFFIX"} <= names
        assert proj["LogsConfig"]["CloudWatchLogs"]["Status"] == "ENABLED"
    tag_suffix = {p["Environment"]["Type"]: {v["Name"]: v["Value"] for v in p["Environment"]["EnvironmentVariables"]} for p in (arm, x86)}
    assert tag_suffix["ARM_CONTAINER"]["DOCKER_PLATFORM"] == "linux/arm64"
    assert tag_suffix["LINUX_CONTAINER"]["DOCKER_PLATFORM"] == "linux/amd64"


def test_template_lambda_function_and_url(tpl):
    fns = resources_of(tpl, "AWS::Lambda::Function")
    main = fns["Function"]
    assert main["Condition"] == "HasImage"
    p = main["Properties"]
    assert p["PackageType"] == "Image"
    assert p["Architectures"] == ["arm64"]
    assert p["MemorySize"] == 3008 and p["Timeout"] == 60
    assert p["Code"] == {"ImageUri": {"Ref": "ImageUri"}}
    rce = p["ReservedConcurrentExecutions"]["Fn::If"]
    assert rce[0] == "HasReservedConcurrency" and rce[1] == {"Ref": "ReservedConcurrency"}
    env = p["Environment"]["Variables"]
    assert env["INSPECTAGENT_STORAGE"] == "aws"
    assert env["INSPECTAGENT_S3_BUCKET"] == {"Ref": "ArtifactBucket"}
    assert env["INSPECTAGENT_DDB_TABLE"] == {"Ref": "Table"}
    assert env["INSPECTAGENT_BEDROCK_MODEL_ID"] == {"Ref": "BedrockModelId"}
    assert p["LoggingConfig"]["LogGroup"] == {"Ref": "FunctionLogGroup"}
    x86 = fns["FunctionX86"]
    assert x86["Condition"] == "HasX86Image" and x86["Properties"]["Architectures"] == ["x86_64"]
    assert x86["Properties"]["FunctionName"] == {"Fn::Sub": "${StackPrefix}-x86"}

    urls = resources_of(tpl, "AWS::Lambda::Url")
    for url in urls.values():
        up = url["Properties"]
        assert up["AuthType"] == "NONE"
        assert up["Cors"]["AllowOrigins"] == ["*"]
    perms = [r["Properties"] for r in resources_of(tpl, "AWS::Lambda::Permission").values()]
    assert any(pp["Action"] == "lambda:InvokeFunctionUrl" and pp["FunctionUrlAuthType"] == "NONE" for pp in perms)
    assert any(pp["Action"] == "lambda:InvokeFunction" and pp.get("InvokedViaFunctionUrl") is True for pp in perms)

    groups = resources_of(tpl, "AWS::Logs::LogGroup")
    assert len(groups) >= 2 and all(g["Properties"]["RetentionInDays"] == 14 for g in groups.values())


def test_template_iam_least_privilege(tpl):
    stmts = iam_statements(tpl)
    allowed_star_resource = {
        "ecr:GetAuthorizationToken", "ecr-public:GetAuthorizationToken", "sts:GetServiceBearerToken",
        "aws-marketplace:ViewSubscriptions", "aws-marketplace:Subscribe",
        "bedrock:ListFoundationModels", "bedrock:ListInferenceProfiles",  # read-only discovery, no resource ARNs
    }
    for rid, st, _cond in stmts:
        assert st["Effect"] == "Allow"
        actions = as_list(st["Action"])
        assert all("*" not in a for a in actions), (rid, actions)
        if "*" in as_list(st["Resource"]):
            assert set(actions) <= allowed_star_resource, (rid, actions)
    fn = [(st, c) for rid, st, c in stmts if rid == "FunctionRole"]
    acts = {a: (st, c) for st, c in fn for a in as_list(st["Action"])}
    # logs only on the function log groups
    assert {"logs:CreateLogStream", "logs:PutLogEvents"} <= set(acts)
    assert "logs:CreateLogGroup" not in acts
    # S3 object access only under traces/ and artifacts/
    s3_res = [r["Fn::Sub"] for r in as_list(acts["s3:GetObject"][0]["Resource"])]
    assert sorted(s3_res) == ["${ArtifactBucket.Arn}/artifacts/*", "${ArtifactBucket.Arn}/traces/*"]
    assert "s3:PutObject" in as_list(acts["s3:GetObject"][0]["Action"])
    # DynamoDB CRUD scoped to the table
    for a in ("dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem", "dynamodb:Query"):
        assert a in acts
    assert {"Fn::GetAtt": ["Table", "Arn"]} in as_list(acts["dynamodb:GetItem"][0]["Resource"])
    # Bedrock (Converse is authorized by bedrock:InvokeModel) on model + inference-profile ARNs, only when enabled
    st, cond = acts["bedrock:InvokeModel"]
    assert cond == "HasBedrock"
    res = " ".join(r["Fn::Sub"] for r in as_list(st["Resource"]))
    assert "foundation-model/*" in res and "inference-profile/*" in res
    # marketplace is opt-in
    st, cond = acts["aws-marketplace:Subscribe"]
    assert cond == "BedrockMarketplace"
    assert tpl["Parameters"]["AllowMarketplaceSubscribe"]["Default"] == "false"
    # CodeBuild can push only to the stack's repository
    cb = {a: st for rid, st, _ in stmts if rid == "CodeBuildRole" for a in as_list(st["Action"])}
    assert cb["ecr:PutImage"]["Resource"] == {"Fn::GetAtt": ["Repository", "Arn"]}
    assert cb["s3:GetObject"]["Resource"] == {"Fn::Sub": "${ArtifactBucket.Arn}/source/*"}


def test_template_outputs(tpl):
    outs = tpl["Outputs"]
    for key in ("FunctionUrl", "BucketName", "TableName", "RepoUri", "ProjectName"):
        assert key in outs, key
    assert outs["FunctionUrl"]["Condition"] == "HasImage"
    assert outs["FunctionUrl"]["Value"] == {"Fn::GetAtt": ["FunctionUrl", "FunctionUrl"]}
    assert outs["FunctionUrlX86"]["Condition"] == "HasX86Image"


def test_template_matches_deploy_script(tpl, dep):
    params, outs = set(tpl["Parameters"]), set(tpl["Outputs"])
    for spec in dep.ARCH_SPECS.values():
        assert spec.image_param in params
        assert {spec.project_output, spec.url_output, spec.function_output} <= outs
    d = dep.Deployer(object(), "inspectagent", "us-east-1", log=lambda m: None)
    assert set(d.base_parameters("", 5, dep.DeployOptions())) <= params
    assert len(TEMPLATE.read_bytes()) < dep.TEMPLATE_BODY_LIMIT
    # buildspec tag suffixes == deploy.py's expectations
    projects = {p["Properties"]["Environment"]["Type"]: p["Properties"] for p in resources_of(tpl, "AWS::CodeBuild::Project").values()}
    suffix = {t: {v["Name"]: v["Value"] for v in p["Environment"]["EnvironmentVariables"]}["TAG_SUFFIX"] for t, p in projects.items()}
    assert suffix == {"ARM_CONTAINER": dep.ARCH_SPECS["arm64"].tag_suffix, "LINUX_CONTAINER": dep.ARCH_SPECS["x86_64"].tag_suffix}


def test_template_passes_cfn_lint():
    api = pytest.importorskip("cfnlint.api")
    matches = api.lint_file(TEMPLATE, config=api.ManualArgs(regions=["us-east-1"]))
    problems = [f"{m.rule.id} {m.message} (line {m.linenumber})" for m in matches if m.rule.id[0] in ("E", "W")]
    assert not problems, "\n".join(problems)


# ------------------------------------------------------------------------------------------------
# buildspec, Dockerfiles, lock file
# ------------------------------------------------------------------------------------------------
def test_buildspec_parses_and_builds_lambda_image():
    spec = yaml.safe_load(BUILDSPEC.read_text(encoding="utf-8"))
    assert str(spec["version"]) == "0.2"
    assert set(spec["phases"]) == {"pre_build", "build", "post_build"}
    dockerfile = spec["env"]["variables"]["DOCKERFILE"]
    assert (ROOT / dockerfile).is_file() and dockerfile == "deploy/Dockerfile.lambda"
    cmds = "\n".join(c for ph in spec["phases"].values() for c in ph["commands"])
    assert "docker build" in cmds and '-f "${DOCKERFILE}"' in cmds
    assert "--provenance=false" in cmds  # Lambda rejects attestation manifests
    assert 'docker push "${ECR_REPO_URI}:${IMAGE_TAG}${TAG_SUFFIX}"' in cmds
    assert 'docker push "${ECR_REPO_URI}:latest${TAG_SUFFIX}"' in cmds
    assert "${SAFE_GIT_SHA}${TAG_SUFFIX}" in cmds
    assert "cp -f deploy/.dockerignore .dockerignore" in cmds and DOCKERIGNORE.is_file()
    assert "aws ecr get-login-password" in cmds
    assert "set -e" not in cmds  # CodeBuild runs all commands in one shell; never change its options


def _instructions(path: Path) -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) of a Dockerfile, with line continuations joined."""
    out: list[tuple[str, str]] = []
    buf = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not buf and (not line or line.startswith("#")):
            continue
        if line.startswith("#"):
            continue
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        full = (buf + line).strip()
        buf = ""
        instr, _, args = full.partition(" ")
        out.append((instr.upper(), args.strip()))
    return out


def _copy_sources(args: str) -> list[str]:
    if args.startswith("["):
        tokens = json.loads(args)
    else:
        tokens = [t for t in shlex.split(args) if not t.startswith("--")]
    return tokens[:-1]


@pytest.mark.parametrize("name", ["Dockerfile.lambda", "Dockerfile"])
def test_dockerfile_copy_sources_exist_in_build_context(name, dep):
    ignore = dep.DockerIgnore.from_file(DOCKERIGNORE)
    context = set(dep.iter_context_files(ROOT, ignore))
    ins = _instructions(DEPLOY / name)
    copies = [args for i, args in ins if i == "COPY"]
    assert copies, "no COPY instructions"
    for args in copies:
        for src in _copy_sources(args):
            rel = src.rstrip("/")
            path = ROOT / rel
            assert path.exists(), f"{name}: COPY source {src} does not exist"
            if path.is_dir():
                in_context = (not ignore.is_ignored(rel)) or any(f.startswith(rel + "/") for f in context)
            else:
                in_context = rel in context
            assert in_context, f"{name}: COPY source {src} is excluded by deploy/.dockerignore"


def test_dockerfile_lambda_contract():
    ins = _instructions(DEPLOY / "Dockerfile.lambda")
    froms = [a for i, a in ins if i == "FROM"]
    assert froms == ["public.ecr.aws/lambda/python:3.13"]
    runs = " ".join(a for i, a in ins if i == "RUN")
    assert "pip install --no-cache-dir" in runs and "requirements.lock.txt" in runs
    cmd = [a for i, a in ins if i == "CMD"]
    assert cmd and json.loads(cmd[-1]) == ["inspectagent.api.lambda_handler.handler"]
    copied = " ".join(a for i, a in ins if i == "COPY")
    for d in ("inspectagent/", "samples/", "models/"):
        assert d in copied
    handler_file = ROOT / "inspectagent" / "api" / "lambda_handler.py"
    if handler_file.is_file():  # written by the API module; check the CMD target really exists
        assert re.search(r"^handler\s*=|^def handler\(", handler_file.read_text(encoding="utf-8"), re.M)


def test_dockerfile_generic_contract():
    ins = _instructions(DEPLOY / "Dockerfile")
    args = [a for i, a in ins if i == "ARG"]
    assert "PYTHON_IMAGE=python:3.13-slim" in args
    assert [a for i, a in ins if i == "FROM"] == ["${PYTHON_IMAGE}"]
    users = [a for i, a in ins if i == "USER"]
    assert users and users[-1] not in ("root", "0")
    assert "8080" in [a for i, a in ins if i == "EXPOSE"]
    cmd = json.loads([a for i, a in ins if i == "CMD"][-1])
    assert "uvicorn inspectagent.api.app:app" in " ".join(cmd) and "8080" in " ".join(cmd)
    assert "requirements.lock.txt" in " ".join(a for i, a in ins if i == "RUN")


def _pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name, sep, ver = line.partition("==")
        assert sep, f"{path.name}: unpinned requirement {line!r}"
        pins[re.sub(r"[-_.]+", "-", name.strip().lower())] = ver.strip()
    return pins


def test_lock_file_pins_all_direct_requirements():
    lock = ROOT / "requirements.lock.txt"
    header = lock.read_text(encoding="utf-8")
    assert "--python-platform aarch64-manylinux_2_28" in header and "--python-version 3.13" in header
    direct, locked = _pins(ROOT / "requirements.txt"), _pins(lock)
    for name, ver in direct.items():
        assert locked.get(name) == ver, f"{name}=={ver} not pinned identically in the lock"
    assert len(locked) > len(direct)  # transitive dependencies are pinned too
    assert "pywin32" not in locked  # Linux lock


# ------------------------------------------------------------------------------------------------
# deploy.py: CLI, ignore rules, source zip, helpers
# ------------------------------------------------------------------------------------------------
def test_deploy_help_works():
    res = subprocess.run([sys.executable, str(DEPLOY_PY), "--help"], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    for flag in ("--region", "--stack-name", "--bedrock-model", "--arch", "--destroy", "--skip-build"):
        assert flag in res.stdout
    assert "auto|none|<id>" in res.stdout and "both" in res.stdout


def test_dockerignore_semantics(dep):
    DI = dep.DockerIgnore
    ig = DI(["# comment", "", "*", "!keep", "keep/drop", "!keep/drop/again"])
    assert ig.is_ignored("a.txt") and ig.is_ignored("dir/x")
    assert not ig.is_ignored("keep/x.py")
    assert ig.is_ignored("keep/drop") and ig.is_ignored("keep/drop/y")
    assert not ig.is_ignored("keep/drop/again/z")
    assert DI(["!a", "*"]).is_ignored("a")  # later rules win
    log = DI(["**/*.log"])
    assert log.is_ignored("x.log") and log.is_ignored("a/b/x.log") and not log.is_ignored("x.logs")
    assert DI(["docs"]).is_ignored("docs/a/b.md")  # a rule matching a parent excludes the subtree
    assert DI(["/abs/"]).is_ignored("abs/f")
    assert DI(["file[0-9].txt"]).is_ignored("file5.txt") and not DI(["file[0-9].txt"]).is_ignored("fileA.txt")
    assert DI(["a?c"]).is_ignored("abc") and not DI(["a?c"]).is_ignored("a/c")
    assert DI(["*.py[cod]"]).is_ignored("m.pyc") and not DI(["*.py[cod]"]).is_ignored("m.py")
    allow = DI(["*", "!scripts/bench_cv.py"])
    assert not allow.can_skip_dir("scripts") and allow.can_skip_dir(".venv")
    with pytest.raises(ValueError):
        DI(["!"])


TREE = {
    "inspectagent/__init__.py": "v = 1\n",
    "inspectagent/api/app.py": "app = None\n",
    "inspectagent/__pycache__/x.cpython-313.pyc": "pyc",
    "inspectagent/vision/__pycache__/y.pyc": "pyc",
    "samples/samples.json": "[]",
    "samples/pcb1/a.jpg": "jpg",
    "samples/.env": "SECRET=1",
    "models/backbone_resnet18.onnx": "onnx",
    "models/pcb1/stat.npz": "npz",
    "results/eval_summary.json": "{}",
    "results/figures/f.png": "png",
    "results/cache/tmp.bin": "cache",
    "scripts/__init__.py": "",
    "scripts/bench_cv.py": "x = 1\n",
    "scripts/evaluate.py": "x = 2\n",
    "requirements.txt": "numpy==2.5.3\n",
    "requirements.lock.txt": "numpy==2.5.3\n",
    "requirements-dev.txt": "-r requirements.txt\n",
    "deploy/Dockerfile": "FROM x\n",
    "deploy/aws/template.yaml": "Resources: {}\n",
    "deploy/aws/.build/source.zip": "old zip",
    ".venv/Lib/site.py": "venv",
    ".git/HEAD": "ref",
    "data/raw/VisA.tar": "tar",
    "tests/test_x.py": "def test(): pass\n",
    "docs/report.md": "# r\n",
    "README.md": "# readme\n",
    ".env": "AWS_SECRET=1",
    "runs/trace.json": "{}",
}
EXPECTED_IN_ZIP = {
    "inspectagent/__init__.py", "inspectagent/api/app.py", "samples/samples.json", "samples/pcb1/a.jpg",
    "models/backbone_resnet18.onnx", "models/pcb1/stat.npz", "results/eval_summary.json", "results/figures/f.png",
    "scripts/__init__.py", "scripts/bench_cv.py", "requirements.txt", "requirements.lock.txt", "requirements-dev.txt",
    "deploy/Dockerfile", "deploy/.dockerignore", "deploy/aws/template.yaml",
}


def make_tree(root: Path) -> Path:
    for rel, content in TREE.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content.encode("utf-8"))  # exact bytes (no CRLF translation on Windows)
    (root / "deploy").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DOCKERIGNORE, root / "deploy" / ".dockerignore")
    return root


def test_source_zip_respects_dockerignore_and_is_deterministic(dep, work_dir):
    root = make_tree(work_dir / "repo")
    z1 = dep.build_source_zip(root, work_dir / "out" / "a.zip", root / "deploy" / ".dockerignore")
    with zipfile.ZipFile(z1.path) as zf:
        names = set(zf.namelist())
        infos = zf.infolist()
        assert zf.read("scripts/bench_cv.py") == b"x = 1\n"
    assert names == EXPECTED_IN_ZIP
    assert set(z1.files) == EXPECTED_IN_ZIP
    assert all(i.date_time == (1980, 1, 1, 0, 0, 0) and (i.external_attr >> 16) == 0o100644 for i in infos)
    assert z1.sha256 == hashlib.sha256(z1.path.read_bytes()).hexdigest() and z1.tag == f"src-{z1.sha256[:12]}"
    # same content -> byte-identical archive, even with new mtimes; content change -> new hash
    os.utime(root / "inspectagent" / "__init__.py", (1_700_000_000, 1_700_000_000))
    z2 = dep.build_source_zip(root, work_dir / "out" / "b.zip", root / "deploy" / ".dockerignore")
    assert z2.sha256 == z1.sha256
    (root / "inspectagent" / "__init__.py").write_bytes(b"v = 2\n")
    z3 = dep.build_source_zip(root, work_dir / "out" / "c.zip", root / "deploy" / ".dockerignore")
    assert z3.sha256 != z1.sha256
    # the archive being written inside the tree is never zipped into itself
    z4 = dep.build_source_zip(root, root / "inspectagent" / "self.zip", root / "deploy" / ".dockerignore")
    assert "inspectagent/self.zip" not in z4.files


def test_prefix_and_reserved_concurrency_helpers(dep):
    assert dep.prefix_from_stack_name("inspectagent") == "inspectagent"
    assert dep.prefix_from_stack_name("InspectAgent-Dev_2") == "inspectagent-dev-2"
    assert dep.prefix_from_stack_name("1-very-long-stack-name-for-tests-xyz") == "very-long-stack-name"
    with pytest.raises(dep.DeployError):
        dep.prefix_from_stack_name("9")
    ch = dep.choose_reserved_concurrency
    assert ch(1000, 1000, 5, 2) == 5
    assert ch(10, 10, 5, 1) == 0  # new account: quota 10, nothing can be reserved
    assert ch(1000, 105, 5, 2) == 0 and ch(1000, 110, 5, 2) == 5
    assert ch(1000, 1000, 0, 1) == 0


PROFILES = [
    {"inferenceProfileId": "us.anthropic.claude-3-5-haiku-20241022-v1:0", "status": "ACTIVE"},
    {"inferenceProfileId": "us.anthropic.claude-3-7-sonnet-20250219-v1:0", "status": "ACTIVE"},
    {"inferenceProfileId": "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "status": "ACTIVE"},
    {"inferenceProfileId": "global.anthropic.claude-sonnet-4-5-20250929-v1:0", "status": "ACTIVE"},
    {"inferenceProfileId": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "status": "ACTIVE"},
    {"inferenceProfileId": "us.anthropic.claude-opus-4-1-20250805-v1:0", "status": "ACTIVE"},
    {"inferenceProfileId": "us.amazon.nova-pro-v1:0", "status": "ACTIVE"},
    {"inferenceProfileId": "us.amazon.nova-micro-v1:0", "status": "ACTIVE"},
]
MODELS = [
    {"modelId": "amazon.nova-lite-v1:0", "inferenceTypesSupported": ["ON_DEMAND"], "outputModalities": ["TEXT"],
     "modelLifecycle": {"status": "ACTIVE"}},
    {"modelId": "amazon.nova-lite-v1:0:300k", "inferenceTypesSupported": ["PROVISIONED"], "outputModalities": ["TEXT"]},
    {"modelId": "anthropic.claude-3-haiku-20240307-v1:0", "inferenceTypesSupported": ["ON_DEMAND"],
     "outputModalities": ["TEXT"], "modelLifecycle": {"status": "LEGACY"}},
    {"modelId": "anthropic.claude-sonnet-4-5-20250929-v1:0", "inferenceTypesSupported": ["INFERENCE_PROFILE"],
     "outputModalities": ["TEXT"], "modelLifecycle": {"status": "ACTIVE"}},
]


def test_rank_bedrock_candidates(dep):
    ranked = dep.rank_bedrock_candidates(PROFILES, MODELS, "us-east-1")
    assert ranked[0] == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"  # newest generation, Sonnet over Haiku, geo profile
    assert ranked[1] == "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
    assert ranked[2] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert ranked.index("us.anthropic.claude-3-7-sonnet-20250219-v1:0") < ranked.index("us.anthropic.claude-3-5-haiku-20241022-v1:0")
    assert all("opus" not in m and "micro" not in m for m in ranked)  # too costly / no image input
    assert "anthropic.claude-3-haiku-20240307-v1:0" not in ranked and "amazon.nova-lite-v1:0:300k" not in ranked
    assert ranked.index("us.amazon.nova-pro-v1:0") > ranked.index("us.anthropic.claude-3-5-haiku-20241022-v1:0")
    assert ranked[-1] == "amazon.nova-lite-v1:0"
    assert dep.rank_bedrock_candidates([], [], "us-east-1") == []


# ------------------------------------------------------------------------------------------------
# deploy / redeploy / destroy against an in-memory AWS behind real boto3 clients
# ------------------------------------------------------------------------------------------------
class _AwsError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class FakeAws:
    """Minimal stateful AWS for deploy.py. botocore serializes + validates each request against the real API
    model; the ``before-call`` hook then answers here instead of sending it (same mechanism as botocore's Stubber)."""

    ACCOUNT = "123456789012"

    def __init__(self, region: str = "us-east-1", account_limit: int = 1000, fail_builds: bool = False,
                 write_during_delete: bool = False):
        self.region = region
        self.account_limit = account_limit
        self.fail_builds = fail_builds
        self.write_during_delete = write_during_delete
        self.calls: list[tuple[str, str, dict]] = []
        self.stacks: dict[str, dict] = {}
        self.objects: dict[tuple[str, str], int] = {}
        self.images: dict[tuple[str, str], str] = {}
        self.builds: dict[str, dict] = {}
        self.http_calls: list[str] = []
        self._n = 0
        tpl = load_template()
        self.tpl_params = {k: str(v.get("Default", "")) for k, v in tpl["Parameters"].items()}
        self.tpl_outputs = set(tpl["Outputs"])
        self.template_body = TEMPLATE.read_text(encoding="utf-8")

    # -- wiring ------------------------------------------------------------------------------
    def session(self):
        boto3 = pytest.importorskip("boto3")
        s = boto3.Session(aws_access_key_id="testing", aws_secret_access_key="testing", region_name=self.region)
        s.events.register("before-parameter-build.*.*", self._capture)
        s.events.register("before-call.*.*", self._answer)
        return s

    @staticmethod
    def _capture(params, context, **_kw):
        context["fake_api_params"] = params

    def _answer(self, model, context, **_kw):
        from botocore.awsrequest import AWSResponse

        service = model.service_model.service_name
        params = context.get("fake_api_params", {})
        self.calls.append((service, model.name, params))
        handler = getattr(self, f"_{service.replace('-', '_')}_{model.name}", None)
        try:
            if handler is None:
                raise _AwsError("NotImplemented", f"fake has no {service}.{model.name}", 501)
            body = handler(params)
            return AWSResponse("https://fake.local", 200, {}, None), {**body, "ResponseMetadata": {"HTTPStatusCode": 200}}
        except _AwsError as e:
            parsed = {"Error": {"Code": e.code, "Message": e.message}, "ResponseMetadata": {"HTTPStatusCode": e.status}}
            return AWSResponse("https://fake.local", e.status, {}, None), parsed

    def ops(self, service: str | None = None) -> list[str]:
        return [op for s, op, _ in self.calls if service is None or s == service]

    def params_of(self, op: str) -> list[dict]:
        return [p for _, o, p in self.calls if o == op]

    def _id(self) -> str:
        self._n += 1
        return f"{self._n:06d}"

    @staticmethod
    def _now() -> datetime:
        return datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)

    # -- STS / Lambda account ---------------------------------------------------------------
    def _sts_GetCallerIdentity(self, p):
        return {"Account": self.ACCOUNT, "Arn": f"arn:aws:iam::{self.ACCOUNT}:user/tester", "UserId": "AIDATEST"}

    def _lambda_GetAccountSettings(self, p):
        return {"AccountLimit": {"ConcurrentExecutions": self.account_limit, "UnreservedConcurrentExecutions": self.account_limit,
                                 "TotalCodeSize": 80530636800, "CodeSizeUnzipped": 262144000, "CodeSizeZipped": 52428800},
                "AccountUsage": {"TotalCodeSize": 0, "FunctionCount": 0}}

    def _lambda_GetFunctionConcurrency(self, p):
        for st in self.stacks.values():
            if p["FunctionName"] in (st["outputs"].get("FunctionName"), st["outputs"].get("FunctionNameX86")):
                return {"ReservedConcurrentExecutions": int(st["params"]["ReservedConcurrency"])}
        raise _AwsError("ResourceNotFoundException", "Function not found", 404)

    # -- CloudFormation ---------------------------------------------------------------------
    def _event(self, st: dict, status: str, logical: str | None = None, rtype: str = "AWS::CloudFormation::Stack", reason: str = ""):
        ev = {"StackId": st["id"], "EventId": self._id(), "StackName": st["name"], "LogicalResourceId": logical or st["name"],
              "ResourceType": rtype, "ResourceStatus": status, "Timestamp": self._now() + timedelta(seconds=self._n)}
        if reason:
            ev["ResourceStatusReason"] = reason
        st["events"].insert(0, ev)

    def _resolve_params(self, given: list[dict], previous: dict | None) -> dict[str, str]:
        out = dict(self.tpl_params)
        for item in given:
            key = item["ParameterKey"]
            if key not in self.tpl_params:
                raise _AwsError("ValidationError", f"Parameters: [{key}] do not exist in the template")
            if item.get("UsePreviousValue"):
                if previous is None or key not in previous:
                    raise _AwsError("ValidationError", f"UsePreviousValue for unknown parameter {key}")
                out[key] = previous[key]
            else:
                out[key] = item.get("ParameterValue", "")
        return out

    def _outputs(self, params: dict[str, str]) -> dict[str, str]:
        pre, acct, reg = params["StackPrefix"], self.ACCOUNT, self.region
        outs = {
            "BucketName": f"{pre}-artifacts-{acct}-{reg}", "TableName": f"{pre}-table", "RepoName": pre,
            "RepoUri": f"{acct}.dkr.ecr.{reg}.amazonaws.com/{pre}", "ProjectName": f"{pre}-build-arm64",
            "ProjectNameX86": f"{pre}-build-x86_64", "BuildLogGroupName": f"/aws/codebuild/{pre}",
        }
        if params["ImageUri"]:
            outs |= {"FunctionUrl": f"https://{pre}-arm64.lambda-url.{reg}.on.aws/", "FunctionName": pre}
        if params["ImageUriX86"]:
            outs |= {"FunctionUrlX86": f"https://{pre}-x86.lambda-url.{reg}.on.aws/", "FunctionNameX86": f"{pre}-x86"}
        assert set(outs) <= self.tpl_outputs, set(outs) - self.tpl_outputs
        return outs

    def _stack(self, name: str) -> dict:
        st = self.stacks.get(name)
        if st is None:
            raise _AwsError("ValidationError", f"Stack with id {name} does not exist")
        return st

    def _advance(self, st: dict) -> None:
        if not st["status"].endswith("_IN_PROGRESS"):
            return
        if st["pending"] > 0:
            st["pending"] -= 1
            return
        if st["status"] == "DELETE_IN_PROGRESS":
            bucket = st["outputs"]["BucketName"]
            if any(b == bucket for b, _k in self.objects):
                st["status"] = "DELETE_FAILED"
                self._event(st, "DELETE_FAILED", "ArtifactBucket", "AWS::S3::Bucket", "The bucket you tried to delete is not empty")
                self._event(st, "DELETE_FAILED")
            else:
                self._event(st, "DELETE_COMPLETE")
                del self.stacks[st["name"]]
            return
        st["status"] = st["status"].replace("_IN_PROGRESS", "_COMPLETE")
        st["outputs"] = self._outputs(st["params"])
        self._event(st, st["status"])

    def _cloudformation_DescribeStacks(self, p):
        st = self._stack(p["StackName"])
        self._advance(st)
        st = self._stack(p["StackName"])
        return {"Stacks": [{
            "StackName": st["name"], "StackId": st["id"], "CreationTime": self._now(), "StackStatus": st["status"],
            "Parameters": [{"ParameterKey": k, "ParameterValue": v} for k, v in st["params"].items()],
            "Outputs": [{"OutputKey": k, "OutputValue": v} for k, v in st["outputs"].items()],
        }]}

    def _cloudformation_DescribeStackEvents(self, p):
        return {"StackEvents": list(self._stack(p["StackName"])["events"])}

    def _check_template(self, p):
        assert p["TemplateBody"] == self.template_body
        assert p["Capabilities"] == ["CAPABILITY_IAM"]

    def _cloudformation_CreateStack(self, p):
        if p["StackName"] in self.stacks:
            raise _AwsError("AlreadyExistsException", "Stack already exists")
        self._check_template(p)
        params = self._resolve_params(p.get("Parameters", []), None)
        st = {"name": p["StackName"], "id": f"arn:aws:cloudformation:{self.region}:{self.ACCOUNT}:stack/{p['StackName']}/x",
              "status": "CREATE_IN_PROGRESS", "pending": 1, "params": params, "outputs": {}, "events": []}
        self.stacks[st["name"]] = st
        self._event(st, "CREATE_IN_PROGRESS")
        self._event(st, "CREATE_IN_PROGRESS", "ArtifactBucket", "AWS::S3::Bucket")
        return {"StackId": st["id"]}

    def _cloudformation_UpdateStack(self, p):
        st = self._stack(p["StackName"])
        self._check_template(p)
        params = self._resolve_params(p.get("Parameters", []), st["params"])
        if params == st["params"]:
            raise _AwsError("ValidationError", "No updates are to be performed.")
        st.update(params=params, status="UPDATE_IN_PROGRESS", pending=1)
        self._event(st, "UPDATE_IN_PROGRESS")
        return {"StackId": st["id"]}

    def _cloudformation_DeleteStack(self, p):
        st = self.stacks.get(p["StackName"])
        if st is None:
            return {}
        if self.write_during_delete:  # the still-running function writes one more trace
            self.write_during_delete = False
            self.objects[(st["outputs"]["BucketName"], "traces/late.json")] = 2
        st.update(status="DELETE_IN_PROGRESS", pending=1)
        self._event(st, "DELETE_IN_PROGRESS")
        return {}

    # -- S3 ---------------------------------------------------------------------------------
    def _s3_HeadObject(self, p):
        size = self.objects.get((p["Bucket"], p["Key"]))
        if size is None:
            raise _AwsError("404", "Not Found", 404)
        return {"ContentLength": size}

    def _s3_PutObject(self, p):
        body = p["Body"]
        data = body.read() if hasattr(body, "read") else bytes(body)
        self.objects[(p["Bucket"], p["Key"])] = len(data)
        return {"ETag": '"fake"'}

    def _s3_ListObjectVersions(self, p):
        keys = sorted(k for b, k in self.objects if b == p["Bucket"])
        return {"Versions": [{"Key": k, "VersionId": "null", "IsLatest": True} for k in keys], "IsTruncated": False}

    def _s3_DeleteObjects(self, p):
        for o in p["Delete"]["Objects"]:
            self.objects.pop((p["Bucket"], o["Key"]), None)
        return {"Deleted": [{"Key": o["Key"]} for o in p["Delete"]["Objects"]]}

    # -- CodeBuild / ECR / Logs -------------------------------------------------------------
    def _codebuild_StartBuild(self, p):
        st = next(iter(self.stacks.values()))
        assert p["projectName"] in (st["outputs"]["ProjectName"], st["outputs"]["ProjectNameX86"])
        bucket, key = p["sourceLocationOverride"].split("/", 1)
        assert (bucket, key) in self.objects, "build started before the source was uploaded"
        env = {e["name"]: e["value"] for e in p["environmentVariablesOverride"]}
        bid = f"{p['projectName']}:{self._id()}"
        suffix = "-x86_64" if p["projectName"].endswith("x86_64") else ""
        self.builds[bid] = {"polls": 1, "env": env, "suffix": suffix, "repo": st["outputs"]["RepoName"]}
        return {"build": {"id": bid, "buildStatus": "IN_PROGRESS", "currentPhase": "SUBMITTED"}}

    def _codebuild_BatchGetBuilds(self, p):
        out = []
        for bid in p["ids"]:
            b = self.builds[bid]
            if b["polls"] > 0:
                b["polls"] -= 1
                out.append({"id": bid, "buildStatus": "IN_PROGRESS", "currentPhase": "BUILD"})
                continue
            status = "FAILED" if self.fail_builds else "SUCCEEDED"
            if status == "SUCCEEDED" and "done" not in b:
                tag = b["env"]["IMAGE_TAG"] + b["suffix"]
                digest = "sha256:" + hashlib.sha256(tag.encode()).hexdigest()
                self.images[(b["repo"], tag)] = digest
                self.images[(b["repo"], "latest" + b["suffix"])] = digest
                b["done"] = True
            item = {"id": bid, "buildStatus": status, "currentPhase": "COMPLETED", "startTime": self._now(),
                    "endTime": self._now() + timedelta(seconds=95),
                    "logs": {"groupName": "/aws/codebuild/x", "streamName": "s/1", "deepLink": "https://console/logs"}}
            if status == "FAILED":
                item["phases"] = [{"phaseType": "BUILD", "phaseStatus": "FAILED", "contexts": [{"message": "exit status 1"}]}]
            out.append(item)
        return {"builds": out}

    def _ecr_DescribeImages(self, p):
        tag = p["imageIds"][0]["imageTag"]
        digest = self.images.get((p["repositoryName"], tag))
        if digest is None:
            raise _AwsError("ImageNotFoundException", f"image {tag} not found")
        return {"imageDetails": [{"imageDigest": digest, "imageTags": [tag], "repositoryName": p["repositoryName"]}]}

    def _logs_GetLogEvents(self, p):
        return {"events": [{"timestamp": 1, "message": "pip: error: no matching distribution"}]}

    # -- Bedrock ----------------------------------------------------------------------------
    def _bedrock_ListInferenceProfiles(self, p):
        assert p.get("typeEquals") == "SYSTEM_DEFINED"
        return {"inferenceProfileSummaries": [
            {"inferenceProfileId": pr["inferenceProfileId"], "inferenceProfileName": pr["inferenceProfileId"],
             "inferenceProfileArn": f"arn:aws:bedrock:{self.region}:{self.ACCOUNT}:inference-profile/{pr['inferenceProfileId']}",
             "models": [], "status": "ACTIVE", "type": "SYSTEM_DEFINED"} for pr in PROFILES]}

    def _bedrock_ListFoundationModels(self, p):
        return {"modelSummaries": [{"modelArn": f"arn:aws:bedrock:{self.region}::foundation-model/{m['modelId']}", **m} for m in MODELS]}

    def _bedrock_runtime_Converse(self, p):
        if p["modelId"] != "us.anthropic.claude-haiku-4-5-20251001-v1:0":  # only Haiku 4.5 has access here
            raise _AwsError("AccessDeniedException", "You don't have access to the model", 403)
        return {"output": {"message": {"role": "assistant", "content": [{"text": "OK"}]}}, "stopReason": "end_turn",
                "usage": {"inputTokens": 4, "outputTokens": 1, "totalTokens": 5}, "metrics": {"latencyMs": 100}}

    # -- Function URL -----------------------------------------------------------------------
    def http_get(self, url: str, timeout: float) -> tuple[int, bytes]:
        self.http_calls.append(url)
        if url.endswith("/healthz"):
            arch = "x86_64" if "-x86." in url else "aarch64"
            body = {"ok": True, "version": "0.1.0", "cv": {"opencv_version": "5.0.0", "arch": arch, "runtime": "lambda"}}
            if len(self.http_calls) == 1:  # first call: cold start still initialising
                return 503, b'{"message": "Service Unavailable"}'
            return 200, json.dumps(body).encode()
        return 404, b"{}"


def _deployer(dep, fake: FakeAws, work_dir: Path, logs: list[str]):
    d = dep.Deployer(fake.session(), "inspectagent", fake.region, log=logs.append, sleep=lambda s: None,
                     http_get=fake.http_get, poll_seconds=0.0)
    d.build_dir = work_dir / "build"
    d.results_dir = work_dir / "results"
    return d


def test_deploy_redeploy_destroy_flow_against_fake_aws(dep, work_dir):
    pytest.importorskip("boto3")
    fake = FakeAws(write_during_delete=True)
    root = make_tree(work_dir / "repo")
    logs: list[str] = []
    d = _deployer(dep, fake, work_dir, logs)
    opts = dep.DeployOptions(archs=("arm64", "x86_64"), bedrock_model="auto", root=root, use_planner_picker=False)

    res = d.deploy(opts)

    # phase 1 created the stack without images, phase 2 set digest-pinned images for both architectures
    create = fake.params_of("CreateStack")[0]
    p1 = {x["ParameterKey"]: x.get("ParameterValue") for x in create["Parameters"]}
    assert p1["ImageUri"] == "" and p1["ImageUriX86"] == "" and p1["StackPrefix"] == "inspectagent"
    assert p1["ReservedConcurrency"] == "5"
    assert p1["BedrockModelId"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # best model the account can call
    stack = fake.stacks["inspectagent"]
    assert stack["status"] == "UPDATE_COMPLETE"
    repo_uri = f"{fake.ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/inspectagent"
    assert re.fullmatch(re.escape(repo_uri) + r"@sha256:[0-9a-f]{64}", stack["params"]["ImageUri"])
    assert stack["params"]["ImageUriX86"].startswith(repo_uri + "@sha256:")
    assert stack["params"]["ImageUri"] != stack["params"]["ImageUriX86"]
    # source zip uploaded once, content-addressed, then one build per architecture with the right overrides
    uploads = [k for (_b, k) in fake.objects if k.startswith("source/")]
    assert len(uploads) == 1 and re.fullmatch(r"source/src-[0-9a-f]{12}\.zip", uploads[0])
    builds = fake.params_of("StartBuild")
    assert sorted(b["projectName"] for b in builds) == ["inspectagent-build-arm64", "inspectagent-build-x86_64"]
    for b in builds:
        env = {e["name"]: e["value"] for e in b["environmentVariablesOverride"]}
        assert env["IMAGE_TAG"] == uploads[0][len("source/"):-len(".zip")]
        assert "GIT_SHA" in env and b["sourceTypeOverride"] == "S3"
    # smoke test retried through the cold start and hit /healthz on both URLs
    assert res.urls == {"arm64": "https://inspectagent-arm64.lambda-url.us-east-1.on.aws/",
                        "x86_64": "https://inspectagent-x86.lambda-url.us-east-1.on.aws/"}
    assert res.health["arm64"]["cv"]["arch"] == "aarch64" and res.health["x86_64"]["cv"]["arch"] == "x86_64"
    assert sum(u.endswith("/healthz") for u in fake.http_calls) == 3
    summary = json.loads((work_dir / "build" / "last_deploy.json").read_text(encoding="utf-8"))
    assert summary["urls"] == res.urls
    assert any("healthz OK" in m for m in logs) and any("phase BUILD" in m for m in logs)

    # redeploy with identical source: no new stack, upload or build; both updates are no-ops
    n_calls = len(fake.calls)
    res2 = d.deploy(opts)
    new_ops = [op for _s, op, _p in fake.calls[n_calls:]]
    assert "CreateStack" not in new_ops and "StartBuild" not in new_ops and "PutObject" not in new_ops
    assert new_ops.count("UpdateStack") == 2 and fake.stacks["inspectagent"]["status"] == "UPDATE_COMPLETE"
    assert res2.images == res.images and sum("already up to date" in m for m in logs) == 2

    # --skip-build reuses latest<suffix> (same digests) and keeps the x86 function when only arm64 is requested
    res3 = d.deploy(dep.DeployOptions(archs=("arm64",), bedrock_model="none", skip_build=True, root=root, use_planner_picker=False))
    assert res3.images["arm64"] == res.images["arm64"]
    assert fake.stacks["inspectagent"]["params"]["BedrockModelId"] == ""
    assert fake.stacks["inspectagent"]["params"]["ImageUriX86"] == res.images["x86_64"]

    # destroy: bucket emptied (and again after a late write made the first deletion fail), stack gone
    d.destroy()
    assert "inspectagent" not in fake.stacks
    assert not [k for (b, k) in fake.objects]
    assert fake.ops("cloudformation").count("DeleteStack") == 2
    d.destroy()  # idempotent
    assert any("nothing to destroy" in m for m in logs)


def test_bedrock_auto_uses_planner_picker_when_available(dep):
    pytest.importorskip("boto3")
    try:
        from inspectagent.agent.planner_bedrock import pick_bedrock_model  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - written concurrently by the agent module
        pytest.skip(f"planner_bedrock.pick_bedrock_model not importable: {exc}")
    fake = FakeAws()
    logs: list[str] = []
    d = dep.Deployer(fake.session(), "inspectagent", fake.region, log=logs.append, sleep=lambda s: None, poll_seconds=0.0)
    picked = d.resolve_bedrock_model("auto", verify=True, use_planner_picker=True)
    assert picked == "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # the only candidate the fake grants access to
    assert any("planner_bedrock.pick_bedrock_model" in m for m in logs) or any("built-in picker" in m for m in logs)
    assert d.resolve_bedrock_model("none") == "" and d.resolve_bedrock_model("my.model-id") == "my.model-id"


def test_deploy_new_account_failed_build_and_skip_build_errors(dep, work_dir):
    pytest.importorskip("boto3")
    root = make_tree(work_dir / "repo")
    logs: list[str] = []
    fake = FakeAws(account_limit=10, fail_builds=True)
    d = _deployer(dep, fake, work_dir, logs)
    with pytest.raises(dep.DeployError, match="image build failed"):
        d.deploy(dep.DeployOptions(archs=("arm64",), bedrock_model="none", root=root, use_planner_picker=False))
    p1 = {x["ParameterKey"]: x.get("ParameterValue") for x in fake.params_of("CreateStack")[0]["Parameters"]}
    assert p1["ReservedConcurrency"] == "0"  # quota 10: reserving would be rejected by Lambda
    assert any("no matching distribution" in m for m in logs) and any("full log:" in m for m in logs)
    assert "UpdateStack" not in fake.ops()  # phase 2 never ran

    fake2 = FakeAws()
    d2 = _deployer(dep, fake2, work_dir, [])
    with pytest.raises(dep.DeployError, match="--skip-build"):
        d2.deploy(dep.DeployOptions(archs=("arm64",), bedrock_model="none", skip_build=True, root=root, use_planner_picker=False))
