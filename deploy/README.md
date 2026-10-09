# Deploying InspectAgent on AWS (Graviton Lambda + S3 + DynamoDB + Bedrock)

One command deploys the whole demo: a public HTTPS **Lambda Function URL** that serves the web UI and the
JSON API, running the OpenCV 5 agent in a **container image built natively for arm64 (Graviton)** by
CodeBuild. Traces and image artifacts go to **S3**, review items to **DynamoDB**, and the optional Bedrock
planner calls **Amazon Bedrock** (Converse API).

```
 browser / MCP client ──HTTPS──> Lambda Function URL (AuthType NONE, CORS *)
                                   │
                                   ▼
                       Lambda "inspectagent" (container, arm64, 3008 MB, 60 s, reserved concurrency 5)
                       FastAPI + Mangum, OpenCV 5, model bundles baked into the image
                         │            │                  │
                         ▼            ▼                  ▼
          S3 traces/ + artifacts/   DynamoDB pk/sk     Bedrock Converse (only if a model id is set)
          (private, SSE-S3, 30 d)   (on-demand)

 deploy.py ──zip (deploy/.dockerignore rules)──> S3 source/ ──> CodeBuild ARM_CONTAINER ──push──> ECR
                                                                 (x86 project for --arch both)
```

> **Status (honest):** no AWS credentials were available on the development machine, so this stack has
> **not been deployed from this repository yet**. Everything is validated statically: `cfn-lint 1.57.2`
> reports no errors or warnings for `aws/template.yaml`, and `tests/test_deploy_static.py` exercises
> `deploy.py` end-to-end (deploy, redeploy, `--skip-build`, failed build, destroy) against an in-memory AWS
> fake that sits behind real boto3 clients, so every API call is validated against the real service models.
> The first real deployment may still surface account-specific issues (quotas, model access); see
> *Troubleshooting*.

## Files

| file | purpose |
|---|---|
| `Dockerfile.lambda` | Lambda image: `public.ecr.aws/lambda/python:3.13`, installs `requirements.lock.txt` (binary wheels only), copies `inspectagent/`, `scripts/bench_cv.py`, `samples/`, `models/`, `results/`; `CMD ["inspectagent.api.lambda_handler.handler"]`. The build fails if the handler cannot be imported. |
| `Dockerfile` | Generic multi-arch image: `python:3.13-slim`, uvicorn on port 8080, non-root user `app`, health check on `/healthz`. For local use, ECS/Fargate or App Runner. |
| `.dockerignore` | Allow-list of the build context (code, samples, models, results, lock files, deploy/). Docker only reads it from the context root, so the buildspec copies it there; `deploy.py` zips the source with exactly these rules. |
| `aws/template.yaml` | CloudFormation: bucket, table, ECR repo, two CodeBuild projects, IAM roles, log groups, Lambda function(s) + Function URL(s) behind `HasImage` / `HasX86Image`. |
| `aws/buildspec.yml` | CodeBuild: `docker build` for the project's platform and push `:<src-hash>`, `:<git-sha>`, `:latest` (`-x86_64` suffix on the x86 project). |
| `aws/deploy.py` | Idempotent deployer (boto3), see below. |

## Prerequisites

* An AWS account and credentials for a principal that may create CloudFormation stacks, IAM roles, S3,
  DynamoDB, ECR, CodeBuild, Lambda and CloudWatch Logs resources (e.g. `AdministratorAccess` on a sandbox
  account). Configure them with `aws configure`, `aws sso login`, or environment variables.
* Python 3.11+ with `boto3` (the project venv already has it). Docker is **not** needed locally: images are
  built by CodeBuild, natively on Graviton.
* A region where Lambda arm64, CodeBuild ARM and Bedrock are available (e.g. `us-east-1`, `us-west-2`,
  `eu-central-1`).
* Optional, for the Bedrock planner: model access in the Bedrock console (Anthropic models need the one-time
  use-case form). Without it the stack still works with the deterministic rule planner.
* The model bundles in `models/` should be built first (`python scripts/build_models.py`), because they
  are baked into the image. Without bundles the API still starts, but inspections need a bundle.

## Deploy

From the repository root:

```bash
# Graviton (arm64) function, Bedrock model picked automatically
python deploy/aws/deploy.py --region us-east-1

# + an identical x86_64 function (<prefix>-x86) and a remote latency benchmark on both
python deploy/aws/deploy.py --region us-east-1 --arch both --bench 5

# redeploy the latest image already in ECR (no zip/upload/build), Bedrock disabled
python deploy/aws/deploy.py --skip-build --bedrock-model none

# tear everything down (empties the bucket first; ECR images are removed by EmptyOnDelete)
python deploy/aws/deploy.py --destroy --yes
```

What `deploy.py` does (every step prints its progress and is safe to re-run):

1. **Inputs**: caller identity; Bedrock model (`--bedrock-model auto` uses
   `inspectagent.agent.planner_bedrock.pick_bedrock_model` when importable, otherwise a built-in equivalent:
   newest Claude Sonnet/Haiku inference profile for the region, else Amazon Nova; Opus is skipped for cost.
   It confirms access with one 5-token Converse call per candidate, at most 5 candidates. Use `--no-verify-model` to skip the
   check); reserved concurrency (5 per function if the account quota allows, else 0).
2. **Stack, phase 1**: creates or updates the stack *without a new image*: bucket, table, repository, build
   projects, roles. Functions that already exist keep their current image (`UsePreviousValue`). A stack left
   in `ROLLBACK_COMPLETE` by a failed first attempt is deleted and re-created automatically.
3. **Images**: zips the repository with the `deploy/.dockerignore` rules into a deterministic archive. Identical
   source gives identical bytes and the same `src-<sha256[:12]>` tag. If ECR already has that tag,
   the build is skipped. Otherwise the zip goes to `s3://<bucket>/source/`, CodeBuild starts (ARM, and x86
   for `--arch both`), and the phases are streamed. On failure the tail of the build log is printed.
4. **Stack, phase 2**: sets `ImageUri` (and `ImageUriX86`) to the digest-pinned image (`repo@sha256:...`),
   which creates/updates the function(s), the Function URL(s) and their public-invoke permissions.
5. **Smoke test**: `GET <FunctionUrl>/healthz` (retried through the cold start) must return `{"ok": true}`;
   prints OpenCV version and `arch` (`aarch64` on Graviton).
6. **Summary**: prints the URL(s) and writes `deploy/aws/.build/last_deploy.json` (git-ignored).

| option | default | meaning |
|---|---|---|
| `--region` | `AWS_REGION` / profile / `us-east-1` | target region |
| `--profile` | – | named AWS CLI profile |
| `--stack-name` | `inspectagent` | stack name; also the resource prefix (lowercased) |
| `--bedrock-model auto\|none\|<id>` | `auto` | Bedrock model or inference-profile id for the planner |
| `--arch arm64\|x86_64\|both` | `arm64` | which function(s) to build and deploy |
| `--skip-build` | off | reuse `latest[-x86_64]` from ECR |
| `--reserved-concurrency N` | auto | per-function cap (0 = none) |
| `--allow-marketplace-subscribe` | off | let the function auto-enable Marketplace-backed Bedrock models (via Bedrock only) |
| `--build-compute small\|large` | `small` | CodeBuild compute size |
| `--bench N` | 0 | after deploying, `GET /api/bench?n=N` on each function, saved to `results/bench_lambda_<arch>.json` |
| `--no-smoke-test`, `--destroy`, `--yes`, `--poll-seconds` | | |

## Graviton vs x86

`--arch both` deploys the **same source** built natively on both architectures (`inspectagent` = arm64,
`inspectagent-x86` = x86_64, same memory/timeout/env). `--bench N` then runs `scripts/bench_cv.run_bench`
inside each function (camera capture, `assess_quality`, registration, fast and deep detectors, full
rule-planner loop: median/p90 ms) and stores both reports under `results/`. Locally:

```bash
python scripts/bench_cv.py --n 20          # -> results/bench_<arch>_<backend>.json (e.g. bench_x86_64_stock.json)
```

The bench uses the first bundled sample with a model bundle. Without bundles it falls back to a synthetic PCB
scene and a small, quickly fitted template (`source.kind == "synthetic"` in the report). Compare runs only
when their `source` matches.

## Resources and security

* **S3** `<prefix>-artifacts-<account>-<region>`: all public access blocked, bucket-owner-enforced, SSE-S3,
  TLS-only bucket policy; lifecycle: `traces/` and `artifacts/` expire after 30 days, `source/` after 14.
* **DynamoDB** `<prefix>-table`: `pk`/`sk` strings, on-demand, TTL attribute `expires_at`.
* **ECR** `<prefix>`: scan on push, keep the 10 most recent images, emptied on stack deletion.
* **CodeBuild** `<prefix>-build-arm64` (`ARM_CONTAINER`, `aws/codebuild/amazonlinux-aarch64-standard:3.0`) and
  `<prefix>-build-x86_64` (`LINUX_CONTAINER`, `aws/codebuild/amazonlinux-x86_64-standard:5.0`), privileged
  (Docker), S3 zip source; role can read `source/*`, push to this repository only, and write its log group.
* **Lambda role** (least privilege): write to its own log groups; `s3:GetObject`/`PutObject` on `traces/*` and
  `artifacts/*` only; DynamoDB item CRUD + Query on this table only; `bedrock:InvokeModel` (+ `...WithResponseStream`) on foundation-model and inference-profile ARNs **only when a Bedrock model id is set**.
  The Converse API is authorized by `bedrock:InvokeModel`. There is no `bedrock:Converse` IAM action, and
  cfn-lint rejects it. When a model id is set, the role can also call the read-only `bedrock:ListFoundationModels` /
  `bedrock:ListInferenceProfiles`, so a `BedrockModelId` of `auto` can be resolved at runtime. No
  Marketplace permissions unless `--allow-marketplace-subscribe`.
* **Function URL** is public (`AuthType NONE`, CORS `*`) for the demo. Anyone with the URL can run inspections.
  Reserved concurrency caps parallel executions and cost. Run `--destroy` after the demo, or switch to
  `AWS_IAM` auth for anything beyond a demo.
* Log groups `/aws/lambda/<prefix>[-x86]` and `/aws/codebuild/<prefix>` keep 14 days.

## Local container builds

```bash
cp deploy/.dockerignore .dockerignore
# Lambda image + Runtime Interface Emulator (built into the base image)
docker build -f deploy/Dockerfile.lambda -t inspectagent:lambda .
docker run --rm -p 9000:8080 -e INSPECTAGENT_STORAGE=local inspectagent:lambda
# generic image
docker build -f deploy/Dockerfile -t inspectagent:local .
docker run --rm -p 8080:8080 inspectagent:local        # http://localhost:8080/
```

## Dependency lock

`requirements.lock.txt` is the full transitive lock for Linux py3.13, used by both Dockerfiles:

```bash
uv pip compile requirements.txt --python-version 3.13 --python-platform aarch64-manylinux_2_28 -o requirements.lock.txt
uv pip compile requirements.txt --python-version 3.13 --python-platform x86_64-manylinux_2_28 -o /tmp/lock_x86.txt   # must resolve identically
```

Both platforms currently resolve to the same 40 pins, and binary wheels exist for all of them on aarch64 and
x86_64 (checked with `uv pip install --dry-run --only-binary :all:`).

## Static validation

```bash
.venv/Scripts/cfn-lint deploy/aws/template.yaml                     # cfn-lint 1.57.2: no errors or warnings
.venv/Scripts/python -m pytest tests/test_deploy_static.py tests/test_bench.py -q
```

## Troubleshooting

* **`ReservedConcurrentExecutions ... below its minimum value`**: the account's concurrency quota is too small.
  `deploy.py` detects this and uses 0, or you can pass `--reserved-concurrency 0` / request a quota increase.
* **Stack `ROLLBACK_COMPLETE`** after a failed first deploy: re-run `deploy.py`. It deletes the failed stack and
  starts again. The failing resource and reason were printed with the stack events.
* **CodeBuild fails**: the last 40 log lines and a console link are printed. Common causes are an ECR Public
  rate limit (retry) or a missing `requirements.lock.txt`.
* **"image manifest ... not supported"** from Lambda: the image was pushed with provenance attestations. The
  buildspec disables them (`--provenance=false`) whenever buildx is available.
* **Bedrock `AccessDeniedException`**: enable model access for the chosen model, or deploy with `--bedrock-model
  none`. The planner falls back to the rule planner when Bedrock fails.
* **`UPDATE_ROLLBACK_FAILED` / `DELETE_FAILED`**: fix the stack in the CloudFormation console (continue rollback or
  retry the delete), or run `--destroy`. A destroy that hits "bucket not empty" because of a late write empties the
  bucket again and retries once.
