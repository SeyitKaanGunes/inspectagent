"""Shared pytest configuration and fixtures (offline, synthetic data only).

* Puts the project root on ``sys.path`` so ``import inspectagent`` / ``import tests.synth`` work
  without installing the package.
* ``board``: one normal synthetic PCB (``tests.synth.make_board(0)``, BGR uint8 900x600).
* ``defective_board``: ``(image, mask)`` — ``make_board(1)`` with an injected ``scratch``
  defect (``tests.synth.inject_defect(..., "scratch", seed=0)``), mask uint8 0/255, scene frame.
  The tuple matches ``inject_defect``'s return value; ``defect_image`` / ``defect_mask`` give the
  parts separately.
Fixtures return fresh copies, so tests may modify them freely.

If pytest's default temp root (``<tmp>/pytest-of-<user>``) exists but is not accessible (seen on
the Windows dev box, where it was created by another process), ``tmp_path`` would fail for every
test; ``pytest_configure`` then points ``--basetemp`` at a private per-process directory that is
removed at the end of the session. An explicit ``--basetemp`` always wins.
"""

from __future__ import annotations

import getpass
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.synth import inject_defect, make_board  # noqa: E402

_OWN_BASETEMP: Path | None = None


def _default_tmp_root_unusable() -> bool:
    """True when pytest's default ``<tmp>/pytest-of-<user>`` exists but cannot be used.

    On the Windows dev box that directory was created by another (elevated) process and is not
    listable, which makes every ``tmp_path`` fixture fail with ``PermissionError``.
    """
    try:
        user = getpass.getuser()
    except Exception:
        user = "unknown"
    root = Path(tempfile.gettempdir()) / f"pytest-of-{user}"
    if not root.exists():
        return False
    try:
        next(iter(os.scandir(root)), None)
        probe = root / f".probe-{os.getpid()}"
        probe.mkdir()
        probe.rmdir()
        return False
    except OSError:
        return True


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Fall back to a private ``--basetemp`` when the default temp root is unusable."""
    global _OWN_BASETEMP
    if getattr(config.option, "basetemp", None) or not _default_tmp_root_unusable():
        return
    _OWN_BASETEMP = Path(tempfile.gettempdir()) / f"inspectagent-pytest-{os.getpid()}"
    config.option.basetemp = str(_OWN_BASETEMP)


def pytest_unconfigure(config: pytest.Config) -> None:
    if _OWN_BASETEMP is not None:
        shutil.rmtree(_OWN_BASETEMP, ignore_errors=True)


@pytest.fixture(scope="session")
def _board_cache() -> np.ndarray:
    return make_board(0)


@pytest.fixture(scope="session")
def _defect_cache() -> tuple[np.ndarray, np.ndarray]:
    return inject_defect(make_board(1), "scratch", seed=0)


@pytest.fixture
def board(_board_cache: np.ndarray) -> np.ndarray:
    """A normal synthetic board: ``tests.synth.make_board(0)``."""
    return _board_cache.copy()


@pytest.fixture
def defective_board(_defect_cache: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``(image, mask)`` of ``make_board(1)`` with an injected scratch (``inject_defect(..., seed=0)``)."""
    img, mask = _defect_cache
    return img.copy(), mask.copy()


@pytest.fixture
def defect_image(defective_board: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Image part of ``defective_board``."""
    return defective_board[0]


@pytest.fixture
def defect_mask(defective_board: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Ground-truth mask (uint8 0/255) of ``defective_board``."""
    return defective_board[1]


@pytest.fixture
def project_root() -> Path:
    """Absolute path of the repository root."""
    return ROOT
