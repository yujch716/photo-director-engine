"""세션 기반 저장 경로 해석 (공용).

앱이 모든 API에 같은 session_id를 넘기면, 그 세션의 산출물을
drone-data/<session_id>/<step>/ 아래로 모은다. session_id가 없으면
기존 경로(legacy)를 그대로 써서 하위호환을 유지한다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

DRONE_DATA_DIR = Path("drone-data")


def make_ts() -> str:
    """밀리초 타임스탬프 폴더명."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _safe_session(session_id: str) -> str:
    """경로 탈출 방지: 슬래시/역슬래시/.. 제거."""
    s = session_id.replace("/", "_").replace("\\", "_").replace("..", "_").strip()
    return s or "session"


def session_root(session_id: str, data_dir: Path | None = None) -> Path:
    """세션 루트 폴더 경로: <data>/<safe_session_id>/ (생성은 안 함)."""
    base = data_dir or DRONE_DATA_DIR
    return base / _safe_session(session_id)


def _unique(folder: Path) -> Path:
    """이미 있으면 _1, _2 ... 붙여 유일한 경로 반환(생성은 안 함)."""
    if not folder.exists():
        return folder
    n = 1
    while (folder.parent / f"{folder.name}_{n}").exists():
        n += 1
    return folder.parent / f"{folder.name}_{n}"


def resolve_save_dir(
    session_id: str | None,
    step: str,
    legacy: Path,
    data_dir: Path | None = None,
) -> Path:
    """저장 폴더를 정해 생성 후 반환한다.

    session_id 있으면: <data>/<session_id>/<step>/  (고정 이름, 이미 있으면 재사용).
      - step은 "3_detail-move/lateral"처럼 중첩 경로 가능.
      - 같은 세션의 여러 단계가 한 폴더를 공유할 수 있음(예: scan-peak/scan-result → 2_scan).
      - 같은 step 재호출 시 파일은 덮어쓰기(고정 이름 유지).
    없으면:            legacy 경로               (중복 시 _n, 기존 동작).
    """
    base = data_dir or DRONE_DATA_DIR
    if session_id and session_id.strip():
        folder = base / _safe_session(session_id) / step
        folder.mkdir(parents=True, exist_ok=True)  # 고정 이름 폴더 재사용
    else:
        folder = _unique(legacy)
        folder.mkdir(parents=True)
    return folder


def _next_index(parent: Path) -> int:
    """parent 아래 숫자 폴더(1,2,3...) 중 다음 번호. 없으면 1."""
    parent.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name) for p in parent.iterdir() if p.is_dir() and p.name.isdigit()]
    return (max(nums) + 1) if nums else 1


def resolve_indexed_save_dir(
    session_id: str | None,
    step: str,
    legacy: Path,
    data_dir: Path | None = None,
) -> Path:
    """반복 호출용: 세션이면 <data>/<session_id>/<step>/<순번>/ (1,2,3... 순서대로).

    같은 세션에서 여러 번 호출되면 lateral/1, lateral/2 ... 처럼 쌓인다.
    세션이 없으면 legacy 경로(기존 타임스탬프 폴더).
    """
    base = data_dir or DRONE_DATA_DIR
    if session_id and session_id.strip():
        parent = base / _safe_session(session_id) / step
        folder = parent / str(_next_index(parent))
        folder.mkdir(parents=True)
    else:
        folder = _unique(legacy)
        folder.mkdir(parents=True)
    return folder
