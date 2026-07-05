"""드론 이동 오프셋(픽셀) 계산.

best 크롭 박스(1x 이미지 픽셀 좌표)와 1x 이미지 크기로부터
"현재 화면 중심 대비 best 크롭이 얼마나 벗어나 있는지"를 픽셀 오프셋으로 산출한다.

직교 성분(dx, dy)과 극좌표(dr, theta_deg)를 함께 낸다. 극좌표는 "θ 방향으로 dr만큼
한 번에" 이동하는 대각선 단일 이동을 표현한다. 실제 월드 좌표/속도 변환은 이후
AirSim 단계에서 별도로 처리한다.
"""

from __future__ import annotations

import math
from typing import Any


def compute_offset(
    best_source_box: list[float] | tuple[float, float, float, float] | None,
    image_1x_size: tuple[int, int] | None,
) -> dict[str, Any] | None:
    """best 크롭 박스와 1x 이미지 크기로 드론 이동 오프셋을 계산한다.

    줌(전진/후진)은 계산하지 않는다: 후보 창이 항상 2배율 크기(W/zoom_ratio)라
    best 박스도 늘 같은 크기 → 줌 변화가 없어 산출 불가. 팬(대각선 이동)만 낸다.

    Args:
        best_source_box: 1x 좌표계의 best 크롭 박스 [left, top, right, bottom].
            report.json의 "best_source_box_in_1x" 값을 그대로 넘기면 된다.
        image_1x_size: 1x 원본 이미지 크기 (W, H).

    Returns:
        dict 또는 None(입력 부족 시):
          -- 직교(픽셀) --
          dx        : best 박스 중심 − 이미지 중심 (픽셀). +면 오른쪽으로 이동해야 함
          dy        : best 박스 중심 − 이미지 중심 (픽셀). +면 아래로 이동해야 함
          dx_norm   : dx / W  (해상도 무관 정규화 값, -0.5 ~ 0.5)
          dy_norm   : dy / H
          -- 극좌표(거리·각도): "θ 방향으로 dr만큼 한 번에" = 대각선 단일 이동 --
          dr        : 이동 거리 = sqrt(dx^2 + dy^2) (픽셀)
          dr_norm   : dr / W  (해상도 무관 정규화)
          theta_deg : 이동 방향(도). 오른쪽(+x)을 0°로, 반시계 방향 양수.
                      0°=오른쪽, 90°=위, 180°=왼쪽, -90°(=270°)=아래.
                      (수학 표준 atan2. 화면 위가 +방향이 되도록 dy 부호를 뒤집어 계산)
    """
    if best_source_box is None or image_1x_size is None:
        return None

    x1, y1, x2, y2 = (float(v) for v in best_source_box)
    W = float(image_1x_size[0])
    H = float(image_1x_size[1])
    if W <= 0 or H <= 0:
        return None

    best_cx = (x1 + x2) / 2.0
    best_cy = (y1 + y2) / 2.0
    img_cx = W / 2.0
    img_cy = H / 2.0

    dx = best_cx - img_cx
    dy = best_cy - img_cy

    dr = math.hypot(dx, dy)
    # 오른쪽(+x)을 0°로, 반시계 양수(위=90°)로 재는 수학 표준 각도.
    # 이미지 좌표는 y가 아래로 +이므로 -dy로 뒤집어 atan2(-dy, dx)를 쓴다.
    theta_deg = math.degrees(math.atan2(-dy, dx))

    return {
        "dx": dx,
        "dy": dy,
        "dx_norm": dx / W,
        "dy_norm": dy / H,
        "dr": dr,
        "dr_norm": dr / W,
        "theta_deg": theta_deg,
    }
