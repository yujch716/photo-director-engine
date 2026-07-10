"""화면 픽셀 오프셋 계산 (드론 좌표계 비종속).

best 크롭 박스(1x 이미지 픽셀 좌표)와 1x 이미지 크기로부터
"피사체(best 구도)가 화면 중심 대비 어디에 있는지"를 순수 화면 좌표로 산출한다.

서버는 여기까지만 한다: "화면에서 피사체가 어느 방향/거리"인지.
특정 드론(AirSim ENU forward, DJI 등)으로의 이동 방향/전후진 변환은 하지 않는다.
그 변환은 각 클라이언트(앱)가 자기 좌표계에 맞춰 담당한다.

좌표계: 화면 픽셀 그대로. x=오른쪽 +, y=아래 + (부호 뒤집기 없음).
직교 성분(dx, dy)과 극좌표(dr, theta_deg)를 함께 낸다.
"""

from __future__ import annotations

import math
from typing import Any


def compute_offset(
    best_source_box: list[float] | tuple[float, float, float, float] | None,
    image_1x_size: tuple[int, int] | None,
) -> dict[str, Any] | None:
    """best 크롭 박스와 1x 이미지 크기로 '화면 픽셀 오프셋'을 계산한다.

    피사체(best 구도)가 화면 중심에서 얼마나(dr) 어느 방향(theta_deg)으로 벗어났는지를
    순수 화면 좌표로만 낸다. 특정 드론의 forward/전후진 등으로 변환하지 않는다(클라이언트 몫).
    줌(전진/후진)도 계산하지 않는다: 후보 창이 항상 1.4배율 크기라 best 박스도 늘 같은 크기.

    Args:
        best_source_box: 1x 좌표계의 best 크롭 박스 [left, top, right, bottom].
            report.json의 "best_source_box_in_1x" 값을 그대로 넘기면 된다.
        image_1x_size: 1x 원본 이미지 크기 (W, H).

    Returns:
        dict 또는 None(입력 부족 시). 모두 '화면 픽셀 좌표'(x=오른쪽+, y=아래+) 기준:
          -- 직교(픽셀) --
          dx        : 피사체 중심 − 화면 중심의 가로 오프셋. +=오른쪽, -=왼쪽
          dy        : 세로 오프셋. +=아래, -=위  (화면 좌표 그대로, 부호 뒤집기 없음)
          dx_norm   : dx / W  (해상도 무관 정규화, -0.5 ~ 0.5)
          dy_norm   : dy / H
          -- 극좌표(거리·방향) --
          dr        : 오프셋 벡터 크기 = sqrt(dx^2 + dy^2) (픽셀)
          dr_norm   : dr / W  (해상도 무관 정규화)
          theta_deg : 화면 좌표 기준 오프셋 방향(도) = degrees(atan2(dy, dx)).
                      dy를 뒤집지 않고 화면 그대로 사용:
                        0°   = 오른쪽 (+x)
                        +90° = 아래   (+y)
                        ±180°= 왼쪽   (-x)
                        -90° = 위     (-y)
                      즉 양의 각도는 화면상 시계방향(오른쪽→아래).
                      ※ 이건 "화면에서 피사체가 어느 방향"일 뿐, 드론 forward/기수방향과 무관.
                        각 클라이언트가 자기 좌표계(AirSim ENU, DJI 등)로 변환해서 쓸 것.
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
    # 화면 좌표 그대로: x=오른쪽+, y=아래+. dy 부호를 뒤집지 않는다.
    #   0°=오른쪽, +90°=아래, ±180°=왼쪽, -90°=위. (양수 = 화면상 시계방향)
    # 특정 드론 좌표계(forward 등)로의 변환은 서버가 하지 않음(클라이언트 담당).
    theta_deg = math.degrees(math.atan2(dy, dx))

    return {
        "dx": dx,
        "dy": dy,
        "dx_norm": dx / W,
        "dy_norm": dy / H,
        "dr": dr,
        "dr_norm": dr / W,
        "theta_deg": theta_deg,
    }
