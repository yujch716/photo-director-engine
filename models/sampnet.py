"""SAMP-Net(Saliency-Augmented Multi-pattern Pooling) 구도 점수 — lazy singleton 로더.

CADB 논문 "Image Composition Assessment with Saliency-augmented Multi-pattern
Pooling"의 SAMP-Net을 감싸, 사진 한 장의 **구도(composition) 점수(1~5)**를 낸다.
NIMA(전반적 미학) / GAIC(크롭 후보 점수)와 달리 "이 구도가 얼마나 좋은가"를 평가한다.

번들 위치(이미 저장소에 포함):
  models/samp_net/Image-Composition-Assessment-Dataset-CADB-main/SAMPNet  (코드)
  models/samp_net/models/samp_net.pth                                     (가중치)

환경변수:
  SAMPNET_WEIGHT : .pth 경로 (기본 위 번들 가중치)

가중치가 없으면 안내 메시지를 담아 RuntimeError.
"""

from __future__ import annotations

import io
import os
import sys
import types
from pathlib import Path
from typing import Any

_MODELS_DIR = Path(__file__).resolve().parent
_BUNDLE = _MODELS_DIR / "samp_net" / "Image-Composition-Assessment-Dataset-CADB-main" / "SAMPNet"

SAMPNET_WEIGHT = os.environ.get(
    "SAMPNET_WEIGHT",
    str(_MODELS_DIR / "samp_net" / "models" / "samp_net.pth"),
)

# 학습 때 쓴 전처리 상수(cadb_dataset.py와 동일)
_IMAGE_SIZE = 224
_IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
_IMAGE_NET_STD = [0.229, 0.224, 0.225]

# 해석용 라벨(config.py의 attribute_types / pattern_list)
ATTRIBUTE_TYPES = ["RuleOfThirds", "BalacingElements", "DoF", "Object", "Symmetry", "Repetition"]
PATTERN_LIST = [1, 2, 3, 4, 5, 6, 7, 8]


class _Cfg:
    """번들 SAMPNet(cfg)가 참조하는 필드만 담은 최소 설정(학습 config.py의 기본값 그대로).

    원본 config.py는 import 시점에 하드코딩된 데이터셋 경로를 assert 하므로 그대로는
    못 쓴다. 여기서 필요한 값만 재현한다."""
    score_level = 5
    use_saliency = True
    use_multipattern = True
    use_pattern_weight = True
    use_channel_attention = True
    use_attribute = True
    use_weighted_loss = True
    resnet_layers = 18
    dropout = 0.5
    pool_dropout = 0.5
    num_attributes = len(ATTRIBUTE_TYPES)
    pattern_list = list(PATTERN_LIST)
    pattern_fuse = "sum"


_model: Any | None = None
_device: str | None = None
_transform: Any | None = None
_detect_saliency: Any | None = None


def _get_model():
    global _model, _device, _transform, _detect_saliency
    if _model is not None:
        return _model

    try:
        import torch
        import torchvision.transforms as transforms
    except Exception as exc:
        raise RuntimeError("SAMP-Net은 torch/torchvision이 필요합니다.") from exc

    if not os.path.exists(SAMPNET_WEIGHT):
        raise RuntimeError(
            f"SAMP-Net 가중치 없음: {SAMPNET_WEIGHT}\n"
            "       models/samp_net/models/ 에 samp_net.pth 를 두세요."
        )
    if not _BUNDLE.exists():
        raise RuntimeError(f"SAMP-Net 코드 번들 없음: {_BUNDLE}")

    # 번들의 samp_net.py는 `from config import Config`를 하는데, 그 config.py가
    # import 시 dataset 경로를 assert 한다. 우리 최소 cfg를 담은 stub을 먼저 주입.
    if "config" not in sys.modules:
        stub = types.ModuleType("config")
        stub.Config = _Cfg
        sys.modules["config"] = stub

    if str(_BUNDLE) not in sys.path:
        sys.path.insert(0, str(_BUNDLE))  # samp_net / cadb_dataset import 위해

    from samp_net import SAMPNet          # 번들 모델 정의
    from cadb_dataset import detect_saliency  # cv2 스펙트럴 residual saliency(모델 불필요)

    # MPS는 SAMP-Net의 non-divisible adaptive_avg_pool2d를 지원 안 함(PyTorch 제약)
    # → mps 제외하고 cuda>cpu.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sampnet] loading resnet{_Cfg.resnet_layers} on {device} ({SAMPNET_WEIGHT})")

    model = SAMPNet(_Cfg(), pretrained=False)  # 전체 state_dict를 실으므로 imagenet 사전학습 불필요
    state = torch.load(SAMPNET_WEIGHT, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state)
    model.eval().to(device)

    _transform = transforms.Compose([
        transforms.Resize((_IMAGE_SIZE, _IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGE_NET_MEAN, std=_IMAGE_NET_STD),
    ])
    _detect_saliency = detect_saliency
    _device = device
    _model = model
    return _model


def score_image(image_bytes: bytes) -> dict[str, Any]:
    """이미지 한 장의 SAMP-Net 구도 점수를 반환한다.

    반환:
      score            : 구도 점수(1~5, 분포 가중평균)
      distribution     : 점수 1~5의 확률 분포(길이 5)
      pattern_weights  : 8개 구도 패턴 가중치(softmax) — 어떤 구도로 보는지
      dominant_pattern : 가중치가 가장 큰 패턴 번호(1~8)
      attributes       : 구도 속성 6종 원점수(RuleOfThirds 등)
      device           : 추론 디바이스
    """
    import numpy as np
    import torch
    from PIL import Image

    model = _get_model()

    src = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _transform(src).unsqueeze(0).to(_device)  # (1,3,224,224)

    # saliency는 원본 RGB에서 생성(학습과 동일) → (1,1,224,224)
    sal = _detect_saliency(np.asarray(src).copy(), target_size=(_IMAGE_SIZE, _IMAGE_SIZE))
    s = torch.from_numpy(sal.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(_device)

    with torch.no_grad():
        weight, attribute, dist = model(x, s)  # dist는 이미 softmax 분포(1~5)

    dist = dist.squeeze(0).cpu()
    levels = torch.arange(1, _Cfg.score_level + 1, dtype=dist.dtype)
    score = float(torch.sum(dist * levels))

    pattern_weights = None
    dominant_pattern = None
    if weight is not None:
        w = torch.softmax(weight.squeeze(0).cpu(), dim=0)
        pattern_weights = [round(float(v), 4) for v in w]
        dominant_pattern = int(PATTERN_LIST[int(torch.argmax(w))])

    attributes = None
    if attribute is not None:
        vals = attribute.squeeze(0).cpu().tolist()
        attributes = {name: round(float(v), 4) for name, v in zip(ATTRIBUTE_TYPES, vals)}

    return {
        "score": round(score, 4),
        "distribution": [round(float(v), 4) for v in dist],
        "pattern_weights": pattern_weights,
        "dominant_pattern": dominant_pattern,
        "attributes": attributes,
        "device": _device,
    }
