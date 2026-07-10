# GAIC crop bundle (컴파일 불필요, 순수 PyTorch)

## 설치
    pip install -r requirements-min.txt
    # GPU면 CUDA용 torch 설치, 없으면 CPU torch 도 동작 (자동 CPU 폴백)

## 가중치
GitHub에서 받은 가중치를 pretrained_models/ 에 그대로 넣기:
    pretrained_models/GAIC-vgg16-reddim32.pth
    pretrained_models/GAIC-mobilenetv2-reddim16.pth
    pretrained_models/GAIC-shufflenetv2-reddim32.pth
(backbone 별로 reddim 이 정해져 있음: vgg16/shufflenetv2=32, mobilenetv2=16)

## CLI 사용
    python crop_image.py /path/photo.jpg
    python crop_image.py /path/photo.jpg --topk 3 --ratio 0.8
    python crop_image.py /path/photo.jpg --object 560 150 720 470   # 객체 안 잘리게

## 코드에서 사용 (다른 노트북/스크립트에서 import)
    import cv2
    from gaic_scorer import GaicScorer, generate_fixed_crops
    scorer = GaicScorer("vgg16", "pretrained_models/GAIC-vgg16-reddim32.pth", device="cuda")
    img = cv2.imread("photo.jpg")
    boxes = generate_fixed_crops(img.shape, ratio=0.8, n=48)        # 또는 contain=(x1,y1,x2,y2)
    scores = scorer.score_crops(img, boxes)
    i, s = scorer.best_crop(img, boxes)

주의: import 하는 파일(또는 sys.path)에서 이 폴더가 루트여야 함.
crop_image.py / gaic_scorer.py 는 자기 폴더를 sys.path 에 자동 추가함.
