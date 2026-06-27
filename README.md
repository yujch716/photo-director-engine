# photo-director-engine

### 프로젝트 환경 구성 (최초 1회만 실행)
```bash
conda create -n photo-director-engine python=3.10  
conda activate photo-director-engine
pip install fastapi uvicorn python-multipart
pip install ultralytics
pip install pyiqa
pip install realesrgan
pip install transformers
```

Real-ESRGAN weights are downloaded on first `lower_not_similar` preprocess request.
If automatic download fails, download this file manually and set `REAL_ESRGAN_MODEL_PATH`:

```bash
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

Logo/arrow removal requires custom YOLO weights trained for SNS logos/arrows.
Put the weights at `models/logo_yolo.pt`, or set `LOGO_YOLO_MODEL_PATH`.
YOLO is still required for automatic logo/arrow detection. OpenCV inpaint only removes pixels from a mask, so it cannot find the logo/arrow location by itself.

### Logo/arrow YOLO training
Put labeled images in YOLO format:

```bash
data/logo_yolo/
  images/train/
  images/val/
  labels/train/
  labels/val/
  dataset.yaml
```

Classes:

```text
0 logo
1 arrow
```

Train and save `models/logo_yolo.pt`:

```bash
python models/logo_yolo_train.py --epochs 80 --imgsz 640 --batch 8
```

Label format per image is YOLO bbox text:

```text
class_id x_center y_center width height
```

All coordinates are normalized from 0 to 1.

### YOLO + CLIP landmark classification
The landmark classifier reuses `models/yolo.py` to crop detected regions, then compares each crop with CLIP text candidates.

Current output candidates:

```text
상생의 손(바다)
상생의 손(육지)
호미곶 등대
호미곶 광장
새천년기념관
```

Default CLIP model is `openai/clip-vit-base-patch32`.
Set `LANDMARK_CLIP_MODEL` to use another Hugging Face CLIP-compatible model.


### 프로젝트 실행
```bash
uvicorn main:app --reload
```


### 화면 접속
http://127.0.0.1:8000/


### 프로젝트 구조
```bash
photo-director-engine/
 ├── main.py           ← FastAPI 앱, 엔드포인트 (교통정리 해주는 곳)
 ├── models/            ← 모델들 정의하는 폴더
 │    ├── nima.py       ← 점수 매기는 모델
 │    ├── yolo.py       ← 객체 탐지
 │    └── clip.py       ← 랜드마크 식별
 ├── static/
 │    └── index.html       ← API TEST 할 수 있는 페이지
 └── venv/
```
