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
