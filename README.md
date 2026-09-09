# Forshot

DJI 드론(Mini 4 Pro) 또는 폰 카메라로 피사체를 촬영하면, AI 서버(YOLO 객체 탐지 + 구도 분석)와 통신하며 드론을 자동으로 움직여 **삼분할 구도(Rule of Thirds)**에 맞는 사진을 찍어주는 Android 앱 서버입니다.
- 클라이언트 레포지토리 : https://github.com/yujch716/forshot-client-app

## 핵심 기능

- **드론 / 폰 카메라 모드**: DJI 드론의 FPV 스트림을 사용하는 드론 모드와, 드론 없이 테스트할 수 있는 폰 카메라 모드를 지원합니다.
- **객체 탐지 및 선택**: 서버의 YOLO 탐지 결과를 화면에 오버레이로 표시하고, 탭으로 촬영 대상 객체를 선택합니다.
- **자동 촬영 파이프라인**: 촬영 버튼 하나로 `후진 → 구도 스캔 → 정점 복귀 → 세부조정 → 짐벌 틸트 → 최종 촬영`까지 전 과정을 자동 수행합니다.
- **비상정지**: 드론 모드에서 언제든 즉시 스틱을 0으로 만들고 진행 중인 동작을 취소할 수 있습니다.



## 프로젝트 환경 구성 (최초 1회만 실행)
```bash
conda create -n photo-director-engine python=3.10  
conda activate photo-director-engine

pip install fastapi uvicorn python-multipart
pip install ultralytics
pip install pyiqa
pip install realesrgan
pip install transformers
pip install python-dotenv
```



## 프로젝트 실행
```bash
uvicorn main:app --reload

uvicorn main:app --reload --host 0.0.0.0 --port 8000 
```


## 화면 접속
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
