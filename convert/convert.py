from ultralytics import YOLO

# 1. 일반 YOLOv8n (파일 없으면 자동 다운로드)
m1 = YOLO("yolov8n.pt")
m1.export(format="tflite")

# 2. 커스텀 (이 .pt 파일 경로를 실제 위치로)
m2 = YOLO("models/homigot_hand_yolo8n.pt")  # 서버 폴더에서 찾아서 경로 지정
m2.export(format="tflite")