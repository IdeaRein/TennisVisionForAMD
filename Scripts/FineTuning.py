# ①YOLOv8（Ultralytics）をインストール
# !pip install ultralytics --upgrade -q

# ②YOLOv8をインポート
# from ultralytics import YOLO

# ３ファインチューニングの実行
# model = YOLO("yolov8x.pt")  # yolov8n.pt / yolov8s.pt / yolov8m.pt / yolov8l.pt なども可
# model.train(
#     data="/content/drive/MyDrive/sample/data.yaml",
#     epochs=50,
#     imgsz=640,
#     batch=16,
#     name="tennis_ball_model"
# )