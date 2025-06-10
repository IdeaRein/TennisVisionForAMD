from ultralytics import YOLO

# モデルをロード
model = YOLO("E:/TennisVision/TennisVisionForAMD/LeaeningModels/best.pt")

# ONNX形式でエクスポート（GPU最適化ONNX）
model.export(format="onnx", dynamic=True, opset=12)