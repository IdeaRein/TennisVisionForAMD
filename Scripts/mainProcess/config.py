# config.py

VIDEO_PATH = "Datas/tennis_sample3.mp4" # 動画ファイルのパス
MODEL_PATH = "E:/TennisVision/TennisVisionForAMD/LeaeningModels/best.onnx" # ONNXモデルのパス
CLASS_NAMES = {0: 'tennisball'} # クラス名（テニスボールのみ）
CONF_THRESH = 0.3 # 閾値設定