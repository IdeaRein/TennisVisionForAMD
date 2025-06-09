import onnxruntime as ort
import numpy as np
import cv2
import time

import onnxruntime as ort
print("Available providers:", ort.get_available_providers())

# モデル読み込み
session = ort.InferenceSession("E:/TennisVision/TennisVisionForAMD/Models/yolov8x.onnx", providers=['DmlExecutionProvider'])  # AMD GPU

# 使用プロバイダー確認
print("利用可能なプロバイダー:", ort.get_available_providers())
print("このセッションで使われているプロバイダー:", session.get_providers())