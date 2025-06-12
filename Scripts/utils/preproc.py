# utils/preproc.py
# 動画のフレームをモデル入力に適した形式に変換する処理をまとめたモジュール。
import cv2
import numpy as np

def preprocess_frame(frame, size=(640, 640)):  
    resized = cv2.resize(frame, size)   
    img_input = resized.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0
    return img_input
