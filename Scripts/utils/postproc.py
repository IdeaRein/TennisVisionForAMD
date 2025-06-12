# utils/postproc.py
# モデルの推論結果を解釈し、使いやすい形に変換したり不要な重複検出を除去する処理をまとめたモジュール。
import cv2

def nms(boxes, scores, iou_threshold=0.5, conf_threshold=0.2):
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    return indices.flatten() if len(indices) > 0 else []

def scale_coords(pred, frame_shape):
    h, w = frame_shape[:2]
    cx, cy, box_w, box_h = pred[0:4]
    x1 = int((cx - box_w / 2) * w / 640)
    y1 = int((cy - box_h / 2) * h / 640)
    x2 = int((cx + box_w / 2) * w / 640)
    y2 = int((cy + box_h / 2) * h / 640)
    return [x1, y1, x2 - x1, y2 - y1]
