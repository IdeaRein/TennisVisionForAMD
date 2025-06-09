import onnxruntime as ort
import numpy as np
import cv2
import time

# モデル読み込み（AMD GPU用）
session = ort.InferenceSession("E:/TennisVision/TennisVisionForAMD/Models/yolov8x.onnx", providers=['DmlExecutionProvider'])

# 動画読み込み
video_path = "Datas/sample.mov"
cap = cv2.VideoCapture(video_path)

# クラス名（テニスボールのみ）
CLASS_NAMES = {32: 'sports ball'}

# 閾値設定
CONF_THRESH = 0.3

# 非最大抑制（簡易）
def nms(boxes, scores, iou_threshold=0.5):
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, iou_threshold)
    return indices.flatten() if len(indices) > 0 else []

# メインループ
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # 前処理
    img_resized = cv2.resize(frame, (640, 640))
    img_input = img_resized.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0

    # 推論
    start = time.time()
    output = session.run(None, {session.get_inputs()[0].name: img_input})[0]
    end = time.time()

    preds = np.squeeze(output).T  # shape: (8400, 84)

    boxes = []
    confidences = []
    class_ids = []

    for pred in preds:
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence > CONF_THRESH and class_id == 32:  # 🎯 テニスボールのみ検出
            cx, cy, w_box, h_box = pred[0:4]
            x1 = int((cx - w_box / 2) * w / 640)
            y1 = int((cy - h_box / 2) * h / 640)
            x2 = int((cx + w_box / 2) * w / 640)
            y2 = int((cy + h_box / 2) * h / 640)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # NMSで重複除去
    indices = nms(boxes, confidences)

    # 結果描画
    for i in indices:
        box = boxes[i]
        class_id = class_ids[i]
        label = CLASS_NAMES.get(class_id, str(class_id))
        conf = confidences[i]

        x, y, w_box, h_box = box
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 255), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # FPS表示（任意）
    fps = 1 / (end - start)
    cv2.putText(frame, f'FPS: {fps:.2f}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Ball Detection Only", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
