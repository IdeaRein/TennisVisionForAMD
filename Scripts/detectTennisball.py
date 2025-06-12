import onnxruntime as ort
import numpy as np
import cv2
import time

# モデル読み込み（AMD GPU用）
session = ort.InferenceSession(
    "E:/TennisVision/TennisVisionForAMD/LeaeningModels/best.onnx",
    providers=['DmlExecutionProvider']
)

# GPU利用状況を確認
print("Available providers:", ort.get_available_providers())
print("Current session providers:", session.get_providers())

# 動画読み込み
video_path = "Datas/tennis_sample3.mp4"
cap = cv2.VideoCapture(video_path)

# クラス名（テニスボールのみ）
CLASS_NAMES = {0: 'tennisball'}

# 閾値設定
CONF_THRESH = 0.3

# 非最大抑制（簡易）
def nms(boxes, scores, iou_threshold=0.5):
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, iou_threshold)
    return indices.flatten() if len(indices) > 0 else []

# 軌跡記録用
ball_positions = []
frame_idx = 0

# メインループ
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    h, w = frame.shape[:2]

    # 前処理
    img_resized = cv2.resize(frame, (640, 640))
    img_input = img_resized.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0

    # 推論
    start = time.time()
    output = session.run(None, {session.get_inputs()[0].name: img_input})[0]
    end = time.time()

    preds = np.squeeze(output).T  # shape: (8400, num_classes + 4)

    boxes, confidences, class_ids = [], [], []

    for pred in preds:
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence > CONF_THRESH and class_id == 0:  # 🎯 テニスボールのみ検出
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

    for i in indices:
        x, y, w_box, h_box = boxes[i]
        center_x = x + w_box // 2
        center_y = y + h_box // 2
        ball_positions.append((frame_idx, center_x, center_y))

        label = CLASS_NAMES.get(class_ids[i], str(class_ids[i]))
        conf = confidences[i]
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 255), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 軌跡の描画
    for i in range(1, len(ball_positions)):
        _, x1, y1 = ball_positions[i - 1]
        _, x2, y2 = ball_positions[i]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # FPS表示
    fps = 1 / (end - start)
    cv2.putText(frame, f'FPS: {fps:.2f}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # ウィンドウ表示
    cv2.imshow("Ball Detection & Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()
