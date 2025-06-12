# main.py
# アプリケーションのメイン処理をまとめたファイル。
# 動画を読み込み、各フレームごとに検出と追跡を行い画面に表示する。
import sys
import os

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)
    
import cv2
import time
from config import VIDEO_PATH, CLASS_NAMES, CONF_THRESH
from detector import BallDetector
from tracker import BallTracker
from utils.preproc import preprocess_frame
from utils.postproc import nms, scale_coords



def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    detector = BallDetector()
    tracker = BallTracker()

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        img_input = preprocess_frame(frame)

        start = time.time()
        preds = detector.detect(img_input)
        end = time.time()

        h, w = frame.shape[:2]
        boxes, confidences, class_ids = [], [], []

        for pred in preds:
            class_scores = pred[4:]
            class_id = class_scores.argmax()
            confidence = class_scores[class_id]

            if confidence > CONF_THRESH and class_id == 0:
                box = scale_coords(pred, (h, w))
                boxes.append(box)
                confidences.append(float(confidence))
                class_ids.append(class_id)

        indices = nms(boxes, confidences)

        for i in indices:
            x, y, box_w, box_h = boxes[i]
            center_x = x + box_w // 2
            center_y = y + box_h // 2
            tracker.update(frame_idx, center_x, center_y)

            label = CLASS_NAMES.get(class_ids[i], str(class_ids[i]))
            conf = confidences[i]
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 255), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        tracker.draw(frame)

        fps = 1 / (end - start)
        cv2.putText(frame, f'FPS: {fps:.2f}', (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Ball Detection & Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
