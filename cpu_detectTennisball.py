from ultralytics import YOLO
import cv2

# model = YOLO("Models/yolov8m.pt")  # 精度の良いモデルを指定
model = YOLO("LeaeningModels/best.pt")  # 精度の良いモデルを指定

# video_path = "Datas/tennis_sample2.mp4"
video_path = "Datas/tennis_sample2.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # conf=0.3で、人間(0)とテニスボール(32)のみ検出
    results = model.predict(frame, conf=0.3, classes=[0, 32])

    annotated_frame = results[0].plot()
    cv2.imshow("Tennis Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

