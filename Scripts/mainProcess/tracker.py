# tracker.py
# 検出したボールの位置情報を蓄積し、軌跡（動いた線）をフレームに描画する機能をまとめたクラス。
import cv2
import numpy as np

class KalmanFilter2D:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        dt = 1.0
        self.kalman.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0 ],
            [0, 0, 0, 1 ]
        ], np.float32)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kalman.statePre = np.zeros((4,1), np.float32)
        self.kalman.statePost = np.zeros((4,1), np.float32)
        self.initialized = False

    def predict(self):
        return self.kalman.predict()

    def correct(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        return self.kalman.correct(measurement)

    def update(self, x, y):
        if not self.initialized:
            self.kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True
        else:
            self.predict()
            self.correct(x, y)
        return self.kalman.statePost[:2].flatten()


class BallTracker:
    def __init__(self, max_length=3):
        self.positions = []
        self.max_length = max_length
        self.kalman_filter = KalmanFilter2D()

    def update(self, frame_idx, x, y):
        filtered_pos = self.kalman_filter.update(x, y)
        fx, fy = int(filtered_pos[0]), int(filtered_pos[1])
        self.positions.append((frame_idx, fx, fy))
        if len(self.positions) > self.max_length:
            self.positions.pop(0)

    def draw(self, frame):
        for i in range(1, len(self.positions)):
            _, x1, y1 = self.positions[i-1]
            _, x2, y2 = self.positions[i]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


