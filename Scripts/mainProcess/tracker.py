# tracker.py
# 検出したボールの位置情報を蓄積し、軌跡（動いた線）をフレームに描画する機能をまとめたクラス。
import cv2

class BallTracker:
    def __init__(self):
        self.positions = []

    def update(self, frame_idx, x, y):
        self.positions.append((frame_idx, x, y))

    def draw(self, frame):
        for i in range(1, len(self.positions)):
            _, x1, y1 = self.positions[i - 1]
            _, x2, y2 = self.positions[i]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
