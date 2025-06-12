# detector.py
# ONNXモデルの読み込みと推論を行うクラスを定義。
import onnxruntime as ort
from config import MODEL_PATH
import numpy as np

class BallDetector:
    def __init__(self):
        self.session = ort.InferenceSession(MODEL_PATH, providers=['DmlExecutionProvider'])
        print("Loaded ONNX model.")
    
    def detect(self, img_input):
        output = self.session.run(None, {self.session.get_inputs()[0].name: img_input})[0]
        preds = np.squeeze(output).T
        return preds
