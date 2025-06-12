from ultralytics import YOLO

def main():
    # 前回の学習で得たbest.ptを使って再学習
    # model = YOLO('Models/yolov8m.pt')
    model = YOLO("E:/TennisVision/TennisVisionForAMD/Models/yolov8m.pt")

    # 再学習の実行
    model.train(
        data="E:/TennisVision/TennisVisionForAMD/Datasets/tennis_custom/data.yaml",
        epochs=10,      # エポック数は必要に応じて変更
        imgsz=1280,      # 画像サイズは必要に応じて変更  
        batch=4,       # 必要に応じて変更
        device='cpu',   # AMDなどCUDA非対応GPUの場合は 'cpu' を指定
    )

if __name__ == '__main__':
    main()
