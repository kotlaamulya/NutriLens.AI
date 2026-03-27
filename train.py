from ultralytics import YOLO
import os

def main():
    # Path where your training runs are saved
    run_name = "nutrilens_yolov8n_cpu2"   # change if needed
    run_dir = f"runs/detect/{run_name}/weights"
    last_ckpt = os.path.join(run_dir, "last.pt")

    # Define your total target epochs
    target_epochs = 30   # 👈 you can set this once

    # If last checkpoint exists, resume
    if os.path.exists(last_ckpt):
        print(f"🔄 Resuming training from {last_ckpt}...")
        model = YOLO(last_ckpt)
        results = model.train(
            data="IndianFoodNet.v1i.yolov8/data.yaml",
            epochs=target_epochs,   # YOLO handles resume properly
            resume=True,
            imgsz=640,
            batch=8,
            device="cpu"
        )
    else:
        # No checkpoint found → start fresh
        print("🚀 Starting new training from yolov8n.pt...")
        model = YOLO("yolov8n.pt")
        results = model.train(
            data="IndianFoodNet.v1i.yolov8/data.yaml",
            epochs=target_epochs,
            imgsz=640,
            batch=8,
            device="cpu",
            name=run_name
        )

    print("✅ Training complete or resumed successfully!")

if __name__ == "__main__":
    main()

