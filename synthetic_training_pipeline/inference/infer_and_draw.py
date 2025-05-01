from ultralytics import YOLO
import cv2
from pathlib import Path
import random

# Paths
model_path = "/home/aaugus11/Projects/cse598/EgoGrasp/synthetic_training_pipeline/runs/obb/jenga_obb_aug/weights/best.pt"
val_dir = Path("/home/aaugus11/Projects/cse598/EgoGrasp/glb_to_obj_pipeline/lego_rendered_obb/YOLOv11_Dataset/test/images")
output_dir = Path("/home/aaugus11/Projects/cse598/EgoGrasp/glb_to_obj_pipeline/lego_rendered_obb/infer_draw")
label_out_dir = output_dir / "labels"
output_dir.mkdir(parents=True, exist_ok=True)
label_out_dir.mkdir(parents=True, exist_ok=True)

# Load model
model = YOLO(model_path)

# Get up to 10 random images from val
image_paths = list(val_dir.glob("*.png"))
random.shuffle(image_paths)
image_paths = image_paths[:10]

# Inference + Save predictions
for img_path in image_paths:
    name = img_path.stem
    results = model(img_path,task='obb')
    pred_img = results[0].plot()

    # Save prediction image
    out_img_path = output_dir / f"{name}_pred.png"
    cv2.imwrite(str(out_img_path), pred_img)
    print(f"Saved: {out_img_path}")

    # Save label file in YOLOv11 OBB format
    label_lines = []
    h, w = cv2.imread(str(img_path)).shape[:2]

    for box in results[0].obb.xyxyxyxy:
        box = box.flatten()  # ensure it's 1D
        normalized = []
        for i, coord in enumerate(box):
            val = float(coord)
            norm = val / w if i % 2 == 0 else val / h
            normalized.append(norm)
        line = "0 " + " ".join(f"{v:.6f}" for v in normalized)
        label_lines.append(line)

    label_path = label_out_dir / f"{name}.txt"
    with open(label_path, 'w') as f:
        f.write("\n".join(label_lines) + "\n")
    print(f"Saved label file: {label_path}")
