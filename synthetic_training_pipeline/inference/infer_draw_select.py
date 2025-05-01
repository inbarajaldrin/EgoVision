#!/usr/bin/env python3
from ultralytics import YOLO
import cv2
from pathlib import Path
import sys

# ——— CONFIG ———
if len(sys.argv) != 2:
    print(f"Usage: python3 {Path(__file__).name} <image_path>")
    sys.exit(1)

img_path = Path(sys.argv[1])
if not img_path.is_file():
    print(f"❌ File not found: {img_path}")
    sys.exit(1)

model_path    = "/home/aaugus11/Projects/cse598/EgoGrasp/synthetic_training_pipeline/runs/obb/jenga_obb_aug/weights/best.pt"
output_dir    = Path("/home/aaugus11/Projects/cse598/EgoGrasp/glb_to_obj_pipeline/lego_rendered_obb/infer_draw")
label_out_dir = output_dir / "labels"

output_dir.mkdir(parents=True, exist_ok=True)
label_out_dir.mkdir(parents=True, exist_ok=True)

# ——— LOAD MODEL ———
model = YOLO(model_path)

# ——— RUN OBB INFERENCE ———
name    = img_path.stem
results = model(img_path, task='obb')
pred_img = results[0].plot()

# ——— SAVE VISUALIZATION ———
out_img = output_dir / f"{name}_pred.png"
cv2.imwrite(str(out_img), pred_img)
print(f"✅ Saved visualization to {out_img}")

# ——— SAVE LABELS ———
h, w = cv2.imread(str(img_path)).shape[:2]
lines = []
for poly, cls in zip(results[0].obb.xyxyxyxy, results[0].obb.cls):
    pts = poly.cpu().numpy().reshape(-1)
    normed = [
        (pts[i] / w) if (i % 2 == 0) else (pts[i] / h)
        for i in range(len(pts))
    ]
    lines.append(f"{int(cls)} " + " ".join(f"{x:.6f}" for x in normed))

label_file = label_out_dir / f"{name}.txt"
with open(label_file, 'w') as f:
    f.write("\n".join(lines) + "\n")
print(f"✅ Saved label file to {label_file}")
