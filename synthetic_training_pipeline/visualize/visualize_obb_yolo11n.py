import os
import random
from PIL import Image, ImageDraw

# Base directories
base_dir = "/home/aaugus11/Desktop/Blender_OBB_Dataset/YOLOv11_Dataset"
test_img_dir = os.path.join(base_dir, "test", "images")
test_lbl_dir = os.path.join(base_dir, "test", "labels")
output_dir = os.path.join(base_dir, "test_visualized")
os.makedirs(output_dir, exist_ok=True)

# Class name mapping (optional)
class_names = {
    0: "Jenga_Block",
}

def draw_yolo11_obb(draw, points, outline="red"):
    draw.polygon(points, outline=outline, width=2)

def visualize_yolo11(image_path, label_path, output_path):
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        w_img, h_img = img.size

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 17:
                continue  # Skip malformed lines

            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            points = [(coords[i] * w_img, coords[i+1] * h_img) for i in range(0, 16, 2)]

            draw_yolo11_obb(draw, points)

            label = class_names.get(class_id, str(class_id))
            draw.text((points[0][0], points[0][1] - 10), label, fill="yellow")

        img.save(output_path)
        print(f"Saved visualization: {output_path}")

# List available test images
image_files = [f for f in os.listdir(test_img_dir) if f.endswith(".png")]
sampled_images = random.sample(image_files, min(10, len(image_files)))

# Process each image
for image_file in sampled_images:
    base = os.path.splitext(image_file)[0]
    image_path = os.path.join(test_img_dir, image_file)
    label_path = os.path.join(test_lbl_dir, base + ".txt")
    output_path = os.path.join(output_dir, image_file)

    if os.path.exists(image_path) and os.path.exists(label_path):
        visualize_yolo11(image_path, label_path, output_path)
    else:
        print(f"Skipping {image_file}: missing image or label.")
