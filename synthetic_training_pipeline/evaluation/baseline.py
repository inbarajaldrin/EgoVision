import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # << Add this line
from pathlib import Path
import os

try:
    from shapely.geometry import Polygon
except ImportError:
    raise ImportError("Please install shapely (pip install shapely) to compute IoU.")

# Paths
pred_label_dir = Path("/home/aaugus11/Desktop/Blender_OBB_Dataset/codes/train_yolo11n/output/labels")
gt_label_dir = Path("/home/aaugus11/Desktop/Blender_OBB_Dataset/YOLOv11_Dataset/test/labels")
img_dir = Path("/home/aaugus11/Desktop/Blender_OBB_Dataset/YOLOv11_Dataset/test/images")
output_dir = Path("/home/aaugus11/Desktop/Blender_OBB_Dataset/codes/train_yolo11n/output/overlay_baseline")
output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------ Helper Functions ------------------------
def draw_obb(image, points, color, thickness=2):
    """Draw an oriented bounding box (8-point polygon) on the image."""
    points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)

def convert_16_to_8(coords, img_shape):
    """
    Convert 16-point GT label to 8-point OBB via minAreaRect.
    `coords` are 16 coords (x1,y1,...,x8,y8) normalized [0,1].
    """
    h, w = img_shape
    coords = [float(c) for c in coords]
    points = np.array([[coords[i] * w, coords[i+1] * h] 
                       for i in range(0, 16, 2)], dtype=np.float32)
    rect = cv2.minAreaRect(points)  # (center, (w,h), angle)
    box = cv2.boxPoints(rect)       # 4 corner points
    return [(pt[0] / w, pt[1] / h) for pt in box]

def polygon_iou(pts1, pts2):
    """
    Compute IoU between two polygons (each is a list of (x, y) in image coords).
    Uses shapely for intersection/union area.
    """
    poly1 = Polygon(pts1).buffer(0)
    poly2 = Polygon(pts2).buffer(0)
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou

def get_centroid(pts):
    """Return the (x, y) centroid of an 8-point polygon."""
    pts_arr = np.array(pts, dtype=np.float32)
    centroid = np.mean(pts_arr, axis=0)
    return centroid  # (cx, cy)

def centroid_distance(pts1, pts2):
    c1 = get_centroid(pts1)
    c2 = get_centroid(pts2)
    return np.linalg.norm(c1 - c2)

def get_minarearect_angle(pts):
    """
    Return the angle (in degrees) from cv2.minAreaRect for the given polygon.
    Typically ranges from -90 to 0 (OpenCV convention).
    """
    pts_arr = np.array(pts, dtype=np.float32)
    rect = cv2.minAreaRect(pts_arr)
    return rect[2]

def angle_difference(pts1, pts2):
    """
    Compute absolute difference between angles of two OBBs 
    (derived via minAreaRect).
    """
    angle1 = get_minarearect_angle(pts1)
    angle2 = get_minarearect_angle(pts2)
    return abs(angle1 - angle2)

def create_bounding_box_image(base_img, boxes, color):
    """
    Returns a copy of `base_img` with all the bounding boxes in `boxes` drawn in `color`.
    `boxes` is a list of 8-pt polygons in (x, y) image space.
    """
    out_img = base_img.copy()
    for pts in boxes:
        draw_obb(out_img, pts, color=color, thickness=2)
    return out_img

# ------------------------ Main Code ------------------------
pred_label_files = sorted(pred_label_dir.glob("*.txt"))[:10]  # up to 10 samples

for pred_file in pred_label_files:
    base_name = pred_file.stem
    gt_file = gt_label_dir / f"{base_name}.txt"
    img_file = img_dir / f"{base_name}.png"
    out_img_path = output_dir / f"{base_name}_overlay.png"
    
    # We'll also define a new path for the 4-column table figure:
    table_fig_path = output_dir / f"{base_name}_comparison_table.png"

    if not (gt_file.exists() and img_file.exists()):
        print(f"Skipping {base_name}: missing image or ground truth.")
        continue

    # Read the raw image
    img = cv2.imread(str(img_file))
    if img is None:
        print(f"Skipping {base_name}: could not read image.")
        continue
    h, w = img.shape[:2]
    
    # Convert to RGB for Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Read predicted OBBs
    pred_obbs = []
    with open(pred_file, "r") as f:
        for line in f:
            line = line.strip().split()
            # First entry is class, next 8 entries are polygon coords
            parts = list(map(float, line[1:]))
            pts = [(parts[i]*w, parts[i+1]*h) for i in range(0, 8, 2)]
            pred_obbs.append(pts)

    # Read ground-truth OBBs (need to convert from 16 -> 8)
    gt_obbs = []
    with open(gt_file, "r") as f:
        for line in f:
            line = line.strip().split()
            # First entry is class, next 16 entries are 8 coords
            parts = list(map(float, line[1:]))
            if len(parts) != 16:
                continue  # skip malformed lines
            converted = convert_16_to_8(parts, (h, w))  # normalized
            # scale to image coords
            pts = [(x * w, y * h) for (x, y) in converted]
            gt_obbs.append(pts)

    # Create images with drawn boxes
    # (1) Ground Truth boxes (green)
    gt_img = create_bounding_box_image(img_rgb, gt_obbs, color=(0, 255, 0))
    # (2) Predicted boxes (red)
    pred_img = create_bounding_box_image(img_rgb, pred_obbs, color=(255, 0, 0)) 
    #    Note the color tuple is (R,G,B) in OpenCV space,
    #    but after conversion for matplotlib it will appear as "red"

    # ------------- Compute Metrics -------------
    print(f"\n=== {base_name} ===")
    metrics_list = []
    num_to_compare = min(len(pred_obbs), len(gt_obbs))
    for i in range(num_to_compare):
        pred_pts = pred_obbs[i]
        gt_pts = gt_obbs[i]
        
        iou_val = polygon_iou(pred_pts, gt_pts)
        cdist_val = centroid_distance(pred_pts, gt_pts)
        angle_diff_val = angle_difference(pred_pts, gt_pts)
        metrics_list.append((iou_val, cdist_val, angle_diff_val))

        print(f"  Box #{i+1}:")
        print(f"    IoU:        {iou_val:.4f}")
        print(f"    CentroidDist (px):  {cdist_val:.2f}")
        print(f"    AngleDiff (deg):    {angle_diff_val:.2f}")

    if len(pred_obbs) != len(gt_obbs):
        print(f"  [WARNING] #PredBoxes={len(pred_obbs)} != #GTBoxes={len(gt_obbs)}")

    # Save the overlay image (pred + GT) if you still want it
    # (already done in your original code – optional)
    overlay_img = img.copy()
    for pts in pred_obbs:
        draw_obb(overlay_img, pts, (0, 0, 255), 2)
    for pts in gt_obbs:
        draw_obb(overlay_img, pts, (0, 255, 0), 2)
    cv2.imwrite(str(out_img_path), overlay_img)
    print(f"Saved overlay: {out_img_path}")

    # ============= CREATE THE 4-COLUMN FIGURE =============
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Column 1: Raw image
    axs[0].imshow(img_rgb)
    axs[0].set_title("Raw Image")
    axs[0].axis("off")

    # Column 2: Ground Truth
    axs[1].imshow(gt_img)
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    # Column 3: Prediction
    axs[2].imshow(pred_img)
    axs[2].set_title("Prediction")
    axs[2].axis("off")

    # Column 4: Text Table (IoU, distance, angle)
    # Build a text string for all boxes
    table_str = ""
    for i, (iou_val, cdist_val, angle_diff_val) in enumerate(metrics_list, start=1):
        table_str += (
            f"Box #{i}:\n"
            f"IoU: {iou_val:.4f}\n"
            f"Dist(px): {cdist_val:.2f}\n"
            f"AngleDiff(deg): {angle_diff_val:.2f}\n\n"
        )
    # If there's a mismatch in # of boxes, mention it
    if len(pred_obbs) != len(gt_obbs):
        table_str += f"[WARNING] #PredBoxes={len(pred_obbs)} != #GTBoxes={len(gt_obbs)}"

    axs[3].axis("off")
    axs[3].text(
        0.5, 0.5, table_str,
        fontsize=12, ha='center', va='center',
        transform=axs[3].transAxes
    )

    plt.tight_layout()
    plt.savefig(str(table_fig_path), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved 4-column comparison: {table_fig_path}")
