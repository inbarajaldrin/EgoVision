
import os
import math
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

# Directories
input_dir = "/home/aaugus11/Projects/cse598/EgoGrasp/glb_to_obj_pipeline/lego_rendered_obb"
output_dir = os.path.join(input_dir, "output_with_bboxes")
os.makedirs(output_dir, exist_ok=True)

def parse_bounding_boxes(filepath):
    """Parse bounding box data from a file."""
    bounding_boxes = {}
    with open(filepath, 'r') as file:
        for line in file:
            name, bbox = line.strip().split(": ")
            bbox = [
                tuple(map(int, point.strip("()").split(",")))
                for point in bbox.strip("[]").split("), (")
            ]
            bounding_boxes[name] = bbox
    return bounding_boxes

def calculate_oriented_bbox(points):
    """Calculate the oriented bounding box (OBB) that tightly fits the points."""
    # Compute the convex hull
    hull = ConvexHull(points)
    hull_points = [points[vertex] for vertex in hull.vertices]

    # Minimum bounding rectangle variables
    min_area = float('inf')
    best_box = None

    for i in range(len(hull_points)):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % len(hull_points)]

        # Calculate edge vector and angle of rotation
        edge_vector = (p2[0] - p1[0], p2[1] - p1[1])
        angle = -math.atan2(edge_vector[1], edge_vector[0])

        # Rotate all points by the negative angle
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        rotated_points = [
            (
                cos_angle * (p[0] - p1[0]) - sin_angle * (p[1] - p1[1]),
                sin_angle * (p[0] - p1[0]) + cos_angle * (p[1] - p1[1])
            )
            for p in hull_points
        ]

        # Find min/max x and y in rotated space
        min_x = min(p[0] for p in rotated_points)
        max_x = max(p[0] for p in rotated_points)
        min_y = min(p[1] for p in rotated_points)
        max_y = max(p[1] for p in rotated_points)

        # Calculate area
        area = (max_x - min_x) * (max_y - min_y)
        if area < min_area:
            min_area = area
            best_box = (min_x, max_x, min_y, max_y, angle, p1)

    # Retrieve the best box parameters
    min_x, max_x, min_y, max_y, angle, origin = best_box

    # Calculate the corner points of the bounding box
    corners = [
        (min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)
    ]
    cos_angle = math.cos(-angle)
    sin_angle = math.sin(-angle)
    unrotated_corners = [
        (
            cos_angle * p[0] - sin_angle * p[1] + origin[0],
            sin_angle * p[0] + cos_angle * p[1] + origin[1]
        )
        for p in corners
    ]

    return unrotated_corners

def overlay_bounding_boxes(image_path, bbox_path, output_path):
    """Overlay bounding boxes on the image."""
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)

        # Parse bounding boxes
        bounding_boxes = parse_bounding_boxes(bbox_path)

        for name, bbox in bounding_boxes.items():
            # Calculate and draw the oriented bounding box
            obb_corners = calculate_oriented_bbox(bbox)
            draw.polygon(obb_corners, outline="red", width=3)

            # Add the object name near the first corner
            draw.text((obb_corners[0][0], obb_corners[0][1] - 10), f"{name}", fill="yellow")

        # Save the image with bounding boxes
        img.save(output_path)
        print(f"Saved image with bounding boxes: {output_path}")

# Get all image and bounding box files
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith("_rgb0001.png")])
bbox_files = sorted([f for f in os.listdir(input_dir) if f.endswith("_obb0001.txt")])

# Ensure proper one-to-one matching
if len(image_files) != len(bbox_files):
    raise ValueError("Number of images and bounding box files do not match!")

# Process each pair
for image_file, bbox_file in zip(image_files, bbox_files):
    # Ensure the base filenames match
    base_image = os.path.splitext(image_file)[0].replace("_rgb0001", "")
    base_bbox = os.path.splitext(bbox_file)[0].replace("_obb0001", "")
    if base_image != base_bbox:
        raise ValueError(f"Mismatched files: {image_file} and {bbox_file}")

    # File paths
    image_path = os.path.join(input_dir, image_file)
    bbox_path = os.path.join(input_dir, bbox_file)
    output_path = os.path.join(output_dir, image_file)

    # Overlay bounding boxes and save
    overlay_bounding_boxes(image_path, bbox_path, output_path)
