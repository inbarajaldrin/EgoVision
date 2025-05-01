# from ultralytics import YOLO

# # Load YOLOv11n-OBB model
# model = YOLO('yolo11n-obb.pt')

# # Train
# model.train(
#     task='obb',
#     data='/home/aaugus11/Desktop/Blender_OBB_Dataset/YOLOv11_Dataset/dataset.yaml',
#     epochs=100,
#     imgsz=640,
#     project='runs/obb',
#     name='jenga_obb'
# )


from ultralytics import YOLO

# Load YOLOv11n-OBB model
model = YOLO('yolo11n-obb.pt')

# Train with geometric + photometric augmentations
model.train(
    task='obb',
    data='/home/aaugus11/Projects/cse598/EgoGrasp/glb_to_obj_pipeline/lego_rendered_obb/YOLOv11_Dataset/dataset.yaml',
    epochs=100,
    imgsz=640,
    project='runs/obb',
    name='jenga_obb_aug',

    # Geometric Augmentations
    degrees=10,
    translate=0.1,
    scale=0.5,
    shear=10,
    perspective=0.0005,
    flipud=0.1,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.2,

    # Photometric Augmentations
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    augment=True  # Enables full augmentation pipeline
)
