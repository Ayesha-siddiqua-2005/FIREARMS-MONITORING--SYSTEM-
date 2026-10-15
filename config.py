import torch

# Device configuration
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Thresholds
CONFIDENCE_THRESHOLD = 0.80
IOU_THRESHOLD = 0.3

# --- WEIGHT PATHS ---
# Your NEW Custom Model
FASTER_RCNN_WEIGHTS = "models/weights/multi_weapon_model.pth"
# Standard COCO Model
EFFICIENTDET_WEIGHTS = "models/weights/tf_efficientdet_d0.pth"

# --- CLASS MAPPINGS ---
# Standard COCO (EfficientDet)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Custom Weapon Classes (Internal ID Mapping)
# We will map these to 100+ in the code to avoid collisions
WEAPON_CLASSES = {
    101: 'PISTOL',
    102: 'KNIFE'
}

# Person class in COCO (index 1). This is used for person-weapon relationship checks.
PERSON_CLASS_ID = 1

# Restrict ensemble outputs to classes used by threat logic.
ENSEMBLE_TARGET_CLASSES = {PERSON_CLASS_ID, *WEAPON_CLASSES.keys()}