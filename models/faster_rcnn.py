import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import config

class FasterRCNNDetector:
    def __init__(self):
        print("[INFO] Loading Custom Weapon Detector (Pistol/Knife)...")
        
        # 1. Load Architecture
        self.model = fasterrcnn_resnet50_fpn(pretrained=False)
        
        # 2. Configure Head for 3 Classes (Background, Pistol, Knife)
        num_classes = 3 
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # 3. Load Your Custom Weights
        try:
            state_dict = torch.load(config.FASTER_RCNN_WEIGHTS, map_location=config.DEVICE)
            self.model.load_state_dict(state_dict)
            self.model.to(config.DEVICE)
            self.model.eval()
            print("[INFO] Weapon Model Loaded Successfully.")
        except Exception as e:
            print(f"[CRITICAL ERROR] Could not load weights: {e}")
            print("Did you place 'multi_weapon_model.pth' in models/weights/?")

    def predict(self, image):
        # Normalize and Convert
        image_tensor = torchvision.transforms.functional.to_tensor(image).to(config.DEVICE)
        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(image_tensor)

        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        # --- THE OFFSET HACK ---
        # Shift labels by 100 to avoid collision with COCO labels
        # 1 (Pistol) -> 101
        # 2 (Knife)  -> 102
        labels = labels + 100

        # Filter by Confidence
        valid_indices = scores >= config.CONFIDENCE_THRESHOLD
        
        return {
            'boxes': boxes[valid_indices],
            'scores': scores[valid_indices],
            'labels': labels[valid_indices]
        }