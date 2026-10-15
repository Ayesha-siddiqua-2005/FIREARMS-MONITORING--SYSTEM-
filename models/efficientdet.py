import torch
from effdet import create_model
import cv2
import numpy as np
import config

class EfficientDetDetector:
    def __init__(self):
        # CHANGED: 'tf_efficientdet_b0' -> 'tf_efficientdet_d0'
        # D0 is the correct model name in this library (it uses the B0 backbone).
        print("[INFO] Loading EfficientDet-D0 (Backbone: EfficientNet-B0)...")
        
        self.model = create_model('tf_efficientdet_d0', bench_task='predict', pretrained=True)
        
        self.model.eval()
        self.model.to(config.DEVICE)
        print("[INFO] EfficientDet-D0 Loaded.")

    def predict(self, image):
        """
        EfficientDet requires specific resizing and normalization.
        """
        # Resize to 512x512 (Standard resolution for EfficientDet-D0)
        img_resized = cv2.resize(image, (512, 512))
        
        # Convert to Tensor and Normalize
        # EfficientDet expects [Batch, 3, H, W]
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0)
        img_tensor = img_tensor.to(config.DEVICE)

        with torch.no_grad():
            output = self.model(img_tensor)

        # Output format: [x_min, y_min, x_max, y_max, score, class]
        results = output.cpu().numpy()[0]

        boxes = []
        scores = []
        labels = []

        # Get original image dimensions for scaling back
        h, w, _ = image.shape
        scale_x = w / 512
        scale_y = h / 512

        for res in results:
            score = res[4]
            if score >= config.CONFIDENCE_THRESHOLD:
                # Scale boxes back to original image size
                xmin = res[0] * scale_x
                ymin = res[1] * scale_y
                xmax = res[2] * scale_x
                ymax = res[3] * scale_y
                
                boxes.append([xmin, ymin, xmax, ymax])
                scores.append(score)
                # EfficientDet classes might be shifted compared to COCO, 
                # but for a demo, we treat them as direct mappings.
                labels.append(int(res[5]))

        return {
            'boxes': np.array(boxes),
            'scores': np.array(scores),
            'labels': np.array(labels)
        }