import numpy as np
import config

class EnsembleModel:
    def __init__(self):
        pass

    def _clip_and_validate_boxes(self, boxes, image_w, image_h):
        if len(boxes) == 0:
            return boxes

        clipped = boxes.copy().astype(np.float32)
        clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, image_w - 1)
        clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, image_h - 1)
        return clipped

    def _nms_indices(self, boxes, scores, iou_threshold):
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(scores)[::-1]
        keep = []

        while order.size > 0:
            current = int(order[0])
            keep.append(current)

            if order.size == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(x1[current], x1[rest])
            yy1 = np.maximum(y1[current], y1[rest])
            xx2 = np.minimum(x2[current], x2[rest])
            yy2 = np.minimum(y2[current], y2[rest])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h

            union = areas[current] + areas[rest] - inter
            iou = np.zeros_like(union, dtype=np.float32)
            valid_union = union > 0
            iou[valid_union] = inter[valid_union] / union[valid_union]

            order = rest[iou <= iou_threshold]

        return keep

    def _filter_relevant_classes(self, pred):
        if len(pred['labels']) == 0:
            return pred

        labels = pred['labels'].astype(int)
        keep_mask = np.array([label in config.ENSEMBLE_TARGET_CLASSES for label in labels], dtype=bool)

        if not np.any(keep_mask):
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'labels': np.array([]),
            }

        return {
            'boxes': pred['boxes'][keep_mask],
            'scores': pred['scores'][keep_mask],
            'labels': pred['labels'][keep_mask],
        }

    def fuse_results(self, predictions_list, image_shape, weights=None):
        """
        Merge predictions from both models and apply class-wise NMS to remove overlaps.
        """
        h, w, _ = image_shape
        all_boxes = []
        all_scores = []
        all_labels = []

        for pred in predictions_list:
            pred = self._filter_relevant_classes(pred)
            if len(pred['boxes']) == 0:
                continue

            boxes = self._clip_and_validate_boxes(pred['boxes'], w, h)
            scores = pred['scores'].astype(np.float32)
            labels = pred['labels'].astype(np.int32)

            valid_mask = (
                (boxes[:, 2] > boxes[:, 0]) &
                (boxes[:, 3] > boxes[:, 1]) &
                (scores >= float(config.CONFIDENCE_THRESHOLD))
            )
            if not np.any(valid_mask):
                continue

            all_boxes.append(boxes[valid_mask])
            all_scores.append(scores[valid_mask])
            all_labels.append(labels[valid_mask])

        if not all_boxes:
            return np.array([]), np.array([]), np.array([])

        boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
        scores = np.concatenate(all_scores, axis=0).astype(np.float32)
        labels = np.concatenate(all_labels, axis=0).astype(np.int32)

        kept_boxes = []
        kept_scores = []
        kept_labels = []

        for class_id in np.unique(labels):
            class_mask = labels == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            class_keep = self._nms_indices(class_boxes, class_scores, float(config.IOU_THRESHOLD))

            if not class_keep:
                continue

            kept_boxes.extend(class_boxes[class_keep].tolist())
            kept_scores.extend(class_scores[class_keep].tolist())
            kept_labels.extend([int(class_id)] * len(class_keep))

        if not kept_boxes:
            return np.array([]), np.array([]), np.array([])

        kept_boxes = np.array(kept_boxes, dtype=np.float32)
        kept_scores = np.array(kept_scores, dtype=np.float32)
        kept_labels = np.array(kept_labels, dtype=np.int32)

        order = np.argsort(kept_scores)[::-1]
        return kept_boxes[order], kept_scores[order], kept_labels[order]