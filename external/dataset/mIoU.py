import numpy as np


def eval_miou(results, targets, num_classes, ignore_index=255):
    total_area_intersect = np.zeros((num_classes,), dtype=np.float64)
    total_area_union = np.zeros((num_classes,), dtype=np.float64)
    total_area_pred = np.zeros((num_classes,), dtype=np.float64)
    total_area_label = np.zeros((num_classes,), dtype=np.float64)

    for result, target in zip(results, targets):
        mask = (target != ignore_index)
        pred = result[mask]
        label = target[mask]

        intersect = pred[pred == label]
        area_intersect, _ = np.histogram(intersect.astype(float), bins=num_classes, range=(0, num_classes - 1))
        area_pred, _ = np.histogram(pred.astype(float), bins=num_classes, range=(0, num_classes - 1))
        area_label, _ = np.histogram(label.astype(float), bins=num_classes, range=(0, num_classes - 1))
        area_union = area_pred + area_label - area_intersect

        total_area_intersect += area_intersect
        total_area_pred += area_intersect
        total_area_label += area_label
        total_area_union += area_union

    iou_per_class = total_area_intersect / total_area_union
    return iou_per_class


if __name__ == '__main__':
    results = [
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ]
    targets = [
        np.array([[1, 2, 3], [1, 1, 2], [255, 255, 255]])
    ]
    eval_miou(results, targets, 19)
