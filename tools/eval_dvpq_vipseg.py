import argparse
import os

import mmcv
import numpy as np
import six
import multiprocessing as mp

CLASSES = [
    {"id": 0, "name": "wall", "isthing": 0, "color": [120, 120, 120]},
    {"id": 1, "name": "ceiling", "isthing": 0, "color": [180, 120, 120]},
    {"id": 2, "name": "door", "isthing": 1, "color": [6, 230, 230]},
    {"id": 3, "name": "stair", "isthing": 0, "color": [80, 50, 50]},
    {"id": 4, "name": "ladder", "isthing": 1, "color": [4, 200, 3]},
    {"id": 5, "name": "escalator", "isthing": 0, "color": [120, 120, 80]},
    {"id": 6, "name": "Playground_slide", "isthing": 0, "color": [140, 140, 140]},
    {"id": 7, "name": "handrail_or_fence", "isthing": 0, "color": [204, 5, 255]},
    {"id": 8, "name": "window", "isthing": 1, "color": [230, 230, 230]},
    {"id": 9, "name": "rail", "isthing": 0, "color": [4, 250, 7]},
    {"id": 10, "name": "goal", "isthing": 1, "color": [224, 5, 255]},
    {"id": 11, "name": "pillar", "isthing": 0, "color": [235, 255, 7]},
    {"id": 12, "name": "pole", "isthing": 0, "color": [150, 5, 61]},
    {"id": 13, "name": "floor", "isthing": 0, "color": [120, 120, 70]},
    {"id": 14, "name": "ground", "isthing": 0, "color": [8, 255, 51]},
    {"id": 15, "name": "grass", "isthing": 0, "color": [255, 6, 82]},
    {"id": 16, "name": "sand", "isthing": 0, "color": [143, 255, 140]},
    {"id": 17, "name": "athletic_field", "isthing": 0, "color": [204, 255, 4]},
    {"id": 18, "name": "road", "isthing": 0, "color": [255, 51, 7]},
    {"id": 19, "name": "path", "isthing": 0, "color": [204, 70, 3]},
    {"id": 20, "name": "crosswalk", "isthing": 0, "color": [0, 102, 200]},
    {"id": 21, "name": "building", "isthing": 0, "color": [61, 230, 250]},
    {"id": 22, "name": "house", "isthing": 0, "color": [255, 6, 51]},
    {"id": 23, "name": "bridge", "isthing": 0, "color": [11, 102, 255]},
    {"id": 24, "name": "tower", "isthing": 0, "color": [255, 7, 71]},
    {"id": 25, "name": "windmill", "isthing": 0, "color": [255, 9, 224]},
    {"id": 26, "name": "well_or_well_lid", "isthing": 0, "color": [9, 7, 230]},
    {"id": 27, "name": "other_construction", "isthing": 0, "color": [220, 220, 220]},
    {"id": 28, "name": "sky", "isthing": 0, "color": [255, 9, 92]},
    {"id": 29, "name": "mountain", "isthing": 0, "color": [112, 9, 255]},
    {"id": 30, "name": "stone", "isthing": 0, "color": [8, 255, 214]},
    {"id": 31, "name": "wood", "isthing": 0, "color": [7, 255, 224]},
    {"id": 32, "name": "ice", "isthing": 0, "color": [255, 184, 6]},
    {"id": 33, "name": "snowfield", "isthing": 0, "color": [10, 255, 71]},
    {"id": 34, "name": "grandstand", "isthing": 0, "color": [255, 41, 10]},
    {"id": 35, "name": "sea", "isthing": 0, "color": [7, 255, 255]},
    {"id": 36, "name": "river", "isthing": 0, "color": [224, 255, 8]},
    {"id": 37, "name": "lake", "isthing": 0, "color": [102, 8, 255]},
    {"id": 38, "name": "waterfall", "isthing": 0, "color": [255, 61, 6]},
    {"id": 39, "name": "water", "isthing": 0, "color": [255, 194, 7]},
    {"id": 40, "name": "billboard_or_Bulletin_Board", "isthing": 0, "color": [255, 122, 8]},
    {"id": 41, "name": "sculpture", "isthing": 1, "color": [0, 255, 20]},
    {"id": 42, "name": "pipeline", "isthing": 0, "color": [255, 8, 41]},
    {"id": 43, "name": "flag", "isthing": 1, "color": [255, 5, 153]},
    {"id": 44, "name": "parasol_or_umbrella", "isthing": 1, "color": [6, 51, 255]},
    {"id": 45, "name": "cushion_or_carpet", "isthing": 0, "color": [235, 12, 255]},
    {"id": 46, "name": "tent", "isthing": 1, "color": [160, 150, 20]},
    {"id": 47, "name": "roadblock", "isthing": 1, "color": [0, 163, 255]},
    {"id": 48, "name": "car", "isthing": 1, "color": [140, 140, 140]},
    {"id": 49, "name": "bus", "isthing": 1, "color": [250, 10, 15]},
    {"id": 50, "name": "truck", "isthing": 1, "color": [20, 255, 0]},
    {"id": 51, "name": "bicycle", "isthing": 1, "color": [31, 255, 0]},
    {"id": 52, "name": "motorcycle", "isthing": 1, "color": [255, 31, 0]},
    {"id": 53, "name": "wheeled_machine", "isthing": 0, "color": [255, 224, 0]},
    {"id": 54, "name": "ship_or_boat", "isthing": 1, "color": [153, 255, 0]},
    {"id": 55, "name": "raft", "isthing": 1, "color": [0, 0, 255]},
    {"id": 56, "name": "airplane", "isthing": 1, "color": [255, 71, 0]},
    {"id": 57, "name": "tyre", "isthing": 0, "color": [0, 235, 255]},
    {"id": 58, "name": "traffic_light", "isthing": 0, "color": [0, 173, 255]},
    {"id": 59, "name": "lamp", "isthing": 0, "color": [31, 0, 255]},
    {"id": 60, "name": "person", "isthing": 1, "color": [11, 200, 200]},
    {"id": 61, "name": "cat", "isthing": 1, "color": [255, 82, 0]},
    {"id": 62, "name": "dog", "isthing": 1, "color": [0, 255, 245]},
    {"id": 63, "name": "horse", "isthing": 1, "color": [0, 61, 255]},
    {"id": 64, "name": "cattle", "isthing": 1, "color": [0, 255, 112]},
    {"id": 65, "name": "other_animal", "isthing": 1, "color": [0, 255, 133]},
    {"id": 66, "name": "tree", "isthing": 0, "color": [255, 0, 0]},
    {"id": 67, "name": "flower", "isthing": 0, "color": [255, 163, 0]},
    {"id": 68, "name": "other_plant", "isthing": 0, "color": [255, 102, 0]},
    {"id": 69, "name": "toy", "isthing": 0, "color": [194, 255, 0]},
    {"id": 70, "name": "ball_net", "isthing": 0, "color": [0, 143, 255]},
    {"id": 71, "name": "backboard", "isthing": 0, "color": [51, 255, 0]},
    {"id": 72, "name": "skateboard", "isthing": 1, "color": [0, 82, 255]},
    {"id": 73, "name": "bat", "isthing": 0, "color": [0, 255, 41]},
    {"id": 74, "name": "ball", "isthing": 1, "color": [0, 255, 173]},
    {"id": 75, "name": "cupboard_or_showcase_or_storage_rack", "isthing": 0, "color": [10, 0, 255]},
    {"id": 76, "name": "box", "isthing": 1, "color": [173, 255, 0]},
    {"id": 77, "name": "traveling_case_or_trolley_case", "isthing": 1, "color": [0, 255, 153]},
    {"id": 78, "name": "basket", "isthing": 1, "color": [255, 92, 0]},
    {"id": 79, "name": "bag_or_package", "isthing": 1, "color": [255, 0, 255]},
    {"id": 80, "name": "trash_can", "isthing": 0, "color": [255, 0, 245]},
    {"id": 81, "name": "cage", "isthing": 0, "color": [255, 0, 102]},
    {"id": 82, "name": "plate", "isthing": 1, "color": [255, 173, 0]},
    {"id": 83, "name": "tub_or_bowl_or_pot", "isthing": 1, "color": [255, 0, 20]},
    {"id": 84, "name": "bottle_or_cup", "isthing": 1, "color": [255, 184, 184]},
    {"id": 85, "name": "barrel", "isthing": 1, "color": [0, 31, 255]},
    {"id": 86, "name": "fishbowl", "isthing": 1, "color": [0, 255, 61]},
    {"id": 87, "name": "bed", "isthing": 1, "color": [0, 71, 255]},
    {"id": 88, "name": "pillow", "isthing": 1, "color": [255, 0, 204]},
    {"id": 89, "name": "table_or_desk", "isthing": 1, "color": [0, 255, 194]},
    {"id": 90, "name": "chair_or_seat", "isthing": 1, "color": [0, 255, 82]},
    {"id": 91, "name": "bench", "isthing": 1, "color": [0, 10, 255]},
    {"id": 92, "name": "sofa", "isthing": 1, "color": [0, 112, 255]},
    {"id": 93, "name": "shelf", "isthing": 0, "color": [51, 0, 255]},
    {"id": 94, "name": "bathtub", "isthing": 0, "color": [0, 194, 255]},
    {"id": 95, "name": "gun", "isthing": 1, "color": [0, 122, 255]},
    {"id": 96, "name": "commode", "isthing": 1, "color": [0, 255, 163]},
    {"id": 97, "name": "roaster", "isthing": 1, "color": [255, 153, 0]},
    {"id": 98, "name": "other_machine", "isthing": 0, "color": [0, 255, 10]},
    {"id": 99, "name": "refrigerator", "isthing": 1, "color": [255, 112, 0]},
    {"id": 100, "name": "washing_machine", "isthing": 1, "color": [143, 255, 0]},
    {"id": 101, "name": "Microwave_oven", "isthing": 1, "color": [82, 0, 255]},
    {"id": 102, "name": "fan", "isthing": 1, "color": [163, 255, 0]},
    {"id": 103, "name": "curtain", "isthing": 0, "color": [255, 235, 0]},
    {"id": 104, "name": "textiles", "isthing": 0, "color": [8, 184, 170]},
    {"id": 105, "name": "clothes", "isthing": 0, "color": [133, 0, 255]},
    {"id": 106, "name": "painting_or_poster", "isthing": 1, "color": [0, 255, 92]},
    {"id": 107, "name": "mirror", "isthing": 1, "color": [184, 0, 255]},
    {"id": 108, "name": "flower_pot_or_vase", "isthing": 1, "color": [255, 0, 31]},
    {"id": 109, "name": "clock", "isthing": 1, "color": [0, 184, 255]},
    {"id": 110, "name": "book", "isthing": 0, "color": [0, 214, 255]},
    {"id": 111, "name": "tool", "isthing": 0, "color": [255, 0, 112]},
    {"id": 112, "name": "blackboard", "isthing": 0, "color": [92, 255, 0]},
    {"id": 113, "name": "tissue", "isthing": 0, "color": [0, 224, 255]},
    {"id": 114, "name": "screen_or_television", "isthing": 1, "color": [112, 224, 255]},
    {"id": 115, "name": "computer", "isthing": 1, "color": [70, 184, 160]},
    {"id": 116, "name": "printer", "isthing": 1, "color": [163, 0, 255]},
    {"id": 117, "name": "Mobile_phone", "isthing": 1, "color": [153, 0, 255]},
    {"id": 118, "name": "keyboard", "isthing": 1, "color": [71, 255, 0]},
    {"id": 119, "name": "other_electronic_product", "isthing": 0, "color": [255, 0, 163]},
    {"id": 120, "name": "fruit", "isthing": 0, "color": [255, 204, 0]},
    {"id": 121, "name": "food", "isthing": 0, "color": [255, 0, 143]},
    {"id": 122, "name": "instrument", "isthing": 1, "color": [0, 255, 235]},
    {"id": 123, "name": "train", "isthing": 1, "color": [133, 255, 0]}
]

CLASSES_THING = [
    {'id': 2, 'name': 'door', 'isthing': 1, 'color': [6, 230, 230]},
    {'id': 4, 'name': 'ladder', 'isthing': 1, 'color': [4, 200, 3]},
    {'id': 8, 'name': 'window', 'isthing': 1, 'color': [230, 230, 230]},
    {'id': 10, 'name': 'goal', 'isthing': 1, 'color': [224, 5, 255]},
    {'id': 41, 'name': 'sculpture', 'isthing': 1, 'color': [0, 255, 20]},
    {'id': 43, 'name': 'flag', 'isthing': 1, 'color': [255, 5, 153]},
    {'id': 44, 'name': 'parasol_or_umbrella', 'isthing': 1, 'color': [6, 51, 255]},
    {'id': 46, 'name': 'tent', 'isthing': 1, 'color': [160, 150, 20]},
    {'id': 47, 'name': 'roadblock', 'isthing': 1, 'color': [0, 163, 255]},
    {'id': 48, 'name': 'car', 'isthing': 1, 'color': [140, 140, 140]},
    {'id': 49, 'name': 'bus', 'isthing': 1, 'color': [250, 10, 15]},
    {'id': 50, 'name': 'truck', 'isthing': 1, 'color': [20, 255, 0]},
    {'id': 51, 'name': 'bicycle', 'isthing': 1, 'color': [31, 255, 0]},
    {'id': 52, 'name': 'motorcycle', 'isthing': 1, 'color': [255, 31, 0]},
    {'id': 54, 'name': 'ship_or_boat', 'isthing': 1, 'color': [153, 255, 0]},
    {'id': 55, 'name': 'raft', 'isthing': 1, 'color': [0, 0, 255]},
    {'id': 56, 'name': 'airplane', 'isthing': 1, 'color': [255, 71, 0]},
    {'id': 60, 'name': 'person', 'isthing': 1, 'color': [11, 200, 200]},
    {'id': 61, 'name': 'cat', 'isthing': 1, 'color': [255, 82, 0]},
    {'id': 62, 'name': 'dog', 'isthing': 1, 'color': [0, 255, 245]},
    {'id': 63, 'name': 'horse', 'isthing': 1, 'color': [0, 61, 255]},
    {'id': 64, 'name': 'cattle', 'isthing': 1, 'color': [0, 255, 112]},
    {'id': 65, 'name': 'other_animal', 'isthing': 1, 'color': [0, 255, 133]},
    {'id': 72, 'name': 'skateboard', 'isthing': 1, 'color': [0, 82, 255]},
    {'id': 74, 'name': 'ball', 'isthing': 1, 'color': [0, 255, 173]},
    {'id': 76, 'name': 'box', 'isthing': 1, 'color': [173, 255, 0]},
    {'id': 77, 'name': 'traveling_case_or_trolley_case', 'isthing': 1, 'color': [0, 255, 153]},
    {'id': 78, 'name': 'basket', 'isthing': 1, 'color': [255, 92, 0]},
    {'id': 79, 'name': 'bag_or_package', 'isthing': 1, 'color': [255, 0, 255]},
    {'id': 82, 'name': 'plate', 'isthing': 1, 'color': [255, 173, 0]},
    {'id': 83, 'name': 'tub_or_bowl_or_pot', 'isthing': 1, 'color': [255, 0, 20]},
    {'id': 84, 'name': 'bottle_or_cup', 'isthing': 1, 'color': [255, 184, 184]},
    {'id': 85, 'name': 'barrel', 'isthing': 1, 'color': [0, 31, 255]},
    {'id': 86, 'name': 'fishbowl', 'isthing': 1, 'color': [0, 255, 61]},
    {'id': 87, 'name': 'bed', 'isthing': 1, 'color': [0, 71, 255]},
    {'id': 88, 'name': 'pillow', 'isthing': 1, 'color': [255, 0, 204]},
    {'id': 89, 'name': 'table_or_desk', 'isthing': 1, 'color': [0, 255, 194]},
    {'id': 90, 'name': 'chair_or_seat', 'isthing': 1, 'color': [0, 255, 82]},
    {'id': 91, 'name': 'bench', 'isthing': 1, 'color': [0, 10, 255]},
    {'id': 92, 'name': 'sofa', 'isthing': 1, 'color': [0, 112, 255]},
    {'id': 95, 'name': 'gun', 'isthing': 1, 'color': [0, 122, 255]},
    {'id': 96, 'name': 'commode', 'isthing': 1, 'color': [0, 255, 163]},
    {'id': 97, 'name': 'roaster', 'isthing': 1, 'color': [255, 153, 0]},
    {'id': 99, 'name': 'refrigerator', 'isthing': 1, 'color': [255, 112, 0]},
    {'id': 100, 'name': 'washing_machine', 'isthing': 1, 'color': [143, 255, 0]},
    {'id': 101, 'name': 'Microwave_oven', 'isthing': 1, 'color': [82, 0, 255]},
    {'id': 102, 'name': 'fan', 'isthing': 1, 'color': [163, 255, 0]},
    {'id': 106, 'name': 'painting_or_poster', 'isthing': 1, 'color': [0, 255, 92]},
    {'id': 107, 'name': 'mirror', 'isthing': 1, 'color': [184, 0, 255]},
    {'id': 108, 'name': 'flower_pot_or_vase', 'isthing': 1, 'color': [255, 0, 31]},
    {'id': 109, 'name': 'clock', 'isthing': 1, 'color': [0, 184, 255]},
    {'id': 114, 'name': 'screen_or_television', 'isthing': 1, 'color': [112, 224, 255]},
    {'id': 115, 'name': 'computer', 'isthing': 1, 'color': [70, 184, 160]},
    {'id': 116, 'name': 'printer', 'isthing': 1, 'color': [163, 0, 255]},
    {'id': 117, 'name': 'Mobile_phone', 'isthing': 1, 'color': [153, 0, 255]},
    {'id': 118, 'name': 'keyboard', 'isthing': 1, 'color': [71, 255, 0]},
    {'id': 122, 'name': 'instrument', 'isthing': 1, 'color': [0, 255, 235]},
    {'id': 123, 'name': 'train', 'isthing': 1, 'color': [133, 255, 0]}
]

CLASSES_STUFF = [
    {'id': 0, 'name': 'wall', 'isthing': 0, 'color': [120, 120, 120]},
    {'id': 1, 'name': 'ceiling', 'isthing': 0, 'color': [180, 120, 120]},
    {'id': 3, 'name': 'stair', 'isthing': 0, 'color': [80, 50, 50]},
    {'id': 5, 'name': 'escalator', 'isthing': 0, 'color': [120, 120, 80]},
    {'id': 6, 'name': 'Playground_slide', 'isthing': 0, 'color': [140, 140, 140]},
    {'id': 7, 'name': 'handrail_or_fence', 'isthing': 0, 'color': [204, 5, 255]},
    {'id': 9, 'name': 'rail', 'isthing': 0, 'color': [4, 250, 7]},
    {'id': 11, 'name': 'pillar', 'isthing': 0, 'color': [235, 255, 7]},
    {'id': 12, 'name': 'pole', 'isthing': 0, 'color': [150, 5, 61]},
    {'id': 13, 'name': 'floor', 'isthing': 0, 'color': [120, 120, 70]},
    {'id': 14, 'name': 'ground', 'isthing': 0, 'color': [8, 255, 51]},
    {'id': 15, 'name': 'grass', 'isthing': 0, 'color': [255, 6, 82]},
    {'id': 16, 'name': 'sand', 'isthing': 0, 'color': [143, 255, 140]},
    {'id': 17, 'name': 'athletic_field', 'isthing': 0, 'color': [204, 255, 4]},
    {'id': 18, 'name': 'road', 'isthing': 0, 'color': [255, 51, 7]},
    {'id': 19, 'name': 'path', 'isthing': 0, 'color': [204, 70, 3]},
    {'id': 20, 'name': 'crosswalk', 'isthing': 0, 'color': [0, 102, 200]},
    {'id': 21, 'name': 'building', 'isthing': 0, 'color': [61, 230, 250]},
    {'id': 22, 'name': 'house', 'isthing': 0, 'color': [255, 6, 51]},
    {'id': 23, 'name': 'bridge', 'isthing': 0, 'color': [11, 102, 255]},
    {'id': 24, 'name': 'tower', 'isthing': 0, 'color': [255, 7, 71]},
    {'id': 25, 'name': 'windmill', 'isthing': 0, 'color': [255, 9, 224]},
    {'id': 26, 'name': 'well_or_well_lid', 'isthing': 0, 'color': [9, 7, 230]},
    {'id': 27, 'name': 'other_construction', 'isthing': 0, 'color': [220, 220, 220]},
    {'id': 28, 'name': 'sky', 'isthing': 0, 'color': [255, 9, 92]},
    {'id': 29, 'name': 'mountain', 'isthing': 0, 'color': [112, 9, 255]},
    {'id': 30, 'name': 'stone', 'isthing': 0, 'color': [8, 255, 214]},
    {'id': 31, 'name': 'wood', 'isthing': 0, 'color': [7, 255, 224]},
    {'id': 32, 'name': 'ice', 'isthing': 0, 'color': [255, 184, 6]},
    {'id': 33, 'name': 'snowfield', 'isthing': 0, 'color': [10, 255, 71]},
    {'id': 34, 'name': 'grandstand', 'isthing': 0, 'color': [255, 41, 10]},
    {'id': 35, 'name': 'sea', 'isthing': 0, 'color': [7, 255, 255]},
    {'id': 36, 'name': 'river', 'isthing': 0, 'color': [224, 255, 8]},
    {'id': 37, 'name': 'lake', 'isthing': 0, 'color': [102, 8, 255]},
    {'id': 38, 'name': 'waterfall', 'isthing': 0, 'color': [255, 61, 6]},
    {'id': 39, 'name': 'water', 'isthing': 0, 'color': [255, 194, 7]},
    {'id': 40, 'name': 'billboard_or_Bulletin_Board', 'isthing': 0, 'color': [255, 122, 8]},
    {'id': 42, 'name': 'pipeline', 'isthing': 0, 'color': [255, 8, 41]},
    {'id': 45, 'name': 'cushion_or_carpet', 'isthing': 0, 'color': [235, 12, 255]},
    {'id': 53, 'name': 'wheeled_machine', 'isthing': 0, 'color': [255, 224, 0]},
    {'id': 57, 'name': 'tyre', 'isthing': 0, 'color': [0, 235, 255]},
    {'id': 58, 'name': 'traffic_light', 'isthing': 0, 'color': [0, 173, 255]},
    {'id': 59, 'name': 'lamp', 'isthing': 0, 'color': [31, 0, 255]},
    {'id': 66, 'name': 'tree', 'isthing': 0, 'color': [255, 0, 0]},
    {'id': 67, 'name': 'flower', 'isthing': 0, 'color': [255, 163, 0]},
    {'id': 68, 'name': 'other_plant', 'isthing': 0, 'color': [255, 102, 0]},
    {'id': 69, 'name': 'toy', 'isthing': 0, 'color': [194, 255, 0]},
    {'id': 70, 'name': 'ball_net', 'isthing': 0, 'color': [0, 143, 255]},
    {'id': 71, 'name': 'backboard', 'isthing': 0, 'color': [51, 255, 0]},
    {'id': 73, 'name': 'bat', 'isthing': 0, 'color': [0, 255, 41]},
    {'id': 75, 'name': 'cupboard_or_showcase_or_storage_rack', 'isthing': 0, 'color': [10, 0, 255]},
    {'id': 80, 'name': 'trash_can', 'isthing': 0, 'color': [255, 0, 245]},
    {'id': 81, 'name': 'cage', 'isthing': 0, 'color': [255, 0, 102]},
    {'id': 93, 'name': 'shelf', 'isthing': 0, 'color': [51, 0, 255]},
    {'id': 94, 'name': 'bathtub', 'isthing': 0, 'color': [0, 194, 255]},
    {'id': 98, 'name': 'other_machine', 'isthing': 0, 'color': [0, 255, 10]},
    {'id': 103, 'name': 'curtain', 'isthing': 0, 'color': [255, 235, 0]},
    {'id': 104, 'name': 'textiles', 'isthing': 0, 'color': [8, 184, 170]},
    {'id': 105, 'name': 'clothes', 'isthing': 0, 'color': [133, 0, 255]},
    {'id': 110, 'name': 'book', 'isthing': 0, 'color': [0, 214, 255]},
    {'id': 111, 'name': 'tool', 'isthing': 0, 'color': [255, 0, 112]},
    {'id': 112, 'name': 'blackboard', 'isthing': 0, 'color': [92, 255, 0]},
    {'id': 113, 'name': 'tissue', 'isthing': 0, 'color': [0, 224, 255]},
    {'id': 119, 'name': 'other_electronic_product', 'isthing': 0, 'color': [255, 0, 163]},
    {'id': 120, 'name': 'fruit', 'isthing': 0, 'color': [255, 204, 0]},
    {'id': 121, 'name': 'food', 'isthing': 0, 'color': [255, 0, 143]}
]

NO_OBJ = 0
NO_OBJ_HB = 255
DIVISOR_PAN = 100
DIVISOR_NEW = 1000
NUM_THING = 58
NUM_STUFF = 66
THING_B_STUFF = False


def vip2hb(pan_map):
    assert not THING_B_STUFF, "VIPSeg only supports stuff -> thing"
    pan_new = - np.ones_like(pan_map)
    vip2hb_thing = {itm['id'] + 1: idx for idx, itm in enumerate(CLASSES_THING)}
    vip2hb_stuff = {itm['id'] + 1: idx for idx, itm in enumerate(CLASSES_STUFF)}
    for idx in np.unique(pan_map):
        if idx == NO_OBJ or idx == 200:
            pan_new[pan_map == idx] = NO_OBJ_HB * DIVISOR_NEW
        elif idx > 128:
            cls_id = idx // DIVISOR_PAN
            cls_new_id = vip2hb_thing[cls_id]
            inst_id = idx % DIVISOR_PAN
            # since stuff -> thing
            cls_new_id += NUM_STUFF
            pan_new[pan_map == idx] = cls_new_id * DIVISOR_NEW + inst_id + 1
        else:
            pan_new[pan_map == idx] = vip2hb_stuff[idx] * DIVISOR_NEW
    assert -1. not in np.unique(pan_new)
    return pan_new


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation of DSTQ')
    parser.add_argument('result_path')
    parser.add_argument('--gt-path', default='data/kitti-step')
    parser.add_argument('--split', default='val')
    parser.add_argument(
        '--depth',
        action='store_true',
        help='eval depth')
    parser.add_argument('--nproc', default=32, type=int, help='number of process')
    args = parser.parse_args()
    return args


def vpq_eval(element):
    pred_ids, gt_ids = element
    max_ins = 2 ** 16
    ign_id = 255
    offset = 2 ** 30
    num_cat = NUM_THING + NUM_STUFF + 1

    iou_per_class = np.zeros(num_cat, dtype=np.float64)
    tp_per_class = np.zeros(num_cat, dtype=np.float64)
    fn_per_class = np.zeros(num_cat, dtype=np.float64)
    fp_per_class = np.zeros(num_cat, dtype=np.float64)

    def _ids_to_counts(id_array):
        ids, counts = np.unique(id_array, return_counts=True)
        return dict(six.moves.zip(ids, counts))

    pred_areas = _ids_to_counts(pred_ids)
    gt_areas = _ids_to_counts(gt_ids)

    void_id = ign_id * max_ins
    ign_ids = {
        gt_id for gt_id in six.iterkeys(gt_areas)
        if (gt_id // max_ins) == ign_id
    }

    int_ids = gt_ids.astype(np.int64) * offset + pred_ids.astype(np.int64)
    int_areas = _ids_to_counts(int_ids)

    def prediction_void_overlap(pred_id):
        void_int_id = void_id * offset + pred_id
        return int_areas.get(void_int_id, 0)

    def prediction_ignored_overlap(pred_id):
        total_ignored_overlap = 0
        for _ign_id in ign_ids:
            int_id = _ign_id * offset + pred_id
            total_ignored_overlap += int_areas.get(int_id, 0)
        return total_ignored_overlap

    gt_matched = set()
    pred_matched = set()

    for int_id, int_area in six.iteritems(int_areas):
        gt_id = int(int_id // offset)
        gt_cat = int(gt_id // max_ins)
        pred_id = int(int_id % offset)
        pred_cat = int(pred_id // max_ins)
        if gt_cat != pred_cat:
            continue
        union = (
                gt_areas[gt_id] + pred_areas[pred_id] - int_area -
                prediction_void_overlap(pred_id)
        )
        iou = int_area / union
        if iou > 0.5:
            tp_per_class[gt_cat] += 1
            iou_per_class[gt_cat] += iou
            gt_matched.add(gt_id)
            pred_matched.add(pred_id)

    for gt_id in six.iterkeys(gt_areas):
        if gt_id in gt_matched:
            continue
        cat_id = gt_id // max_ins
        if cat_id == ign_id:
            continue
        fn_per_class[cat_id] += 1

    for pred_id in six.iterkeys(pred_areas):
        if pred_id in pred_matched:
            continue
        if (prediction_ignored_overlap(pred_id) / pred_areas[pred_id]) > 0.5:
            continue
        cat = pred_id // max_ins
        fp_per_class[cat] += 1

    return iou_per_class, tp_per_class, fn_per_class, fp_per_class


def read_to_eval(element):
    max_ins = 2 ** 16

    pred_list, gt_list = element
    pred_cat = [mmcv.imread(image[0], flag='unchanged').astype(np.int32) for image in pred_list]
    pred_ins = [mmcv.imread(image[1], flag='unchanged').astype(np.int32) for image in pred_list]
    pred_cat = np.concatenate(pred_cat, axis=1)
    pred_ins = np.concatenate(pred_ins, axis=1)
    pred = pred_cat.astype(np.int32) * max_ins + pred_ins.astype(np.int32)

    gt_pan = [mmcv.imread(image, flag='unchanged').astype(np.int64) for image in gt_list]
    gt_pan = np.concatenate(gt_pan, axis=1)
    gt_pan = vip2hb(gt_pan)

    gt_cls = gt_pan // DIVISOR_NEW
    gt_ins = gt_pan % DIVISOR_NEW

    gt = gt_cls * max_ins + gt_ins
    result = vpq_eval([pred, gt])

    return result


def eval_dvpq(result_dir, gt_dir, split='val', k=1, with_depth=True):
    if with_depth:
        raise NotImplementedError
    ann_folders = mmcv.list_from_file(os.path.join(gt_dir, "{}.txt".format(split)),
                                      prefix=os.path.join(gt_dir, 'panomasks') + '/')
    seq_ids = np.arange(0, len(ann_folders)).tolist()

    iou_per_class_all = []
    tp_per_class_all = []
    fn_per_class_all = []
    fp_per_class_all = []

    for seq_id in seq_ids:
        gt_names = list(mmcv.scandir(ann_folders[seq_id]))
        gt_pan_names = sorted(list(filter(lambda x: '.png' in x, gt_names)))
        if not os.path.exists(os.path.join(result_dir, 'panoptic', str(seq_id))):
            print("Error when seq_id is {}. But cal existing seqs.".format(seq_id))
            break
        pred_name_panoptic = list(mmcv.scandir(os.path.join(result_dir, 'panoptic', str(seq_id))))
        pred_ins_names = sorted(list(filter(lambda x: 'ins' in x, pred_name_panoptic)))
        pred_cls_names = sorted(list(filter(lambda x: 'cat' in x, pred_name_panoptic)))
        if len(gt_pan_names) != len(pred_ins_names):
            print("Error when seq_id is {}. But cal existing seqs.".format(seq_id))
            break
        elements = []
        assert len(pred_ins_names) == len(pred_cls_names)
        assert len(pred_cls_names) == len(gt_pan_names)
        len_seq = len(pred_ins_names)

        k = min(k, len_seq)

        for idx in range(len_seq):
            if idx + k - 1 >= len_seq:
                break
            pred = []
            gt = []
            for j in range(k):
                pred_cur = (os.path.join(result_dir, 'panoptic', str(seq_id), pred_cls_names[idx + j]),
                            os.path.join(result_dir, 'panoptic', str(seq_id), pred_cls_names[idx + j]))
                gt_cur = os.path.join(ann_folders[seq_id], gt_pan_names[idx + j])
                pred.append(pred_cur)
                gt.append(gt_cur)
            elements.append((pred, gt))

        N = mp.cpu_count()
        with mp.Pool(processes=N) as p:
            results = p.map(read_to_eval, elements)

        iou_per_class = np.stack([result[0] for result in results])
        iou_per_class_all.append(iou_per_class)
        tp_per_class = np.stack([result[1] for result in results])
        tp_per_class_all.append(tp_per_class)
        fn_per_class = np.stack([result[2] for result in results])
        fn_per_class_all.append(fn_per_class)
        fp_per_class = np.stack([result[3] for result in results])
        fp_per_class_all.append(fp_per_class)

    epsilon = 1e-10
    iou_per_class_all = np.concatenate(iou_per_class_all, axis=0).sum(axis=0)[:NUM_THING + NUM_STUFF]
    tp_per_class_all = np.concatenate(tp_per_class_all, axis=0).sum(axis=0)[:NUM_THING + NUM_STUFF]
    fn_per_class_all = np.concatenate(fn_per_class_all, axis=0).sum(axis=0)[:NUM_THING + NUM_STUFF]
    fp_per_class_all = np.concatenate(fp_per_class_all, axis=0).sum(axis=0)[:NUM_THING + NUM_STUFF]

    sq = iou_per_class_all / (tp_per_class_all + epsilon)
    rq = tp_per_class_all / (tp_per_class_all + 0.5 * fn_per_class_all + 0.5 * fp_per_class_all + epsilon)
    pq = sq * rq
    spq = pq[:NUM_STUFF]
    tpq = pq[NUM_STUFF:]
    print(
        r'PQ : {:.3f} PQ_thing : {:.3f} PQ_stuff : {:.3f}'.format(
            pq.mean() * 100,
            tpq.mean() * 100,
            spq.mean() * 100)
    )


# usage python eval_dstq_vipseg.py /opt/data/results/test --gt-path /opt/data/VIPSeg
if __name__ == '__main__':
    args = parse_args()
    result_path = args.result_path
    gt_path = args.gt_path
    split = args.split
    for k in [1, 2, 4, 6]:
        print("k={}".format(k))
        eval_dvpq(result_path, gt_path, split=split, with_depth=args.depth, k=k)
