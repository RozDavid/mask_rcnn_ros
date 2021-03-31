import numpy as np


class COCODataset:

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: CLASS_NAMES.index('teddy bear')
    CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    FOCUSED_NAMES = ['BG', 'person', 'backpack', 'umbrella', 'handbag',
                   'suitcase', 'frisbee', 'bottle', 'wine glass', 'cup', 'bowl',
                   'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop',
                   'keyboard',  'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase']

    SUNRGBD_CLASS_COLORS = [(0, 0, 0), (119, 119, 119), (244, 243, 131),
                    (137, 28, 157), (150, 255, 255), (54, 114, 113),
                    (0, 0, 176), (255, 69, 0), (87, 112, 255), (0, 163, 33),
                    (255, 150, 255), (255, 180, 10), (101, 70, 86),
                    (38, 230, 0), (255, 120, 70), (117, 41, 121),
                    (150, 255, 0), (132, 0, 255), (24, 209, 255),
                    (191, 130, 35), (219, 200, 109), (154, 62, 86),
                    (255, 190, 190), (255, 0, 255), (192, 79, 212),
                    (152, 163, 55), (230, 230, 230), (53, 130, 64),
                    (155, 249, 152), (87, 64, 34), (214, 209, 175),
                    (170, 0, 59), (255, 0, 0), (193, 195, 234), (70, 72, 115),
                    (255, 255, 0), (52, 57, 131), (12, 83, 45)]

def _get_colormap(n):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n, 3), dtype='uint8')
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    return cmap