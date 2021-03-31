from config import Config

class SunRGBDConfig(Config):

    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "sun"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 13  # Background + balloon

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

    CLASS_NAMES = ['BG', 'bed', 'books', 'ceiling', 'chair', 'floor',
                   'furniture', 'objects', 'picture', 'sofa', 'table',
                   'tv', 'wall', 'window']

    FOCUSED_NAMES = ['BG', 'bed', 'books', 'ceiling', 'chair', 'floor',
                   'furniture', 'objects', 'picture', 'sofa', 'table',
                   'tv', 'wall', 'window']

    CLASS_COLORS = [(0, 0, 0), (119, 119, 119), (244, 243, 131),
                    (137, 28, 157), (150, 255, 255), (54, 114, 113),
                    (0, 0, 176), (255, 69, 0), (87, 112, 255), (0, 163, 33),
                    (255, 150, 255), (255, 180, 10), (101, 70, 86),
                    (38, 230, 0)]