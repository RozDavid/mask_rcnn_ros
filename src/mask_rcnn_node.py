#!/usr/bin/env python3
import os
import sys
import argparse
import threading
import numpy as np
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath + "/src/mask_rcnn_ros")
COCO_MODEL_PATH = '../models/mask_rcnn_coco.h5'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import coco
import sunrgbd
import utils
import model as modellib
import visualize
from mask_rcnn_ros.msg import Result

RGB_TOPIC = '/hp_laptop/color/image_color'

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MaskRCNNNode(object):
    def __init__(self):
        self._cv_bridge = CvBridge()

        parser = argparse.ArgumentParser()

        # Parse arguments for global params
        parser.add_argument('--input_rgb_topic', type=str, default='/hp_laptop/color/image_color',
                            help='RGB images to run inference on')
        parser.add_argument('--model_path', type=str, default=COCO_MODEL_PATH,
                            help='MASK RCNN checkpoint path - download from the matterport github repo')
        parser.add_argument('--dataset_name', type=str, default="coco",
                            help='Either Coco or SunRGBD training set and output labels')
        parser.add_argument("--visualization", default=True, type=utils.str2bool, nargs='?',
                            help="If we would like to visualize the results with masks, names and confidence")
        parser.add_argument("--semantic_topic_name", default='/semantics/semantic_image',
                            help="The output topic name for the semantically segmented image")
        self.args = parser.parse_args()

        self._rgb_input_topic = self.args.input_rgb_topic
        self._model_path = self.args.model_path
        self._visualization = self.args.visualization
        self._semantic_topic_name = self.args.semantic_topic_name
        self._dataset_name = self.args.dataset_name

        if self._dataset_name == "coco":
            config = coco.CocoConfig()
            config.display()

        elif self._dataset_name == "sunrgbd":
            config = sunrgbd.SunRGBDConfig()
            config.display()
        else:
            raise ValueError('Unsupported models type. Should be "sunrgbd" or "coco"')

        # Create models object in inference mode.
        self._model = modellib.MaskRCNN(mode="inference", model_dir="",
                                        config=config)

        # Load weights trained on MS-COCO
        self._model.load_weights(self._model_path, by_name=True)

        self._class_names = config.CLASS_NAMES
        self._focused_names = config.FOCUSED_NAMES
        self._class_colors = config.CLASS_COLORS


        self._last_msg = None
        self._msg_lock = threading.Lock()

        self._publish_rate = 20

    def run(self):
        self._result_pub = rospy.Publisher('~result', Result, queue_size=1)
        vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)
        semantics_publisher = rospy.Publisher(self._semantic_topic_name, Image, queue_size=1)

        rospy.Subscriber(self._rgb_input_topic, Image,
                         self._image_callback, queue_size=1)

        rate = rospy.Rate(self._publish_rate)
        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if msg is not None:

                np_image = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8').astype(np.float32)

                # Run detection
                results = self._model.detect([np_image], verbose=0)
                result = results[0]
                result_msg = self._build_result_msg(msg, result)
                self._result_pub.publish(result_msg)

                # Visualize results
                if self._visualization:
                    visualization_result = self._visualize_plt(result, np_image)
                    image_msg = self._cv_bridge.cv2_to_imgmsg(visualization_result, 'rgb8')
                    vis_pub.publish(image_msg)

                bg_image = np.zeros(np_image.shape, dtype=np_image.dtype)
                semantic_result = self._semantic_plt(result, bg_image).astype(np.uint8)
                image_msg = self._cv_bridge.cv2_to_imgmsg(semantic_result, 'rgb8')
                image_msg.header = msg.header
                semantics_publisher.publish(image_msg)

            rate.sleep()

    def _build_result_msg(self, msg, result):
        result_msg = Result()
        result_msg.header = msg.header
        for i, (y1, x1, y2, x2) in enumerate(result['rois']):
            box = RegionOfInterest()
            box.x_offset = x1.item()
            box.y_offset = y1.item()
            box.height = (y2 - y1).item()
            box.width = (x2 - x1).item()
            result_msg.boxes.append(box)

            class_id = result['class_ids'][i]
            result_msg.class_ids.append(class_id)

            class_name = self._class_names[class_id]
            if class_name in self._focused_names:
                result_msg.class_names.append(class_name)

            score = result['scores'][i]
            result_msg.scores.append(score)

            mask = Image()
            mask.header = msg.header
            mask.height = result['masks'].shape[0]
            mask.width = result['masks'].shape[1]
            mask.encoding = "mono8"
            mask.is_bigendian = False
            mask.step = mask.width
            mask.data = (result['masks'][:, :, i] * 255).tobytes()
            result_msg.masks.append(mask)
        return result_msg

    def _visualize(self, result, image):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        axes = fig.gca()
        visualize.display_instances(image, result['rois'], result['masks'],
                                    result['class_ids'], self._class_names,
                                    result['scores'], ax=axes,
                                    class_colors=self._class_colors)
        fig.tight_layout()
        canvas.draw()
        result = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

        _, _, w, h = fig.bbox.bounds
        result = result.reshape((int(h), int(w), 3))
        return result

    def _get_fig_ax(self):
        """Return a Matplotlib Axes array to be used in
        all visualizations. Provide a
        central point to control graph sizes.

        Change the default size attribute to control the size
        of rendered images
        """
        fig, ax = plt.subplots(1)
        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        return fig, ax

    def _visualize_plt(self, result, image):
        fig, ax = self._get_fig_ax()
        image = visualize.display_instances_plt(
            image,
            result['rois'],
            result['masks'],
            result['class_ids'],
            self._class_names,
            result['scores'],
            fig=fig,
            ax=ax)

        return image

    def _semantic_plt(self, result, image):

        image = visualize.display_masks_plt(
            image,
            result['masks'],
            result['class_ids'],
            self._class_names,
            result['scores'],
            class_colors=self._class_colors)

        return image

    def _image_callback(self, msg):
        rospy.logdebug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()


def main():
    rospy.init_node('mask_rcnn')

    node = MaskRCNNNode()
    node.run()


if __name__ == '__main__':
    main()
