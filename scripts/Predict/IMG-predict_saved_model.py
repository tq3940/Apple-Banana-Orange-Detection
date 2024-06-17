import time
print('Importing... ')
start_time = time.time()


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import re
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings


end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {:.2f} seconds\n'.format(elapsed_time))


def Load_model(PATH_TO_SAVED_MODEL):
    '''
        加载 SAVED_MODEL 模式的模型
    '''
    print('Loading model... ')
    start_time = time.time()

        
    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.compat.v2.distribute.MirroredStrategy()

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    detection_model = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {:.2f} seconds'.format(elapsed_time))

    return detection_model


def load_image_into_numpy_array(path):
    '''
        用numpy数组加载image
    '''
    img = Image.open(path)
    # print(img.mode)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def inference(image_path, detection_model):
    '''
        return detections, image_np
    '''

    print('Running inference for {}... '.format(image_path))
    start_time = time.time()
    
    image_np = load_image_into_numpy_array(image_path)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # 模型预测
    detections = detection_model(input_tensor)


    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {:.2f} seconds\n'.format(elapsed_time))

    return detections, image_np


def infer_and_show_result(IMAGE_PATHS, detection_model, category_index):

    for image_path in IMAGE_PATHS:

        detections, image_np = inference(image_path, detection_model)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 0
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.40,
                agnostic_mode=False)

        plt.figure()
        plt.imshow(image_np_with_detections)

    plt.show()

if __name__ == "__main__":

    # 待修改为相对路径
    IMG_DIR = "PATH_TO_IMAMGE"
  
    IMAGE_PATHS= [os.path.join(IMG_DIR,f) for f in os.listdir(IMG_DIR)
                if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.jpg|.jpeg|.png)$', f)]

    PATH_TO_MODEL_DIR = "PATH_TO_MODEL_DIR"
    PATH_TO_LABELS = "PATH_TO_LABEL_MAP"


    PATH_TO_SAVED_MODEL = os.path.join(PATH_TO_MODEL_DIR ,"saved_model")

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
    detection_model = Load_model(PATH_TO_SAVED_MODEL)

    # IMAGE_PATHS = IMAGE_PATHS[:5]


    infer_and_show_result(IMAGE_PATHS, detection_model, category_index)

