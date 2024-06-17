import time
print('Importing... ')
start_time = time.time()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import re
from object_detection.utils import visualization_utils as viz_utils
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
import numpy as np
from PIL import Image
import evaluate_util as eval
import xml.etree.ElementTree as ET
from tabulate import tabulate

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

    return detections, image_np.shape[0], image_np.shape[1]


def get_DTboxes(detections, im_width, im_height, confidence_threshold=0.5):
    '''
        读取detections中预测框信息
        返回一张图片的DTboxes
    '''

    num_detections = int(detections["num_detections"])
    DTboxes = []

    for i in range(num_detections):
        confidence = float(detections['detection_scores'][0][i])
        if confidence < confidence_threshold:
            continue

        label = int(detections['detection_classes'][0][i]) 

        box = detections['detection_boxes'][0][i]
        # ymin, xmin, ymax, xmax = box
        y_min = int( box[0]*im_height) 
        x_min = int( box[1]*im_width ) 
        y_max = int( box[2]*im_height) 
        x_max = int( box[3]*im_width ) 

        DTbox = (label, confidence, x_min, y_min, x_max, y_max)
        DTboxes.append(DTbox)

    return DTboxes


def get_GTboxes(image_path, label_to_int):
    '''
        读取与img相同目录下、相同名字的xml
        返回一张图片的GTboxes
    '''

    GTboxes = []
    xml_path = re.sub(r"(.jpg|.jpeg|.png)$",".xml", image_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for object in root.findall('object'):
        # Get the box information
        name = object.find('name').text
        xmin = int(object.find('bndbox').find('xmin').text)
        ymin = int(object.find('bndbox').find('ymin').text)
        xmax = int(object.find('bndbox').find('xmax').text)
        ymax = int(object.find('bndbox').find('ymax').text)

        # Add the box to the list
        GTbox = (label_to_int[name], xmin, ymin, xmax, ymax)
        GTboxes.append(GTbox)

    return GTboxes

def eval_images(IMAGE_PATHS, detection_model, label_to_int, eval_index):
    '''
        评测一组图片
    '''

    GTboxesList = []
    DTboxesList = []

    for image_path in IMAGE_PATHS:
        
        detections, im_height, im_width = inference(image_path, detection_model)

        DTboxes = get_DTboxes(detections, im_width, im_height)
        GTboxes = get_GTboxes(image_path, label_to_int)
        GTboxesList.append(GTboxes)
        DTboxesList.append(DTboxes)
   
    print("GTboxesList:")
    for GTboxes in GTboxesList:
        print(GTboxes)

    print("\nDTboxesList:")
    for DTboxes in DTboxesList:
        print(DTboxes)

    evalution_result = eval.evaluta_main(GTboxesList, DTboxesList, eval_index)
    return evalution_result


def load_img_paths(IMG_DIR):
    '''
        加载图片地址，要求所有图片都在同一目录下
    '''

    img_paths_list = []
    for fruit in ["apple", "banana", "orange", "mixed"]:
        img_paths_list.append( [IMG_DIR+"\\"+f for f in os.listdir(IMG_DIR)  
                                if re.search(fruit+'_\d\d.jpg', f)] )
    return img_paths_list


def output_result(eval_list, IMG_DIR):
    '''
        输出表格以及保存为csv
    '''
    table = [['Class', 'IoU', 'Precision', 'Recall', 'Mean Average Precision'],
             ['All']    +   eval_list[0],
             ['Apple']  +   eval_list[1],
             ['Banana'] +   eval_list[2],
             ['Orange'] +   eval_list[3]]
    
    print("\n\n\n****** Final Evaluation Result ******\n")
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    import csv
    with open("./eval-result.csv", "a", encoding="utf-8", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(table)
        csv_writer.writerow(["Images from:", IMG_DIR])
        csv_writer.writerow([""])
        print("Eval result is saved in "+os.path.join(os.getcwd(),"eval-result.csv"))
        file.close()


if __name__ == "__main__":

    # IMG_DIR = "H:\\Fruit_Detection\\workspace\\images\\Extended_Dataset\\test"
    # IMG_DIR = "H:\\Fruit_Detection\\workspace\\scripts\\eval\\eval_test_dataset"
    # IMG_DIR = "H:\\Fruit_Detection\\workspace\\\images\\Kaggle\\test"

    IMG_DIR = os.path.join(os.getcwd(), "test")
    PATH_TO_SAVED_MODEL = os.path.join(os.getcwd(), "model", "saved_model")

    if os.path.exists(IMG_DIR) == False:
        print("Images dir doesn't exist at "+IMG_DIR+"!")
        raise FileNotFoundError
    if os.path.exists(PATH_TO_SAVED_MODEL) == False:
        print("Saved_model dir doesn't exist at "+PATH_TO_SAVED_MODEL+"!")
        raise FileNotFoundError



    # PATH_TO_MODEL_DIR = "H:\\Fruit_Detection\\workspace\\eval-model\\exported"
    # PATH_TO_MODEL_DIR = "H:\\Fruit_Detection\\workspace\\exported-models\\My_d3_4_exported"

    # PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

    label_to_int = {"apple":1, "banana":2, "orange":3}
    
    detection_model = Load_model(PATH_TO_SAVED_MODEL)
    
    img_paths_list = load_img_paths(IMG_DIR)


    # apple_paths= [IMG_DIR+"\\apple\\"+f for f in os.listdir(IMG_DIR+"\\apple\\")
    #             if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.jpg|.jpeg|.png)$', f)]
    # banana_paths= [IMG_DIR+"\\banana\\"+f for f in os.listdir(IMG_DIR+"\\banana\\")
    #             if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.jpg|.jpeg|.png)$', f)]
    # orange_paths= [IMG_DIR+"\\orange\\"+f for f in os.listdir(IMG_DIR+"\\orange\\")
    #             if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.jpg|.jpeg|.png)$', f)]
    # mix_paths= [IMG_DIR+"\\mix\\"+f for f in os.listdir(IMG_DIR+"\\mix\\")
    #             if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.jpg|.jpeg|.png)$', f)]

    eval_list = []
    for i in range(4):
        eval_result = eval_images(img_paths_list[i], detection_model, label_to_int, i)
        eval_list.append( [eval_result["IOU"], eval_result["P"], eval_result["R"], eval_result["mAP"]] )

    # apple_eval  = eval_images(apple_paths, detection_model, label_to_int, [0])
    # banana_eval = eval_images(banana_paths, detection_model, label_to_int,[1])
    # orange_eval = eval_images(orange_paths, detection_model, label_to_int,[2])
    # mix_eval    = eval_images(mix_paths, detection_model, label_to_int,   [0,1,2]  )

    # apple_eval_list  = [apple_eval["IOU"], apple_eval["P"], apple_eval["R"], apple_eval["mAP"]]
    # banana_eval_list = [banana_eval["IOU"], banana_eval["P"], banana_eval["R"], banana_eval["mAP"]]
    # orange_eval_list = [orange_eval["IOU"], orange_eval["P"], orange_eval["R"], orange_eval["mAP"]]
    # mix_eval_list    = [mix_eval["IOU"], mix_eval["P"], mix_eval["R"], mix_eval["mAP"]]

    output_result(eval_list, IMG_DIR)


