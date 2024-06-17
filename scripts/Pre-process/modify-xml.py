
import re
import xml.etree.ElementTree as ET  
import os

IMG_DIR = "PATH_TO_IMAGE_DIR"

img_names = [f for f in os.listdir(IMG_DIR)
            if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.jpg|.jpeg|.png)$', f)]


for img_name in img_names:
    xml_name = re.sub(r"(.jpg|.jpeg|.png)$",".xml", img_name)

    img_path = IMG_DIR+"\\"+img_name
    xml_path = IMG_DIR+"\\"+xml_name

    tree=ET.parse(xml_path)
    root = tree.getroot()

    for child in root:
        if child.tag == 'filename':
            child.text = img_name
    
    # 注意要提前在目录下新建 new_xml 文件夹
    new_xml_path = IMG_DIR + "\\new_xml\\" + xml_name
    tree.write(new_xml_path)

    

