from PIL import Image
import os
import re
path = "PATH_TO_IMAGE_DIR"# 原始路径
save_path = "PATH_TO_SAVE_IMAGE_DER"# 保存路径

all_images = [f for f in os.listdir(path)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.jpg|.jpeg|.png)$', f)]

for image in all_images:
    image_path = os.path.join(path, image)
    img = Image.open(image_path)  # 打开图片
    if img.mode != "RGB":
        print(img.format, img.size, img.mode)#打印出原图格式
        img = img.convert("RGB")  # 4通道转化为rgb三通道
        img.save(save_path + image)
