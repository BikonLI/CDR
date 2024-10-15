# 导入模块
import os
import json
import cv2
from ultralytics import YOLO
from tqdm import tqdm
from pose_detect import getKeyPoints, writeKeyPoints
# from recognize_character import *


test_stage_images = r"test_stage2\test\images"
json_store_path = r"results_stage2\recognization.json"


os.makedirs(os.path.split(json_store_path)[0], exist_ok=True)


def init_with_m1():
    """这段代码将会构建一个json文件，然后输出全部为-1.
    """
    jsondata = {}
    
    folders_list = os.listdir(test_stage_images)
    for folder in folders_list:
        jsondata[folder] = -1
        
    with open(json_store_path, "w", encoding="utf-8") as jsonfile:
        json.dump(jsondata, jsonfile)
    
    
def recognize0(imgpath: str):
    model = YOLO("runs/detect/train/weights/best.pt")
    results = model(imgpath, show=True)
    cv2.waitKey(0)
    
    
pointer = 0
def output_to_result():
    jsondata = {}
    for folders in os.listdir(r"D:\CDR\test_stage2\test\images")[pointer:]:
        folder = os.path.join(r"D:\CDR\test_stage2\test\images", folders)
        print(folder)
        number = getKeyPoints(writeKeyPoints(folder))
        jsondata[f"{folders}"] = int(number)

        with open(json_store_path, "w", encoding="utf-8") as f:
            json.dump(jsondata, f)
    


if __name__ == "__main__":
    pointer = 43
    # init_with_m1()      # 单纯为了上榜。
    # result = recognize(read_img("val_stage2/val/images/0/0_1.jpg"))
    # print(result)
    # recognize0("train_stage2/train/images/0/0_1.jpg")
    output_to_result()
    