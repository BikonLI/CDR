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
record_progress_path = "progress.json"

if not os.path.exists(record_progress_path):
    data = {}
    with open(record_progress_path, "a", encoding="utf-8") as f:
        data["folder"] = 0
        data["file"] = 0
        json.dump(data, f)

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
file = 0
def output_to_result():
    jsondata = {}
    for folders in os.listdir(r"D:\CDR\test_stage2\test\images")[pointer:]:
        
        with open(record_progress_path, "r", encoding="utf-8") as f:
            data = json.load(f) 
            
        with open(record_progress_path, "w", encoding="utf-8") as f:
            data["folder"] = os.listdir(r"D:\CDR\test_stage2\test\images").index(folders)
            json.dump(data, f)
        
            
        folder = os.path.join(r"D:\CDR\test_stage2\test\images", folders)
        print(folder)
        number = getKeyPoints(writeKeyPoints(folder), file)
        
        with open(json_store_path, "r", encoding="utf-8") as f:
            jsondata = json.load(f)
            jsondata[f"{folders}"] = int(number)

        with open(json_store_path, "w", encoding="utf-8") as f:
            json.dump(jsondata, f)
    

if __name__ == "__main__":
    
    with open(record_progress_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        folder = data["folder"]
        file = data["file"]
        
    pointer = folder
    file = file
    output_to_result()
    # init_with_m1()      # 单纯为了上榜。
    # result = recognize(read_img("val_stage2/val/images/0/0_1.jpg"))
    # print(result)
    # recognize0("train_stage2/train/images/0/0_1.jpg")
    