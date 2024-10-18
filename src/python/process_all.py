import os
os.environ['YOLO_VERBOSE'] = str(False)
from ultralytics import YOLO
from OCR import predict1
from pose_detect import *
import cv2

# 一个trackId的球员矩形照片放到一个文件夹当中
def analyze_video(folder: str):
    
    players = {}
    
    model = YOLO("best.pt")
    for i in range(len(os.listdir(folder))):
        imgPath = os.path.join(folder, f"frame_{i:04}.jpg")
        
        img = cv2.imread(imgPath)
        result = model.track(img, persist=True)[0]
        
        xyxy_list = result.boxes.xyxy.tolist()
        id_list = result.boxes.id.tolist()
        
        for i in range(len(id_list)):
            people = xyxy_list[i].tolist()
            trackId = int(id_list[i])
            
            people_img = getRectangle(img, people)
            
        
        
