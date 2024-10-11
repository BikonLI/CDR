import multiprocessing.process
import os
import shutil
import json
import cv2
from tqdm import tqdm
from multitask import MultiTask
import multiprocessing
import random

root_train_folder = "train_stage2/train"
images_root = os.path.join(root_train_folder, "images")     # train_stage2/train/images
labels_root = os.path.join(root_train_folder, "labels")     # train_stage2/train/labels

dataset = "datasets"
train_images_folder = os.path.join(dataset, "train", "images")  # dataset/train/images
train_labels_folder = os.path.join(dataset, "train", "labels")  # dataset/train/labels
val_images_folder = os.path.join(dataset, "val", "images")      # dataset/val/images
val_labels_folder = os.path.join(dataset, "val", "labels")      # dataset/val/labels

work_list = os.listdir(labels_root)                 # ['0', '1' ...]   仅为有效文件夹编号

# ---
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)


def init(work_list: list, processid: int=0):
    """将数据初始化为yolo可识别的格式

    Args:
        work_list (list): ['0', '1', '2' ...]
    """
    pgb = tqdm(work_list, desc=f"__init__ {processid:02}", total=len(work_list))
    
    for label_folder in pgb:
        index_folder = label_folder     # '0', '1' ...
        
        # train_stage2/train\labels\0
        label_folder = os.path.join(labels_root, label_folder)
        
        for label in os.listdir(label_folder):
            json_name = label   # 0_1.json
            pic_name = os.path.splitext(json_name)[0]
            
            # train_stage2/train\labels\0\0_1.json
            label = os.path.join(label_folder, f"{pic_name}.json")
            pic = os.path.join(images_root, index_folder, f"{pic_name}.jpg")
            
            # destination path
            label_des = os.path.join(train_labels_folder, f"{pic_name}.txt")
            pic_des = os.path.join(train_images_folder, f"{pic_name}.jpg")
            
            # operations
            rcode = labelConvert(label, label_des)
            if rcode == 0:
                rcode = imgcpy(pic, pic_des)


def imgcpy(path1, path2):
    try:
        shutil.copy(path1, path2)
        return 0
    except Exception as e:
        print(f"图片拷贝出现问题 {path1} {path2} {e}")
        return 1
        
        
def labelConvert(path1, path2):
    with open(path1, "r", encoding="utf-8") as f:
        data = json.load(f)
        
        try:
            objects = data["shapes"]
            het = data["imageHeight"]
            wid = data["imageWidth"]
            
            with open(path2, "w", encoding="utf-8") as f2:
            
                for ob in objects:
                    num = ob["label"]
                    
                    x0, y0 = ob["points"][0]
                    x1, y1 = ob["points"][1]
                    
                    cx, cy, w, h = xywhToYolo(x0, y0, x1, y1, wid, het)
                    print(f"{int(num)} {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}", file=f2)
            return 0
            
        except Exception as e:
            print(f"标签转化出现问题 {path1} {path2} {e}")
            return -1
        
 
def xywhToYolo(x0, y0, x1, y1, wid, het):
    """返回归一化的cx, cy, w, h

    Args:
        x0 (f): tf_x
        y0 (f): tf_y
        wid (f): width
        het (f): height

    Returns:
        tuple[f]: cx, cy, w, h
    """
    centerx = (x0 + x1) / 2
    centery = (y0 + y1) / 2
    w = x1 - x0
    h = y1 - y0
    
    return centerx / wid, centery / het, w / wid, h / het


def copyToVal(count: int=1000):
    """随机选出一些照片作为验证集

    Args:
        count (int, optional): 需要移动的照片和标签数量. Defaults to 1000.

    Returns:
        None: 0
    """
    images = os.listdir(train_images_folder)
    index_list = random.sample(range(len(images)), count)
    pgb = tqdm(index_list, desc="拷贝")
    for index in pgb:
        imgName = images[index]
        name = os.path.splitext(imgName)[0]
        
        prim_pic = os.path.join(train_images_folder, f"{name}.jpg")
        prim_lab = os.path.join(train_labels_folder, f"{name}.txt")
        
        dst_pic = os.path.join(val_images_folder, f"{name}.jpg")
        dst_lab = os.path.join(val_labels_folder, f"{name}.txt")
        
        movePicAndLab(prim_pic, dst_pic, prim_lab, dst_lab)
        
        
def movePicAndLab(path1, path2, path3, path4):
    try:
        shutil.move(path1, path2)
        shutil.move(path3, path4)
        
    except Exception as e:
        print(f"剪切照片和标签出现错误 {path1} {path2} {e}")
        

class MultiT(MultiTask):
    def __init__(self, tasksNum: int, totalWork: list = ...) -> None:
        super().__init__(tasksNum, totalWork)
        self.index = 1
    
    def mallocWork(self, totalWork):
        works = []
        process_work_num = len(totalWork) // self.tasksNum
        j = 0
        for i in range(self.tasksNum):
            works.append(totalWork[j:j+process_work_num])
            j += process_work_num

        return works
    
    def createTask(self, work):
        p = multiprocessing.Process(target=init, args=(work, self.index))
        self.index += 1
        return p
    

if __name__ == "__main__":
    mt = MultiT(4, work_list)
    # mt.start()
    copyToVal(50)
    
    