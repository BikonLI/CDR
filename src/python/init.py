import os
import shutil
import json
import cv2

root_train_folder = "train_stage2/train"
images_root = os.path.join(root_train_folder, "images")     # train_stage2/train/images
labels_root = os.path.join(root_train_folder, "labels")     # train_stage2/train/labels

dataset = "dataset"
train_images_folder = os.path.join(dataset, "train", "images")  # dataset/train/images
train_labels_folder = os.path.join(dataset, "train", "labels")  # dataset/train/labels
val_images_folder = os.path.join(dataset, "val", "images")      # dataset/val/images
val_labels_folder = os.path.join(dataset, "val", "labels")      # dataset/val/labels

work_list = os.listdir(labels_root)                 # ['0', '1' ...]   仅为有效文件夹编号

# ---
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)


def init(work_list: list):
    for label_folder in work_list:
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
            labelConvert(label, label_des)

            

def imgcpy(path1, path2):
    shutil.copy(path1, path2)
    
def labelConvert(path1, path2):
    with open(path1, "r", encoding="utf-8") as f:
        data = json.load(f)
        
        try:
            objects = data["shapes"]
            for ob in objects:
                num = ob["label"]
                
                x0, y0 = ob["points"][0]
                x1, y1 = ob["points"][1]
                print(num, x0, x1, y0, y1)
            
        except Exception as e:
            print(f"标签转化出现问题 {path1} {path2} {e}")
        
        
            
            

if __name__ == "__main__":
    init(work_list)