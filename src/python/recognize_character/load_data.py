import torch
import os
import cv2
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset

uninit_dataset_dir = r"D:\Projects\Big-data-competition\train_stage2\train"

dataset_dir = os.path.join(os.path.split(__file__)[0], "dataset")
os.makedirs(dataset_dir, exist_ok=True)

def load_data():
    """数据集格式如下
    (
    [
        image0,
        image1,
        image2,
        ...
    ],
    
    [
        label0,
        label1,
        label2,
        ...   
    ]
    )
    
    """
    
    npy_file_images = os.path.join(dataset_dir, "images.npy")
    npy_file_label = os.path.join(dataset_dir, "labels.npy")
    
    if os.path.exists(npy_file_images) and os.path.exists(npy_file_label):
        return np.load(npy_file_images), np.load(npy_file_label)
    
    gtjson = os.path.join(uninit_dataset_dir, "train_gt.json")
    
    with open(gtjson, "r", encoding="utf-8") as f:
        data: dict = json.load(f)
        
    
    valid_group = []
    for key, value in data.items():
        if value != -1:
            valid_group.append(key)
    
    pic_list = []
    lable_list = []
    
    images_dir = os.path.join(uninit_dataset_dir, "images")
    for folders in os.listdir(images_dir):
        if folders not in valid_group:
            continue
        
        folder = os.path.join(images_dir, folders)
        print(folder)
        for pics in os.listdir(folder):
            pic = os.path.join(folder, pics)
        
            img = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
            
            x, y = img.shape
            if x * y < 4375:        # 对图像的大小进行筛选
                img = cv2.resize(img, (50, 125))
                pic_list.append(img)
                number = f"{data[folders]}"
                num_list = [int(number[0]), int(number[1])] if len(number) == 2 else [0, int(number[0])]
                lable_list.append(num_list)
            
    pic_array = np.array(pic_list)
    lable_array = np.array(lable_list)
    
    print(pic_array.shape)
    print(lable_array.shape)
    np.save(npy_file_images, pic_array)
    np.save(npy_file_label, lable_array)
    
    return pic_array, lable_array
    
class DigitDataset(Dataset):
    
    def __init__(self, dataloader, transform=None):
        self.pic_array, self.lable_array = dataloader()
        self.transform = transform

    def __len__(self):
        return len(self.pic_array)

    def __getitem__(self, idx):
        img, label = torch.tensor(self.pic_array[idx], dtype=torch.float32), torch.tensor(self.lable_array[idx], dtype=torch.long)
        img = img / 255
        img = img.unsqueeze(0)
        return img, label


dtset = DigitDataset(load_data)
dataloader = DataLoader(dtset, shuffle=True)

if __name__ == "__main__":
    load_data()