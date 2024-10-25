from pathlib import Path
import cv2

import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import math
from typing import *
import json
from OCR import getRectangle as get_rectangle_img




def getKeyPoint(jsonpath: Path):
    with open(jsonpath, "r", encoding="utf-8") as f:
        data = json.load(f)
        try: 
            points_conf = data["people"][0]["pose_keypoints_2d"]
        except (KeyError, IndexError) as e:
            return None
            
    points = [points_conf[i:i+3] for i in range(0, len(points_conf) - 3, 3)]
        
    rsholder = points[2][:2]
    lsholder = points[5][:2]
    rhip = points[9][:2]
    lhip = points[12][:2]
    mt = points[1][:2]
    md = points[8][:2]

    return rsholder, lsholder, rhip, lhip, mt, md


def get_rectangle(rsholder, lsholder, rhip, lhip, mt, md):
    x, y = md
    if x != 0 and y != 0:
        x1, y1 = x - 15, y - 10
        x2, y2 = x + 15, y + 15
        return (x1, y1), (x2, y2)
    
    else:
        x, y = mt
        x1, y1 = x - 15, y - 10
        x2, y2 = x + 15, y + 15
        return (x1, y1), (x2, y2)


class Container:
    
    def __init__(self, root_folder: Path, pose_root_folder: Path, thresh = 10) -> None:
        """分类

        Args:
            root_folder (Path): 球员存放根目录  (players/0/1/*.jpg)
            thresh (int, optional): 阈值 (向量夹角绝对值)。 Defaults to 10.
        """
        self.thresh = thresh
        self.classes = {"0": [], "1": [], "-1": []}
        self.root_folder = root_folder
        self.pose_root_folder = pose_root_folder
        self.name    = []
        self.feature = []
        
        self.classify()
        
    def get_dominant_color(self, img, k=2):
        
        try:
            img = img.reshape((-1, 3))
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(img)
            
            counts = Counter(kmeans.labels_)

            dominant_color_index = counts.most_common(1)[0][0]
            dominant_color = kmeans.cluster_centers_[dominant_color_index]
            
        except Exception:
            return None
        
        return dominant_color
    
    @staticmethod
    def calculate_differ(color1, color2):
        r, g, b = color1
        r1, g1, b1 = color2

        distance = math.sqrt((r - r1) ** 2 + (g - g1) ** 2 + (b - b1) ** 2)
        
        return distance
    
    def get_feature(self, folder_img: Path, folder_pose: Path) -> Tuple["real_color", "name"]: # type: ignore
        dominant_colors = []
        keypoints_backup = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
        for pic in folder_img.iterdir():
            img = cv2.imread(pic)
            pic_name = pic.name
            # cv2.imshow("name", img)
            # cv2.waitKey(0)
            keyPjson_name = pic_name.strip(".jpg") + "_keypoints.json"
            json_path = folder_pose / keyPjson_name
            
            keypoints = getKeyPoint(json_path)
            if keypoints is None:
                keypoints = keypoints_backup
            else:
                keypoints_backup = keypoints
            
            img = get_rectangle_img(img, get_rectangle(*keypoints))
            color = self.get_dominant_color(img)
            if color is not None:
                dominant_colors.append(color)

        r, g, b = 0, 0, 0
        for i in range(len(dominant_colors)):
            r += dominant_colors[i][0]
            g += dominant_colors[i][1]
            b += dominant_colors[i][2]

        real_dominant_color = [
            r / (len(dominant_colors) + 1e-8), 
            g / (len(dominant_colors) + 1e-8), 
            b / (len(dominant_colors) + 1e-8)
        ]
        
        r, g, b = real_dominant_color
        if r > 255 or g > 255 or b > 255:
            return (255, 255, 255), folder_img.name
        
        return real_dominant_color, folder_img.name
    
    def get_all_features(self, folder_list: List[Path], pose_list: List[Path]):
        for i, folder in enumerate(folder_list):
            print("照片路径以及poseresult ", folder, pose_list[i])
            dominant_color, name = self.get_feature(folder, pose_list[i])
            print(dominant_color, name)
            self.feature.append(dominant_color)
            self.name.append(name)
    
    def classify(self):
        
        folder_list = []
        pose_list = []
        folders = [path for path in self.root_folder.iterdir()]
        
        for i in range(len(folders)):
            folder = folders[i]
            pose_folder = self.pose_root_folder / folder.name
            folder_list.append(folder)
            pose_list.append(pose_folder)
        
        self.get_all_features(folder_list, pose_list)
        
        # start classify
        first_class, second_class, other = self.classify_colors_with_labels(self.feature, self.name)
        
        self.classes["0"] = first_class
        self.classes["1"] = second_class
        self.classes["-1"] = other
        
        return self.classes


    def __str__(self):
        return f"{self.classes}"
    
    def __getitem__(self, index):
        return self.classes[index]

    
    @staticmethod
    def classify_colors_with_labels(colors, labels, threshold=50):
        # Step 1: 使用 KMeans 将颜色分成两类
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(colors)
        
        # 获取每个类别的中心（即每个类的平均颜色）
        cluster_centers = kmeans.cluster_centers_
        
        # Step 2: 计算每个颜色到类中心的距离
        cluster_labels = kmeans.labels_
        distances = [np.linalg.norm(colors[i] - cluster_centers[cluster_labels[i]]) for i in range(len(colors))]
        
        # Step 3: 根据距离分成三类
        class_0_labels = []
        class_1_labels = []
        outlier_labels = []
        
        for i, color in enumerate(colors):
            if distances[i] > threshold:  # 大于阈值，归为第三类
                outlier_labels.append(labels[i])
            elif cluster_labels[i] == 0:
                class_0_labels.append(labels[i])
            else:
                class_1_labels.append(labels[i])
        
        return class_0_labels, class_1_labels, outlier_labels
