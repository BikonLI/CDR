from pathlib import Path
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import math
from typing import *

class Container:
    
    def __init__(self, root_folder: Path, thresh = 10) -> None:
        """分类

        Args:
            root_folder (Path): 球员存放根目录  (players/0/1/*.jpg)
            thresh (int, optional): 阈值 (向量夹角绝对值)。 Defaults to 10.
        """
        self.thresh = thresh
        self.classes = {"0": [], "1": [], "-1": []}
        self.root_folder = root_folder
        self.name    = []
        self.feature = []
        
        self.classify()
        
    def get_dominant_color(self, img, k=2):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(img)
        
        counts = Counter(kmeans.labels_)

        dominant_color_index = counts.most_common(1)[0][0]
        dominant_color = kmeans.cluster_centers_[dominant_color_index]
        
        return dominant_color
    
    def calculate_differ(self, color1, color2):
        r, g, b = color1
        r1, g1, b1 = color2
        cosa = (
            (r * r1 + g * g1 + b * b1) / 
            (math.sqrt(r ** 2 + g ** 2 + b ** 2) * 
            math.sqrt(r1 ** 2 + g1 ** 2 + b1 ** 2))
        )
        a = math.acos(cosa)
        return abs(a)
    
    def get_feature(self, folder: Path) -> Tuple["real_color", "name"]: # type: ignore
        dominant_colors = []
        for pic in folder.iterdir():
            img = cv2.imread(pic)
            color = self.get_dominant_color(img)
            dominant_colors.append(color)

        r, g, b = 0, 0, 0
        for i in range(len(dominant_colors)):
            r += dominant_colors[i][0]
            g += dominant_colors[i][1]
            b += dominant_colors[i][2]

        real_dominant_color = [
            r / len(dominant_colors), 
            g / len(dominant_colors), 
            b / len(dominant_colors)
        ]
        
        return real_dominant_color, folder.name
    
    def get_all_features(self, folder_list: List[Path]):
        for folder in folder_list:
            dominant_color, name = self.get_feature(folder)
            self.feature.append(dominant_color)
            self.name.append(name)
            
    def find_class_feature(self):
        count = len(self.feature)
        same_num = [0 for i in range(count)]
        
        for i in range(count):
            for j in range(count):
                feature1 = self.feature[i]
                feature2 = self.feature[j]
                
                if i != j and self.calculate_differ(feature1, feature2) <= self.thresh:
                    same_num[i] += 1
                    
        top_two = self.find_top_two_indices(same_num)
        if not top_two:
            return None
        else:
            first_i, second_i = top_two
        
        first_color = self.feature[first_i]
        second_color = self.feature[second_i]
        
        return first_color, second_color
    
    def classify(self):
        folder_list = [path for path in self.root_folder.iterdir()]
        self.get_all_features(folder_list)
        class_feature = self.find_class_feature()
        
        if class_feature:
            first_color, second_color = class_feature
        else:
            return {"0": self.name, "1": [], "-1": []}
        
        # start classify
        for i in range(len(self.name)):
            feature = self.feature[i]
            name = self.name[i]
            
            if self.calculate_differ(feature, first_color) <= self.thresh:
                self.classes["0"].append(name)
            elif self.calculate_differ(feature, second_color) <= self.thresh:
                self.classes["1"].append(name)
            else:
                self.classes["-1"].append(name)
                
    def __str__(self):
        return f"{self.classes}"
    
    def __getitem__(self, index):
        return self.classes[index]
    
    @staticmethod
    def find_top_two_indices(lst):
        if len(lst) < 2:
            return 0

        max_value = max(lst)
        max_indices = [i for i, x in enumerate(lst) if x == max_value]

        if len(max_indices) > 1:
            return max_indices

        filtered_lst = [x for x in lst if x != max_value]
        second_max_value = max(filtered_lst)
        second_max_index = lst.index(second_max_value)

        return max_indices + [second_max_index]

# 示例
lst = [3, 5, 7, 7, 2, 6]
print(find_top_two_indices(lst))  # 输出最大数和第二大的数的索引                    
        