from pathlib import Path
from typing import *
import math
import numpy as np


class Event:
    
    def __init__(self, analyze: Path) -> None:
        self.analyze = analyze
        
        with open(self.analyze, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
            
        self.frame_info = [[]]
        frame_count_n = 0
        for line in self.lines:
            line = line.split(" ")
            frame_count = int(line[0])
            
            if frame_count != frame_count_n:
                frame_count_n += 1
                self.frame_info.append([])
            
            self.frame_info[frame_count].append(line[1:])
    
    
    def score_detect(self):
        score_rate = []
        for frame in self.frame_info:
            points = []
            for person in frame:
                points.append(person[:2], person[2:])        
            score_rate.append(1 /self.cal_gather_rate(points) * len(frame))
        
            
            
        return     
        
    @staticmethod    
    def cal_gather_rate(points: List[tuple]):
        """计算集中率

        Args:
            points ([(x, y), (x, y), (x, y), (x, y)]): 两个人的xyxy
        """
            
        peoples = []
        
        for i in range(0, len(points), 2):
            point1 = points[i]
            point2 = points[i + 1]
            x1, y1 = point1
            x2, y2 = point2
            
            middle = ((x1 + y1) / 2, (x2 + y2) / 2)
            peoples.append(middle)
         
        total_distance = 0    
        for middle1 in peoples:
            for middle2 in peoples:
                x1, y1 = middle1
                x2, y2 = middle2
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                total_distance += distance
        
        average_distance = total_distance / len(peoples)
        
        return average_distance
     
    