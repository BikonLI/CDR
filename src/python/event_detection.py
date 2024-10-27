from pathlib import Path
from typing import *
import math
import numpy as np
from pose_json_parse import *


class Event:
    
    def __init__(self, analyze: Path, left_team_name="left-team", right_team_name="right-team") -> None:
        self.analyze = analyze
        self.left_team = left_team_name
        self.right_team = right_team_name
        
        with open(self.analyze, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
            
        self.frame_info = [[]]
        frame_count_n = 0
        for line in self.lines:
            line = line.split(" ")
            line[0] = int(line[0])
            line[1] = int(line[1]) 
            line[2] = float(line[2]) 
            line[3] = float(line[3]) 
            line[4] = float(line[4])
            line[5] = float(line[5]) 
            line[6] = Path(line[6].strip()) 
            
            frame_count = int(line[0])
            
            if frame_count != frame_count_n:
                frame_count_n += 1
                self.frame_info.append([])
            
            self.frame_info[frame_count].append(line[1:])
    
    
    def score_detect(self):
        gather_rates = []
        people_counts = []
        
        w1 = .7 
        w2 = .3
        for frame in self.frame_info:
            points = []
            for person in frame:
                points.extend([person[1:3], person[3:5]])
            frame_gather_rate = self.cal_gather_rate(points)
            people_count = len(frame)
            
            gather_rates.append(frame_gather_rate)
            people_counts.append(people_count)
            
        min_gather_rate = min(gather_rates)
        score_probability = []
        for i in range(len(gather_rates)):
            gather_rate = gather_rates[i]
            people_count = people_counts[i]
            
            probability = w1 * (min_gather_rate - gather_rate) + w2 * people_count            
            score_probability.append(probability)
            
        max_probability = max(score_probability)
    
        return [
            (i + 40, "score") for i in range(len(score_probability)) 
            if score_probability[i] == max_probability
        ] 
        
    def attact_detect(self, thresh=100): # 阈值越高，越难检测出进攻事件， 向左进攻和向右。
        last_frame = self.frame_info[0]
        total_offset_list = []
        
        for frame in self.frame_info[1:]:
            offset_list = []
            count = 0
            for player in frame:
                trackId = int(player[0])
                for person in last_frame:
                    trackId_last = int(person[0])                    

                    if trackId == trackId_last:
                        count += 1
                        centerx1 = (player[1] + player[3]) / 2
                        centerx2 = (person[1] + person[3]) / 2
                        offsetx = centerx1 - centerx2       # 末减初
                        offset_list.append(offsetx)
                    else: pass
            
            last_frame = frame
            try:
                average_offset = sum(offset_list) / count
            except ZeroDivisionError:
                average_offset = 0
            total_offset_list.append(average_offset)
                    
        result_list = []
        count = 0
        
        for i in range(len(total_offset_list) - 20):
            total_offset_of_20_frames = sum(total_offset_list[i:i+20])
            if abs(total_offset_of_20_frames) >= thresh:
                if total_offset_of_20_frames > 0:
                    result_list.append((i, "left"))
                else:
                    result_list.append((i, "right"))
            
        current_event = None if len(result_list) == 0 else result_list[0]        
        final_result_list = [] if len(result_list) == 0 else [result_list[0]]
        for i in range(len(result_list)):
            iter_event = result_list[i]
            
            if iter_event[0] - current_event[0] <= 20 and iter_event[1] == current_event[1]:
                pass
            else:
                final_result_list.append(iter_event)
                current_event = iter_event
               
        current_event = None if len(final_result_list) == 0 else final_result_list[0]        
        final_final_result_list = [] if len(final_result_list) == 0 else [final_result_list[0]] 
        for i in range(len(final_result_list)):
            iter_event = final_result_list[i]
            
            if iter_event[0] - current_event[0] <= 50 and iter_event[1] == current_event[1]:
                pass
            else:
                final_final_result_list.append(iter_event)
                current_event = iter_event
                    
        return final_final_result_list
    
    def goal_detect(self):
        total_hands_up_count_list = []
        for frame in self.frame_info:
            total_hands_up_count = 0
            for people in frame:
                pose_file = people[-1]
                parser = get_keypoint(pose_file)
                hands_up_count = raising_hand(parser)
                total_hands_up_count += hands_up_count
            total_hands_up_count_list.append(total_hands_up_count)
        
        
        max_count = max(total_hands_up_count_list)
            
        if max_count <= 1:
            return None
        
        most_probably_goal_frame = [
            (index, "goal") for index in range(len(total_hands_up_count_list) - 1, -1, -1) 
            if total_hands_up_count_list[index] == max_count
        ]
        
        return most_probably_goal_frame        
        
    @staticmethod
    def if_same_sign(a, b):
        if a < 0 and b < 0:
            return True
        elif a > 0 and b > 0:
            return True
        elif a == 0 and b == 0:
            return True
        else: return False
        
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
        
        try:
            average_distance = total_distance / len(peoples)
        except ZeroDivisionError:
            average_distance = 100000
        
        return average_distance
    
    def detect(self):
        attaction_result = self.attact_detect()
        score_result = self.score_detect()
        goal_result = self.goal_detect()
        
        event_list = attaction_result + score_result + goal_result
        event_list = sorted(event_list, key=lambda x: x[0])
        
        final_result = []
        per = True
        for event in event_list[::-1]:
            if event[0] >= (len(self.frame_info) - 30):
                continue
                
            
            if event[1] == "goal":
                if per:
                    final_result.append(event)
                    per = False
                else:
                    continue
            else:
                final_result.append(event)
        
        final_result = final_result[::-1]
                                
        prompt_list = []
        attack_event = []
        score_event = []
        goal_event = []
        
        for index, event in enumerate(final_result):
            
            if event[1] in ["left", "right"]:
                try:
                    if index >= 1:
                        if final_result[index - 1][1] == "score":
                            if final_result[index - 1][0] == "left":
                                prompt_list.append(f"{self.left_team}尝试扑球")
                            else:
                                prompt_list.append(f"{self.right_team}尝试扑球")
                            continue
                    
                    if event[1] != attack_event[-1][1]:
                        if event[1] == "left":
                            prompt_list.append(f"{self.right_team}防守")
                        else:
                            prompt_list.append(f"{self.left_team}防守")
                    else:
                        if event[1] == "left":
                            prompt_list.append(f"{self.right_team}持续进攻")
                        else:
                            prompt_list.append(f"{self.left_team}持续进攻")
                except:
                    if event[1] == "left":
                        prompt_list.append(f"{self.right_team}开始进攻了")
                    else:
                        prompt_list.append(f"{self.left_team}开始进攻了")
                
                attack_event.append(event)
            
            if event[1] == "score":
                if len(attack_event) >= 2:
                    attack = attack_event[-2]
                    if attack[1] == "left":
                        prompt_list.append(f"{self.right_team}射门")
                    else:
                        prompt_list.append(f"{self.left_team}射门")
                else:
                    continue
                score_event.append(event)
                
            if event[1] == "goal":
                if len(score_event) == 0:
                    continue
                else:
                    if len(attack_event) == 0:
                        continue
                    else:
                        if attack_event[1] == "left":
                            prompt_list.append(f"{self.left_team}进球了")
                        else:
                            prompt_list.append(f"{self.right_team}进球了")
                goal_event.append(event)
        
        return prompt_list
        

if __name__ == "__main__":
    event = Event(Path("analyze/0.txt"))
    print(event.detect())
    