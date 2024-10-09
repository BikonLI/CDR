# 导入模块
import os
import json
from recognize_character import *


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
    

if __name__ == "__main__":
    # init_with_m1()      # 单纯为了上榜。
    result = recognize(read_img("val_stage2/val/images/0/0_1.jpg"))
    print(result)
    
    