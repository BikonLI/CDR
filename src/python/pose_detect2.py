from pose_detect import *
from bayes_model_new import *
"""贝叶斯训练脚本相关"""


if __name__ == "__main__":
    
    # with open(r"C:\Users\li\Desktop\tasks0.txt", "w", encoding="utf-8") as f:
    #     folders_list = os.listdir(r"D:\CDR\train_stage2\train\images")
    #     for folder in folders_list[:int(len(folders_list) / 2)]:
    #         print(folder, file=f)
            
    # with open(r"C:\Users\li\Desktop\tasks1.txt", "w", encoding="utf-8") as f:
    #     folders_list = os.listdir(r"D:\CDR\train_stage2\train\images")
    #     for folder in folders_list[int(len(folders_list) / 2):]:
    #         print(folder, file=f)
    
    IMAGES_ROOT = r"D:\CDR\train_stage2\train\images"
    
    
    
    with open("tasks.txt", "r", encoding="utf-8") as f:
        folder_list = f.readlines()
        
    # 训练集文件夹
    folder_list = [folder.strip('\n') for folder in folder_list]   


    ## --- 训练集初始化代码（posedetect）
    # for folders in tqdm(folder_list):
    #     folder = os.path.join(IMAGES_ROOT, folders)
    #     writeKeyPoints(folder)

    
    # writeKeyPointsPatck(IMAGES_ROOT, os.path.join(OPENPOSE_ROOT, "results"))
    ## ---

        
    with open("train_stage2/train/train_gt.json", "r", encoding="utf-8") as f:
        actual_number = json.load(f)

    i = 0
    for folders in folder_list:
        # if i >= 1:
        #     break
        folder = os.path.join(IMAGES_ROOT, folders)
        name = os.path.split(folder)[-1]
        if actual_number[name] == -1:
            print("无数字")
            continue
        else:
            print(folder)
            print(f"实际数字={actual_number[name]}")
        number = getKeyPoints(resultFolder=writeKeyPoints(folder),index=0, imgs_folder=IMAGES_ROOT,train=True)
        print(f"number={number}")
        with open("test.txt", "w", encoding="utf-8") as f:
            print(number, file=f)
        train(getPriors(), actual_number[name])
        reset_priors()
        i += 1
        