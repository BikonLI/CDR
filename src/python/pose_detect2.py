from pose_detect import *
from bayes_model_new import *


if __name__ == "__main__":
    
    # with open(r"C:\Users\li\Desktop\tasks0.txt", "w", encoding="utf-8") as f:
    #     folders_list = os.listdir(r"D:\CDR\train_stage2\train\images")
    #     for folder in folders_list[:int(len(folders_list) / 2)]:
    #         print(folder, file=f)
            
    # with open(r"C:\Users\li\Desktop\tasks1.txt", "w", encoding="utf-8") as f:
    #     folders_list = os.listdir(r"D:\CDR\train_stage2\train\images")
    #     for folder in folders_list[int(len(folders_list) / 2):]:
    #         print(folder, file=f)
    
    
    with open("tasks.txt", "r", encoding="utf-8") as f:
        folder_list = f.readlines()
    folder_list = [folder.strip('\n') for folder in folder_list]

    for folders in tqdm(folder_list):
        folder = os.path.join(r"D:\CDR\train_stage2\train\images", folders)
        writeKeyPoints(folder)
        
        
    with open("train_stage2/train/train_gt.json", "r", encoding="utf-8") as f:
        actual_number = json.load(f)

    for folders in folder_list:
        folder = os.path.join(r"D:\CDR\train_stage2\train\images", folders)
        name = os.path.split(folder)
        number = getKeyPoints(writeKeyPoints(folder), resetPriors=False)
        train(getPriors(), actual_number[name])
        