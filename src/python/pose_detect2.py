from pose_detect import *


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
    for folders in tqdm(folder_list):
        folder = os.path.join(r"D:\CDR\train_stage2\train\images", folders)
        writeKeyPoints(folder)