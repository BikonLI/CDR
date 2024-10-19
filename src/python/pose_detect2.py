from pose_detect import *


if __name__ == "__main__":
    
    for folders in tqdm(os.listdir(r"D:\CDR\train_stage2\train\images")):
        folder = os.path.join(r"D:\CDR\train_stage2\train\images", folders)
        writeKeyPoints(folder)