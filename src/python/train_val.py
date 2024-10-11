# 对预训练模型继续训练  未完成
from ultralytics import YOLO

model = YOLO("cfg/yolo11.yaml")

if __name__ == "__main__":

    result = model.train(
        data = "cfg/data_stage2.yaml", 
        epochs = 30,            # 学习轮次
        imgsz = 720,
        patience = 10,
        batch = -1,            # GPU占用率
        save = True,            # 保存轮次
        # cache = True,           # 将图片缓存到内存中，提升学习速度
        pretrained = True,
        lr0 = 1e-4,             # 初始学习率
        lrf = 0.15,             # 最终学习率
        momentum = 0.93,        # 帮助模型平滑更新
        weight_decay = 0.0005,
        # resume = True,
    )


    result = model.val(
        data = "cfg/data.yaml", 
        conf = 0.01,
        plots = True,
    )
