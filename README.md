# Chacarter-Detection&Recognization

### 方案描述

打榜部分：
使用`yolo11`检测人物目标，然后将人物的检测框截取出来，再使用`OpenPose`对当中的人物进行姿态分析，截取背部的球衣号，最后通过`parseq`和`tesseract`进行文字识别。将识别结果投入到贝叶斯概率更新模型当中过滤掉噪音，获取概率最大的数字作为该球员的球衣编号。

智能体部分：
使用`bilix`下载视频，然后按照打榜部分分析出`yolo11`给出的`trackId`与球衣编号的映射关系。
通常映射关系为`{"Index1": [trackId1, trackId2, ...]}`。
然后分析事件（传球，射门，进球），并将这些事件与球员进行关联。最后工作流结束，智能体输出结果。

### 路径配置标准
```
root/
    project/
        src/
            python/
                configs/
                strhub/
                *.py
            c/
    test_stage2/
    train_stage2/
    val_stage2/
    *.json    # 程序运行产生的文件，通常是用来记录脚本运行进度的，防止程序被杀停之后的重新运行。
    raw_videos/     # 原视频下载目录
    frames/         # 视频帧提取目录

    README.md

---
openpose/    # 需要配置环境变量，若要访问此文件夹下的文件，需要使用绝对路径
    results/
        1200/
        1201/
            *.json
        ...
        name.txt

---
```
工作流工作内容->
下载视频->提取帧->检测加识别并保存结果到磁盘->OCR检测ID并更新磁盘中的结果