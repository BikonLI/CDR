from OCR import predict_character_ocr, getRectangle, predict
from pathlib import Path
import cv2

def detect_team_name(img_path: Path):
    img = cv2.imread(img_path)
    if img is None:
        return 0
    
    team_1 = getRectangle(img, ((2445, 115), (2715, 190)))
    team_1_score = getRectangle(img, ((2720, 115), (2775, 190)))
    team_2_score = getRectangle(img, ((2900, 115), (2960, 190)))
    team_2 = getRectangle(img, ((2970, 115), (3150, 190)))
        
    return [
        predict_character_ocr(team_1).strip(),
        predict(team_1_score).strip(),
        predict(team_2_score).strip(),
        predict_character_ocr(team_2).strip(),
    ]
    
    
if __name__ == "__main__":
    print(detect_team_name("frame_video/0/00000.jpg"))
    