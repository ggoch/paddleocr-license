import os
import platform
import pylab as plt
import cv2
import numpy as np
import time
import fps_plot
import torch

from paddleocr import PaddleOCR, draw_ocr

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from collections import defaultdict

ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory


def text(img, text, xy=(0, 0), color=(0, 0, 0), size=20, stroke_width=2):
    pil = Image.fromarray(img)
    s = platform.system()

    if s == "Linux":
        font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", size)
    elif s == "Darwin":
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", size)
    else:
        font = ImageFont.truetype("simsun.ttc", size)

    ImageDraw.Draw(pil).text(xy, text, color, font=font, stroke_width=stroke_width)
    return np.array(pil)

def text_paddleocr(img):
    result = ocr.ocr(img, cls=True)[0]
    texts = []

    if result is not None:
    
        for line in result:
            txt = line[1][0]
            xy = line[0]
            texts.append(txt)
    
    return texts



model = YOLO("models/first-train/weights/best.pt")  # load a custom model

cap = cv2.VideoCapture("datas/videos/Barataria Morvant Exit.mp4")
# cap = cv2.VideoCapture("datas/videos/Vehicle style and Plate recolonization for moving cars.mp4")

statrTime = time.time()
frame_count = 0
skipFrameCount = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    boxes = []

    annotated_frame = frame[:, :, ::-1].copy()

    if frame_count % skipFrameCount == 0:
        # # 處理每一幀（以下代碼與處理單張圖片相同）
        img = frame[:, :, ::-1].copy()  # 轉換顏色空間
        # results = model.predict(img, save=False)
        results = model.track(img, conf=0.3, iou=0.5,persist=True)
        boxes = results[0].boxes.xyxy

        annotated_frame = results[0].plot()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            gray_img = annotated_frame[y1:y2, x1:x2].copy()
            tmp = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
            texts = text_paddleocr(tmp)

            for i, txt in enumerate(texts):
                annotated_frame = text(annotated_frame, txt, (x1, y1 - 20 - i * 20), (255, 0, 0), 30)
    
    frame_count += 1

    annotated_frame = fps_plot.get_fps(annotated_frame, statrTime, frame_count, (0, 0, 255))

    # # 顯示結果
    cv2.imshow('YOLO + OCR', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    # 按 'q' 鍵退出循環
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 釋放視頻捕獲對象
cap.release()
cv2.destroyAllWindows()
