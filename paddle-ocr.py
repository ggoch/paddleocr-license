import os
import platform
import pylab as plt
import cv2
import numpy as np
import time
import fps_plot
import show_fn
import torch
from video_player import VideoPlayer

from paddleocr import PaddleOCR, draw_ocr

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from collections import defaultdict

ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory

model = YOLO("models/first-train/weights/best.pt")  # load a custom model

video_player = VideoPlayer("datas/videos/Barataria Morvant Exit.mp4")

def text_paddleocr(img):
    result = ocr.ocr(img, cls=True)[0]
    texts = []

    if result is not None:
    
        for line in result:
            txt = line[1][0]
            xy = line[0]
            texts.append(txt)
    
    return texts

def predict(frame):
    img = frame[:, :, ::-1].copy()  # 转换颜色空间
    results = model.track(img, conf=0.3, iou=0.5,persist=True)
    boxes = results[0].boxes.xyxy

    annotated_frame = results[0].plot()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        gray_img = annotated_frame[y1:y2, x1:x2].copy()
        tmp = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        texts = text_paddleocr(tmp)

        for i, txt in enumerate(texts):
            annotated_frame = show_fn.drow_text(annotated_frame, txt, (x1, y1 - 20 - i * 20), (255, 0, 0), 30)

    return annotated_frame

video_player.play(predict)
