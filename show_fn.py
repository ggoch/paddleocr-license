import cv2
import time
from PIL import Image, ImageDraw, ImageFont
import platform
import numpy as np


def get_fps(img, start_time, frame_count=1,color=(255, 255, 255)):

    current_time = time.time()
    elapsed_time = current_time - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # 在圖片上繪製 FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), font, 1, color, 2, cv2.LINE_AA)
    return img

def drow_text(img, text, xy=(0, 0), color=(0, 0, 0), size=20, stroke_width=2):
    '''
    在圖片上繪製文字，並返回np.array
    '''
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