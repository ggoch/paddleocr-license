import cv2
import time


def get_fps(img, start_time, frame_count=1,color=(255, 255, 255)):
    """
    計算並返回當前的 FPS。

    :param start_time: 開始處理視頻的時間。
    :param frame_count: 到目前為止處理的幀數。
    :return: 當前的 FPS 值。
    """
    current_time = time.time()
    elapsed_time = current_time - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # 在圖片上繪製 FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), font, 1, color, 2, cv2.LINE_AA)
    return img
