import cv2
import time
import show_fn

class VideoPlayer:
    def __init__(self, video_path,skipFrameCount=1,fast_forward_frame_count=30,rewind_frame_count=30):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.paused = False
        self.startTime = time.time()
        self.frameCount = 0
        self.skipFrameCount = skipFrameCount
        self.fast_forward_frame_count = fast_forward_frame_count
        self.rewind_frame_count = rewind_frame_count
        self.last_results = None
        self.show_progress_bar = True

    def play(self,callback):
        while self.cap.isOpened():
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                annotated_frame = frame.copy() 

                self.frameCount += 1
                annotated_frame = self.process_frame(frame,callback)

            # 在每一帧都重新计算并绘制FPS
            annotated_frame = show_fn.get_fps(annotated_frame, self.startTime, self.frameCount,color=(0, 0, 255))

            # 繪製進度條
            if self.show_progress_bar:
                annotated_frame = self.draw_progress_bar(annotated_frame)

            # # 顯示結果
            cv2.imshow('YOLO', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))


            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("k"):  # 暂停/播放
                self.pause()
            elif key == ord("j") and self.frameCount > 1:  # 倒退一帧
                self.rewind()
            elif key == ord("l"):  # 前进一帧
                self.fast_forward()
            elif key == ord("h"):  # 切换进度条的显示
                self.show_progress_bar = not self.show_progress_bar

            # 如果我们处于暂停状态并且没有按键操作，就在这里等待
            while self.paused:
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("k"):  # 再次按 'k' 继续播放
                        self.paused = False
                        break
                    elif key == ord("q"):  # 退出
                        return
                    elif key == ord("j") and self.frameCount > 1:  # 倒退一帧
                        self.rewind()
                        self.update_frame(callback)
                    elif key == ord("l"):  # 前进一帧
                        self.fast_forward()
                        self.update_frame(callback)
                    elif key == ord("h"):  # 切换进度条的显示
                        self.show_progress_bar = not self.show_progress_bar
                        # self.update_frame(callback)
                    time.sleep(0.1)

            if self.paused:  # 如果退出暂停循环是因为按了退出键，那么也应该退出外层循环
                break
            
        # 釋放視頻捕獲對象
        self.cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame,callback):
        if self.frameCount % self.skipFrameCount == 0:
            # 保存预测结果供后续帧使用
            self.last_results = callback(frame)
            return self.last_results
        else:
            # 如果有最后的预测结果，绘制它们
            if self.last_results is not None:
                return self.last_results[0].plot()
            else:
                return frame[:, :, ::-1].copy()
            
    def update_frame(self, callback):
        # 讀取並顯示當前幀
        ret, frame = self.cap.read()
        if ret:
            annotated_frame = frame.copy()
            annotated_frame = self.process_frame(frame, callback)
            annotated_frame = show_fn.get_fps(annotated_frame, self.startTime, self.frameCount,color=(0, 0, 255))
            if self.show_progress_bar:
                annotated_frame = self.draw_progress_bar(annotated_frame)
            cv2.imshow('YOLO', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            
    def pause(self):
        # 暂停视频的逻辑
        self.paused = not self.paused

    def fast_forward(self):
        # 计算新的帧数，不能超过视频的总帧数
        self.frameCount = min(self.cap.get(cv2.CAP_PROP_FRAME_COUNT), self.frameCount + self.fast_forward_frame_count)
        # 设置视频的当前帧位置
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameCount)

    def rewind(self):
        # 计算新的帧数，不能小于0
        self.frameCount = max(0, self.frameCount - self.rewind_frame_count)
        # 设置视频的当前帧位置
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameCount)

    def draw_progress_bar(self,frame):
        if self.show_progress_bar:
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = int(self.frameCount / total_frames * frame.shape[1])

            # 创建一个透明图层来绘制进度条
            overlay = frame.copy()
            overlay_height = 50  # 进度条的高度
            cv2.rectangle(overlay, (0, frame.shape[0] - overlay_height), (progress, frame.shape[0]), (255, 255, 0), -1)  # 绿色进度条

            # 将透明图层叠加到原始帧上
            alpha = 0.7  # 进度条的透明度
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame