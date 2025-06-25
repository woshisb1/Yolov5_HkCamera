import cv2
import numpy as np
import torch
import queue
import threading
from ultralytics import YOLO
import time

import tensorrt as trt
print(f"TensorRT 版本: {trt.__version__}")
print(f"GPU 加速支持: {trt.Builder(trt.Logger()).platform_has_fast_gpu}")

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")


# 初始化模型 - 使用TensorRT加速
def initialize_model(model_path="yolov8n.pt", device="cuda"):
    try:
        model = YOLO(model_path.replace('.pt', '.engine'))
        print("已加载TensorRT优化模型")
    except:
        model = YOLO(model_path)
        model.export(format="engine", half=True, device=0)  # FP16精度
        model = YOLO(model_path.replace('.pt', '.engine'))
        print("模型已转换为TensorRT引擎")

    model = model.to(device)
    return model


# 异步处理器
class AsyncProcessor:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.stream = torch.cuda.Stream()

    def preprocess(self, frame):
        with torch.cuda.stream(self.stream):
            tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            tensor = tensor.to(self.device, non_blocking=True)
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(640, 640), mode='bilinear', align_corners=False
            ).squeeze(0)
        return tensor

    def inference(self, tensor):
        with torch.cuda.stream(self.stream):
            results = self.model(tensor.unsqueeze(0), verbose=False)
        return results

    def postprocess(self, results, frame):
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                boxes_xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls in zip(boxes_xyxy, confs, classes):
                    if conf > 0.2:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{self.model.names[cls]} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame


# 多线程视频读取器
class VideoReader:
    def __init__(self, rtsp_url):
        self.cap = cv2.VideoCapture(rtsp_url)
        self.queue = queue.Queue(maxsize=5)
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.queue.full():
                self.queue.get_nowait()
            self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def release(self):
        self.cap.release()


def main():
    # 初始化模型
    model = initialize_model("yolov8n.pt", device)

    # 初始化视频读取器
    reader = VideoReader("rtsp://admin:Aa12345678@192.168.1.64:554/Streaming/Channels/1?tcp")

    # 初始化异步处理器
    processor = AsyncProcessor(model, device)

    # 创建窗口
    cv2.namedWindow("YOLOv8 检测", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 检测", 1280, 720)

    # 性能监控
    last_time = time.time()
    frame_count = 0
    fps = 0

    try:
        while True:
            # 读取帧（非阻塞）
            frame = reader.read()

            # 预处理
            tensor = processor.preprocess(frame)

            # 推理
            results = processor.inference(tensor)

            # 等待当前流完成
            torch.cuda.current_stream().wait_stream(processor.stream)

            # 后处理
            processed_frame = processor.postprocess(results, frame.copy())

            # 计算FPS
            current_time = time.time()
            frame_count += 1
            if current_time - last_time >= 1.0:
                fps = frame_count / (current_time - last_time)
                frame_count = 0
                last_time = current_time

            # 显示FPS
            cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 显示画面
            cv2.imshow("YOLOv8 检测", processed_frame)

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"发生错误: {e}")

    finally:
        # 释放资源
        reader.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()