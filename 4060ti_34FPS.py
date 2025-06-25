import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 初始化模型
model = YOLO("yolov8n.pt")
model = model.to(device)

# 打开 RTSP 流
cap = cv2.VideoCapture("rtsp://admin:Aa12345678@192.168.1.64:554/Streaming/Channels/1?tcp")

# 创建窗口
cv2.namedWindow("YOLOv8 检测", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 检测", 1280, 720)

# 性能监控
last_time = time.time()
frame_count = 0
fps = 0

try:
    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧，退出...")
            break

        # 推理（降低置信度到0.2）
        results = model(frame, conf=0.2, device=device)

        # 处理结果
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # 提取检测框
                boxes_xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                # 绘制检测框
                for box, conf, cls in zip(boxes_xyxy, confs, classes):
                    x1, y1, x2, y2 = map(int, box)

                    # 绘制边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 绘制标签
                    label = f"{model.names[cls]} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 计算FPS
        current_time = time.time()
        frame_count += 1
        if current_time - last_time >= 1.0:
            fps = frame_count / (current_time - last_time)
            frame_count = 0
            last_time = current_time

        # 显示FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示画面
        cv2.imshow("YOLOv8 检测", frame)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"发生错误: {e}")

finally:
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()