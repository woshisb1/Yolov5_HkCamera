import cv2
import numpy as np
import torch
import torchvision  # 添加 torchvision 导入
from ultralytics import YOLO
import threading
import time

# RTSP配置（增加缓冲区和超时设置）
rtsp_url = "rtsp://admin:Aa12345678@192.168.1.64:554/Streaming/Channels/1?tcp"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # 减少缓冲区大小，降低延迟
cap.set(cv2.CAP_PROP_FPS, 15)  # 限制帧率

# 检查相机连接
if not cap.isOpened():
    print("无法连接到相机")
    exit()

# 获取相机原始分辨率（用于显示）
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 模型初始化 - 指定使用 CUDA 设备
model = YOLO("yolov8n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"使用设备: {device}")

# 帧缓冲区和状态标志
frame_buffer = None
stop_event = threading.Event()
detection_results = None
last_detection_time = 0
detection_interval = 0.5  # 检测间隔(秒)


# 平滑跟踪类
class SmoothTracker:
    def __init__(self, alpha=0.6, max_age=5):
        self.alpha = alpha  # 平滑系数
        self.max_age = max_age  # 最大跟踪丢失帧数
        self.tracks = {}  # 跟踪的目标

    def update(self, new_boxes):
        """更新跟踪结果，平滑处理检测框"""
        # 衰减现有跟踪的年龄
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            # 移除长时间未更新的跟踪
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]

        # 匹配新检测框和已有跟踪
        for box in new_boxes:
            x1, y1, x2, y2, conf, cls = box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            # 寻找最接近的已有跟踪
            best_match_id = None
            min_distance = float('inf')
            for track_id, track in self.tracks.items():
                if track['cls'] != cls:  # 只匹配同一类别的目标
                    continue
                dist = np.sqrt((track['center'][0] - center[0]) ** 2 +
                               (track['center'][1] - center[1]) ** 2)
                if dist < min_distance and dist < 100:  # 阈值100像素
                    min_distance = dist
                    best_match_id = track_id

            # 更新已有跟踪或创建新跟踪
            if best_match_id is not None:
                # 平滑处理
                prev_box = self.tracks[best_match_id]['box']
                smoothed_box = [
                    prev_box[0] * (1 - self.alpha) + x1 * self.alpha,
                    prev_box[1] * (1 - self.alpha) + y1 * self.alpha,
                    prev_box[2] * (1 - self.alpha) + x2 * self.alpha,
                    prev_box[3] * (1 - self.alpha) + y2 * self.alpha,
                    conf, cls
                ]
                self.tracks[best_match_id] = {
                    'box': smoothed_box,
                    'center': center,
                    'age': 0,
                    'cls': cls
                }
            else:
                # 创建新跟踪
                self.tracks[len(self.tracks) + 1] = {
                    'box': [x1, y1, x2, y2, conf, cls],
                    'center': center,
                    'age': 0,
                    'cls': cls
                }

        return [track['box'] for track in self.tracks.values()]


# 初始化跟踪器
tracker = SmoothTracker(alpha=0.7, max_age=5)


# 视频读取线程函数
def read_frames():
    global frame_buffer
    while not stop_event.is_set():
        try:
            ret, frame = cap.read()
            if not ret:
                print("无法获取帧")
                time.sleep(0.1)  # 短暂等待后重试
                continue

            # 只保留最新帧
            frame_buffer = frame

            # 控制读取速度，避免缓冲区堆积
            time.sleep(0.03)  # 约30FPS的读取速度

        except Exception as e:
            print(f"读取线程错误: {e}")
            time.sleep(0.1)

    cap.release()


# 启动视频读取线程
thread = threading.Thread(target=read_frames)
thread.daemon = True
thread.start()

# 主循环
try:
    last_display_time = 0
    fps = 15  # 目标显示帧率
    frame_delay = 1 / fps  # 每帧显示时间间隔

    # 创建可调整大小的窗口
    cv2.namedWindow("YOLOv8", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8", 1280, 720)  # 设置初始窗口大小

    while not stop_event.is_set():
        current_time = time.time()

        # 控制显示帧率
        if current_time - last_display_time < frame_delay:
            time.sleep(frame_delay - (current_time - last_display_time))

        # 检查是否有新帧
        if frame_buffer is None:
            continue

        # 复制当前帧用于处理
        current_frame = frame_buffer.copy()

        # 显示原始分辨率画面
        display_frame = current_frame.copy()

        # 创建小尺寸图像用于检测（保持原始宽高比）
        detection_size = (640, 480)  # 增大检测尺寸提高清晰度
        small_frame = cv2.resize(current_frame, detection_size)

        # 控制检测频率
        if current_time - last_detection_time > detection_interval:
            # 执行检测 - 指定设备（如果可用则使用 CUDA）
            results = model.predict(source=small_frame, show=False, conf=0.4, device=device)
            new_detection = []

            # 提取检测框信息
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)

                # 应用NMS（使用原生实现）
                if len(boxes) > 0:
                    # 按类别分别应用NMS
                    unique_classes = np.unique(classes)
                    nms_indices = []

                    for cls in unique_classes:
                        cls_mask = (classes == cls)
                        cls_boxes = boxes[cls_mask]
                        cls_scores = scores[cls_mask]

                        if len(cls_boxes) > 0:
                            # 转换为PyTorch张量以使用原生NMS
                            cls_boxes_tensor = torch.tensor(cls_boxes, dtype=torch.float32, device=device)
                            cls_scores_tensor = torch.tensor(cls_scores, dtype=torch.float32, device=device)

                            # 执行NMS（修复后的代码）
                            cls_indices = torchvision.ops.nms(cls_boxes_tensor, cls_scores_tensor, iou_threshold=0.5)

                            nms_indices.extend(np.where(cls_mask)[0][cls_indices.cpu().numpy()])

                    # 应用NMS后过滤检测结果
                    for i in nms_indices:
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        conf = float(scores[i])
                        cls = classes[i]
                        new_detection.append([x1, y1, x2, y2, conf, cls])

            # 更新跟踪结果
            if new_detection:
                detection_results = tracker.update(new_detection)

            last_detection_time = current_time

        # 在原始分辨率画面上绘制结果
        if detection_results:
            # 计算缩放比例
            scale_x = frame_width / detection_size[0]
            scale_y = frame_height / detection_size[1]

            for box in detection_results:
                x1, y1, x2, y2, conf, cls = box
                # 将检测框坐标映射回原始分辨率
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                # 绘制检测框（增加线条粗细提高可见性）
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # 绘制标签背景
                label = f"{model.names[cls]} {conf:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label,
                                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(display_frame, (x1, y1 - label_height - 10),
                              (x1 + label_width, y1), (0, 255, 0), -1)

                # 绘制标签文本（增加字体大小和粗细）
                cv2.putText(display_frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 显示FPS
        fps_text = f"FPS: {1 / (current_time - last_display_time):.1f}"
        cv2.putText(display_frame, fps_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 显示处理时间
        process_time = (current_time - last_display_time) * 1000
        cv2.putText(display_frame, f"Processing Time: {process_time:.1f}ms", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 显示画面
        cv2.imshow("YOLOv8", display_frame)
        last_display_time = current_time

        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

except KeyboardInterrupt:
    pass

finally:
    stop_event.set()
    thread.join(timeout=1.0)
    cv2.destroyAllWindows()