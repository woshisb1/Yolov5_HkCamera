import cv2
import numpy as np
import torch
import torchvision
import queue
import threading
import os
from ultralytics import YOLO

# 设置CUDA环境
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")


# 优化的模型初始化 - 使用INT8量化和TensorRT
def initialize_model(model_name="yolov8n.pt", device="cuda"):
    # 尝试加载优化后的模型
    try:
        model = YOLO(model_name.replace('.pt', '.engine'))
        print(f"已加载优化模型: {model_name.replace('.pt', '.engine')}")
    except:
        model = YOLO(model_name)
        # 仅在CUDA可用时进行模型优化
        if device == "cuda":
            # 使用INT8量化（需要TensorRT支持）
            model.export(format="engine", half=False, int8=True)
            model = YOLO(model_name.replace('.pt', '.engine'))
            print(f"模型已转换为TensorRT INT8格式")

    if device == "cuda":
        model = model.to(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.fastest = True

        # 预热模型
        warmup_tensor = torch.zeros(1, 3, 640, 640).to(device)
        with torch.no_grad():
            model(warmup_tensor)

        print(f"模型已优化并预热，运行在 {device} 上")

    return model


# 简化的图像增强 - 仅保留必要的预处理
class ImageEnhancer:
    def __init__(self, device="cuda"):
        self.device = device

    def enhance(self, frame_tensor):
        # 仅调整大小，移除对比度增强和锐化以提高性能
        return frame_tensor


# 异步批处理推理引擎
class BatchInferenceEngine:
    def __init__(self, model, batch_size=1, device="cuda"):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.buffer = []
        self.results = queue.Queue(maxsize=5)
        self.lock = threading.Lock()
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.enhancer = ImageEnhancer(device)

    def enqueue(self, frame):
        with self.lock:
            # 预处理帧
            frame_tensor = self._preprocess(frame)
            self.buffer.append(frame_tensor)

            if len(self.buffer) >= self.batch_size:
                batch = torch.stack(self.buffer).to(self.device)
                self.buffer = []
                return batch
            return None

    def dequeue(self):
        try:
            return self.results.get(block=False)
        except queue.Empty:
            return None

    def _processing_loop(self):
        while True:
            batch = self.enqueue(None)  # 获取当前批次
            if batch is not None:
                # 异步推理
                with torch.no_grad():
                    results = self.model(batch, verbose=False)
                    self.results.put(results)
            else:
                time.sleep(0.001)  # 短暂休眠避免CPU占用过高

    def _preprocess(self, frame):
        # 转换为张量
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        # 简化预处理
        frame = torch.nn.functional.interpolate(
            frame.unsqueeze(0), size=(640, 640), mode='bilinear', align_corners=False
        ).squeeze(0)

        return frame


# 高性能视频读取器
class VideoReader:
    def __init__(self, rtsp_url, buffer_size=1):
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        # 配置相机获取最高FPS
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # 启动读取线程
        self.thread = threading.Thread(target=self._read_frames)
        self.thread.daemon = True
        self.thread.start()

    def _read_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            try:
                # 保持队列最新帧
                self.buffer.get_nowait()
            except queue.Empty:
                pass

            try:
                self.buffer.put_nowait(frame)
            except queue.Full:
                pass

    def read(self):
        try:
            return self.buffer.get_nowait()
        except queue.Empty:
            return None

    def release(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.cap.release()


# 平滑跟踪类
class SmoothTracker:
    def __init__(self, alpha=0.7, max_age=3):  # 减少max_age以提高响应速度
        self.alpha = alpha
        self.max_age = max_age
        self.tracks = {}
        self.track_count = 0

    def update(self, new_boxes):
        # 衰减现有跟踪
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]

        # 匹配新检测
        if not new_boxes:
            return []

        for box in new_boxes:
            x1, y1, x2, y2, conf, cls = box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            # 快速匹配（仅比较最近的跟踪）
            best_match_id = None
            min_distance = float('inf')

            for track_id, track in self.tracks.items():
                if track['cls'] != cls:
                    continue

                dist = np.sqrt((track['center'][0] - center[0]) ** 2 +
                               (track['center'][1] - center[1]) ** 2)

                if dist < min_distance and dist < 100:
                    min_distance = dist
                    best_match_id = track_id

            # 更新或创建跟踪
            if best_match_id is not None:
                prev_box = self.tracks[best_match_id]['box']
                self.tracks[best_match_id] = {
                    'box': [
                        prev_box[0] * (1 - self.alpha) + x1 * self.alpha,
                        prev_box[1] * (1 - self.alpha) + y1 * self.alpha,
                        prev_box[2] * (1 - self.alpha) + x2 * self.alpha,
                        prev_box[3] * (1 - self.alpha) + y2 * self.alpha,
                        conf, cls
                    ],
                    'center': center,
                    'age': 0,
                    'cls': cls
                }
            else:
                self.tracks[len(self.tracks) + 1] = {
                    'box': [x1, y1, x2, y2, conf, cls],
                    'center': center,
                    'age': 0,
                    'cls': cls
                }
                self.track_count += 1

        return [track['box'] for track in self.tracks.values()]


# 并行后处理
class PostProcessor(threading.Thread):
    def __init__(self, model, tracker, results_queue, display_queue):
        super().__init__()
        self.model = model
        self.tracker = tracker
        self.results_queue = results_queue
        self.display_queue = display_queue
        self.daemon = True
        self.start()

    def run(self):
        while True:
            results, frame = self.results_queue.get()
            if results is None:
                break

            # 处理结果
            if isinstance(results, list) and len(results) > 0:
                first_result = results[0]
                if hasattr(first_result, 'boxes'):
                    boxes = first_result.boxes
                    if len(boxes) > 0:
                        # 提取检测框
                        boxes_np = boxes.xyxy.cpu().numpy()
                        confs_np = boxes.conf.cpu().numpy()
                        classes_np = boxes.cls.cpu().numpy().astype(int)

                        # 过滤低置信度检测
                        detections = []
                        for box, conf, cls in zip(boxes_np, confs_np, classes_np):
                            if conf > 0.3:  # 降低阈值提高检测率
                                detections.append([*box, conf, cls])

                        # 更新跟踪
                        if detections:
                            tracked_boxes = self.tracker.update(detections)

                            # 绘制跟踪结果
                            for box in tracked_boxes:
                                x1, y1, x2, y2, conf, cls = box
                                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                                # 绘制边界框（使用更高效的绘制方法）
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                # 绘制标签
                                label = f"{self.model.names[cls]} {conf:.2f}"
                                cv2.putText(frame, label, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 将处理后的帧放入显示队列
            self.display_queue.put(frame)


# 性能测试函数
def test_fps(model, device):
    dummy_input = torch.zeros(1, 3, 640, 640).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)

    # 测试FPS
    start_time = time.time()
    num_frames = 100

    with torch.no_grad():
        for _ in range(num_frames):
            model(dummy_input)

    end_time = time.time()
    fps = num_frames / (end_time - start_time)
    print(f"模型纯推理FPS: {fps:.2f}")

    return fps


# 主程序
def main():
    # 初始化模型
    model = initialize_model("yolov8n.pt", device)

    # 测试模型性能
    model_fps = test_fps(model, device)
    print(f"理论最大FPS: {model_fps:.2f}")

    # 初始化视频读取器
    reader = VideoReader("rtsp://admin:Aa12345678@192.168.1.64:554/Streaming/Channels/1?tcp")

    # 初始化批处理推理引擎
    inference_engine = BatchInferenceEngine(model, batch_size=1, device=device)

    # 初始化跟踪器
    tracker = SmoothTracker(alpha=0.7, max_age=3)

    # 创建队列用于并行处理
    results_queue = queue.Queue(maxsize=5)
    display_queue = queue.Queue(maxsize=5)

    # 启动后处理器
    post_processor = PostProcessor(model, tracker, results_queue, display_queue)

    # 创建窗口
    cv2.namedWindow("YOLOv8 优化版", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 优化版", 1280, 720)

    # 性能监控
    last_time = time.time()
    frame_count = 0
    display_fps = 0

    try:
        while True:
            # 读取最新帧
            frame = reader.read()
            if frame is None:
                time.sleep(0.001)
                continue

            # 提交推理任务
            batch = inference_engine.enqueue(frame)
            if batch is not None:
                # 处理推理结果
                results = inference_engine.dequeue()
                if results is not None:
                    try:
                        results_queue.put((results, frame.copy()), block=False)
                    except queue.Full:
                        pass  # 队列满时丢弃结果

            # 显示处理后的帧
            try:
                display_frame = display_queue.get(block=False)

                # 计算并显示FPS
                current_time = time.time()
                frame_count += 1

                if current_time - last_time >= 1.0:
                    display_fps = frame_count / (current_time - last_time)
                    frame_count = 0
                    last_time = current_time

                cv2.putText(display_frame, f"FPS: {display_fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 显示CUDA内存使用情况
                if device == "cuda":
                    allocated = torch.cuda.memory_allocated(0) / 1024 ** 2
                    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
                    cv2.putText(display_frame, f"GPU Memory: {allocated:.2f}/{total:.2f} MB",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("YOLOv8 优化版", display_frame)
            except queue.Empty:
                # 没有可用的帧，显示原始帧
                cv2.putText(frame, f"FPS: {display_fps:.2f} (Processing...)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("YOLOv8 优化版", frame)

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"发生错误: {e}")

    finally:
        # 释放资源
        reader.release()
        results_queue.put((None, None))  # 停止后处理器
        post_processor.join(timeout=1.0)
        cv2.destroyAllWindows()

        # 释放CUDA资源
        if device == "cuda":
            torch.cuda.empty_cache()
            print("CUDA资源已释放")


if __name__ == "__main__":
    main()