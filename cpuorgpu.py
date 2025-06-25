import torch
import platform


def check_torch_gpu():
    """检查PyTorch是否能使用GPU，并显示相关信息"""
    print("===== PyTorch GPU 检测 =====")
    print(f"Python 版本: {platform.python_version()}")
    print(f"PyTorch 版本: {torch.__version__}")

    # 检查CUDA（NVIDIA GPU）
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA 可用: {cuda_available}")
    if cuda_available:
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024:.0f} MB")
    else:
        print("未检测到NVIDIA GPU或CUDA驱动未安装")

    # 检查MPS（Apple Silicon GPU）
    if hasattr(torch, 'has_mps'):
        mps_available = torch.has_mps
        print(f"\nMPS (Apple Silicon GPU) 可用: {mps_available}")
        if mps_available:
            print("警告: MPS支持仍在实验阶段，可能存在兼容性问题")
    else:
        print("\nMPS 检测不可用 (需要PyTorch 2.0或更高版本)")

    # 检查Metal（macOS GPU）
    if hasattr(torch.backends, 'metal'):
        metal_available = torch.backends.metal.is_available()
        print(f"\nMetal (macOS GPU) 可用: {metal_available}")
    else:
        print("\nMetal 检测不可用")

    # 测试张量计算
    print("\n=== 测试张量计算 ===")
    try:
        # 创建CPU张量
        cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
        print(f"CPU 张量: {cpu_tensor.device}")

        # 创建GPU张量（如果可用）
        if cuda_available:
            gpu_tensor = cpu_tensor.cuda()
            print(f"CUDA 张量: {gpu_tensor.device}")
        elif hasattr(torch, 'has_mps') and torch.has_mps:
            mps_tensor = cpu_tensor.to('mps')
            print(f"MPS 张量: {mps_tensor.device}")
        else:
            print("无法创建GPU张量 - 没有可用的GPU")

        # 简单计算测试
        if cuda_available or (hasattr(torch, 'has_mps') and torch.has_mps):
            device = 'cuda' if cuda_available else 'mps'
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            start_time = torch.cuda.Event(enable_timing=True) if cuda_available else time.time()
            end_time = torch.cuda.Event(enable_timing=True) if cuda_available else None

            if cuda_available:
                start_time.record()
            else:
                start_time = time.time()

            c = torch.matmul(a, b)

            if cuda_available:
                end_time.record()
                torch.cuda.synchronize()  # 等待所有GPU操作完成
                elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # 秒
            else:
                elapsed_time = time.time() - start_time

            print(f"GPU 矩阵乘法耗时: {elapsed_time:.4f} 秒")

        print("\n测试完成!")
    except Exception as e:
        print(f"测试过程中出错: {e}")
        print("提示: 确保CUDA驱动与PyTorch版本兼容")


if __name__ == "__main__":
    check_torch_gpu()