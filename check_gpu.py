"""
GPU 诊断脚本
"""

import sys
import os

print("=" * 60)
print("GPU 诊断")
print("=" * 60)

# 检查 Python 环境
print(f"\nPython 路径: {sys.executable}")
print(f"Python 版本: {sys.version}")

# 检查 PyTorch
try:
    import torch
    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"PyTorch 路径: {torch.__file__}")
    
    # CUDA 检查
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA 可用: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("\nCUDA 不可用！可能原因:")
        print("  1. PyTorch 安装的是 CPU 版本")
        print("  2. CUDA 驱动问题")
        print("  3. 环境变量问题")
        
        # 尝试找出原因
        print("\n详细检查:")
        
        # 检查是否有 CUDA 库
        try:
            from torch.utils.cpp_extension import CUDA_HOME
            print(f"  CUDA_HOME: {CUDA_HOME}")
        except Exception as e:
            print(f"  CUDA_HOME: 未设置 ({e})")
        
        # 检查环境变量
        cuda_path = os.environ.get('CUDA_PATH', '未设置')
        print(f"  CUDA_PATH 环境变量: {cuda_path}")
        
        # 检查 cudnn
        try:
            print(f"  cuDNN 版本: {torch.backends.cudnn.version()}")
        except Exception as e:
            print(f"  cuDNN: 不可用 ({e})")
            
except ImportError as e:
    print(f"\nPyTorch 未安装: {e}")
    print("   请运行: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "=" * 60)
