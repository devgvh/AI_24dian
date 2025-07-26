#!/usr/bin/env python3
"""
GPU环境检查脚本
用于诊断PyTorch和CUDA环境
"""
import sys
import subprocess

def check_python_packages():
    """检查Python包版本"""
    print("=" * 60)
    print("🐍 Python 环境检查")
    print("=" * 60)
    
    packages = ['torch', 'transformers', 'datasets']
    for package in packages:
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'show', package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                version = next((line.split(': ')[1] for line in lines if line.startswith('Version: ')), 'Unknown')
                print(f"✅ {package}: {version}")
            else:
                print(f"❌ {package}: 未安装")
        except Exception as e:
            print(f"❌ {package}: 检查失败 - {e}")

def check_cuda_installation():
    """检查CUDA安装"""
    print("\n" + "=" * 60)
    print("🔧 CUDA 环境检查")
    print("=" * 60)
    
    # 检查nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi 可用")
            print(result.stdout)
        else:
            print("❌ nvidia-smi 不可用")
    except FileNotFoundError:
        print("❌ nvidia-smi 未找到 - 可能未安装NVIDIA驱动")
    
    # 检查nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvcc 可用")
            lines = result.stdout.strip().split('\n')
            version_line = next((line for line in lines if 'release' in line), 'Unknown')
            print(f"   {version_line}")
        else:
            print("❌ nvcc 不可用")
    except FileNotFoundError:
        print("❌ nvcc 未找到 - 可能未安装CUDA Toolkit")

def check_pytorch_cuda():
    """检查PyTorch CUDA支持"""
    print("\n" + "=" * 60)
    print("🔥 PyTorch CUDA 检查")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
        print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA 版本: {torch.version.cuda}")
            print(f"✅ cuDNN 版本: {torch.backends.cudnn.version()}")
            print(f"✅ GPU 数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # 测试GPU计算
            print("\n🧪 GPU 计算测试:")
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                z = torch.mm(x, y)
                print("✅ GPU 矩阵乘法测试通过")
            except Exception as e:
                print(f"❌ GPU 计算测试失败: {e}")
                
        else:
            print("❌ CUDA 不可用")
            print("💡 可能的原因:")
            print("   - PyTorch 未安装 CUDA 版本")
            print("   - NVIDIA 驱动未安装或版本不匹配")
            print("   - CUDA Toolkit 未安装")
            
    except ImportError:
        print("❌ PyTorch 未安装")

def check_training_environment():
    """检查训练环境"""
    print("\n" + "=" * 60)
    print("🎯 训练环境检查")
    print("=" * 60)
    
    try:
        import torch
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        
        print("✅ Transformers 库可用")
        
        # 测试模型加载
        print("📦 测试模型加载...")
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        print("✅ T5 模型加载成功")
        
        # 测试GPU移动
        if torch.cuda.is_available():
            print("🚀 测试模型GPU移动...")
            model = model.cuda()
            print(f"✅ 模型已移动到: {next(model.parameters()).device}")
            
            # 测试前向传播
            print("⚡ 测试GPU前向传播...")
            input_ids = torch.randint(0, 1000, (1, 10)).cuda()
            with torch.no_grad():
                outputs = model(input_ids)
            print("✅ GPU 前向传播测试通过")
        else:
            print("⚠️  跳过GPU测试 (CUDA不可用)")
            
    except Exception as e:
        print(f"❌ 训练环境检查失败: {e}")

def main():
    print("🔍 24点训练项目 GPU 环境诊断")
    print("=" * 60)
    
    check_python_packages()
    check_cuda_installation()
    check_pytorch_cuda()
    check_training_environment()
    
    print("\n" + "=" * 60)
    print("📋 总结和建议")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print("🎉 您的环境支持GPU训练!")
            print("💡 建议使用 train_improved.py 进行训练")
        else:
            print("⚠️  您的环境不支持GPU训练")
            print("💡 建议:")
            print("   1. 安装NVIDIA驱动")
            print("   2. 安装CUDA Toolkit")
            print("   3. 安装PyTorch CUDA版本: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    except ImportError:
        print("❌ PyTorch未安装，请先安装PyTorch")

if __name__ == "__main__":
    main() 