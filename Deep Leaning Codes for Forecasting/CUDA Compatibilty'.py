#!/usr/bin/env python3
"""
GPU Setup and Verification Script for PyTorch
Automatically detects, installs, and verifies CUDA-enabled PyTorch
"""

import subprocess
import sys
import os
import importlib
import time


def run_command(command, shell=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            command,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def check_current_pytorch():
    """Check current PyTorch installation."""
    print("=" * 60)
    print("CHECKING CURRENT PYTORCH INSTALLATION")
    print("=" * 60)

    try:
        import torch
        print(f"✓ PyTorch found: {torch.__version__}")

        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
            return True, torch.__version__
        else:
            print("✗ CUDA not available - need to reinstall PyTorch")
            return False, torch.__version__

    except ImportError:
        print("✗ PyTorch not found")
        return False, None
    except Exception as e:
        print(f"✗ Error checking PyTorch: {e}")
        return False, None


def check_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    print("\n" + "=" * 60)
    print("CHECKING NVIDIA GPU")
    print("=" * 60)

    # Check nvidia-smi
    success, stdout, stderr = run_command("nvidia-smi")
    if success:
        print("✓ NVIDIA GPU detected:")
        lines = stdout.split('\n')
        for line in lines:
            if 'RTX' in line or 'GTX' in line or 'GeForce' in line:
                print(f"  {line.strip()}")
        return True
    else:
        print("✗ nvidia-smi not found or GPU not detected")
        print("Make sure NVIDIA drivers are installed")
        return False


def uninstall_pytorch():
    """Uninstall current PyTorch installation."""
    print("\n" + "=" * 60)
    print("UNINSTALLING CURRENT PYTORCH")
    print("=" * 60)

    packages_to_remove = ["torch", "torchvision", "torchaudio"]

    for package in packages_to_remove:
        print(f"Uninstalling {package}...")
        success, stdout, stderr = run_command(f"{sys.executable} -m pip uninstall {package} -y")
        if success:
            print(f"✓ {package} uninstalled")
        else:
            print(f"✗ Failed to uninstall {package}: {stderr}")

    # Clear import cache
    for module in list(sys.modules.keys()):
        if module.startswith('torch'):
            del sys.modules[module]


def install_cuda_pytorch():
    """Install CUDA-enabled PyTorch."""
    print("\n" + "=" * 60)
    print("INSTALLING CUDA-ENABLED PYTORCH")
    print("=" * 60)

    # Try different CUDA versions
    cuda_versions = [
        ("cu121", "CUDA 12.1"),
        ("cu118", "CUDA 11.8"),
        ("cu124", "CUDA 12.4")
    ]

    for cuda_ver, cuda_name in cuda_versions:
        print(f"\nTrying {cuda_name}...")
        command = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_ver}"

        print("Installing... (this may take a few minutes)")
        success, stdout, stderr = run_command(command)

        if success:
            print(f"✓ Installation completed for {cuda_name}")
            return True, cuda_ver
        else:
            print(f"✗ Installation failed for {cuda_name}")
            print(f"Error: {stderr}")

    return False, None


def verify_installation():
    """Verify the CUDA PyTorch installation."""
    print("\n" + "=" * 60)
    print("VERIFYING INSTALLATION")
    print("=" * 60)

    try:
        # Force reimport
        import importlib
        if 'torch' in sys.modules:
            importlib.reload(sys.modules['torch'])
        else:
            import torch

        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")

            # Test basic GPU operations
            print("\nTesting basic GPU operations...")
            device = torch.device('cuda')

            # Test tensor creation
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)

            # Test matrix multiplication
            start_time = time.time()
            z = torch.matmul(x, y)
            gpu_time = time.time() - start_time

            print(f"✓ GPU matrix multiplication test passed ({gpu_time:.4f}s)")

            # Test memory info
            memory_allocated = torch.cuda.memory_allocated(0) / 1024 ** 2
            memory_reserved = torch.cuda.memory_reserved(0) / 1024 ** 2
            print(f"✓ GPU memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB")

            # Clean up
            del x, y, z
            torch.cuda.empty_cache()

            return True
        else:
            print("✗ CUDA still not available after installation")
            return False

    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def create_gpu_test_script():
    """Create a test script for future GPU verification."""
    test_script = '''#!/usr/bin/env python3
"""
Quick GPU Test Script
Run this anytime to verify GPU is working
"""

import torch
import time

print("="*50)
print("GPU STATUS CHECK")
print("="*50)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

    device = torch.device('cuda')

    # Performance test
    print("\\nRunning performance test...")
    x = torch.randn(2000, 2000, device=device)
    y = torch.randn(2000, 2000, device=device)

    start_time = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()  # Wait for completion
    gpu_time = time.time() - start_time

    print(f"GPU matrix multiplication (2000x2000): {gpu_time:.4f}s")

    # Memory info
    memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
    print(f"GPU memory allocated: {memory_allocated:.1f}MB")

    print("\\n✓ GPU is working correctly!")

    # Cleanup
    del x, y, z
    torch.cuda.empty_cache()
else:
    print("\\n✗ GPU not available")
'''

    with open("gpu_test.py", "w") as f:
        f.write(test_script)
    print("✓ Created 'gpu_test.py' for future testing")


def main():
    """Main setup process."""
    print("PyTorch CUDA Setup Script")
    print("This script will set up CUDA-enabled PyTorch on your system")
    print("=" * 60)

    # Step 1: Check NVIDIA GPU
    if not check_nvidia_gpu():
        print("\n❌ NVIDIA GPU not detected. Please:")
        print("1. Install NVIDIA drivers from: https://www.nvidia.com/drivers/")
        print("2. Restart your computer")
        print("3. Run this script again")
        return False

    # Step 2: Check current PyTorch
    cuda_works, current_version = check_current_pytorch()

    if cuda_works:
        print("\n✅ CUDA PyTorch already working correctly!")
        create_gpu_test_script()
        return True

    # Step 3: Ask user for permission to reinstall
    if current_version:
        response = input(f"\nCurrent PyTorch {current_version} doesn't have CUDA support.")
        response = input("Do you want to reinstall with CUDA support? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Installation cancelled by user")
            return False

    # Step 4: Uninstall current PyTorch
    if current_version:
        uninstall_pytorch()

    # Step 5: Install CUDA PyTorch
    install_success, cuda_version = install_cuda_pytorch()

    if not install_success:
        print("\n❌ Failed to install CUDA PyTorch")
        print("You may need to:")
        print("1. Update pip: python -m pip install --upgrade pip")
        print("2. Try manual installation from: https://pytorch.org/")
        return False

    # Step 6: Verify installation
    if verify_installation():
        print("\n✅ SUCCESS! CUDA PyTorch is now installed and working")
        create_gpu_test_script()

        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("=" * 60)
        print("1. Your GPU is now ready for PyTorch")
        print("2. Run 'python gpu_test.py' anytime to verify GPU status")
        print("3. Your GNN drought forecasting code should now use GPU")
        print("4. Expected speedup: 3-5x faster training")

        return True
    else:
        print("\n❌ Installation completed but verification failed")
        print("You may need to restart Python or your IDE")
        return False


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Setup complete! GPU is ready for your GNN model.")
        else:
            print("\n❌ Setup failed. Check error messages above.")
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()