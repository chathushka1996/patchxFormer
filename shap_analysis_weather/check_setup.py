"""
Setup Verification Script for SHAP Analysis

This script checks if your environment is ready for SHAP analysis.
Run this before attempting SHAP analysis to avoid common issues.

Usage:
    python check_setup.py
"""

import sys
import os
import importlib
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80)


def print_status(item, status, message=""):
    """Print status with color"""
    status_symbol = "✓" if status else "✗"
    status_text = "OK" if status else "FAIL"
    
    print(f"[{status_symbol}] {item:40s} {status_text:10s} {message}")
    return status


def check_python_version():
    """Check Python version"""
    print_header("PYTHON VERSION CHECK")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    is_ok = version.major == 3 and version.minor >= 8
    
    print_status(
        f"Python {version_str}",
        is_ok,
        "Required: 3.8+"
    )
    
    return is_ok


def check_packages():
    """Check required packages"""
    print_header("REQUIRED PACKAGES CHECK")
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'scikit-learn',
        'shap': 'SHAP',
        'tqdm': 'tqdm',
        'scipy': 'SciPy'
    }
    
    all_ok = True
    
    for package, name in required_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print_status(f"{name} ({package})", True, f"v{version}")
        except ImportError:
            print_status(f"{name} ({package})", False, "NOT INSTALLED")
            all_ok = False
    
    if not all_ok:
        print("\n⚠ Install missing packages:")
        print("   pip install -r shap_analysis_weather/requirements.txt")
    
    return all_ok


def check_torch_gpu():
    """Check PyTorch GPU availability"""
    print_header("GPU AVAILABILITY CHECK")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print_status("CUDA Available", cuda_available)
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print_status(f"GPU Devices", True, f"{device_count} device(s)")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print(f"    GPU {i}: {device_name}")
        else:
            print("\n⚠ No GPU detected. Analysis will run on CPU (slower).")
        
        return True
        
    except Exception as e:
        print_status("PyTorch GPU Check", False, str(e))
        return False


def check_dataset():
    """Check dataset availability"""
    print_header("DATASET CHECK")
    
    dataset_paths = [
        'dataset/sl_piliyandala/train.csv',
        'dataset/sl_piliyandala/val.csv',
        'dataset/sl_piliyandala/test.csv',
    ]
    
    all_ok = True
    
    for path in dataset_paths:
        exists = os.path.exists(path)
        
        if exists:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print_status(path, True, f"{size_mb:.2f} MB")
        else:
            print_status(path, False, "NOT FOUND")
            all_ok = False
    
    if not all_ok:
        print("\n⚠ Dataset files missing!")
        print("   Expected location: dataset/sl_piliyandala/")
    
    return all_ok


def check_model_checkpoints():
    """Check for trained model checkpoints"""
    print_header("MODEL CHECKPOINT CHECK")
    
    checkpoint_dirs = [
        './drive/MyDrive/msc-val/model_log',
        './checkpoints',
        './model_checkpoints'
    ]
    
    found_checkpoints = []
    
    for check_dir in checkpoint_dirs:
        if os.path.exists(check_dir):
            for root, dirs, files in os.walk(check_dir):
                for file in files:
                    if file == 'checkpoint.pth':
                        checkpoint_path = os.path.join(root, file)
                        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
                        found_checkpoints.append((checkpoint_path, size_mb))
    
    if found_checkpoints:
        print(f"\n Found {len(found_checkpoints)} checkpoint(s):\n")
        for path, size in found_checkpoints:
            print(f"   ✓ {path} ({size:.2f} MB)")
        return True
    else:
        print_status("Model checkpoints", False, "NO CHECKPOINTS FOUND")
        print("\n⚠ Train a model first using:")
        print("   bash PatchXFormer.sh")
        return False


def check_output_directory():
    """Check output directory"""
    print_header("OUTPUT DIRECTORY CHECK")
    
    output_dirs = [
        'shap_analysis_weather/results',
        'shap_analysis_weather/results/plots',
        'shap_analysis_weather/results/csv_reports'
    ]
    
    for dir_path in output_dirs:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
                print_status(dir_path, True, "Created")
            except Exception as e:
                print_status(dir_path, False, str(e))
                return False
        else:
            print_status(dir_path, True, "Exists")
    
    return True


def check_memory():
    """Check available memory"""
    print_header("MEMORY CHECK")
    
    try:
        import psutil
        
        # RAM
        ram = psutil.virtual_memory()
        ram_gb = ram.total / (1024**3)
        ram_available_gb = ram.available / (1024**3)
        
        ram_ok = ram_available_gb > 4
        print_status(
            f"RAM: {ram_available_gb:.1f} GB available / {ram_gb:.1f} GB total",
            ram_ok,
            "Recommended: 8+ GB"
        )
        
        # Disk space
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        
        disk_ok = disk_free_gb > 2
        print_status(
            f"Disk: {disk_free_gb:.1f} GB free",
            disk_ok,
            "Recommended: 5+ GB"
        )
        
        return ram_ok and disk_ok
        
    except ImportError:
        print_status("Memory check", False, "psutil not installed (optional)")
        return True


def check_scripts():
    """Check if analysis scripts exist"""
    print_header("ANALYSIS SCRIPTS CHECK")
    
    scripts = [
        'shap_analysis_weather/shap_patchxformer_analysis.py',
        'shap_analysis_weather/run_shap_analysis.py',
        'shap_analysis_weather/shap_simple_kernel.py',
        'shap_analysis_weather/requirements.txt',
        'shap_analysis_weather/README.md'
    ]
    
    all_ok = True
    
    for script in scripts:
        exists = os.path.exists(script)
        print_status(script, exists)
        if not exists:
            all_ok = False
    
    return all_ok


def generate_report(checks):
    """Generate final report"""
    print_header("SETUP VERIFICATION REPORT")
    
    total = len(checks)
    passed = sum(checks.values())
    
    print(f"\n Total Checks: {total}")
    print(f" Passed: {passed}")
    print(f" Failed: {total - passed}")
    print(f" Success Rate: {(passed/total)*100:.1f}%")
    
    print("\n" + "-"*80)
    
    if passed == total:
        print("\n ✓ ALL CHECKS PASSED!")
        print("\n You're ready to run SHAP analysis:")
        print("   python shap_analysis_weather/run_shap_analysis.py")
    else:
        print("\n ✗ SOME CHECKS FAILED")
        print("\n Please fix the issues above before running SHAP analysis.")
        
        # Prioritized recommendations
        if not checks.get('packages', False):
            print("\n CRITICAL: Install required packages first:")
            print("   pip install -r shap_analysis_weather/requirements.txt")
        
        if not checks.get('dataset', False):
            print("\n CRITICAL: Dataset files missing!")
            print("   Ensure dataset is at: dataset/sl_piliyandala/")
        
        if not checks.get('checkpoints', False):
            print("\n IMPORTANT: Train model first:")
            print("   bash PatchXFormer.sh")
        
        if not checks.get('memory', False):
            print("\n WARNING: Low memory detected.")
            print("   Consider using smaller sample sizes:")
            print("   --num_samples 200 --background_samples 50")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print(" SHAP ANALYSIS SETUP VERIFICATION")
    print(" PatchXFormer Solar Power Forecasting Model")
    print("="*80)
    
    checks = {}
    
    # Run all checks
    checks['python'] = check_python_version()
    checks['packages'] = check_packages()
    checks['gpu'] = check_torch_gpu()
    checks['dataset'] = check_dataset()
    checks['checkpoints'] = check_model_checkpoints()
    checks['output'] = check_output_directory()
    checks['memory'] = check_memory()
    checks['scripts'] = check_scripts()
    
    # Generate report
    generate_report(checks)


if __name__ == '__main__':
    main()
