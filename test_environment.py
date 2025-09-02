#!/usr/bin/env python3
"""
Simple test to verify Python
"""

import sys
import subprocess

def test_python_version():
    """Check if Python 3.8+"""
    print("Testing Python version...")
    version = sys.version_info
    print(f"Found Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("Python version is good!")
        return True
    else:
        print("Need Python 3.8+")
        return False

def test_virtual_environment():
    """Check virtual environment"""
    print("Testing virtual environment...")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Virtual environment detected!")
        print(f"Environment path: {sys.prefix}")
        return True
    else:
        print("Not in virtual environment (this is OK for now)")
        return True

def test_basic_packages():
    """Test packcages"""
    print("Testing packages...")
    
    packages_to_test = [
        ('numpy', 'numpy as np'),
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('matplotlib', 'matplotlib.pyplot as plt'),
        ('tqdm', 'tqdm'),
        ('jupyter', 'jupyter')
    ]
    
    all_good = True
    for package_name, import_name in packages_to_test:
        try:
            exec(f"import {import_name}")
            print(f"{package_name} - imported successfully")
        except ImportError as e:
            print(f"{package_name} - not found: {e}")
            all_good = False
        except Exception as e:
            print(f"{package_name} - import error: {e}")
    
    return all_good

def test_torch_gpu():
    """Check if PyTorch can see GPU (if available)"""
    print("\nTesting GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {gpu_name}")
            print(f"GPU count: {gpu_count}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("No GPU detected - will use CPU (this is fine for learning)")
            return True
    except Exception as e:
        print(f"Error checking GPU: {e}")
        return False

def test_huggingface():
    """Test HuggingFace model"""
    print("\nTesting HuggingFace model loading...")
    try:
        from transformers import AutoTokenizer
        # Use a tiny model for testing
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        test_text = "Hello, this is a test!"
        tokens = tokenizer(test_text)
        print(f"Successfully tokenized: '{test_text}'")
        print(f"Token count: {len(tokens['input_ids'])}")
        return True
    except Exception as e:
        print(f"HuggingFace test failed: {e}")
        return False

def main():
    """Run all tests and give overall result"""
    print("ENVIRONMENT VERIFICATION TEST")
    print("=" * 50)
    print("This script checks if your environment is ready for LLM optimization.")
    print()
    
    tests = [
        test_python_version(),
        test_virtual_environment(), 
        test_basic_packages(),
        test_torch_gpu(),
        test_huggingface()
    ]
    
    print("\n" + "=" * 50)
    passed = sum(tests)
    total = len(tests)
    
    if passed == total:
        print(f"ALL TESTS PASSED ({passed}/{total})")
        print("Your environment is ready for LLM optimization!")
    else:
        print(f"SOME TESTS FAILED ({passed}/{total})")
        print("You may need to install missing packages")
        print("Try running: pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
