#!/usr/bin/env python3
"""
Brain Tumor Detection - Demo Script
===================================

This script demonstrates the complete brain tumor detection pipeline.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

def print_banner():
    """Print welcome banner"""
    print("="*60)
    print("🧠 BRAIN TUMOR DETECTION SYSTEM DEMO")
    print("Vision Transformer (ViT) for Medical Image Analysis")
    print("="*60)

def check_environment():
    """Check if environment is properly set up"""
    print("\n🔍 Checking environment...")
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"   Python version: {python_version}")
    
    # Check key dependencies
    dependencies = [
        'torch', 'torchvision', 'timm', 'transformers',
        'numpy', 'pandas', 'matplotlib', 'seaborn',
        'scikit-learn', 'PIL', 'cv2', 'albumentations',
        'fastapi', 'gradio', 'uvicorn'
    ]
    
    missing_deps = []
    for dep in dependencies:
        try:
            if dep == 'PIL':
                import PIL
            elif dep == 'cv2':
                import cv2
            else:
                __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            missing_deps.append(dep)
            print(f"   ❌ {dep}")
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("   ✅ All dependencies installed!")
    return True

def check_dataset():
    """Check dataset availability"""
    print("\n📊 Checking dataset...")
    
    dataset_path = Path("brain_tumor_dataset")
    
    if not dataset_path.exists():
        print(f"   ❌ Dataset not found at {dataset_path}")
        return False
    
    # Check for class directories
    no_tumor_path = dataset_path / "no"
    tumor_path = dataset_path / "yes"
    
    if not no_tumor_path.exists():
        print(f"   ❌ 'no' class directory not found")
        return False
    
    if not tumor_path.exists():
        print(f"   ❌ 'yes' class directory not found")
        return False
    
    # Count images
    no_images = len(list(no_tumor_path.glob("*.jpg")) + list(no_tumor_path.glob("*.jpeg")) + list(no_tumor_path.glob("*.png")))
    tumor_images = len(list(tumor_path.glob("*.jpg")) + list(tumor_path.glob("*.jpeg")) + list(tumor_path.glob("*.png")))
    
    print(f"   ✅ Dataset found with {no_images + tumor_images} images")
    print(f"      - No tumor: {no_images} images")
    print(f"      - Tumor: {tumor_images} images")
    
    return True

def demonstrate_model_creation():
    """Demonstrate model creation"""
    print("\n🤖 Demonstrating model creation...")
    
    try:
        from src.model import create_model
        
        print("   Creating Vision Transformer model...")
        model = create_model(
            model_name="vit_base_patch16_224",
            num_classes=2,
            pretrained=True
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✅ Model created successfully!")
        print(f"      - Architecture: Vision Transformer (ViT-Base)")
        print(f"      - Parameters: {total_params:,}")
        print(f"      - Input size: 224x224x3")
        print(f"      - Output classes: 2 (No Tumor, Tumor)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating model: {e}")
        return False

def demonstrate_data_loading():
    """Demonstrate data loading"""
    print("\n📁 Demonstrating data loading...")
    
    try:
        from src.dataset import BrainTumorDataset, create_data_loaders
        
        print("   Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir="brain_tumor_dataset",
            batch_size=8,
            image_size=224,
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            num_workers=2
        )
        
        print(f"   ✅ Data loaders created successfully!")
        print(f"      - Training samples: {len(train_loader.dataset)}")
        print(f"      - Validation samples: {len(val_loader.dataset)}")
        print(f"      - Test samples: {len(test_loader.dataset)}")
        print(f"      - Batch size: 8")
        
        # Test loading a batch
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        print(f"      - Sample batch shape: {images.shape}")
        print(f"      - Sample labels: {labels}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating data loaders: {e}")
        return False

def check_gpu_availability():
    """Check GPU availability"""
    print("\n🖥️  Checking GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✅ GPU available: {gpu_name}")
            print(f"      - Memory: {gpu_memory:.1f} GB")
            print(f"      - CUDA version: {torch.version.cuda}")
        else:
            print("   ⚠️  No GPU available, will use CPU")
            print("      - Training will be slower on CPU")
            print("      - Consider using Google Colab or GPU instance")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error checking GPU: {e}")
        return False

def show_next_steps():
    """Show next steps for users"""
    print("\n🚀 NEXT STEPS")
    print("-" * 30)
    print("1. 📓 Run EDA Notebook:")
    print("   jupyter notebook notebooks/01_eda_brain_tumor_detection.ipynb")
    
    print("\n2. 🎯 Train the Model:")
    print("   python train_model.py --epochs 50 --batch_size 32")
    
    print("\n3. 🌐 Start API Server:")
    print("   cd api && python main.py")
    
    print("\n4. 🎨 Launch Gradio Interface:")
    print("   cd api && python gradio_app.py")
    
    print("\n5. 📊 Quick Start:")
    print("   ./run.sh")
    
    print("\n📚 Documentation:")
    print("   - README.md: Complete project documentation")
    print("   - notebooks/: Jupyter notebooks with examples")
    print("   - API docs: http://localhost:8000/docs (when API is running)")

def main():
    """Main demo function"""
    print_banner()
    
    # Run checks
    checks = [
        ("Environment", check_environment),
        ("Dataset", check_dataset),
        ("GPU", check_gpu_availability),
        ("Model Creation", demonstrate_model_creation),
        ("Data Loading", demonstrate_data_loading),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{'='*20} {name.upper()} {'='*20}")
        result = check_func()
        results.append((name, result))
        time.sleep(1)  # Small delay for readability
    
    # Summary
    print("\n" + "="*60)
    print("📋 SYSTEM CHECK SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name:<20}: {status}")
        if not result:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("   Your system is ready for brain tumor detection!")
        show_next_steps()
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("   Please resolve the issues above before proceeding.")
    
    print("="*60)

if __name__ == "__main__":
    main()
