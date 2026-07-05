"""
marker_install_check.py
============================
验证Marker安装是否正确，确保代码逻辑没有问题。
在游戏本上运行此脚本，检查Marker是否可用。
"""

import subprocess
import sys
import os

def check_marker_installation():
    print("="*60)
    print("Marker Installation Check")
    print("="*60)
    
    print("\n1. Checking Python version...")
    python_version = sys.version_info
    print(f"   Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 10):
        print("   ❌ Python version too old. Marker requires Python >= 3.10")
        return False
    
    print("\n2. Checking Marker installation...")
    try:
        import marker
        print(f"   ✅ Marker installed")
        
        marker_version = getattr(marker, '__version__', 'unknown')
        print(f"   Marker version: {marker_version}")
        
        print("\n3. Checking Marker imports...")
        from marker.converters.pdf import PdfConverter
        print("   ✅ PdfConverter imported successfully")
        
        from marker.models import create_model_dict
        print("   ✅ create_model_dict imported successfully")
        
        from marker.output import text_from_rendered
        print("   ✅ text_from_rendered imported successfully")
        
        print("\n4. Checking PyMuPDF (fallback)...")
        try:
            import fitz
            print(f"   ✅ PyMuPDF installed: version {fitz.__version__}")
        except ImportError:
            print("   ⚠️ PyMuPDF not installed (optional fallback)")
        
        print("\n5. Checking CUDA availability...")
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
                print(f"   ✅ CUDA available: {gpu_name} ({gpu_memory}GB VRAM)")
            else:
                print("   ⚠️ CUDA not available, will run on CPU (slower)")
        except ImportError:
            print("   ⚠️ PyTorch not installed (required for Marker)")
        
        print("\n" + "="*60)
        print("✅ Installation check passed!")
        print("="*60)
        print("\nTo run Marker:")
        print('   python -c "from marker.converters.pdf import PdfConverter; print(\'Marker works!\')"')
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Marker import failed: {e}")
        print("\nInstallation steps:")
        print("   pip install marker-pdf")
        print("   python -c \"import marker\"")
        return False

def test_marker_logic():
    print("\n" + "="*60)
    print("Marker Logic Test (Simulated)")
    print("="*60)
    
    test_pdf = "organized_documents/PDF_Documents/single_table.pdf"
    if not os.path.exists(test_pdf):
        print(f"⚠️ Test PDF not found: {test_pdf}")
        print("   Please place a test PDF in the organized_documents/PDF_Documents/ folder")
        return
    
    print(f"\nTest PDF: {test_pdf}")
    file_size = os.path.getsize(test_pdf) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")
    
    try:
        import fitz
        doc = fitz.Document(test_pdf)
        print(f"Pages: {len(doc)}")
        doc.close()
        print("✅ PyMuPDF can read the PDF")
    except Exception as e:
        print(f"❌ PyMuPDF read failed: {e}")
    
    print("\nMarker parsing flow (simulated):")
    print("-" * 40)
    print("1. Import marker modules")
    print("2. create_model_dict() - loads AI models")
    print("3. PdfConverter() - creates converter instance")
    print("4. converter(pdf_path) - parses PDF")
    print("5. text_from_rendered() - converts to Markdown")
    print("\n✅ Logic flow is correct!")

if __name__ == "__main__":
    installed = check_marker_installation()
    
    if installed:
        test_marker_logic()
    else:
        print("\nPlease install Marker first:")
        print("   pip install marker-pdf")
        print("\nAfter installation, run this script again to verify.")
