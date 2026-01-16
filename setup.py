"""
Setup script untuk Windows
"""

import subprocess
import sys
import os

def install_requirements():
    """Install semua requirements"""
    print("🔧 Installing requirements...")
    
    requirements = [
        "streamlit==1.28.0",
        "numpy==1.24.0",
        "pandas==2.0.0",
        "opencv-python-headless==4.8.0",
        "Pillow==10.0.0",
        "pyzbar==0.1.9",
        "matplotlib==3.7.0",
        "seaborn==0.12.0",
        "plotly==5.17.0",
        "tldextract==5.1.0",
        "qrcode==7.4.0",
        "imutils==0.5.4",
        "python-multipart==0.0.6"
    ]
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
    
    print("✅ All requirements installed!")

def create_folders():
    """Create necessary folders"""
    folders = [
        "models",
        "utils", 
        "assets/css",
        "assets/images",
        "pages"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")
    
    # Create __init__.py files
    init_files = ["models/__init__.py", "utils/__init__.py"]
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write("# Package initialization\n")
        print(f"Created: {init_file}")

def setup_environment():
    """Setup environment untuk Windows"""
    print("🚀 Setting up QR Code Analyzer on Windows...")
    print("=" * 50)
    
    # Create folders
    create_folders()
    
    # Install requirements
    install_requirements()
    
    print("=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the application: streamlit run app.py")
    print("2. Open browser at: http://localhost:8501")
    print("3. Upload QR code images to analyze")

if __name__ == "__main__":
    setup_environment()