#!/usr/bin/env python3
"""
Setup Script for Employee ITS System
Automates the installation and verification process
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")

def check_python_version():
    """Check if Python version is sufficient"""
    print_info("Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print_error(f"Python 3.8+ required. Current version: {version.major}.{version.minor}.{version.micro}")
        return False

def install_requirements():
    """Install required packages"""
    print_info("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print_success("All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install packages: {str(e)}")
        return False

def verify_installation():
    """Verify that key packages are installed"""
    print_info("Verifying installation...")
    
    required_packages = [
        'pandas',
        'numpy',
        'streamlit',
        'scikit-learn',
        'plotly',
        'seaborn',
        'matplotlib'
    ]
    
    all_installed = True
    
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package} installed")
        except ImportError:
            print_error(f"{package} NOT installed")
            all_installed = False
    
    return all_installed

def check_data_file():
    """Check if data file exists"""
    print_info("Checking for data file...")
    
    data_file = Path("data") / "Data POS-ITS 2.csv"
    
    if data_file.exists():
        print_success(f"Data file found: {data_file}")
        return True
    else:
        print_error(f"Data file not found at {data_file}")
        print_info("Please ensure 'Data POS-ITS 2.csv' is in the 'data' folder")
        return False

def create_directories():
    """Create necessary directories"""
    print_info("Creating project directories...")
    
    directories = ['data', 'outputs', 'models', 'utils', 'pages', 'assets']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print_success("All directories created/verified")
    return True

def display_next_steps():
    """Display next steps for the user"""
    print_header("Setup Complete! 🎉")
    
    print("Next Steps:")
    print()
    print("1️⃣  Start the Streamlit application:")
    print("    streamlit run app.py")
    print()
    print("2️⃣  Run the command-line analysis:")
    print("    python main.py")
    print()
    print("3️⃣  View documentation:")
    print("    README.md - Full documentation")
    print("    QUICKSTART.md - Quick start guide")
    print()
    print("4️⃣  Access the application at:")
    print("    http://localhost:8501")
    print()
    print("="*70)
    print()

def main():
    """Main setup function"""
    print_header("Employee ITS System - Setup Wizard")
    
    print("This script will:")
    print("  • Check Python version")
    print("  • Install required packages")
    print("  • Verify installation")
    print("  • Check data files")
    print("  • Create directories")
    print()
    
    input("Press Enter to continue...")
    
    # Step 1: Check Python version
    print_header("Step 1: Checking Python Version")
    if not check_python_version():
        print_error("Setup cannot continue. Please upgrade Python to 3.8+")
        sys.exit(1)
    
    # Step 2: Create directories
    print_header("Step 2: Creating Project Structure")
    create_directories()
    
    # Step 3: Install packages
    print_header("Step 3: Installing Required Packages")
    print_info("This may take a few minutes...")
    if not install_requirements():
        print_error("Setup cannot continue. Please check error messages above.")
        sys.exit(1)
    
    # Step 4: Verify installation
    print_header("Step 4: Verifying Installation")
    if not verify_installation():
        print_error("Some packages are missing. Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Step 5: Check data file
    print_header("Step 5: Checking Data Files")
    check_data_file()  # Just check, don't exit if missing
    
    # Display next steps
    display_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ An error occurred: {str(e)}")
        sys.exit(1)
