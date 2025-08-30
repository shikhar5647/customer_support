#!/usr/bin/env python3
"""
Quick Start Script for Customer Support System
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_virtual_env():
    """Check if virtual environment is activated"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is active")
        return True
    else:
        print("⚠️  Virtual environment not detected")
        print("   Consider creating one: python -m venv virtual_env")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_gemini_key():
    """Check if Gemini API key is set"""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        print("✅ Gemini API key is configured")
        return True
    else:
        print("❌ Gemini API key not configured")
        print("   Please set GEMINI_API_KEY environment variable")
        print("   or create a .env file with your API key")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    if not env_file.exists():
        print("\n📝 Creating .env file...")
        try:
            with open(env_file, "w") as f:
                f.write("GEMINI_API_KEY=your_gemini_api_key_here\n")
            print("✅ .env file created")
            print("   Please edit .env and add your actual Gemini API key")
            return False
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    return True

def run_tests():
    """Run system tests"""
    print("\n🧪 Running system tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_system.py"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ System tests passed")
            return True
        else:
            print("❌ System tests failed")
            print("   Output:", result.stdout)
            print("   Errors:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def start_streamlit():
    """Start Streamlit application"""
    print("\n🚀 Starting Streamlit application...")
    
    try:
        print("   The application will open in your browser")
        print("   Press Ctrl+C to stop the application")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")

def main():
    """Main quick start function"""
    print("🚀 Customer Support System - Quick Start")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check virtual environment
    check_virtual_env()
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check Gemini API key
    if not check_gemini_key():
        if not create_env_file():
            return False
        print("\n⚠️  Please configure your Gemini API key and run this script again")
        return False
    
    # Run tests
    if not run_tests():
        print("\n⚠️  System tests failed. Please check the errors above.")
        return False
    
    # Start application
    start_streamlit()
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Quick start failed. Please check the errors above.")
        sys.exit(1)
