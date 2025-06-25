#!/usr/bin/env python3
"""
Server startup script for the ML Regression Pipeline
Compatible with Python 3.11.11
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"Error: Python 3.8+ required, found {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'static/plots', 'temp', 'templates']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'flask_socketio', 'pandas', 'numpy', 
        'scikit-learn', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them using: pip install -r requirements.txt")
        return False
    
    return True

def install_dependencies():
    """Install dependencies from requirements.txt"""
    if os.path.exists('requirements.txt'):
        print("Installing dependencies...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("✓ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install dependencies")
            return False
    else:
        print("✗ requirements.txt not found")
        return False

def run_server():
    """Start the Flask server"""
    try:
        print("\n🚀 Starting Flask server...")
        print("Server will be available at: http://localhost:5000")
        print("Press Ctrl+C to stop the server\n")
        
        # Import and run the app
        from app import app, socketio
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure all files are in the correct location:")
        print("  - app.py")
        print("  - pipeline/regression_pipeline.py")
        print("  - utils/data_cleaning.py")
        return False
    except Exception as e:
        print(f"✗ Server error: {e}")
        return False

def main():
    """Main function"""
    print("=== ML Regression Pipeline Server ===\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create necessary directories
    create_directories()
    
    # Check if dependencies are installed
    if not check_dependencies():
        print("\nAttempting to install missing dependencies...")
        if not install_dependencies():
            print("Please install dependencies manually:")
            print("pip install -r requirements.txt")
            sys.exit(1)
    
    # Start server
    if not run_server():
        sys.exit(1)

if __name__ == "__main__":
    main()