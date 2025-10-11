#!/usr/bin/env python3
"""
Verification script to check if Book Chunker is set up correctly.
Run this after installation to verify everything is working.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version is 3.7+"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (need 3.7+)")
        return False

def check_dependencies():
    """Check required packages are installed"""
    required = {
        'PyPDF2': 'PyPDF2',
        'fitz': 'PyMuPDF',
        'qdrant_client': 'qdrant-client',
        'openai': 'openai',
        'dotenv': 'python-dotenv'
    }
    
    results = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package}")
            results.append(True)
        except ImportError:
            print(f"✗ {package} (run: pip install {package})")
            results.append(False)
    
    return all(results)

def check_project_structure():
    """Check project directories exist"""
    required_dirs = [
        'utils',
        'examples',
        'experimental',
        'database',
        'tests',
        'docs'
    ]
    
    results = []
    for directory in required_dirs:
        if Path(directory).is_dir():
            print(f"✓ {directory}/")
            results.append(True)
        else:
            print(f"✗ {directory}/ (missing)")
            results.append(False)
    
    return all(results)

def check_main_scripts():
    """Check main scripts exist and are executable"""
    scripts = [
        'book_extractor.py',
        'vectorize.py'
    ]
    
    results = []
    for script in scripts:
        path = Path(script)
        if path.is_file():
            is_exec = os.access(script, os.X_OK)
            status = "executable" if is_exec else "not executable (run: chmod +x)"
            print(f"✓ {script} ({status})")
            results.append(True)
        else:
            print(f"✗ {script} (missing)")
            results.append(False)
    
    return all(results)

def check_documentation():
    """Check documentation files exist"""
    docs = [
        'README.md',
        'MIGRATION.md',
        'CHANGELOG.md',
        'docs/API.md',
        'docs/ARCHITECTURE.md',
        'docs/TROUBLESHOOTING.md'
    ]
    
    results = []
    for doc in docs:
        if Path(doc).is_file():
            print(f"✓ {doc}")
            results.append(True)
        else:
            print(f"✗ {doc} (missing)")
            results.append(False)
    
    return all(results)

def check_env_setup():
    """Check .env file exists"""
    if Path('.env').is_file():
        print("✓ .env file exists")
        print("  Note: Verify your API keys are set correctly")
        return True
    else:
        print("✗ .env file missing")
        print("  Copy .env.example to .env and add your API keys")
        return False

def main():
    print("=" * 60)
    print("Book Chunker Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Main Scripts", check_main_scripts),
        ("Documentation", check_documentation),
        ("Environment Setup", check_env_setup)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 40)
        results.append(check_func())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ All checks passed! Book Chunker is ready to use.")
        print("\nQuick Start:")
        print("  python book_extractor.py --help")
        print("  python vectorize.py --help")
        print("\nSee README.md for usage examples.")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\nFor help, see docs/TROUBLESHOOTING.md")
    print("=" * 60)
    
    return 0 if all(results) else 1

if __name__ == '__main__':
    sys.exit(main())

