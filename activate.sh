#!/bin/bash
# Quick activation script for the virtual environment
# Usage: source activate.sh

source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""
echo "Available commands:"
echo "  ./book_extractor.py --help"
echo "  ./vectorize.py --help"
echo "  python verify_setup.py"
echo ""
echo "To deactivate: deactivate"

