#!/usr/bin/env python3
"""
Simple launcher for Car Price Prediction System
Usage: python run.py
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import main

if __name__ == "__main__":
    main()
