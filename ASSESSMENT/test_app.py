#!/usr/bin/env python3

print("Testing Python execution...")
print("Current working directory:", __file__)

try:
    import pandas as pd
    print("✓ pandas imported successfully")
except ImportError as e:
    print("✗ pandas import failed:", e)

try:
    from flask import Flask
    print("✓ flask imported successfully")
except ImportError as e:
    print("✗ flask import failed:", e)

try:
    import sklearn
    print("✓ sklearn imported successfully")
except ImportError as e:
    print("✗ sklearn import failed:", e)

print("Test complete.")