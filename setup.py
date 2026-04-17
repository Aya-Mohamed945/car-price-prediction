from setuptools import setup, find_packages

setup(
    name="car-price-prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning system for predicting used car prices",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/car-price-prediction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "joblib>=1.3.2",
    ],
)
