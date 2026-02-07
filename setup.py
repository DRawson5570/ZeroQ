"""
ZeRO-Q: Quantization-Aware Distributed Training

Setup script for installation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="zero-q",
    version="0.1.0",
    author="Zero (Claude Opus 4.5)",
    author_email="drawson@example.com",
    description="Quantization-aware distributed training combining ZeRO-3 and 4-bit quantization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DRawson5570/phoenix-cluster",
    project_urls={
        "Bug Tracker": "https://github.com/DRawson5570/phoenix-cluster/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"zero_q": "src"},
    packages=["zero_q"],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "bitsandbytes>=0.43.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
        "training": [
            "transformers>=4.36.0",
            "peft>=0.7.0",
            "datasets",
            "accelerate",
        ],
    },
)
