"""
Setup configuration for Multi-Server Federated Learning framework.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the requirements from requirements.txt
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path, "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

# Read the README for long description (if exists)
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="multi-server-fl",
    version="0.1.0",
    description="Multi-Server Federated Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Multi-Server FL Team",
    packages=find_packages(exclude=["scripts", "tests", "data", "outputs", "swanlog"]),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="federated-learning machine-learning distributed-systems pytorch",
)
