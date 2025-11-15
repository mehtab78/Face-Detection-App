"""Setup configuration for face detection package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="face-detection-app",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Streamlit-based face detection application using OpenCV",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/face-detection-app",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "data"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.3.3",
            "pytest-cov>=5.0.0",
            "black>=24.8.0",
            "mypy>=1.11.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "face-detection-app=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.xml", "*.toml"],
    },
)
