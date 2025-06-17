from setuptools import setup, find_packages

setup(
    name="edgeyolo_runner",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "onnxruntime>=1.15.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
    ],
    extras_require={
        "gpu": ["onnxruntime-gpu>=1.15.0"],
        "tflite": ["tensorflow>=2.8.0"],
        "coreml": ["coremltools>=6.0"],
        "all": ["tensorflow>=2.8.0", "coremltools>=6.0"],
    },
    author="EdgeYOLO Runner Contributors",
    author_email="your.email@example.com",
    description="A package for running EdgeYOLO models using ONNX, TensorFlow Lite, and CoreML on images and videos",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ramonhollands/edgeyolo_runner",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 