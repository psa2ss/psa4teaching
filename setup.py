from setuptools import setup, find_packages

setup(
    name="psa4teaching",
    version="1.0.0",
    author="PSA Teaching Team",
    author_email="psa@example.edu",
    description="电力系统分析教学Python包",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/psa2ss/psa4teaching",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
)