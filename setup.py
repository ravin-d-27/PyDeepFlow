from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pydeepflow",  # Your package name
    version="0.1.0",  # Initial release version
    author="Ravin D",
    author_email="ravin.d3107@gmail.com",
    description="A deep learning package optimized for performing Deep Learning Tasks, easy to learn and integrate into projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ravin-d-27/PyDeepFlow",  # Repo URL
    packages=find_packages(),  # Automatically discover package directories
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
    classifiers=[
        "Development Status :: 3 - Alpha",  # Development status of the project
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.6',  # Minimum Python version requirement
    install_requires=[
        "numpy==1.23.5",
        "pandas==1.5.3",
        "scikit-learn==1.2.0",
        "jupyter==1.0.0",
        "tqdm==4.64.1",
        "colorama==0.4.6",
        "cupy",  
    ],  
)
