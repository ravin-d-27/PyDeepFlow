from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pydeepflow",
    version="1.0.0",  # Updated version
    author="Ravin D",
    author_email="ravin.d3107@outlook.com",
    description="A deep learning package optimized for performing Deep Learning Tasks, easy to learn and integrate into projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ravin-d-27/PyDeepFlow",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",  # Additional metadata
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.23.5",
        "pandas>=1.5.3",
        "scikit-learn>=1.2.0",
        "jupyter>=1.0.0",
        "tqdm>=4.64.1",
        "colorama>=0.4.6",
        "matplotlib",
    ],
    extras_require={
        "gpu": ["cupy>=9.6.0"],  # Optional GPU support
        "testing": ["pytest>=6.2.5"],  # Dependencies for testing
    },
    entry_points={
        "console_scripts": [
            "pydeepflow-cli=pydeepflow.cli:main",  # CLI tool if applicable
        ],
    },
    keywords="deep-learning artificial-intelligence neural-networks tensorflow pytorch",  # Add relevant keywords
    license="MIT",
    project_urls={
        "Bug Tracker": "https://github.com/ravin-d-27/PyDeepFlow/issues",
        "Source Code": "https://github.com/ravin-d-27/PyDeepFlow",
        "Documentation": "https://github.com/ravin-d-27/PyDeepFlow/wiki",
    },
)
