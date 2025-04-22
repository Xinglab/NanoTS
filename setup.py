import setuptools
import os
from setuptools import setup, find_packages
def read_requirements(filename):
    """Read dependencies from a requirements file."""
    if os.path.isfile(filename):
        with open(filename) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

# Reading dependencies from pip_requirements.txt
#pip_requirements = read_requirements("pip_requirements.txt")


setuptools.setup(
    name="nanoTS",
    version="1.0.0",
    author="Zelin Liu",
    author_email="liuz6@chop.edu",
    description="A nanopore transcriptome SNP caller",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nanoTS",  # Update to your actual repository
    license="GPL-3.0",  
    packages=setuptools.find_packages(where="source"),
    package_dir={"": "source"},
    include_package_data=True,  # Includes config.yaml, LICENSE, etc.
    entry_points={
        "console_scripts": [
            "nanoTS=nanoTS.nanoTS:main",  # Adjust if necessary
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",  # Ensure compatibility with Python 3.12.x
)

