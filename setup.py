import os
import sys
import platform
import setuptools
from setuptools import setup

# --- Hard block non-Linux installs ---
if platform.system() != "Linux":
    sys.exit("ERROR: NanoTS is supported only on Linux systems.")

def read_requirements(filename):
    if os.path.isfile(filename):
        with open(filename) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="nanoTS",
    version="1.0.0",
    author="Zelin Liu",
    author_email="liuz6@chop.edu",
    description="A nanopore transcriptome SNP caller",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Xinglab/NanoTS",
    license="GPL-3.0",
    packages=setuptools.find_packages(where="source"),
    package_dir={"": "source"},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "nanoTS=nanoTS.nanoTS:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Environment :: Console",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    # Strictly Python 3.12.x
    python_requires=">=3.12,<3.13",
)

