from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="solweig-gpu",
    version="1.2.14",
    description="GPU-accelerated SOLWEIG model for urban thermal comfort simulation",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Harsh Kamath, Naveen Sudharsan",
    author_email="harsh.kamath@utexas.edu, naveens@utexas.edu",
    url="https://github.com/nvnsudharsan/SOLWEIG-GPU",
    packages=find_packages(),  
    include_package_data=True, 
    package_data={
        "solweig_gpu": ["landcoverclasses_2016a.txt"], 
    },
    install_requires=[
        "torch",
        "numpy",
        "scipy",
        "pandas",
        "netCDF4",
        "pytz",
        "shapely",
        "timezonefinder",
        "gdal",
        "xarray",
        "tqdm",
        "PyQt5",
        "matplotlib"
    ],
    license="GPL-3.0-or-later",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.10",
    entry_points={
        'console_scripts': [
            'solweig_gpu=solweig_gpu.cli:main',
            'thermal_comfort=solweig_gpu.cli:main',
            'solweig_gpu_gui=solweig_gpu.solweig_gpu_gui:main',
        ],
    },
)
