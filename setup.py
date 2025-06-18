from setuptools import setup, find_packages

setup(
    name="solweig-gpu",
    version="1.1.0",
    description="GPU-accelerated SOLWEIG model for urban thermal comfort simulation",
    author="Harsh Kamath, Naveen Sudharsan",
    author_email="harsh.kamath@utexas.edu, naveens@utexas.edu",
    url="https://github.com/nvnsudharsan/SOLWEIF-GPU",
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
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'thermal_comfort=solweig_gpu.cli:main',
            'solweig_gpu=solweig_gpu.solweig_gpu_gui:main',
        ],
    },
)
