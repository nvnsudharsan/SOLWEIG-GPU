from setuptools import setup, find_packages

setup(
    name="solweig-gpu",
    version="0.1.0",
    description="GPU-accelerated SOLWEIG model for urban thermal comfort simulation",
    author="Harsh Kamath, Naveen Sudharsan",
    author_email="harsh.kamath@utexas.edu, naveens@utexas.edu",
    url="https://github.com/nvnsudharsan/utci-pipeline",
    packages=find_packages(),  # Automatically finds `solweig_gpu`
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
        "tqdm"
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
        ],
    },
)
