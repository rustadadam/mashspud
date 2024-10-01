# setup.py
from setuptools import setup, find_packages

setup(
    name='mashspud',  
    version='0.1',         
    packages=find_packages(), 
    description='Manifold Alignment Methods',
    author='Adam G. Rustad',
    url='https://github.com/rustadadam/SPUD_and_MASH.git',  
    include_package_data=True,  
    install_requires=[  # List of dependencies
        "scipy",
        "graphtools",
        "numpy",
        "matplotlib",
        "igraph",
        "scikit-learn",
        "seaborn",
        "pandas"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    ],
)
