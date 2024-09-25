# setup.py
from setuptools import setup, find_packages

setup(
    name='SPUD_and_MASH',   # Name of your package
    version='0.1',          # Version number
    packages=find_packages(), # Automatically find sub-packages
    description='Manifold Alignment Methods',
    #author='Your Name',
    #author_email='your.email@example.com',
    url='https://github.com/rustadadam/SPUD_and_MASH.git',  # URL for the package (if applicable)
    include_package_data=True,  # To include non-Python files if you have any
    install_requires=[  # List of dependencies
        # Add any dependencies like 'numpy', 'pandas', etc.
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License'  # Or another license
    ],
)
