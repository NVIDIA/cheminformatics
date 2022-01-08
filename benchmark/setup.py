import os
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup (
    name='cuchembm',
    version='0.0.1',
    description='Benchmarking tool for Cheminformatic',
    url='https://github.com/NVIDIA/cheminformatics',
    author='NVIDIA',
    packages=['cuchembm'],
    install_requires=required
)