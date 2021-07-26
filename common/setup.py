import os
from setuptools import setup, find_packages


requirementPath = 'requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()


setup(
    name='cuchem-commons',
    version='0.1',
    packages=find_packages(),
    description='Common components used in Nvidia ChemInformatics',
    install_requires=install_requires
)