from setuptools import find_packages, setup

setup(
    name='nonGenCom',
    packages=find_packages(include=['nonGenCom']),
    version='0.1.0',
    description='Library with independent functions to compare two databases (Forensic Case Database and Missing '
                'Person Database), generating scores based on non-genetic variables.',
    setup_requires=['pandas']
)
