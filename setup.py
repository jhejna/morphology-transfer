import sys, os
from setuptools import setup, find_packages

'''
Bot Transfer Setup Script

Notes for later: see package_data arg if additional files need to be supplied.
'''
if sys.version_info.major != 3:
    print('Please use Python3!')

setup(name='bot_transfer',
        packages=[package for package in find_packages()
                    if package.startswith('bot_transfer')],
        install_requires=[
           ],
        extras_require={
            },
        description='Framework for Morphology Transfer Experiments',
        author='Joey Hejna',
        lisence='MIT',
        version='0.0.1',
        )
