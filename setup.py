#!/usr/bin/env python
from setuptools import setup
from distutils.core import Command
import subprocess
import time


class BuildDENN(Command):

    description = "Build DENN libs"
    user_options = []

    def initialize_options(self):
        pass

    def run(self):
        subprocess.run(
            ["make clean"],
            shell=True
        )
        subprocess.run(
            ["make"],
            shell=True
        )
    
    def finalize_options(self):
        pass
    


setup(
    name='DENN',
    version='1.0.0',
    description='Differential Evolution Op for TensorFlow',
    author='Gabriele Di Bari && Mirco Tracolli',
    author_email='ToDo',
    url='https://github.com/Gabriele91/DENN',
    package_dir={
        'DENN': 'DENN',
        'training': 'DENN/training'
    },
    package_data={
        'DENN': ['*.so']
    },
    packages=[
        'DENN',
        'DENN.training'
    ],
    cmdclass={
        'build_denn': BuildDENN
    },
)
