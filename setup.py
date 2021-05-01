#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='SpeakerRecognitionYarp',
    version='0.0.0',
    description='Speaker recognition model for the iCub robot with YARP',
    author='Jonas Gonzalez',
    author_email='jonas.gonzalez@iit.it',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/PyTorchLightning/pytorch-lightning-conference-seed',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

