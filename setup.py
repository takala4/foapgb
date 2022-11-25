#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

__author__ = 'takala4'

setup(
    name='foapgb',  
    version='1.0.1',  
    description='Python package for solving Fujita and Ogawa (1982) model with the accelerated projected gradient method and the balancing method (APGB).',  
    author='takala4', 
    author_email='takara.sakai.t1@dc.tohoku.ac.jp',  
    url='XXX', 
    license='MIT License', 
    install_requires=[ #  
        'numpy',
        'scipy'
    ],
    packages=find_packages()
)