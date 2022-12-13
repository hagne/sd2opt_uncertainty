#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 13:47:29 2022

@author: hagen
"""
import sys

required_verion = (3,6)
if sys.version_info < required_verion:
    raise ValueError('needs at least python {}! You are trying to install it under python {}'.format('.'.join(str(i) for i in required_verion), sys.version))

# import ez_setup
# ez_setup.use_setuptools()

from setuptools import setup
# from distutils.core import setup
setup(
    name="sd2opt_uncertainty",
    version="0.1",
    packages=['sd2opt_uncertainty'],
    author="Hagen Telg",
    author_email="hagen@hagnet.net",
    description="blablabl",
    license="MIT",
    keywords="sd2opt_uncertainty",
    url="https://github.com/hagne/sd2opt_uncertainty",
    # scripts=['scripts/goes_aws_scaper_surfrad', 
    #           ],
    # install_requires=['numpy','pandas'],
    # extras_require={'plotting': ['matplotlib'],
    #                 'testing': ['scipy']},
    # test_suite='nose.collector',
    # tests_require=['nose'],
)