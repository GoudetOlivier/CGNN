# -*- coding: utf-8 -*-
# Copyright (C) 2016 Olivier Goudet
# Licence: Apache 2.0

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def setup_package():
    setup(name='cgnn',
          version='1.0',
          description='Causal Generative Neural Networks',
          url='https://github.com/GoudetOlivier/CGNN',
          author='Olivier Goudet',
          author_email='olivier.goudet@lri.fr',
          license='Apache 2.0',
          packages=['cgnn'])


if __name__ == '__main__':
    setup_package()
