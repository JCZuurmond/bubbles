#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""


import setuptools


with open('README.md', encoding='UTF-8') as readme_file:
    readme = readme_file.read()


requirements = [
    'scipy>=1.4.0',
]
setup_requirements = []
test_requirements = ['pytest']
extra_requirements = {
    'develop': ['jupyter>=1.0.0'],
}


setuptools.setup(
    name='bubbles',
    author='Cor Zuurmond',
    author_email='jczuurmond@protonmail.com',
    description='',
    url='TODO',
    license='Open source',
    packages=['bubbles'],
    version='0.1.0',
    install_requires=requirements,
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require=extra_requirements,
)
