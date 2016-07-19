# -*- coding: utf-8 -*-
import sys
import io

import imp
fp, pathname, description = imp.find_module('versioneer')
try:
    versioneer = imp.load_module('versioneer', fp, pathname, description)
finally:
    if fp: fp.close()

versioneer.VCS = 'git'
versioneer.versionfile_source = 'sigutils/_version.py'
versioneer.versionfile_build = 'sigutils/_version.py'
versioneer.tag_prefix = ''  # tags are like 1.2.0
versioneer.parentdir_prefix = 'sigutils-'  # dirname like 'myproject-1.2.0'

try:
    from setuptools import setup, find_packages
except ImportError:
    print("Please install or upgrade setuptools or pip")
    sys.exit(1)

readme = io.open('README.rst', mode='r', encoding='utf-8').read()
doclink = """
Documentation
-------------

The full documentation is at http://sigutils.rtfd.org."""
history = io.open('HISTORY.rst', mode='r',
                  encoding='utf-8').read().replace('.. :changelog:', '')

# Use cmdclass.update to add additional commands as necessary. See
# https://docs.python.org/2/distutils/extending.html#integrating-new-commands
cmdclass = versioneer.get_cmdclass()

setup(
    name='sigutils',
    version=versioneer.get_version(),
    description='"A set of utility functions for signal processing."',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    license='MIT',
    author='Ryan Dwyer',
    author_email='ryanpdwyer@gmail.com',
    url='https://github.com/ryanpdwyer/sigutils',
    zip_safe=False,
    include_package_data=True,
    # This lets setuptools include_package_data work with git
    setup_requires=["setuptools_git >= 0.3"],
    packages=find_packages(),

    # Add requirements here. If the requirement is difficult to install,
    # add to docs/conf.py MAGIC_MOCK, and .travis.yml 'conda install ...'
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas'],

    tests_require=['nose'],
    test_suite='nose.collector',
    cmdclass=cmdclass,
    keywords='sigutils',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
)
