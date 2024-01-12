# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 07:49:16 2017

@author: Ali Aghaeifar
"""


from setuptools import setup


setup(name='pyshim', # this will be name of package in packages list : pip list 
      version='0.1.0',
      description='shimming toolbox',
      keywords='twix,reconstruction,mri,shimming',
      author='Ali Aghaeifar',
      author_email='ali.aghaeifar [at] tuebingen.mpg [dot] de',
      license='MIT License',
      packages=['pyshim'],
      install_requires = ['tqdm','numpy','nibabel','torch','twixtools', 'recotwix', 'scipy']
     )
