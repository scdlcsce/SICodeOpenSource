import sys
import os

from setuptools import setup
from setuptools import Command
from setuptools import Extension
REQUIRES = ["numpy"]


setup(name='_cluster',
      ext_modules=[
        Extension('_cluster',
            sources = ['clustermodule.c','cluster.c'],
            include_dirs = ['/usr/local/include','/home/yuanjung/linux-4.16/include/','/home/yuanjung/linux-4.16/include/uapi'],
            library_dirs = ['/usr/local/lib']
            )
        ],
      install_requires=REQUIRES
)