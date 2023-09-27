# -*- coding:utf-8 -*-
from distutils.core import setup
from setuptools import find_packages

setup(
        name='ryu_pytools',
        version='0.1',
        packages=find_packages(),  # 查找包的路径
        include_package_data=False,
        description='ryuuyou\'s common python tools',
        author='ryuuyou',
        author_email='ryuuyou0529@163.com',
        url='https://github.com/RyuuYou0529/pytools',
        license='MIT',
        install_requires=[
            'numpy', 
            'matplotlib',
            ],
        python_requires='>=3.6'
    )