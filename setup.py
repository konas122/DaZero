from setuptools import setup
from dazero import __version__


setup(name='dazero',
    version=__version__,
    license='MIT License',
    install_requires=['numpy', 'matplotlib', 'urllib3'],
    description='Deep Learning Framework from Zero',
    author='konas122',
    author_email='2407626470@qq.com',
    url='https://github.com/konas122/DaZero',
    packages=['dazero'],
)
