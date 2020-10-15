from setuptools import setup, find_packages
from setuptools.command.install import install

setup(name='locgame',
      packages=find_packages(),
      version="0.1.0",
      description='A project that plays unity games with supervised learning',
      author='Satchel Grant',
      author_email='grantsrb@stanford.edu',
      url='https://github.com/grantsrb/locgame.git',
      install_requires= ["numpy",
                         "torch",
                         "tqdm"],
      py_modules=['locgame'],
      long_description='''
            A project that plays a specific unity game that compares
            egomotion to static setups
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X :: Ubuntu',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      )
