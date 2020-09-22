from setuptools import setup, find_packages

# setup.
setup(name='retina-chatbands',
      version='0.1',
      description='detection of retina chatbands in 3d fluorescence microscopy stacks',
      license='BSD',
      packages=find_packages(exclude=[
          'tests',
      ]),
      install_requires=[
          'luigi',
          'pandas',
          'tqdm',
          'tensorflow==1.12',
          'keras==2.2.4',
          'keras-applications==1.0.7',
          'opencv-python>=3.4',
          'matplotlib',
          'czifile',
          'scipy',
          'scikit-image==0.14.2',
          'scikit-learn==0.20.3',
          'absl-py==0.7.0',
      ])
