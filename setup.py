from setuptools import setup

setup(name='dimension_regularisation',
      version="0.1",
      packages=['dimension_regularisation'],
      description='Train a neural network and regularise for the dimensions',
      author='Richard Gerum',
      author_email='richard.gerum@fau.de',
      license='MIT',
      install_requires=[
          'numpy',
          'scipy',
          'tensorflow',
          'pandas',
          'fire',
          'tensorflow_datasets',
          'scikit-learn',
      ],
      )
