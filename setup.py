from setuptools import setup
  
setup(
    name='LinkedIn profile ranking',
    version='0.1',
    description='A package to rank potential customers',
    author='Zihui Ouyang',
    author_email='makingsurgeon@gmail.com',
    packages=['my_package'],
    install_requires=[
        'scikit-learn',
        'pandas',
        'numpy',
        'scipy'
    ],
)