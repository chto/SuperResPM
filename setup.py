from setuptools import setup
from setuptools import setup, find_packages



setup(
   name='SuperResPM',
   version='1.0',
   description='Some fun project',
   author='chto',
   author_email='chunhaoto@gmail.com',
   package_dir={"": "./"},
   packages=find_packages(),
   install_requires=[], #external packages as dependencies
   scripts=[
           ]
)
