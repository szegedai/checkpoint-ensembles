import setuptools

setuptools.setup(
   name='trainutils',
   description='Training utilities.',
   version='1.0',
   author='Leah',
   packages=setuptools.find_packages(),
   install_requires=[
       'matplotlib',
       'numpy',
       'pandas',
       'pillow',
       'torch',#>=2.0.1',
       'torchvision',#>=0.15.2',
       'tqdm'
   ]
)
