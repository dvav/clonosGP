import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='clonosGP',
    version='1.0',
    description='Clonal deconvolution in cancer using longitudinal NGS data',
    long_description=long_description,
    long_description_content_type="text/markdown",      
    url='http://github.com/dvav/clonosGP',
    author='Dimitrios V. Vavoulis',
    author_email='dimitris.vavoulis@gmail.com',
    license='MIT',
    packages=setuptools.find_packages(),    
    python_requires='>=3.7',    
    install_requires=[
        'numpy>=1.17.5',
        'scipy>=1.4.1',
        'pandas>=0.25.3',
        'matplotlib>=3.1.2',
        'theano>=1.0.4',
        'pymc3>=3.8'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ])
