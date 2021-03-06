# Introduction

Tumours are mixtures of phylogenetically related cancer cell populations or *clones*, which are subject to a process of Darwinian evolution in response to selective pressures in their local micro-environment. `clonosGP` is a statistical methodology for tracking this latent heterogeneity continuously in time based on longitudinally collected tumour samples. In technical terms, it combines Dirichlet Process Mixture Models with Gaussian Process Priors to identify clusters of mutations and track their cellular prevalence continuously in time. If only cross-sectional data are available, then it performs standard non-parametric clustering of mutations based on their observed frequency, similarly to other software in the same category. The statistical models underlying `clonosGP` were implemented in the excellent probabilistic programming system `PyMC3` on which we also rely for inference using variational methods.                

# Installation

`clonosGP` requires Python 3.7 or later. It can be easilly installed as follows: 
1. Create a virtual environment: `python3 -m venv myenv`
2. Activate the newly created environment: `source myenv/bin/activate`
3. Install `clonosGP` as follows: `pip install -U clonosGP` 

All necessary dependencies will also be installed. 

# Usage

A guide to start using `clonosGP` quickly is available [here](https://github.com/dvav/clonosGP/blob/master/quickstart.ipynb). A more thorough tutorial can be found [here](https://github.com/dvav/clonosGP/blob/master/tutorial.ipynb).

# Citation

If you find this software or the statistical model behind it useful, then please cite the following paper:

> "[A statistical approach for tracking clonal dynamics in cancer using longitudinal next-generation sequencing data](https://doi.org/10.1093/bioinformatics/btaa672)" by Vavoulis DV, Cutts A, Taylor JC & Schuh A. *Bioinformatics*, 2020
