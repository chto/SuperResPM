#!/bin/bash
cd SuperResPM/diffrax  
pip install .
cd ../../
cd SuperResPM/jax_cosmo 
pip install .
cd ../../
cd SuperResPM/JaxPM
pip install .
cd ../../
cd SuperResPM/numpyro
pip install .
cd ../../
pip install -e .
