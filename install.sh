#!/bin/bash
cd SuperResPM/diffrax  
pip install .
cd ../../
cd SuperResPM/JaxPM
pip install .
cd ../../
pip install -e .
