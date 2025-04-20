#!/bin/bash
cd SuperResPM/diffrax  
pip install -e .
cd ../../
cd SuperResPM/JaxPM
pip install -e .
cd ../../
pip install -e .
