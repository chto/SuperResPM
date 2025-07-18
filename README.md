# SuperResPM

**SuperResPM** is a differentiable, FastPM-style cosmological solver implemented with [Diffrax](https://github.com/patrick-kidger/diffrax).  
The goal is to make it a cosmology-ready code for field-level inferences. 

---

## 🚀 Features

- FastPM-like leapfrog solver using `diffrax`
- Differentiable with reversible adjoint for efficient gradient computation
- Compatible with `jaxpm` forces and `jax_cosmo` cosmology

## Installation 
```
git clone --recurse-submodules -j8 https://github.com/chto/SuperResPM.git
cd SuperResPM
sh install.sh
```
---

## 🔧 Example Usage
See [notebook](notebook/Jaxpmtest.ipynb).
