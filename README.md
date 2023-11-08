# pyCAP

**pyCAP** is a Python package for performing simulations of chiral active Brownian particles with alignment and excluded volume interactions in 2D bulk or under circular confinement.

## Dependencies

Before using this package, ensure you have the following dependencies installed:

- [Python 3](https://www.python.org): The code is written in Python, so you need to have Python 3 installed on your system.

- [NumPy](https://numpy.org): NumPy is used for numerical computations and data structures, making it essential for this package.

- [Cython](https://cython.org): Cython is required for compiling the Cython code in this package.

- [pickle](https://docs.python.org/3/library/pickle.html): The `pickle` module is used for serialization and deserialization of Python objects.

```bash
pip install numpy
pip install cython
pip install pickle
```
## Usage

To get started with the package, create a `Simulation` instance, initialize the simulation, and run it with your desired parameters. You can find more detailed usage information in the package documentation.

```bash
# require simulation.pyx, sim_conf.pyx, setup.py, main.py, parameter.py, parameter.txt
# compile to cython:
>>> python setup.py build_ext --inplace
# run simulation:
>>> python main.py -p parameter.txt
```
## Documentation

This package comes with comprehensive documentation within the code. It provides explanations of class parameters, methods and usage examples.

## License

This project is distributed under the MIT License, which means you are free to use, modify, and distribute it according to the terms of the license.

## Publications

The following works used code in this repository:

- [Collective Hall current in chiral active fluids: Coupling of phase and mass transport through traveling bands](https://arxiv.org/abs/2307.11115)  
Authors: F. Siebers, R. Bebon, A. Jayaram, T. Speck
- [Exploiting compositional disorder in collectives of light-driven circle walkers](https://www.science.org/doi/full/10.1126/sciadv.adf5443)  
Authors: F. Siebers, A. Jayaram, P. Blümler, T. Speck

## Authors

- Frank Siebers
