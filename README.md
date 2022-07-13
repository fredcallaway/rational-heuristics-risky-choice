# Automatically Deriving Resource-Rational Heuristics for Risky Choice

This repository contains code and data for the analyses for "Automatically Deriving Resource-Rational Heuristics for Risky Choice", 
which is currently under review ([preprint](https://psyarxiv.com/mg7dn)).

Model simulations can be generated by running `julia -p auto run_model.jl` in the julia/ directory. Note that this takes a long time even on a server with many cores.

Behavioral analyses can be performed by running 'python3 main.py' in the python/ directory.
You may first create the "risky-choice" virtual environment by going to the python/env/ directory and running 'conda env create -f environment.yml'. main.data_processing() may take ~30 minutes.
