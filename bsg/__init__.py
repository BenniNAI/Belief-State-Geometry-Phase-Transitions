"""bsg: phase transitions in the learning of belief state geometry.

Modules
    processes  generating processes (Mess3, RRXOR, products with interactions, nonergodic mixtures)
    beliefs    sampling, exact Bayesian filtering, reference predictors, probe targets (torch)
    model      small transformer batched over an ensemble of seeds
    analysis   order parameters at each checkpoint (probes, KLs, dimensions, overlaps)
    train      training script;  sweep  grids of runs;  plot / summarize  figures and tables
"""
__version__ = "0.1.0"
