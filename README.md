This Python Code can be used to reproduced the figures of our paper ["Convergence Rates for Non-Log-Concave Sampling and Log-Partition Estimation"](https://arxiv.org/abs/2303.03237). Please cite the paper if you use this code for research purposes. It requires the `torch`, `numpy` and `matplotlib` libraries to run. All figures will be generated by running `python3 experiments.py device`, where `device` can be a PyTorch device name like `cpu` or `cuda:0`. Intermediate results will be saved in order to avoid recomputing them if a plot should be changed.
