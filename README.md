AOT Analysis Code
=================

This repository contains scripts to analyse simulations of AOT using coarse-grained models.
In order to use the JAX-accelerated version of the Willard-Chandler computation, install
this fork of pytim: <https://github.com/a-ws-m/pytim>.

After installing this repo (`pip install -e .`), the analysis code can be
invoked using `aot_cluster`. Try `aot_cluster -h` for help on how to use the
code. It also requires a `.yaml` file that describes the cutoff distance and the
names of the tailgroups for the models you're analysing. An example for the
models described in the paper is in `results.yaml`. The path to the `.tpr` and
`.xtc` files needs to be changed to match your own simulation results.
