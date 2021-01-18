# Morphology Transfer

This repository contains a set of environments and algorithms for transferring reinforcement learning policies across agents. 

The code is based on [the original repository](https://github.com/jhejna/hierarhical_morphology_transfer) for the paper [Hierarchically Decoupled Imitation for Morphological Transfer](https://arxiv.org/abs/2003.01709). The code has been cleaned up in comparison to the original repo and new environments and algorithm variants have been added.

*Warning*: This repository has both environment and algorithmic differences from that of the original publication, and hyper-parameter sweeps have not yet been run. As I have not run every experiment configuration, there may be bugs. For exact comparisons with the aforementioned publication, please use its repository until this one has been benchmarked.

The new features include:
* Support for training low level policies with Hindsight Experience Replay (HER)
* New Sawyer Robot environments courtesy of [Hardware Conditioned Policies](https://arxiv.org/abs/1811.09864) by Tao Chen, Adithyavairavan Murali, and Abhinav Gupta. 
* Cleaner code and algorithm implementations

To see a list of all environments, look at the `bot_transfer/envs/__init__.py` file.
To understand how to addd your own environment, look at `bot_transfer/envs/base.py`.

Example experiments are included in the `configs` directory. Run them from the root directory of the repo. Code will output to the `data` directory. To train low or high level policies using existing models, save those models to a folder called `low_levels` and `high_levels` respectively. The code will search for models in these directories to finetune from.