# Colab-Based Projects

Several of my projects have been work based primarily in Google Colab, and they are collected here. Many of these are still works in progress.

### RNN Multiclass

Code for the project resulting in our [Geometry of Integration in Text Classification RNNs paper](https://arxiv.org/pdf/2010.15114.pdf) (accepted to ICLR 2021). This work uses tools from dynamical systems analysis to better understand recurrent networks trained on text-classiciation tasks. Note this work relies on the [reverse-engineering neural networks (renn) library](https://github.com/google-research/reverse-engineering-neural-networks).

### RNN Seq2Seq

Follow-up work to the above that aims to analyze the dynamics of attention in Transformer-like and recurrent architectures. A paper based on this work has been submitted to ICML 2021.

### Echo-State RNN

Also a follow-up to the multi-class RNN work above. I aim to leverage our understanding of how RNNs trained on such tasks behave in order to better understand the efficacy of echo-state networks. 

### RNN RL

Some early work attempting to apply our dynamical systems analysis to RNNs trained on reinforcement learning tasks.

### Large Learning Rates

Large learning rate work that is a follow-up to [large learning rate and catapult behavior observed in networks with MSE loss](https://arxiv.org/pdf/2003.02218.pdf). We aim to extend such results to network with cross-entropy loss.

