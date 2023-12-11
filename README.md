# ResaPred: A Deep Residual Network with Self-Attention to Predict Protein Flexibility
ResaPred is a two-state (non-strict and strict) flexibility prediction tool with high accuracy. ResaPred combines a variety of features related to flexibility, such as secondary structure, torsion angle, solvent accessibility, etc. ResaPred is a novel deep network based on a modified 1D residual module and self-attention mechanism, which effectively extracts deep key features related to flexibility. The modified 1D residual module consists of three convolution layers, with batchnorm and relu layers added after each layer to prevent gradient explosion or vanishing. The self- attention module facilitates capturing long-range intra or inter-slice dependencies which are often overlooked by convolution layers.
# Requirements
Python 3.7.0 or higher
Pytorch 1.8.0 or higher
