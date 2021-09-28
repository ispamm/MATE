# A Meta-Learning Approach for Training Explainable Graph Neural Networks
**Authors**: Indro Spinelli, Simone Scardapane, Aurelio Uncini

This repotisory contains the code of MATE (MetA-Train to Explain), a meta-learning framework for improving the level of explainability of a Graph Neural Network at training time. Our approach steers the optimization procedure towards more interpretable minima meanwhile optimizing for the original classification task. [Here there is the preprint.](https://arxiv.org/abs/2109.09426)

The code is build upon the repository of [Re: Parameterized Explainer for Graph Neural Networks](https://github.com/LarsHoldijk/RE-ParameterizedExplainerForGraphNeuralNetworks) and we thanks the authors (Maarten Boon, Stijn Henckens, Lars Holdijk and Lysander de Jong) for making their code accessible to everyone.

## IPython Notebooks
- **experiment_model_training**: Meta-trains models with MATE algorithm.
- **experiment_replication**:Evaluate model's explainability.


## Installation
Install required packages using
```pip install -r requirements.txt```
additionally follow the [instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) in order to install PyTorch Geometric.
