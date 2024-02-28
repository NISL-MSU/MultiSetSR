# MSSP: Multi-Set Symbolic Skeleton Prediction for Symbolic Regression

## Description

We present a method that generates univariate symbolic skeletons that aim to describe the functional relation between each variable and the response.
By analyzing multiple sets of synthetic data where one input variable varies while others are fixed, relationships are modeled separately for each input variable. 

To do this, we introduce a new SR problem called Multi-Set symbolic skeleton prediction (MSSP). It receives multiple 
sets of input--response pairs, where all sets correspond to the same functional form but use different equation constants, 
and outputs a common skeleton expression, as follows:

<p align="center">
  <img src="https://raw.githubusercontent.com/GiorgioMorales/MSSP-SymbolicRegression/master/figs/mst.jpg?token=GHSAT0AAAAAACO2QV33BTWYKL2AL7AIUNDIZO6QXOA" alt="alt text" width="400">
</p>

We present a novel transformer model called "Multi-Set Transformer" to solve the MSSP problem. The model is pre-trained 
on a large dataset of synthetic symbolic expressions using an entropy-based loss function. The 
identification process of the functional form between each variable and the system's response is viewed as a sequence 
of MSSP problems:

<p align="center">
  <img src="https://raw.githubusercontent.com/GiorgioMorales/MSSP-SymbolicRegression/master/figs/Skeleton.jpg?token=GHSAT0AAAAAACO2QV33IVMLJAG7XNQVT566ZO6QV7Q" alt="alt text" width="500">
</p>
