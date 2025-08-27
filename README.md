# Overview
This repository contains the python scripts to build and train a **two-headed LeNet** architecture to regress the orientation vector of suspended objects from two perpendicular views.
This methodology is intended to measure the orientation of single suspended particles rotating in viscous shear flows. 
**Given the impossibility to obtain labeled training data for this application, the model here presented was intended to be trained on synthetic data.**

## Organization
Custom models and data loaders can be found in [src](https://github.com/ddg93/LeRing_JFM/tree/main/src)
Run the main.py file on the training data to 

# Data
Training data as well as experimental recordings of the rotation of axis-symmetric particle suspended in a confined shear flow in the viscous and small-inertia regime can be found at [the following repository](https://huggingface.co/datasets/ddg93/LeRing_JFM_experiments/tree/main)

# License
This repository is licensed under a cc-by-4.0 license.  
You are free to use, share, and adapt this dataset for any purpose, provided you give appropriate credit to the original author.

# Citation
If you use this repository in your work or are inspired by our approach, please cite our articles:

[1] [**Di Giusto, D.** and Bergougnoux L. and Guazzelli, É., 2025, **Orientation of flat bodies of revolution in shear flows at low Reynolds number**, Journal of Fluid Mechanics, 1017, p.A41.](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/orientation-of-flat-bodies-of-revolution-in-shear-flows-at-low-reynolds-number/B66C5D71373A2DF5199B3AA399AE78C7#article)

[2] [**Di Giusto, D.** and Bergougnoux, L. and Marchioli, C. and Guazzelli, É., 2024. **Influence of small inertia on Jeffery orbits**, Journal of Fluid Mechanics, 979, p.A42.](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/influence-of-small-inertia-on-jeffery-orbits/8B7D42276607AD9A069E99B91DA9BD1E)
