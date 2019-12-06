# Unsupervised Anomaly Localization using Variational Auto-Encoders
Copyright © German Cancer Research Center (DKFZ), [Division of Medical Image Computing (MIC)](https://www.dkfz.de/en/mic/index.php). Please make sure that your usage of this code is in compliance with the code [license](../master/LICENSE).

Code for the MICCAI19 paper [Unsupervised Anomaly Localization using Variational Auto-Encoders](https://arxiv.org/abs/1907.02796).

*Abstract*:

An assumption-free automatic check of medical images for potentially overseen anomalies would be a valuable assistance for a radiologist. Deep learning and especially Variational Auto-Encoders (VAEs) have shown great potential in the unsupervised learning of data distributions. In principle, this allows for such a check and even the localization of parts in the image that are most suspicious. Currently, however, the reconstruction-based localization by design requires adjusting the model architecture to the specific problem looked at during evaluation. This contradicts the principle of building assumption-free models. We propose complementing the localization part with a term derived from the Kullback-Leibler (KL)-divergence. For validation, we perform a series of experiments on FashionMNIST as well as on a medical task including >1000 healthy and >250 brain tumor patients. Results show that the proposed formalism outperforms the state of the art VAE-based localization of anomalies across many hyperparameter settings and also shows a competitive max performance.

## How to run:

The Fashion-MNIST experiments can simply be reproduced using the *minst_script.py*.

For the brain MRI experiments first download the HCP and BraTS-17 datasets and preprocess them using *utils/preprocess_brain.py*. Then you can reproduce the results using the *brain_script.py* script.

***


