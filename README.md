# CeBr detector image classification

This repository includes my results for multiplicity classification of simulated CeBr data after playing around with a few CNN architectures.
The required data and data preparation steps are available [here](https://github.com/geirtul/event_classification_example).
In particular, the 200k image dataset has to be processed to be fed into the neural network.

The resulting notebook is available in the `notebooks` folder. It is also available as [html](notebooks/CeBr_CNN.html) and [markdown](notebooks/CeBr_CNN.md) for easier viewing online.

The final trained model is [available as well for download](notebooks/weights_best.hdf5) and achieves a validation accuracy of **98.7%**.
