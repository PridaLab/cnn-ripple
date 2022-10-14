# cnn-ripple

# Description

__CNN-ripple__ is a tool designed to detect sharp wave ripples (SWRs), a kind of fast oscillation that appear repetitively in hippocampal electrophysiological signals. During SWR, the sequential firing of ensembles of neurons act to reactivate memory traces of previously encoded experience. SWR-related interventions can influence hippocampal-dependent cognitive function, making their detection crucial to understand underlying mechanisms. However, existing SWR identification tools mostly rely on using spectral methods, which remain suboptimal.

![alt text](https://github.com/RoyVII/cnn-ripple/blob/main/images/example_ripple.png)

__CNN-ripple__ is a 1D convolutional neural network (CNN) operating over high-density LFP recordings to detect hippocampal SWR both offline and online.  It works with recordings from several types of recordings (e.g. linear arrays, high-density probes, ultradense Neuropixels) as well as on open databases that were not used for training. Moreover, __CNN-ripple__ can be used to detect SWRs in real time, by loading it our [custom plug-in](https://github.com/RoyVII/CNNRippleDetectorOEPlugin) to the widely used open system such [Open Ephys](https://open-ephys.org/).

The main advantages of cnn-ripple vs other SWR-detection algorithms are the following:
* Its performance lies within experts performance
* Its performance is much sess threshold-sensitive than spectral methods
* It can detect a wider variety of SWRs than spectral methods
* It can be used as an interpretation tool
* It achieves online prediction several milliseconds in advance

You can check our results in our manuscript ([Navas-Olive, Amaducci et al. eLife 2022](https://elifesciences.org/articles/77772))


## Try it yourself!

[This notebook](https://colab.research.google.com/github/RoyVII/cnn-ripple/blob/main/src/notebooks/cnn-example.ipynb) illustrates one example of sharp-wave ripple detection with __CNN-ripple__. Follow and execute each block to download data and to load the trained CNN. Detection of events and the plot are depicted at the end of the notebook. You will be able to scroll along the recording to visualize detected events

![alt text](https://github.com/RoyVII/cnn-ripple/blob/main/images/example_notebook.png)


## Technical notes

The adapted architecture includes seven convolutional deep layers composed of different filters to process LFP inputs in increasing hierarchical complexity and one output layer delivering the probability of an occurring SWR.

Inputs must meet the following characteristics:
* Sampling frequency: 1250 Hz
* Number of channels: 8
* Channels sorted from top to bottom

![alt text](https://github.com/RoyVII/cnn-ripple/blob/main/images/example_architecture.png)
