# freq-e = (class) frequency estimation 

Use this software to infer the class frequencies in a collection of items (e.g documents or images). 
For example, given all blog posts about Barack Obama during a certain time period, what is the overall positive sentiment towards him? 

In our [academic paper](http://www.aclweb.org/anthology/D18-1487) we show that naive approaches which aggregate the hard labels or soft probabilities outputted from a trained discriminative classifier were often biased. Instead, we use an *implicit likelihood* method which combines a discriminitive classifier in a generative framework and allows for more robust estimation when the true prevalences of the train and test groups differ. See also 

 - Github repository: https://github.com/slanglab/freq-e
 - Research project website: http://slanglab.cs.umass.edu/doc_prevalence/
 - Paper: http://aclweb.org/anthology/D18-1487

This software currently only supports *binary* predictions. Future work will expand this to multiclass. 

# Installation 

Installing the `freq-e` package, assuming Python 3:
1. `pip install freq-e` 
2. Then follow `py_tutorial/tutorial.ipynb`. 

# Usage 

As we specify in `py_tutorial/tutorial.ipynb`, there are three different ways to obtain class frequency estimates:  
1. Create a `FreqEstimator` object and use the built-in training method. 
2. You can also train a scikit-learn classifier yourself and pass it in to freq-e. Here the model class is restricted to scikit-learn models that have a .decision_function() method. 
3. Use the standalone `infer_freq_from_predictions()` method and pass in the predicted probabilities of the positive class of the test set. This may be useful in the cases where you have certain classifier architectures that are not built from sklearn (e.g. an LSTM or CNN). 

# Citing 
If you use this software please cite our paper. Here is the [Bibtex entry](https://kakeith.github.io/bibtex/keith18emnlp.bib). 

# Contact 
Contact the software authors with any questions: Katherine Keith (kkeith@cs.umass.edu) and Brendan O'Connor (brenocon@cs.umass.edu).
