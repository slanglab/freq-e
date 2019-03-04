# freq-e = (class) frequency estimation 

Use this software to infer the class frequencies in a collection of items (e.g documents or images). 
For example, given all blog posts about Barack Obama during a certain time period, what is the overall positive sentiment towards him? 

In our [academic paper](http://www.aclweb.org/anthology/D18-1487), we showed that the naive approaches to this problem that aggregated the hard labels or soft probabilies from a trained discriminative classifier were often biased. Instead, this software uses an *implicit likelihood* method which combines a discriminitive classifier in a generative framework and allows for more robust estimation when the true training and test group class frequencies differ. 

# Installation 

Installing the (test) `freq-e` package: 
1. `python3 -m pip install --index-url https://test.pypi.org/simple/ freq-e-test` 
2. Then follow `py_tutorial/tutorial.ipynb`. 

# Usage 

As we specify in `py_tutorial/tutorial.ipynb`, there are three different ways to obtain class frequency estimates: 
1. Create a `FreqEstimator` object and use the built-in training and 10-fold cross-validation method with logistic regression. 
2. Use the `infer_freq()` method and pass in a pre-trained scikit-learn linear model (e.g. `Logistic_Regression`). Here the model class is restricted to scikit-learn models that have a `.decision_function()` method. 
3. Use the `infer_freq()` method and pass in the predicted probabilities of the positive class of the test set. 

# Citing 
If you use this software please cite our paper. Here is the [Bibtex entry](https://kakeith.github.io/bibtex/keith18emnlp.bib). 

# Contact 
Contact the software authors with any questions: Katherine Keith (kkeith@cs.umass.edu) and Brendan O'Connor (brenocon@cs.umass.edu).
