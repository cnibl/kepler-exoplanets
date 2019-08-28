Analysis
========

In the script ``src/neural_network.py``, the ``MLPClassifier`` class from ``scikit-learn`` is used to train a neural network with a subset of the Kepler data. The model is then validated on the remaining part of the dataset. The aim is to predict the value of the discrete ``koi_pdisposition`` column, which designates whether an object is an exoplanet candidate, or whether it is a false positive. The analysis is thus a classification analysis, where the prediction target takes a value in one of two classes. The data columns used for the prediction are taken to be the eight variables that are mostly correlated with the prediction target column. In the notebook ``src/data-plotting.ipynb`` the data is explored and visualised, and these eight variables are found. 

Target and predictors
---------------------

In the notebook ``data-plotting``, the data is explored and the variables that correlate mostly with the designation of a KOI as an exoplanet or a false candidate are found. From the analysis it is clear that the properties of the host star are not very relevant when determining this. The predictor variables are taken to be the variables that correlate to ``koi_pdisposition`` with a correlation coefficient with an absolute value above 0.2.

**Target**: ``koi_pdisposition``, i.e. the designation of the KOI as an exoplanet or a false candidate.

**Predictors**: The eight variables most correlated with ``koi_pdisposition``. These are the variables ``koi_depth``, ``koi_score``, ``koi_teq``, ``koi_model_snr``, ``koi_tce_plnt_num`` and the four ``koi_fpflag_XX`` variables.

Data treatment
--------------

To maximise the use of the data we have filled in some missing values in some columns. We fill in missing values with the mean value of the respective data column. In order to use the neural network classifier, we need numerical input. Hence, we have used an encoder to assign numerical values to the target variable, which is categorical.

We separate the data set into a training set and a validation set, where the former is used to train the neural network and the latter is used only for validation. This ensures that we do not overfit the classifier to the training set since we have an independent set for validation/testing. The training set is comprised of 80% of the full data, and the validation set the remaining 20%.

Neural network analysis
-----------------------

In the analysis, we use the ``MLPClassifier`` class from the ``scikit-learn`` package, a neural network classifier. Below we describe how we choose the parameters of the model. 

In the script ``layer_tests.py``, we try out a number of combinations for the size of the hidden layers and the number of neurons in each. To make things a bit simpler, we only look at the case where each hidden layer has the same number of neurons. A five-fold cross validation is then performed on the training set, looking at the f1 score (the so-called harmonic mean of precision and recall). The script prints the results to a file ``layer_test.txt`` in the ``results`` directory. From this we find that a good choice is 2 layers with 16 neurons each. 

In ``neural_network.py`` an extensive grid search is then performed over some of the model parameters to find the best combination. Since ``adam`` appears to be the optimal choice of solver, we do not vary the solver to reduce the computational expense of the grid search. We scan over the parameters ``alpha``, ``learning_rate_init``, ``beta_1`` and ``beta2``. We choose activation function ``logistic`` (the sigmoid function). The search picks out the best parameter values. The parameter values finally used are given in the below table.

+------------------------+------------------------+
| Parameter              | Value                  |
+========================+========================+ 
| ``activation``         | "logistic"             |
+------------------------+------------------------+ 
| ``solver``             | "adam"                 |
+------------------------+------------------------+
| ``alpha``              | 0.001                  |
+------------------------+------------------------+
| ``learning_rate_init`` | 0.001                  |
+------------------------+------------------------+
| ``beta_1``             | 0.6                    |
+------------------------+------------------------+
| ``beta_2``             | 0.7                    |
+------------------------+------------------------+
| ``hidden_layer_sizes`` | (16,16)                |
+------------------------+------------------------+


Results
-------
Some results of the analysis are printed out to the ``results`` folder. Some scores, like precision and recall, are also printed out at the end of running the ``neural_network.py`` script. 

- ``layer_tests.txt``: Result of various choices for the size of the hidden layers and the number of neurons in each. Described above.
- ``grid_scores.txt``: The result of the grid search over the various model parameters described above. 
- ``conf_matrix.pdf``: A heatmap of the confusion matrix for the final result, showing true and false positives and negatives. From this plot we can see that the model does fairly well, with a rather small number (some percent) of false positives and false negatives. 


Conclusions
-----------

For the best parameter choice, the neural network is rather good at determining whether a KOI is an exoplanet candidate or a false candidate. The precision and recall scores are both around 0.97, meaning that about 97% of the predicted candidates in the validation set are indeed candidates (precision), and that about 97% of the total number of candidates in the validation set are predicted as candidates (recall). This is likely not good enough for scientific purposes though. 

A somewhat serious issue relates to data contamination. The predictor variable most correlated with the target is ``koi_score``, which is a score of the confidence in the classification of the KOI. That is, this variable is not independent of the target, and should actually be excluded from the predictors. The result of including it is an overly optimistic result for the classification. An updated analysis should take this into account. 