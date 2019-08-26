# Trains a neural network to predict whether an observation in the dataset is 
# an exoplanet candidate or not. Uses data from the Kepler telescope.

# Imports for data handling, plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from pathlib import Path

# Imports for machine learning
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Imputer


# Read input data into a DataFrame and  and features to make prediction
this_script_path = Path(__file__).parent
file_path = (this_script_path / "../inputdata/keplerdata.csv").resolve()
kepler_data = pd.read_csv(file_path)

# The following columns are the ones correlating mostly with the prediction 
# target (see data-plotting.ipynb)
predictors = ["koi_score", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", 
      "koi_fpflag_ec", "koi_depth", "koi_teq", "koi_model_snr", "koi_tce_plnt_num"] 
X = kepler_data[predictors]

# Separate out prediction target (what we want to predict) and make numerical
target_column = ["koi_pdisposition"] 
encoder = LabelEncoder()
y = encoder.fit_transform(kepler_data[target_column].values.ravel())

# Separate into training and test data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
      test_size=0.3, random_state=0)

      
if __name__=="__main__":
   
   # Number of hidden layers and their sizes (same size for all) determined by
   # trial and error on the combinations in layer_tests.py
   no_hidden_layers = 5
   size_per_layer = 8
   layer_sizes = tuple(size_per_layer for i in range(no_hidden_layers))
   
   # Bundle preprocessing and modeling code in a pipeline
   imputer = Imputer(strategy="mean")
   model = MLPClassifier(hidden_layer_sizes=layer_sizes, random_state=0)
   nn_pipeline = Pipeline(steps=[("imputer", imputer),("model", model)])
   
   # Do cross validation with grid search to optimise parameters like alpha 
   # and learning rate
   param_grid = [{"model__alpha": [1e-5, 1e-4, 1e-3], 
         "model__activation": ["logistic", "relu"], 
         "model__learning_rate_init": [1e-5, 1e-4, 1e-3]}]
   
   gs = GridSearchCV(nn_pipeline, param_grid, cv=5)
   gs.fit(X_train, y_train)
   
   print("Best parameters found on training set:")
   print()
   print(gs.best_params_)
   print()
   print("Grid scores on validation set:")
   print()
   means = gs.cv_results_["mean_test_score"]
   stds = gs.cv_results_["std_test_score"]
   for mean, std, params in zip(means, stds, gs.cv_results_["params"]):
       print("%0.3f (+/-%0.03f) for %r"
             % (mean, std * 2, params))
   print()
   print("Detailed classification report:")
   print()
   print("The model is trained on the full training set.")
   print("The scores are computed on the full validation set.")
   print()
   preds = gs.predict(X_valid)
   print(classification_report(y_valid, preds))
   print()
       
       
   # Train model
   
   # Test on 