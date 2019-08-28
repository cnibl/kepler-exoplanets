# Goes through combinations for number of layers and their size and prints
# cross validation scores to a file

# Import all necessary variables etc from neural_network.py
from neural_network import *

# The scoring method used
#print(sorted(sklearn.metrics.SCORERS.keys()))
cv_scoring = "f1"

# Write header to output file
file_name = "../results/layer_tests.txt"
this_script_path = Path(__file__).parent
file_path = (this_script_path / file_name).resolve()
with open(file_path,"w") as f:
   f.write("# Mean of 5-fold cross validation precision scores\n") 
   f.write("# no_hidden_layers, size_per_layer, {}\n".format(cv_scoring))
   
# Loop over layer variants
layer_variants = ((2, 4), (3, 4), (4, 4), (5, 4), (6, 4),  
                  (2, 8), (3, 8), (4, 8), (5, 8), (6, 8),  
                  (2, 16), (3, 16), (4, 16), (5, 16), (6, 16),
                  (2, 100), (3, 100), (4, 100))
for no_hidden_layers, size_per_layer in layer_variants:
   # Number of hidden layers and their sizes (same size for all)
   layer_sizes = tuple(size_per_layer for i in range(no_hidden_layers))
   
   # Bundle preprocessing and modeling code in a pipeline
   imputer = SimpleImputer(strategy="mean")
   model = MLPClassifier(hidden_layer_sizes=layer_sizes, random_state=0)
   nn_pipeline = Pipeline(steps=[("imputer", imputer),("model", model)])
   
   
   cv_scores = cross_val_score(nn_pipeline, X_train, y_train, cv=5, 
         scoring=cv_scoring)
   
   # Print results to file
   with open(file_path,"a") as f:
      f.write("%-1d, %2d, %5.3f\n" % 
            (no_hidden_layers, size_per_layer, cv_scores.mean()))