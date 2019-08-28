# Analyses the model fit to the training data in neural_network.py and 
# outputs plots and results

# Import necessary variables like X_train, y_train etc from neural_network.py
from neural_network import *

from joblib import load

# Load the already fitted model 
nn_model = load("neural_network.joblib")

# Make predictions on vallidation data
preds = nn_model.predict(X_valid)

# Access the parameter values used
best_params = nn_model.best_params_ 

# Make a heatmap of the confusion matrix
fig, ax = plt.subplots()
conf = confusion_matrix(y_true=y_valid, y_pred=preds)
ax = sns.heatmap(conf, cmap="binary", annot=True, fmt="d", 
      xticklabels=encoder.classes_, yticklabels=encoder.classes_)
ax.set_title("alpha={0}, solver={1}, beta_1={2}, beta_2={3}, \nlearning_rate_init={4}, hidden_layer_sizes={5}".format(
      best_params["model__alpha"],
      best_params["model__solver"],
      best_params["model__beta_1"],
      best_params["model__beta_2"],
      best_params["model__learning_rate_init"],
      best_params["model__hidden_layer_sizes"]))
plot_file_name = "../results/conf_matrix.png"
this_script_path = Path(__file__).parent
plot_file_path = (this_script_path / plot_file_name).resolve()
plt.savefig(plot_file_path)

# Write detailed grid scores to output file
file_name = "../results/grid_scores.txt"
this_script_path = Path(__file__).parent
file_path = (this_script_path / file_name).resolve()
with open(file_path,"w") as f:
   f.write("Grid scores on validation set:\n")
   means = nn_model.cv_results_["mean_test_score"]
   stds = nn_model.cv_results_["std_test_score"]
   for mean, std, params in zip(means, stds, nn_model.cv_results_["params"]):
       f.write("%0.3f (+/-%0.03f) for %r\n"
             % (mean, std * 2, params))

# Print out scoring
print()
print("Detailed classification report:")
print()
print("The model is trained on the full training set.")
print("The scores are computed on the full validation set.")
print()
print(classification_report(y_valid, preds))
print()
   