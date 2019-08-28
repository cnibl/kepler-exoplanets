# kepler-exoplanets
Project on exoplanet data from the Kepler telescope. The analysis is contained in the ``src/neural_network.py`` script. When executed, this script trains a neural network on a subset of the Kepler data and validates with the remaining data, with the goal of predicting whether an observed object in the data is an exoplanet candidate or not. 


## Documentation
There is documentation on the dataset and the analysis in the ``docs`` folder. To read this, first **build** the documentation by typing ``make html`` in the ``docs`` directory, then the resulting html documentation can be accessed by typing ``open _build/html/index.html``. Alternatively, a pdf file can be built (using ``pdflatex``) by typing instead ``make latexpdf`` in the ``docs`` directory. The resulting pdf can then be found in the ``_build/latex`` directory.