Dataset
=======

The data comes from NASA's Kepler telescope. Kepler looks for exoplanets, i.e. planets that have similar properties as the Earth and hence could potentially host life. Kepler attempts to see planets orbiting stars by looking at for example the loss in stellar flux due to the planet passing in front of it. Each row contains data on a KOI (Kepler Object of Interest).

Data columns of interest
------------------------
We do not use all the data columns in this analysis since many of them are not so relevant for the analysis we want to make. For example, the parameters describing the host star's properties and position on the sky are not very useful for determining whether a planetary object is an exoplanet or not. Here we provide explanations of the columns that we use and briefly comment on the others. The full explanation of all columns can be found here_.

.. _here: https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html#transit_prop.

In the analysis, we will attempt to predict the value of ``koi_pdisposition`` given the eight numerical variables explained below. These eight are the variables that are most correlated with ``koi_pdisposition``. 

Categorical variables
^^^^^^^^^^^^^^^^^^^^^
- ``koi_pdisposition``: The disposition of of Kepler towards this KOI. Takes values ``CANDIDATE``, which is an object that has passed all tests and is considered an exoplanet candidate by Kepler, and ``FALSE POSITIVE``, which is a KOI that looks like an exoplanet in some way(s) but has failed at least one of the false positive tests.
- ``koi_disposition``: The disposition of of Kepler, combined with information from an exoplanet database (the Exoplanet Archive). Takes on values ``CANDIDATE``, ``FALSE POSITIVE`` and ``CONFIRMED`` where the first two are the same as the ``koi_pdisposition`` values of the KOI and the ``CONFIRMED`` value is assigned to those objects who are also in the Exoplanet Archive.

Numerical variables
^^^^^^^^^^^^^^^^^^^
- ``koi_fpflag_nt``, koi_fpflag_ss``, koi_fpflag_co``, koi_fpflag_ec``: Discrete variable, denoting whether KOI has passed a test corresponding to the particular ``fpflag`` variable. Takes on values 0 and 1.
- ``koi_depth``: Continuous variable. Gives the fraction of stellar flux lost at the minimum of the planetary transit around the star.
- ``koi_score``: Continuous variable between 0 and 1 that indicates the confidence in the assignment of the KOI. Higher value means higher confidence in a ``CANDIDATE`` designated KOI, but a lower confidence in a ``FALSE POSITIVE`` KOI. 
- ``koi_teq``: Approximation for the planet's temperature, under some assumptions about thermodynamic and atmospheric properties.
- ``koi_model_snr``: The transit signal-to-noise. Depth in transit normalised by the mean uncertainty in the flux during transit.
- ``koi_tce_plnt_num``: Planet number. Unclear if needed, also not very correlated with target.

