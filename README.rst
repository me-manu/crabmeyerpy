crabmeyerpy
===========

Python routines for the Meyer et al. (2010) Crab nebula constant B-field model in 1D and now in 3D. 
Also includes routines to calculate the emission in the Kennel & Coroniti (1984) MHD flow model solution. 

All models assume time independence and spherical symmetry. 

Prerequisites
-------------

Python 3.6 or higher and the following packages
    - numpy 
    - scipy
    - astropy
    - numba

If you want to use the fast routines for the 3D model calculations, you need to install numba 
and the fast interp package: https://github.com/dbstein/fast_interp

Installation
------------

Currtently, no installation through package managers is supported. Please clone / fork the repository 
and add the file path to your `PYTHONPATH` variable.

Getting Started
---------------

Please take a look at the example notebooks provided in the `notebooks/` folder.

Acknowledgements
----------------

This development of this package has received support from the European Research Council (ERC) under
the European Union’s Horizon 2020 research and innovation program Grant agreement No. 843800 (GammaRayCascades).

License
-------

This project is licensed under a 3-clause BSD style license - see the
``LICENSE`` file.
