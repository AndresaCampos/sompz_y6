# sompz_y6

## Setting up the environment

You need to set up a Python 3 sompz enviroment and a Jupyter interpreter with the packages listed in *requierments.txt* and the twopoint, which is not currently avaible in conda. In a terminal inside the sompz_y6 folder, run: 

$ module load python
$ conda config --append channels conda-forge \
$ conda create --name sompz python=3.10.9 ipykernel --file requirements.txt \
$ conda activate sompz \
$ pip install twopoint \
$ python -m ipykernel install --user --name sompz --display-name SOMPZ \

If you running at nersc, you need to also need the following 2 commands to make your mpi4py work properly

$ module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu \
$ MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py\

You only need to run the commands above once. Now, when you want to use the sompz enviroment just run

$ conda activate sompz

or select the SOMPZ kernel in your Jupyter notebook. 



## Train Phase

The *train_* files are reference files to train the deep and wide SOMs

- Train deep som
    - Use deep catalog
- Train wide some
    - Use subsample of the wide catalog 


## Assign Phase

The *assign_* files are reference files to assign the deep, balrog and wide catalogs to their respective SOMs

- Assign catalogs to deep som: deep and balrog
- Assign catalogs to wide som: wide and balrog


## Redshifts Estimation

The *compute_redshifts.ipynb* notebook contains all the steps to estimate the N(z)s for the wide catalog

- Read deep, wide and balrog catalogs
- Create dataframes for each catalog and add cell assignment information
- Compute pcchat (transfer matrix) using balrog assignments in deep and wide
- Compute pzc and pzchat
- Compute N(z)s
- Save everything in h5 file (input for the 2pt pipeline) 

## Papers to Cite

Please cite the following papers if you use this code in your research:

1. [A. Campos et al. (DES Collaboration) - Enhancing weak lensing redshift distribution characterization by optimizing the Dark Energy Survey Self-Organizing Map Photo-z method](https://arxiv.org/pdf/2408.00922)
2. [C. Sánchez,  M. Raveri,  A. Alarcon,  G. Bernstein - Propagating sample variance uncertainties in redshift calibration: simulations, theory, and application to the COSMOS2015 data](https://doi.org/10.1093/mnras/staa2542)
3. [R. Buchs, et al. - Phenotypic redshifts with self-organizing maps: A novel method to characterize redshift distributions of source galaxies for weak lensing](https://doi.org/10.1093/mnras/stz2162)
4. [J. Myles,  A. Alarcon, et al. (DES Collaboration) - Dark Energy Survey Year 3 results: redshift calibration of the weak lensing source galaxies](https://doi.org/10.1093/mnras/stab1515)
