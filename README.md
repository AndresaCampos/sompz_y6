# sompz_y6

## Setting up the environment

You need to set up a Python 3 sompz enviroment and a Jupyter interpreter with the packages listed in *requierments.txt* and the twopoint, which is not currently avaible in conda. In a terminal inside the sompz_y6 folder, run: 

$ conda config --append channels conda-forge \
$ conda create --name sompz ipykernel --file requirements.txt \
$ conda activate sompz \
$ pip install twopoint \
$ python -m ipykernel install --user --name sompz --display-name SOMPZ \


You only need to run this once. Now, when you want to use the sompz enviroment just run

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


