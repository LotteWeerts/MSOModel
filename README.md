# MSOModelModule

Welcome to the MSOmodelModule! This is the documentation of the model of the MSO as shown in Weerts (2016). This README file contains instructions on installation and usage and instructions on how to reproduce the graphs in chapter 3, 4 and the appendix. 

## Installation 

To use this package you need to install Python (https://www.python.org/downloads/). This automatically installs pip, a package management system used to install Python packages. 

We assume for all commands given in this README that you are in the folder where this file is located. 

The code in this repository depends on several packages, which you can install via the following command (first move to the folder where you stored this repository):

```
pip install -r requirements.txt
```

This should install all required packages, such as NumPy and the memory_profiler. We recommend using a virtual environment to install the packages (see http://docs.python-guide.org/en/latest/dev/virtualenvs/) 

## Documentation

You can find an HTML version of the documentation in the 'doc' folder. Just open the `index.html` in the doc folder and you will get an overview of all the documented functions and modules within the MSOModule.  

## Usage

### Reproduce figures in the report

All figures that were generated can be reproduced using the notebooks stored in the `src/notebooks` folder. These notebooks use the reduced data set in the `data` folder to generate the figures (although some figures in chapter 3 can be loaded automatically). You need Jupyter notebook to run these notebooks (see http://jupyter.readthedocs.io/en/latest/install.html for installation instructions). If you have Jupyter installed you can run the notebook server as follows:

```
jupyter notebook src/notebooks/
```

Now you can simply click on each of the chapters and click on "run all" to run the code that's displayed. This will generate the figures of that particular chapter. 

### Use the MSO model to perform your own tests

You can run the MSO model through the command line as follows:

```
python src/MSOModelModule/core.py --t 5000 --b 1000 --v 3.0 --i 500 --M 5 --d 0.005 --I '[0.0]' \
       --T 4.0 --memorySavingMode 2 --storeData
```

Here:
* `-t` is the total number of timesteps in milliseconds
* `-b` is the batchsize
* `v` is the threshold of the MSO neurons
* `i` indicates the number of input neurons
* `M` indicates the number of MSO neurons (we recommend to use a low number)
* `d` is the stepsize in milliseconds
* `I` is a string that contains a list with all the ITDs the simulation can choose from (e.g. '[0.0]' sets the ITD to a constant ITD of 0.0, '[-0.5, 0.0, 0.5]' switches between ITDs -0.5, 0.0 and 0.5 every time a new batch starts. To use the task-related ITD, enter 1 instead of a list
* `T` is the period of the input sound in ms
* `memorySavingMode` indicates which memory saving mode should be used (see appendix A in report). We recommend to always use setting 2
* `storeData` indicates whether or not you want the simulation to be stored, which is necessary if you want to perform tests on the model

The MSO neuron will first 'warm up' for 5 batches, that is, it will run without STDP rules being applied. 

Note that the large simulations as presented in the report can take a long time to run (e.g. up to two days for 50 neurons for 1000 seconds in simulation time) so we recommend testing for shorter time periods or a smaller amount of MSO neurons. 

### Use the test package to run tests on trained models 

```
python src/MSOModelModule/test.py --t 200000000 --ITDcorrelation '[-2.0, -1.0, 0.0, 1.0]' \
       --path <insert path>
```

Here
* `t` is the time of the simulation which you would like to test. This should be given in timesteps, and loads the weights in the file t - 1. So if you for example want to load the weights stored in `1999999_w.json.gz`, you enter `--t 2000000`. 
* `ITDcorrelation` indicates which ITDs  you would like to test the MSO model on for 10 s each. Each ITD gets its own output file in the test folder (apologies for the legacy name of this parameter) 
* `path` is the path of the folder where your MSO simulation is stored. This is generally of the form `day/time`. 
This will generate a test folder in the folder that stores a sparse matrix with all the spikes of all the MSO neurons in those 10 ms (note that this will be a big file). 

Please do not run this method on the models stored in the dataset, as the addition of new test files could disrupt the notebooks that generate the figures (see above) (you could however just copy one of the data folders and run the above command on the path of your copied folder).  

* Weerts, L. Modelling Unsupervised Spike-Timing-Dependent Plasticity for the Detection of Interaural Time Differences in the Medial Superior Olive (2016) 
