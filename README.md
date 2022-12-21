# CPPL Framework

This repository contains the Contextual Preselection with Plackett-Luce (CPPL) for realtime algorithm configuration framework. 
The framework currently runs the CaDiCaL, Glucose and CPLEX solver on the problem instances of SAT and MIP problems respectively.

The code implements the CPPL algorithm given by El Mesaoudi-Paul, A., Weiß, D., Bengs, V., Hüllermeier, E., Tierney, K. (2020). Pool-Based Realtime Algorithm Configuration: A Preselection Bandit Approach. In: Kotsireas, I., Pardalos, P. (eds) Learning and Intelligent Optimization. LION 2020. Lecture Notes in Computer Science(), vol 12096. Springer, Cham. https://doi.org/10.1007/978-3-030-53552-0_22 

To run the framework, you need the instance files which are to be solved by the solvers.
The SAT solvers (CaDiCaL and Glucose) solve files with `.cnf` formats and MIP solver (CPLEX) solve `.mip` format instance files.
To run the framework, you need to run the `CPPL_test_run.py` file in a **`Linux`** operating system.
Since the solvers are designed to work in a Linux exnvironment, running them on a Windows or Mac OS will result in an error.
Before running the framework, make sure you have installed `Python 3` in your system by downloading the latest version from [here](https://www.python.org/downloads/).
You can create a virtual environment to install the libraries used in this repository from `requirements.txt` by following the instructions from [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).
You can also install Python version 3 in an anaconda environment by following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
You can create an anaconda environment as well by following the instructions [here](https://datumorphism.leima.is/til/programming/python/python-anaconda-install-requirements/).

After installing Python version 3 and all the packages from the `requirements.txt`, open a Linux terminal in the location where the repository is cloned and run the following command
```bash
    python3 CPPL_test_run.py --help
```
This will give a list of all the arguments which is used as an input by the framework.
If you know which arguments you need, well good for you ^_~. Otherwise if you are fully lost, do not worry, here is an example of a run time command to run the framework:
```bash
python3 CPPL_test_run.py -to 300 -d <problem_instance_directory> -p pws -s <solver_name>
```
where `<problem_instance_directory>` is the directory name of the directory containing the problem instances and `<solver_name>` the name of the solver to be configured. 
You also need a `.csv` containing instance features of the problem instances to be solved. 
If you want to use a solver which is different from the ones already in the `solvers` directory, you will also need to adjust the `tournament.py` in the `tournament_classes`` directory.


The framework uses problem instance features which needs to be saved in `Instance_Features` folder in `.csv` format.
If you have given the `-d` argument as: 
```
python3 CPPL_test_run.py -d problem_instances
```
Then name of the `.csv` file should be of format `Features_problem_instances.csv` in the `Instance_Features` folder.

After solving all the problem instances, the framework will store the execution timings of each problem instance in the file name given in the argument `-tfn` or `--times_file_name`.
The file `Pool_<solver_name>.json` contains the parameters of the solver used as the contenders and the file `Winners_<solver_name>.txt` contains the winner configuration of each problem instance.