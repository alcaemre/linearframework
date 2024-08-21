# linearframework

A repo for calculating qunantities relevant to Jeremy Gunawardena's [Linear Framework](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0036321) using the [Chabotarev-Agaev recurrence](https://arxiv.org/abs/math/0508178) purely in python.
The purpose of the package is to calculate the transient quantities whose [graph-theoretic formulas](https://www.frontiersin.org/journals/cell-and-developmental-biology/articles/10.3389/fcell.2023.1233808/full) were developed by Chris (Kee-Myoung) Nam.

The best way to install this library is by the usual github method:
- make sure you have conda git installed and you are logged in to github.
- in terminal, cd into the folder where you would like to use this library.
- type into the command line `git clone https://github.com/alcaemre/linearframework`
- cd into the folder `linearframework` (the top level folder for the repo after cloning)
- create the conda environment by running `conda env create --name linearframework --file environment.yml` in your command line.
- run the command `pip install -e .` to locally install the package.

Note that this requires networkx-3.3, which is not the default install from conda right now, so you will likely need to run the command `pip install networkx --upgrade`.

A good way to check that the installation is complete is to run the command `pytest .` from within the repo with the environment `linearframework` active.
The output should read 

`==================================== test session starts =====================================`

`platform darwin -- Python 3.12.4, pytest-7.4.4, pluggy-1.0.0`

`rootdir: /your/path/here/linearframework`

`collected 29 items`

`tests/test_ca_recurrence.py ......                                                     [ 20%]`

`tests/test_gen_graphs.py ....                                                          [ 34%]`

`tests/test_generalized_aldous_schepp.py ...                                            [ 44%]`

`tests/test_graph_operations.py .........                                               [ 75%]`

`tests/test_linear_framework_results.py .......                                         [100%]`

`==================================== 29 passed in 24.87s =====================================`

Now you can read tutorial.ipynb to get going!
