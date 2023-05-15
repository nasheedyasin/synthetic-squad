# synthetic-squad
The NLP group project for CSE 635

To create an env run

```
cd ~/location_of_repo
conda env create -f environment.yml
```

Please note that if you are using a CPU based environment: comment out the indicated lines in the `environment.yml` file before running.
After it is done installing and creating the environment, [visit pytorch's website](https://pytorch.org/get-started/previous-versions/) to install the latest version of pytorch <2.0.0 (i.e. pre 2.0.0)

_Every experiment is run on colab notebook or on our personal computer. Tu get the same results run our colab notebooks and change the sources from where data is being read_

# Data Information
 ### Files
- liwc_pos_dep_tr.csv - Contains data with pos sequence, dependency tags sequence and liwc features for training
- liwc_pos_dep_eval.csv - Contains data with pos sequence, dependency tags sequence and liwc features for evaluation
- liwc_pos_dep_reddit.csv - Contains data with pos sequence, dependency tags sequence and liwc features for reddit
