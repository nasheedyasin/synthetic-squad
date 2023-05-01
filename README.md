# synthetic-squad
The NLP group project for CSE 635

To create an env run

```
cd ~/location_of_repo
conda env create -f environment.yml
```

Please note that if you are using a CPU based environment: comment out the indicated lines in the `environment.yml` file before running.
After it is done installing and creating the environment, [visit pytorch's website](https://pytorch.org/get-started/previous-versions/) to install the latest version of pytorch <2.0.0 (i.e. pre 2.0.0)


# Data Information
 ### Files
- no_punc_data_all.csv - Contains data without punctuations
- punc_data_all.csv - Contains Raw Data

 ### Train-Test Data
 - punc_data_tr.csv - Training data (BERT models' data now "new_punc_data_tr.csv" as of 4-29)
 - punc_data_eval.csv - Evaluation data (BERT models' data now "new_punc_data_eval.csv" as of 4-29)
    - There are 10 eval files for Task 2 (HVM): punc_data_eval_1.csv, punc_data_eval_2.csv ..... to punc_data_eval_10.csv

 ### Columns
- title - Title of the Text
- generation - Data Generated by algorithm
- alg- The Algorithm
- src- Data Source (Either from aa_paper or reddit)
