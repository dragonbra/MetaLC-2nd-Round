import os
from sys import argv, path
root_dir = os.path.abspath(os.curdir)
path.append(root_dir)
import json
import numpy as np
import math
import csv
import random
from learning_curve import Learning_Curve


#=== Set RANDOM SEED
random.seed(208)

#=== Verbose mode
verbose = False

def vprint(mode, t):
    """
    Print to stdout, only if in verbose mode.

    Parameters
    ----------
    mode : bool
        True if the verbose mode is on, False otherwise.

    Examples
    --------
    >>> vprint(True, "hello world")
    hello world

    >>> vprint(False, "hello world")

    """

    if(mode):
        print(str(t))

class Meta_Learning_Environment():
    """
    A meta-learning environment which provides access to learning curve and meta-feature data.
    """

    def __init__(self, train_data_dir, validation_data_dir, test_data_dir, meta_features_dir, algorithms_meta_features_dir, output_dir):
        """
        Initialize the meta-learning environment

        Parameters
        ----------
        train_data_dir : str
            Path to learning curve data on the training set
        validation_data_dir : str
            Path to learning curve data on the validation set
        test_data_dir : str
            Path to learning curve data on the test set
        meta_features_dir : str
            Path to meta features of datasets
        algorithms_meta_features_dir : str
            Path to algorithms_meta_features of algorithms
        output_dir : str
            Path to output directory

        """
        self.output_dir = output_dir
        self.train_data_dir = train_data_dir
        self.validation_data_dir = validation_data_dir
        self.test_data_dir = test_data_dir
        self.meta_features_dir = meta_features_dir
        self.algorithms_meta_features_dir = algorithms_meta_features_dir
        self.num_dataset = 1 # the agent works on one dataset at a time
        self.done = False

        #=== List of dataset names
        self.list_datasets = os.listdir(self.test_data_dir)
        if '.DS_Store' in self.list_datasets: # remove junk files
            self.list_datasets.remove('.DS_Store')
        self.list_datasets.sort(key=int)

        #=== List of algorithms
        self.list_algorithms = os.listdir(os.path.join(self.test_data_dir, self.list_datasets[0]))
        if '.DS_Store' in self.list_algorithms: # remove junk files
            self.list_algorithms.remove('.DS_Store')
        self.list_algorithms.sort(key=int)

        self.num_algo = len(self.list_algorithms) # Number of algorithms

        #=== Load data from provided paths
        self.load_all_data()

    def load_all_data(self):
        """
        Load all data
        """
        self.train_learning_curves = {}
        self.validation_learning_curves = {}
        self.test_learning_curves = {}
        self.meta_features = {}
        self.algorithms_meta_features = {}

        #=== Load DATASET META-FEATURES
        vprint(verbose, "[+]Start loading META-FEATURES of datasets")
        # Iterate through all datasets
        for d in os.listdir(self.meta_features_dir):
            if '.DS_Store' not in d:
                dataset_name = d.split('.')[0].split('_')[0]
                dict_temp = {}
                with open(os.path.join(self.meta_features_dir, d), 'r') as f:
                    for line in f:
                        key, value = line.split('=')
                        key, value = key.replace(' ','').replace('\n', ''), value.replace(' ','').replace('\n', '').replace('\'','') #remove whitespaces and special symbols
                        dict_temp[key] = value
                self.meta_features[dataset_name] = dict_temp
        vprint(verbose, "[+]Finished loading META-FEATURES of datasets")

        #=== Load ALGORITHM HYPERPARAMETERS
        vprint(verbose, "[+]Start loading HYPERPARAMETERS of algorithms")
        # Iterate through all datasets
        for d in os.listdir(self.algorithms_meta_features_dir):
            if '.DS_Store' not in d:
                algorithm_name = d.split('.')[0]
                dict_temp = {}
                with open(os.path.join(self.algorithms_meta_features_dir, d), 'r') as f:
                    for line in f:
                        key, value = line.split('=')
                        key, value = key.replace(' ','').replace('\n', ''), value.replace(' ','').replace('\n', '').replace('\'','') #remove whitespaces and special symbols
                        dict_temp[key] = value
                self.algorithms_meta_features[algorithm_name] = dict_temp
        vprint(verbose, "[+]Finished loading HYPERPARAMETERS of algorithms")

        #=== Load TRAIN LEARNING CURVES
        if self.train_data_dir!=None:
            vprint(verbose, "[+]Start loading TRAIN learning curves")
            # Iterate through all datasets
            for dataset_name in self.list_datasets:
                dict_temp = {}
                for algo_name in self.list_algorithms:
                    path_to_algo = os.path.join(self.train_data_dir, dataset_name, algo_name)
                    dict_temp[algo_name] = Learning_Curve(os.path.join(path_to_algo + "/scores.txt"), float(self.meta_features[dataset_name]['time_budget']))
                self.train_learning_curves[dataset_name] = dict_temp
            vprint(verbose, "[+]Finished loading TRAIN learning curves")

        #=== Load VALIDATION LEARNING CURVES
        if self.validation_data_dir!=None:
            vprint(verbose, "[+]Start loading VALIDATION learning curves")
            # Iterate through all datasets
            for dataset_name in self.list_datasets:
                dict_temp = {}
                for algo_name in self.list_algorithms:
                    path_to_algo = os.path.join(self.validation_data_dir, dataset_name, algo_name)
                    dict_temp[algo_name] = Learning_Curve(os.path.join(path_to_algo + "/scores.txt"), float(self.meta_features[dataset_name]['time_budget']))
                self.validation_learning_curves[dataset_name] = dict_temp
            vprint(verbose, "[+]Finished loading VALIDATION learning curves")

        #=== Load TEST LEARNING CURVES
        vprint(verbose, "[+]Start loading TEST learning curves")
        # Iterate through all datasets
        for dataset_name in self.list_datasets:
            dict_temp = {}
            for algo_name in self.list_algorithms:
                path_to_algo = os.path.join(self.test_data_dir, dataset_name, algo_name)
                dict_temp[algo_name] = Learning_Curve(os.path.join(path_to_algo + "/scores.txt"), float(self.meta_features[dataset_name]['time_budget']))
            self.test_learning_curves[dataset_name] = dict_temp
        vprint(verbose, "[+]Finished loading TEST learning curves")

    def reset(self, dataset_name):
        """
        Reset the environment for a new task

        Parameters
        ----------
        dataset_name : str
            Name of the dataset at hand

        Returns
        ----------

        dataset_meta_features : dict of {str : dict of {str : str}}
            The meta-features of the dataset at hand, including:
                usage = 'AutoML challenge 2014'
                name = name of the dataset
                task = 'binary.classification', 'multiclass.classification', 'multilabel.classification', 'regression'
                target_type = 'Binary', 'Categorical', 'Numerical'
                feat_type = 'Binary', 'Categorical', 'Numerical', 'Mixed'
                metric = 'bac_metric', 'auc_metric', 'f1_metric', 'pac_metric', 'a_metric', 'r2_metric'
                time_budget = total time budget for running algorithms on the dataset
                feat_num = number of features
                target_num = number of columns of target file (one, except for multi-label problems)
                label_num = number of labels (number of unique values of the targets)
                train_num = number of training examples
                valid_num = number of validation examples
                test_num = number of test examples
                has_categorical = whether there are categorical variable (yes=1, no=0)
                has_missing = whether there are missing values (yes=1, no=0)
                is_sparse = whether this is a sparse dataset (yes=1, no=0)

        algorithms_meta_features : dict of {str : dict of {str : str}}
            The meta_features of each algorithm, for example:
                meta_feature_0 = 1
                meta_feature_1 = 0.0001

        Examples
        ----------
        >>> dataset_meta_features, algorithms_meta_features = env.reset("dataset01")
        >>> dataset_meta_features
        {'usage': 'AutoML challenge 2014', 'name': 'dataset01', 'task': 'regression',
        'target_type': 'Binary', 'feat_type': 'Binary', 'metric': 'f1_metric',
        'time_budget': '600', 'feat_num': '9', 'target_num': '6', 'label_num': '10',
        'train_num': '17', 'valid_num': '87', 'test_num': '72', 'has_categorical': '1',
        'has_missing': '0', 'is_sparse': '1'}
        >>> algorithms_meta_features
        {'0': {'meta_feature_0': '0', 'meta_feature_1': '0', meta_feature_2 : '0.000001'},
         '1': {'meta_feature_0': '1', 'meta_feature_1': '1', meta_feature_2 : '0.0001'},
         ...
         '39': {'meta_feature_0': '2', 'meta_feature_1': '2', meta_feature_2 : '0.01'},
         }
        """
        self.dataset_name = dataset_name
        dataset_meta_features = self.meta_features[dataset_name]
        self.total_time_budget = float(dataset_meta_features['time_budget'])
        self.remaining_time_budget = self.total_time_budget
        self.done = False

        #Write the header of the output file
        with open(self.output_dir + '/' + self.dataset_name + '.csv', 'a') as f:
            writer = csv.writer(f) # create the csv writer
            writer.writerow(('A', 'p', 't', 'R_train_A_p', 'R_validation_A_p'))

        return dataset_meta_features, self.algorithms_meta_features

    def reveal(self, action):
        """
        Execute an action and reveal new information on the learning curves

        Parameters
        ----------
        action : tuple of (int, float)
            The suggested action consisting of 2 things:
                (2) A: index of the algorithm to be trained and tested
                (3) p: decimal fraction of training data used, with value of p in [0.1, 0.2, 0.3, ..., 1.0]
        Returns
        ----------
        observation : tuple of (int, float, float, float, float)
            An observation containing: (A, p, t, R_train_A_p, R_validation_A_p)
                1) A: index of the algorithm provided in the previous action,
                2) p: decimal fraction of training data used, with value of p in [0.1, 0.2, 0.3, ..., 1.0]
                3) t: amount of time it took to train A with training data size of p,
                      and make predictions on the training/validation/test sets.
                4) R_train_A_p: performance score on the training set
                5) R_validation_A_p: performance score on the validation set

        done : bool
            True if the time budget is exhausted, False otherwise

        Examples
        ----------
        >>> observation, done = env.reveal((9, 0.5))
        >>> observation
            (9, 0.5, 151.73, 0.9, 0.6)
        >>> done
            True
        """

        A, p = action
        p = round(p, 1) # to avoid round-off errors

        #=== Perform the action to get the performance scores and time spent
        R_train_A_p, t = self.train_learning_curves[self.dataset_name][str(A)].get_performance_score(p)
        R_validation_A_p, _ = self.validation_learning_curves[self.dataset_name][str(A)].get_performance_score(p)

        #=== Check exceeding the given time budget
        if t > self.remaining_time_budget:
            R_train_A_p = 'None' # not enough of time to reveal the score
            R_validation_A_p = 'None' # not enough of time to reveal the score
            t = self.remaining_time_budget # cannot exceed the given time budget

        #=== Observation to be sent to the agent
        observation = (A, p, t, R_train_A_p, R_validation_A_p)

        #=== Write observation to the ouput directory
        with open(self.output_dir + '/' + self.dataset_name + '.csv', 'a') as f:
            writer = csv.writer(f) # create the csv writer
            writer.writerow(observation) # write a row to the csv file

        #=== Check done
        if t>=self.remaining_time_budget:
            self.done = True

        #=== Update the remaining time budget
        self.remaining_time_budget = round(self.remaining_time_budget - t, 2)

        return observation, self.done
