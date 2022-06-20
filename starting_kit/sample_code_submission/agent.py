import random
import numpy as np


class Agent():
    def __init__(self, number_of_algorithms):
        """
        Initialize the agent

        Parameters
        ----------
        number_of_algorithms : int
            The number of algorithms

        """
        self.number_of_algorithms = number_of_algorithms

        self.best_index = 0
        self.best_algorithms = []
        self.best_times = [0 for _ in range(number_of_algorithms)]

        self.ratio_list = []
        self.ratio_sum = [0 for _ in range(number_of_algorithms)]
        self.datasets_meta_features = None
        self.algorithms_meta_features = None
        pass

    def reset(self, dataset_meta_features, algorithms_meta_features):
        """
        Reset the agents' memory for a new dataset

        Parameters
        ----------
        dataset_meta_features : dict of {str : str}
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

        algorithms_meta_features : dict of dict of {str : str}
            The meta_features of each algorithm, for example:
                meta_feature_0 = 0, 1, 2, …
                meta_feature_1 = 0, 1, 2, …
                meta_feature_2 = 0.000001, 0.0001 …


        Examples
        ----------
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
        rank_base = 1
        keywords = ['task', 'target_type', 'feat_type', 'metric','time_budget', 'has_categorical', 'has_missing', 'is_sparse']
        keywords_weight = [2, 1, 1, 1, 0.2, 0.5, 0.5, 0.5]
        self.ratio_sum = [0 for _ in range(self.number_of_algorithms)]
        for dataset_idx, dataset_name in enumerate(self.datasets_meta_features.keys()):
            dataset_rank = rank_base
            for idx, feat in enumerate(keywords):
                if self.datasets_meta_features[dataset_name][feat] == dataset_meta_features[feat]:
                    dataset_rank += keywords_weight[idx]
                    pass
            for alg_idx in range(self.number_of_algorithms):
                self.ratio_sum[alg_idx] += dataset_rank * self.ratio_list[dataset_idx][alg_idx]
        
        self.best_algorithms = sorted(range(len(self.ratio_sum)), key=lambda k: self.ratio_sum[k], reverse=True)
        # print("DEBUG: Time_budget ", dataset_meta_features['time_budget'])
        # print("DEBUG: ", self.best_algorithms)

    def meta_train(self, datasets_meta_features, algorithms_meta_features, train_learning_curves, validation_learning_curves, test_learning_curves):
        """
        Start meta-training the agent

        Parameters
        ----------
        datasets_meta_features : dict of {str : dict of {str : str}}
            Meta-features of meta-training datasets

        algorithms_meta_features : dict of {str : dict of {str : str}}
            The meta_features of all algorithms

        train_learning_curves : dict of {str : dict of {str : Learning_Curve}}
            TRAINING learning curves of meta-training datasets

        validation_learning_curves : dict of {str : dict of {str : Learning_Curve}}
            VALIDATION learning curves of meta-training datasets

        test_learning_curves : dict of {str : dict of {str : Learning_Curve}}
            TEST learning curves of meta-training datasets

        Examples:
        To access the meta-features of a specific dataset:
        >>> datasets_meta_features['dataset01']
        {'name':'dataset01', 'time_budget':'1200', ...}

        To access the validation learning curve of Algorithm 0 on the dataset 'dataset01' :

        >>> validation_learning_curves['dataset01']['0']
        <learning_curve.Learning_Curve object at 0x9kwq10eb49a0>

        >>> validation_learning_curves['dataset01']['0'].times
        [196, 319, 334, 374, 409]

        >>> validation_learning_curves['dataset01']['0'].scores
        [0.6465293662860659, 0.6465293748988077, 0.6465293748988145, 0.6465293748988159, 0.6465293748988159]
        """
        self.datasets_meta_features = datasets_meta_features
        self.algorithms_meta_features = algorithms_meta_features
        self.ratio_list = [[0 for __ in range(self.number_of_algorithms)] for _ in range(len(datasets_meta_features))]

        for dataset_idx, dataset_name in enumerate(test_learning_curves.keys()):
            dataset = test_learning_curves[dataset_name]
            dataset_train = train_learning_curves[dataset_name]
            dataset_validation = validation_learning_curves[dataset_name]
            
            max_ratio, best_algorithm = 0, 0
            for alg_idx, alg_name in enumerate(dataset.keys()):
                curve = dataset[alg_name]
                curve_train = dataset_train[alg_name]
                curve_validation = dataset_validation[alg_name]

                idx = 1
                if len(curve.scores) <= idx:
                    continue
                # select the second point to calculate
                ratio = curve.scores[idx] / curve.times[idx] + curve_train.scores[idx] / curve_train.times[idx] \
                    + curve_validation.scores[idx] / curve_validation.times[idx]
                self.ratio_list[dataset_idx][alg_idx] = ratio

                if ratio > max_ratio:
                    max_ratio, best_algorithm = ratio, int(alg_name)

            self.best_times[best_algorithm] += 1
            # normalization on ratio_sum
            for alg_idx in range(self.number_of_algorithms):
                self.ratio_list[dataset_idx][alg_idx] /= max_ratio
                # self.ratio_sum[alg_idx] += self.ratio_list[dataset_idx][alg_idx]
                pass

        # sort best algorithms based on its best performance times    
        self.best_algorithms = sorted(range(len(self.best_times)), key=lambda k: self.best_times[k], reverse=True)

    def suggest(self, observation):
        """
        Return a new suggestion based on the observation

        Parameters
        ----------
        observation : tuple of (int, float, float, float, float)
            An observation containing: (A, p, t, R_train_A_p, R_validation_A_p)
                1) A: index of the algorithm provided in the previous action,
                2) p: decimal fraction of training data used, with value of p in [0.1, 0.2, 0.3, ..., 1.0]
                3) t: amount of time it took to train A with training data size of p,
                      and make predictions on the training/validation/test sets.
                4) R_train_A_p: performance score on the training set
                5) R_validation_A_p: performance score on the validation set

        Returns
        ----------
        action : tuple of (int, float)
            The suggested action consisting of 2 things:
                (2) A: index of the algorithm to be trained and tested
                (3) p: decimal fraction of training data used, with value of p in [0.1, 0.2, 0.3, ..., 1.0]

        Examples
        ----------
        >>> action = agent.suggest((9, 0.5, 151.73, 0.9, 0.6))
        >>> action
        (9, 0.9)
        """
        ### TO BE IMPLEMENTED ###
        if observation == None:
            self.best_index = 0
            # return (self.fastest_index, 0.1)

        # print("DEBUG:", self.dataset_meta_features)
        A = self.best_algorithms[self.best_index]
        p = random.choices([0.1, 0.2, 0.3, 0.4, 0.5], \
            weights=[20 - self.best_index * 4, 5 - self.best_index, self.best_index, self.best_index // 2, self.best_index // 3])[0]
        action = (A, p)
        self.best_index += 1
        if self.best_index == self.number_of_algorithms:
            self.best_index = 0
        return action
