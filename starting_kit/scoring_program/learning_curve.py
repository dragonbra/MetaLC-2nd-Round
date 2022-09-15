import numpy as np
import traceback
import json
import random

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

class Learning_Curve():
    """
    A learning curve of an algorithm on a dataset
    """
    def __init__(self, file_path, time_budget):
        """
        Initialize the learning curve

        Parameters
        ----------
        file_path : str
            Path to the file containing data of the learning_curve

        time_budget : float
            Given time budget for the dataset at hand.

        """
        self.file_path = file_path
        self.time_budget = time_budget
        self.training_data_sizes, self.times, self.scores= self.load_data()

    def load_data(self):
        """
        Load training data sizes, times, and scores from the given path to build a learning curve

        Parameters
        ----------
        file_path : str
            Path to the file containing data of the learning_curve

        Returns
        ----------
        training_data_sizes : list of float in [0.1, 0.2, ..., 1.0]
            Training data size corresponding to each point on a learning curve

        times : list of float
            The amount of time spent for training the algorithm on a certain traning data size

        scores : list of str
            Performance score of the algorithm trained on a certain traning data size
            on the training set, validation set, or test set

        Examples
        ----------
        >>> lc.load_data()
        training_data_sizes: [0.3, 0.5, 0.8]
        times: [97.0, 199.0, 298.0]
        scores: [0.41, 0.36, 0.35]
        """

        training_data_sizes, times, scores = [], [], []

        # Parse data
        try:
            with open(self.file_path, "r") as data:
                lines = data.readlines()
                dictionary = {line.split(":")[0]:line.split(":")[1] for line in lines}
                training_data_sizes = np.around(json.loads(dictionary['training_data_sizes']), decimals=2)
                times = np.around(json.loads(dictionary['times']), decimals=2)
                scores = np.around(json.loads(dictionary['scores']), decimals=2)

        # If the data is missing, set time = 0 and score = 0 as default
        except FileNotFoundError:
            vprint(verbose, "Learning curve does not exist because the algorithm failed to make the first point! (NOT an error)")
            # print(traceback.format_exc())

        return training_data_sizes, times, scores

    def get_performance_score(self, p):
        """
        Return the algorithm's performance on the train/validation/test set,
        after being trained with training_data_size = p

        Parameters
        ----------
        p : float
            decimal fraction of training data used, with value of p in [0.1, 0.2, 0.3, ..., 1.0]

        Returns
        ----------
        score : float
            Performance score achieved
        t : float
            Amount of time it took to train and make predictions to obtain the performance score

        Examples
        ----------
        >>> score, time = lc.get_performance_score(0.2)
        score = 0.55
        t =  151.73
        """
        if p not in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            vprint(verbose, "Invalid value of p!")
            t = 'None'
            score = 'None'
        else:
            if p in self.training_data_sizes:
                index = self.training_data_sizes.tolist().index(p)
                score, t = self.scores[index], self.times[index]
            else:
                score = 'None'
                t = self.time_budget

        return score, t
