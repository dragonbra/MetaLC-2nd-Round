import os
from sys import argv, path
from environment import Meta_Learning_Environment
import random
import os
from sklearn.model_selection import KFold
import time
import datetime
import shutil

import warnings
warnings.filterwarnings("ignore")

#=== Verbose mode
verbose = True

#=== Set RANDOM SEED
random.seed(208)

#=== Setup input/output directories
root_dir = os.getcwd()
default_input_dir = os.path.join(root_dir, "sample_data/")
default_output_dir = os.path.join(root_dir, "output/")
default_program_dir = os.path.join(root_dir, "ingestion_program/")
default_submission_dir = os.path.join(root_dir, "sample_code_submission/")

#=== List valid values of p
list_valid_values_of_p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
    return
    if(mode):
        print(str(t))

def clear_output_dir(output_dir):
    """
    Delete previous output files.

    Parameters
    ----------
    output_dir : str
        Path to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    #=== Delete all .DS_Store
    os.system("find . -name '.DS_Store' -type f -delete")

def meta_training(agent, D_tr):
    """
    Meta-train an agent on a set of datasets.

    Parameters
    ----------
    agent : Agent
        The agent before meta-training.
    D_tr : list of str
        List of dataset indices used for meta-training

    Returns
    -------
    agent : Agent
        The meta-trained agent.
    """

    vprint(verbose, "[+]Start META-TRAINING phase...")
    vprint(verbose, "Meta-training datasets = ")
    vprint(verbose, D_tr)

    #=== Gather all meta_features and learning curves on meta-traning datasets
    train_learning_curves = {}
    validation_learning_curves = {}
    test_learning_curves = {}
    datasets_meta_features = {}
    algorithms_meta_features = {}

    for d_tr in D_tr:
        dataset_name = list_datasets[d_tr]
        datasets_meta_features[dataset_name] = env.meta_features[dataset_name]
        train_learning_curves[dataset_name] = env.train_learning_curves[dataset_name]
        validation_learning_curves[dataset_name] = env.validation_learning_curves[dataset_name]
        test_learning_curves[dataset_name] = env.test_learning_curves[dataset_name]

    #=== Get algorithms_meta_features of algorithms
    algorithms_meta_features = env.algorithms_meta_features

    #=== Start meta-traning the agent
    vprint(verbose, datasets_meta_features)
    vprint(verbose, algorithms_meta_features)
    agent.meta_train(datasets_meta_features, algorithms_meta_features, train_learning_curves, validation_learning_curves, test_learning_curves)

    vprint(verbose, "[+]Finished META-TRAINING phase")

    return agent

def meta_testing(trained_agent, D_te):
    """
    Meta-test the trained agent on a set of datasets. The goal is to maximize
    the area under the agent's learning curve.

    Parameters
    ----------
    trained_agent : Agent
        The agent after meta-training.
    D_te : list of str
        List of dataset indices used for meta-testing
    """

    vprint(verbose, "\n[+]Start META-TESTING phase...")
    vprint(verbose, "meta-testing datasets = " + str(D_te))

    for d_te in D_te:
        dataset_name = list_datasets[d_te]
        meta_features = env.meta_features[dataset_name]

        #=== Reset both the environment and the trained_agent for a new task
        dataset_meta_features, algorithms_meta_features = env.reset(dataset_name=dataset_name)
        trained_agent.reset(dataset_meta_features, algorithms_meta_features)
        vprint(verbose, "\n#===================== Start META-TESTING on dataset: " + dataset_name + " =====================#")
        vprint(verbose, "\n#---Dataset meta-features = " + str(dataset_meta_features))
        vprint(verbose, "\n#---Algorithms meta-features = " + str(algorithms_meta_features))

        #=== Start meta-testing on a dataset step by step until the given total_time_budget is exhausted (done=True)
        done = False
        observation = None
        while not done:
            # Get the agent's suggestion
            action = trained_agent.suggest(observation)

            # Check validity of the action
            A = action[0]
            p = round(action[1], 1) # to avoid round-off errors

            if (str(A) not in list_algorithms) or (p not in list_valid_values_of_p):
                raise Exception("Invalid action: " + "action=" + str(action) +
                 "\nA must be in " + str(list(map(int, list_algorithms))) + "\np must be in " +
                 str(list_valid_values_of_p))

            # Execute the action and observe
            observation, done = env.reveal(action)

            # Check validity of the observation
            t = observation[2]
            if t<0:
                raise Exception("t cannot be less than 0, please report this to the organizers!")

            vprint(verbose, "------------------")
            vprint(verbose, "A = " + str(A))
            vprint(verbose, "p = " + str(p))
            vprint(verbose, "observation = " + str(observation))
            vprint(verbose, "remaining_time_budget = " + str(env.remaining_time_budget))
            vprint(verbose, "done=" + str(done))

    vprint(verbose, "[+]Finished META-TESTING phase")

#################################################
################# MAIN FUNCTION #################
#################################################

if __name__ == "__main__":
    #=== Get input and output directories
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
        program_dir= default_program_dir
        submission_dir= default_submission_dir
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])
        program_dir = os.path.abspath(argv[3])
        submission_dir = os.path.abspath(argv[4])

    train_data_dir = os.path.join(input_dir, 'train')
    validation_data_dir = os.path.join(input_dir, 'validation')
    test_data_dir = os.path.join(input_dir, 'test')
    meta_features_dir = os.path.join(input_dir, 'dataset_meta_features')
    algorithms_meta_features_dir = os.path.join(input_dir, 'algorithms_meta_features')

    vprint(verbose, "Using input_dir: " + input_dir)
    vprint(verbose, "Using output_dir: " + output_dir)
    vprint(verbose, "Using program_dir: " + program_dir)
    vprint(verbose, "Using submission_dir: " + submission_dir)
    vprint(verbose, "Using train_data_dir: " + train_data_dir)
    vprint(verbose, "Using validation_data_dir: " + validation_data_dir)
    vprint(verbose, "Using test_data_dir: " + test_data_dir)
    vprint(verbose, "Using meta_features_dir: " + meta_features_dir)
    vprint(verbose, "Using algorithms_meta_features_dir: " + algorithms_meta_features_dir)

    #=== List of dataset names
    list_datasets = os.listdir(validation_data_dir)
    if '.DS_Store' in list_datasets: #remove junk files
        list_datasets.remove('.DS_Store')
    list_datasets.sort(key=int)

    #=== List of algorithms
    list_algorithms = os.listdir(os.path.join(validation_data_dir, list_datasets[0]))
    if '.DS_Store' in list_algorithms: #remove junk files
        list_algorithms.remove('.DS_Store')
    list_algorithms.sort(key=int)

    #=== Import the agent submitted by the participant
    path.append(submission_dir)
    from agent import Agent

    #=== Clear old output
    clear_output_dir(output_dir)

    #=== Init K-folds cross-validation
    kf = KFold(n_splits=5, shuffle=False)

    ################## MAIN LOOP ##################
    #=== Init a meta-learning environment
    env = Meta_Learning_Environment(train_data_dir, validation_data_dir, test_data_dir, meta_features_dir, algorithms_meta_features_dir, output_dir)

    #=== Start iterating, each iteration involves a meta-training step and a meta-testing step
    iteration = 0
    for D_tr, D_te in kf.split(list_datasets):
        vprint(verbose, "\n********** ITERATION " + str(iteration) + " **********")

        # Init a new agent instance in each iteration
        agent = Agent(number_of_algorithms = len(list_algorithms))

        #=== META-TRAINING
        trained_agent = meta_training(agent, D_tr)

        #=== META-TESTING
        meta_testing(trained_agent, D_te)

        iteration += 1
    ################################################

#=== For debug only
# tmp = os.listdir(os.path.join(root_dir, 'program'))
# # tmp2 = os.listdir(root_dir)
# tmp = ' '.join([str(item) for item in tmp])
# tmp_3 = os.listdir(tmp)
