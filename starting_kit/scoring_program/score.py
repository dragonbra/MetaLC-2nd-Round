import os
from sys import argv, path
root_dir = os.getcwd()
from environment import Meta_Learning_Environment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import base64
import traceback
import math
import random

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000) 

#=== Set RANDOM SEED
random.seed(208)

#=== Verbose mode
verbose = True

#=== Setup input/output directories
root_dir = os.getcwd()
default_input_dir = os.path.join(root_dir, "sample_data/")
default_output_dir = os.path.join(root_dir, "output/")
default_program_dir = os.path.join(root_dir, "ingestion_program/")
default_submission_dir = os.path.join(root_dir, "sample_code_submission/")

#=== Normalize time as implemented in the AutoML challenge
normalize_t = True
t_0 = 60.0 # Hyperparameters used for computing scaled time (t_tilde). It controls how important performing well at the beginning is.

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

def visualize_agent_learning_curve(dataset_name, total_time_budget, df_, alc):
    """
    Visualize agent's learning curves on the test set.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset at hand.
    total_time_budget : float
        Total time budget given to the agent for searching algorithms on the given dataset.
    df_ : Dataframe
        Data for plotting the learning curve, with columns = ['algo_index', 'algo_time_spent', 'cumulative_t', 'score']

    """

    # The dataset with only 1 action with test_score_of_best_algorithm_so_far==None will not be plotted
    df = df_[~df_['test_score_of_best_algorithm_so_far'].isin(['None'])]

    plt.title(label = "Dataset: " + dataset_name)
    min_y = float(df[df['R_test_A_p']!='None'].min()['R_test_A_p'])
    if min_y<0:
        min_y -= 0.1
    else:
        min_y = 0.0
    plt.ylim(min_y, 1.05)
    plt.ylabel('test score')
    if not df.empty:
        plt.step(df['normalized_cumulative_t'], df['test_score_of_best_algorithm_so_far'], where='post', label = dataset_name)
        plt.scatter(df['normalized_cumulative_t'], df['test_score_of_best_algorithm_so_far'], s=8, marker="D")
        L=plt.legend()
        L.get_texts()[0].set_text('ALC = ' + str(np.around(alc, decimals=2)))
        plt.fill_between(df['normalized_cumulative_t'], df['test_score_of_best_algorithm_so_far'], step="post", alpha=0.4, color='dodgerblue')
        plt.grid()

        #=== Normalize time on the plot
        if normalize_t:
            plt.xlim(0, 1)
            plt.xlabel('normalized cumulative time')
            plt.text(1.0, df.iloc[-1]['test_score_of_best_algorithm_so_far'], str(df.iloc[-1]['test_score_of_best_algorithm_so_far']), fontdict=None)
        else:
            plt.xlim(0, total_time_budget)
            plt.xlabel('time (s)')
            plt.text(df['cumulative_t'].max(), df.iloc[-1]['test_score_of_best_algorithm_so_far'], str(df.iloc[-1]['test_score_of_best_algorithm_so_far']), fontdict=None)

    #=== Save figure and clear the plot
    plt.savefig(output_visualizations_dir + dataset_name + ".png", dpi=120, format='png', bbox_inches='tight')
    plt.cla()
    plt.clf()

def write_scores_html(output_dir, output_visualizations_dir, auto_refresh=True, append=False):
    filename = 'scores.html'
    image_paths = glob(os.path.join(output_visualizations_dir, '*.png'))

    try: # Try to sort by numerical values
      image_paths = sorted(image_paths,key=lambda x: int(x.split('/')[-1].split('.')[0]))
    except:
      image_paths = sorted(image_paths)

    if auto_refresh:
      html_head = '<html><head> <meta http-equiv="refresh" content="5"> ' +\
                  '</head><body><pre>'
    else:
      html_head = """<html><body><pre>"""
    html_end = '</pre></body></html>'
    if append:
      mode = 'a'
    else:
      mode = 'w'
    filepath = os.path.join(output_dir, filename)
    with open(filepath, mode) as html_file:
        html_file.write(html_head)
        for image_path in image_paths:
          with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            encoded_string = encoded_string.decode('utf-8')
            s = '<img src="data:image/png;charset=utf-8;base64,{}"/>'\
                .format(encoded_string)
            html_file.write(s + '<br>')
        html_file.write(html_end)
    vprint(verbose, "\n[+] Write agent's learning curve visualizations to {}".format(filepath))

def normalize_time(df, total_time_budget, t_0):
    """
    Normalize time using log functions and t_0.

    Parameters
    ----------
    df : Dataframe
        Learning curve data with 'cumulative_t' column to be normalized.
    total_time_budget : float
        Total time budget for searching algorithms on the dataset at hand
    t_0 : float
        Hyperparameter controlling how important performing well at the beginning is.

    Returns
    ----------
    df : Dataframe
        The dataframe with the normalized time.

    Examples
    ----------

    >>> df
        A    p      t R_train_A_p R_validation_A_p  cumulative_t R_test_A_p  test_score_of_best_algorithm_so_far
    0  18  0.4   24.0         0.9             0.44          24.0       0.36                    0.36
    1  36  0.5   32.0        0.97             0.51          56.0       0.42                    0.42
    2  26  0.4   21.0         1.0             0.49          77.0       0.42                    0.42
    3  25  0.7  623.0        None             None         700.0       None                    0.42



    >>> normalize_time(df, 1100, 60)
        A    p      t R_train_A_p R_validation_A_p  cumulative_t R_test_A_p  test_score_of_best_algorithm_so_far  normalized_cumulative_t
    0  18  0.4   24.0         0.9             0.44          24.0       0.36                    0.36                 0.132523
    1  36  0.5   32.0        0.97             0.51          56.0       0.42                    0.42                 0.259650
    2  26  0.4   21.0         1.0             0.49          77.0       0.42                    0.42                 0.325185
    3  25  0.7  623.0        None             None         700.0       None                    0.42                 1.000000

    """

    df['normalized_cumulative_t'] = df['cumulative_t'].apply(lambda x: round(np.log(1+x/t_0)/np.log(1+total_time_budget/t_0), 2))

    return df

def reveal_score_on_the_test_set(env, dataset_name, df):
    """
    Query the scores on the learning curve on the test set

    Parameters
    ----------
    env : Meta_Learning_Environment
        The environment which contains all the learning curve data.
    dataset_name : str
        Name of the dataset at hand
    df : Dataframe
        Agent's learning curve data

    Returns
    ----------
    df : Dataframe
        The dataframe with newly created columns: cumulative_t, R_test_A_p, test_score_of_best_algorithm_so_far.

    Examples
    ----------
    >>> df
             A    p      t R_train_A_p R_validation_A_p
            18  0.4   24.0         0.9             0.44
            36  0.5   32.0        0.97             0.51
            26  0.4   21.0         1.0             0.49
            25  0.7  623.0        None             None

    >>> reveal_score_on_the_test_set(env, dataset_name, df)
             A    p      t R_train_A_p R_validation_A_p  cumulative_t R_test_A_p  test_score_of_best_algorithm_so_far
         0  18  0.4   24.0         0.9             0.44          24.0       0.36                    0.36
         1  36  0.5   32.0        0.97             0.51          56.0       0.42                    0.42
         2  26  0.4   21.0         1.0             0.49          77.0       0.42                    0.42
         3  25  0.7  623.0        None             None         700.0       None                    0.42
    """

    #=== Iterate through each timestamp and compute the corresponding score on the test learning curve
    test_score_of_best_algorithm_so_far = 'None'
    max_validation_score = 'None'

    cumulative_t = 0.0
    for i, row in df.iterrows():
        cumulative_t += row['t']

        if row['R_train_A_p']!='None':
            #=== Reveal the test scores
            lc = env.test_learning_curves[dataset_name][str(row['A'])]
            R_test_A_p = lc.get_performance_score(float(row['p']))[0]
            df.at[i,'R_test_A_p'] = str(R_test_A_p)

            if test_score_of_best_algorithm_so_far=='None':
                test_score_of_best_algorithm_so_far = float(R_test_A_p)
                max_validation_score = float(row['R_validation_A_p'])
            else:
                if max_validation_score=='None':
                    test_score_of_best_algorithm_so_far = R_test_A_p
                    max_validation_score = float(row['R_validation_A_p'])
                else:

                    if float(row['R_validation_A_p']) >= max_validation_score:
                        test_score_of_best_algorithm_so_far = R_test_A_p
                        max_validation_score = float(row['R_validation_A_p'])
        else:
            df.at[i,'R_test_A_p'] = 'None'

        #=== Update the dataframe
        df.at[i,'cumulative_t'] = cumulative_t
        df.at[i,'test_score_of_best_algorithm_so_far'] = test_score_of_best_algorithm_so_far

    return df

def compute_ALC(df, total_time_budget):
    """
    Compute the area under the agent's learning curve. The goal is to maximize
    this area.

    Parameters
    ----------
    df : Dataframe
        Learning curve data.
    total_time_budget : float
        Total time budget for searching algorithms on the dataset at hand

    Returns
    ----------
    alc : float

    Examples
    ----------
    >>> df
             algo_index  algo_time_spent  cumulative_t  score   test_score_of_best_algorithm_so_far
    0            9            15.28          0.253961        0.39       0.39
    1            9            34.50          0.322162        0.44       0.44
    2            9            34.50          0.455340        0.44
    ...
    26           9           151.73          1.000000        0.49       0.49

    >>> compute_ALC(df, 1200):
    0.35
    """

    alc = 0.0

    for i in range(len(df)):
        if df.iloc[i]['test_score_of_best_algorithm_so_far'] == 'None':
            continue
        if i==0:
            if normalize_t:
                alc += df.iloc[i]['test_score_of_best_algorithm_so_far'] * (1-df.iloc[i]['normalized_cumulative_t'])
            else:
                alc += df.iloc[i]['test_score_of_best_algorithm_so_far'] * (total_time_budget-df.iloc[i]['cumulative_t'])
        elif i>0:
            if normalize_t:
                alc += (df.iloc[i]['test_score_of_best_algorithm_so_far']-df.iloc[i-1]['test_score_of_best_algorithm_so_far']) * (1-df.iloc[i]['normalized_cumulative_t'])
            else:
                alc += (df.iloc[i]['test_score_of_best_algorithm_so_far']-df.iloc[i-1]['test_score_of_best_algorithm_so_far']) * (total_time_budget-df.iloc[i]['cumulative_t'])
    return round(alc, 2)

if __name__ == "__main__":
    #=== Get input and output directories
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
        train_data_dir = os.path.join(input_dir, 'train')
        validation_data_dir = os.path.join(input_dir, 'validation')
        test_data_dir = os.path.join(input_dir, 'test')
        meta_features_dir = os.path.join(input_dir, 'dataset_meta_features')
        algorithms_meta_features_dir = os.path.join(input_dir, 'algorithms_meta_features')
        output_from_ingestion_program_dir = output_dir # Output from the ingestion_program
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])
        train_data_dir = os.path.join(input_dir, 'ref/train')
        validation_data_dir = os.path.join(input_dir, 'ref/validation')
        test_data_dir = os.path.join(input_dir, 'ref/test')
        meta_features_dir = os.path.join(input_dir, 'ref/dataset_meta_features')
        algorithms_meta_features_dir = os.path.join(input_dir, 'ref/algorithms_meta_features')
        output_from_ingestion_program_dir = os.path.join(input_dir, 'res') # Output from the ingestion_program

    vprint(verbose, "Using input_dir: " + input_dir)
    vprint(verbose, "Using output_dir: " + output_dir)
    vprint(verbose, "Using train_data_dir: " + train_data_dir)
    vprint(verbose, "Using validation_data_dir: " + validation_data_dir)
    vprint(verbose, "Using test_data_dir: " + test_data_dir)
    vprint(verbose, "Using meta_features_dir: " + meta_features_dir)
    vprint(verbose, "Using algorithms_meta_features_dir: " + algorithms_meta_features_dir)
    vprint(verbose, "Using output_from_ingestion_program_dir: " + output_from_ingestion_program_dir)

    #=== Directory for visualizations of agent's learning curves
    output_visualizations_dir = os.path.join(output_dir, "output_visualizations/")
    if not os.path.exists(output_visualizations_dir):
        os.makedirs(output_visualizations_dir)

    #=== List of dataset names
    list_datasets = os.listdir(test_data_dir)
    if '.DS_Store' in list_datasets:
        list_datasets.remove('.DS_Store')
    list_datasets.sort(key=int)

    #=== List of algorithms
    list_algorithms = os.listdir(os.path.join(test_data_dir, list_datasets[0]))
    if '.DS_Store' in list_algorithms:
        list_algorithms.remove('.DS_Store')
    list_algorithms.sort(key=int)

    #=== Create files for storing output scores
    score_file = open(os.path.join(output_dir, 'scores.txt'), 'w')
    html_file = open(os.path.join(output_dir, 'scores.html'), 'w')

    ################## MAIN LOOP ##################
    #=== Create an environment
    env = Meta_Learning_Environment(train_data_dir, validation_data_dir, test_data_dir, meta_features_dir, algorithms_meta_features_dir, output_dir)

    #=== As we are using k-fold cross-validation, each dataset is used for testing once
    #=== We iterate through each dataset to compute the agent's test performance on each dataset
    list_final_score, list_alc = [], []
    average_final_score = 0.0
    average_alc = 0.0
    for dataset_name in list_datasets:
        vprint(verbose, "\n#====================== Results on dataset: " + dataset_name + " ======================#")
        alc, final_score = 0.0, 0.0

        #=== Get total_time_budget from meta_features of the dataset
        meta_features = env.meta_features[dataset_name]
        total_time_budget = int(meta_features['time_budget'])

        #=== Read output file from the ingestion program
        #=== The agent's learning curve on a dataset is stored in a csv file
        try:
            output_file = os.path.join(output_from_ingestion_program_dir, dataset_name + '.csv')
            df = pd.read_csv(output_file, header=0, dtype={'A': int, 'p': float, 't': float, 'R_train_A_p': str, 'R_validation_A_p': str})

            #=== Update the dataframe with scores on the learning curve on the validation set
            updated_df = reveal_score_on_the_test_set(env, dataset_name, df)

            #=== Normalizing t
            if normalize_t:
              normalize_time(updated_df, total_time_budget, t_0)
            vprint(verbose, updated_df)

            #=== Get the final score
            final_score = updated_df.iloc[-1]['test_score_of_best_algorithm_so_far']
            if final_score == 'None':
                final_score = 0

            #=== Compute ALC
            alc = compute_ALC(updated_df, total_time_budget)

            #=== Visualization
            visualize_agent_learning_curve(dataset_name, total_time_budget, updated_df, alc)
        except:
          print(traceback.format_exc())

        vprint(verbose, "\nalc = " + str(alc))
        list_alc.append(alc)
        vprint(verbose, "\nfinal score = " + str(final_score))
        list_final_score.append(final_score)

        updated_df.to_csv(os.path.join(output_from_ingestion_program_dir, dataset_name + '_updated.csv'), index=False)
    #############################################

    #=== Compute average final score and average ALC
    if len(list_final_score)!=0:
        average_final_score = round(sum(list_final_score) / len(list_datasets), 2)
    if len(list_alc)!=0:
        average_alc = round(sum(list_alc) / len(list_datasets), 2)

    #=== Write scores.html
    write_scores_html(output_dir, output_visualizations_dir)

    #=== Write out the scores to scores.txt
    score_file.write("average_final_score: " + str(average_final_score) + "\n")
    score_file.write("average_ALC: " + str(average_alc) + "\n")
    vprint(verbose, "\n######################################## FINAL AVERAGE RESULTS ########################################")
    vprint(verbose, "\naverage_ALC = " + str(average_alc))
    vprint(verbose, "\naverage_final_score = " + str(average_final_score))
    vprint(verbose, "\n[+]Finished running scoring program")
