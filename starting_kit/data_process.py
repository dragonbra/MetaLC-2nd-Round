import pandas as pd
import os
from scoring_program.learning_curve import Learning_Curve

#=== Setup data directories
root_dir = os.getcwd()
data_dir = os.path.join(root_dir, "sample_data/")
# Setup for specific data directories
train_data_dir = os.path.join(data_dir, 'train')
validation_data_dir = os.path.join(data_dir, 'validation')
test_data_dir = os.path.join(data_dir, 'test')
meta_features_dir = os.path.join(data_dir, 'dataset_meta_features')
algorithms_meta_features_dir = os.path.join(data_dir, 'algorithms_meta_features')

#=== List of dataset names
list_datasets = os.listdir(test_data_dir)
if '.DS_Store' in list_datasets: # remove junk files
    list_datasets.remove('.DS_Store')
list_datasets.sort(key=int)

#=== List of algorithms
list_algorithms = os.listdir(os.path.join(test_data_dir, list_datasets[0]))
if '.DS_Store' in list_algorithms: # remove junk files
    list_algorithms.remove('.DS_Store')
list_algorithms.sort(key=int)

def get_algorithm_meta_features():
    algorithms_meta_features = {}
    #=== Load ALGORITHM HYPERPARAMETERS
    # Iterate through all datasets
    for d in os.listdir(algorithms_meta_features_dir):
        if '.DS_Store' not in d:
            algorithm_name = d.split('.')[0]
            dict_temp = {}
            with open(os.path.join(algorithms_meta_features_dir, d), 'r') as f:
                for line in f:
                    key, value = line.split('=')
                    key, value = key.replace(' ','').replace('\n', ''), value.replace(' ','').replace('\n', '').replace('\'','') #remove whitespaces and special symbols
                    dict_temp[key] = value
            algorithms_meta_features[algorithm_name] = dict_temp
    return algorithms_meta_features

def get_meta_features_data(algorithms_meta_features):
    """
    这一部分得到的是x3, x4, x5
    e.g.
        x3: meta_feature_0 = 0
        x4: meta_feature_1 = 0
        x5: meta_feature_2 = 0.1
    """
    x3, x4, x5 = [[0 for _ in range(len(list_algorithms))] for __ in range(3)]
    # print(algorithms_meta_features)
    for algo_name in list_algorithms:
        # print(algo_name, algorithms_meta_features[algo_name])
        x3[int(algo_name)] = algorithms_meta_features[algo_name]['meta_feature_0']
        x4[int(algo_name)] = algorithms_meta_features[algo_name]['meta_feature_1']
        x5[int(algo_name)] = algorithms_meta_features[algo_name]['meta_feature_2']

    return x3, x4, x5

def get_score_time_curve():
    meta_features = {}
    test_learning_curves = {}
    #=== Load DATASET META-FEATURES
    # Iterate through all datasets
    for d in os.listdir(meta_features_dir):
        if '.DS_Store' not in d:
            dataset_name = d.split('.')[0].split('_')[0]
            dict_temp = {}
            with open(os.path.join(meta_features_dir, d), 'r') as f:
                for line in f:
                    key, value = line.split('=')
                    key, value = key.replace(' ','').replace('\n', ''), value.replace(' ','').replace('\n', '').replace('\'','') #remove whitespaces and special symbols
                    dict_temp[key] = value
            meta_features[dataset_name] = dict_temp

    #=== Load TEST LEARNING CURVES
    # Iterate through all datasets
    for dataset_name in list_datasets:
        dict_temp = {}
        for algo_name in list_algorithms:
            path_to_algo = os.path.join(test_data_dir, dataset_name, algo_name)
            dict_temp[algo_name] = Learning_Curve(os.path.join(path_to_algo + "/scores.txt"), float(meta_features[dataset_name]['time_budget']))

        test_learning_curves[dataset_name] = dict_temp
    return test_learning_curves

def get_score_time_at_point(test_learning_curves, x=1):
    """
    这一部分得到的是x1, x2
    e.g.
        [x1: score, x2: time]  # 对应在同一个点的数据 e.g. p = 0.2 || 整体的平均
        x1: 0.3618851626026678
        x2: 287.61

    Parameters
    ----------
    x: int
        选取曲线上对应位置的score & time的点, 对应MetaLC中agent给出的p
        默认为p=0.2, 如果不包含这个点
    """
    x1, x2 = [[0 for _ in range(len(list_algorithms))] for __ in range(2)]

    #=== Load TEST LEARNING CURVES
    # Iterate through all datasets
    for dataset_name in list_datasets:
        for algo_name in list_algorithms:
            learning_curve = test_learning_curves[dataset_name][algo_name]
            scores = learning_curve.scores
            times = learning_curve.times

            print(scores)
            print(times)
            print()

    return x1, x2

   
# 三个字段 name, site, age
nme = ["Google", "Runoob", "Taobao", "Wiki"]
st = ["www.google.com", "www.runoob.com", "www.taobao.com", "www.wikipedia.org"]
ag = [90, 40, 80, 98]
   
# 字典
dict = {'name': nme, 'site': st, 'age': ag}
     
df = pd.DataFrame(dict)
 
# 保存 dataframe
df.to_csv('site.csv')


if __name__ == "__main__":

    test_learning_curves = get_score_time_curve();
    x1, x2 = get_score_time_at_point(test_learning_curves)

    algorithms_meta_features = get_algorithm_meta_features();
    x3, x4, x5 = get_meta_features_data(algorithms_meta_features)

    data = {
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'x5': x5
    }
    df = pd.DataFrame(data)
    df.to_csv('meta_lc_data.csv')

    pass