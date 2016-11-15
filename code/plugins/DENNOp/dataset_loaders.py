import numpy as np
import os
import sys

__all__ = ['load_iris_data']

def load_iris_data(path, output=False):
    if not output:
        old_descriptor = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    flower_data = []
    label_data = []

    labels = []
    features_max = []

    print('+ Load data ...')
    with open(path+'.data', 'r') as iris_file:
        for line in iris_file.readlines():
            cur_line = [elm.strip() for elm in line.split(',')]

            if len(cur_line) == 5:
                cur_label = cur_line[-1]
                if cur_label not in labels:
                    labels.append(cur_label)

                label_data.append(labels.index(cur_label))

                features = [float(elm) for elm in cur_line[:-1]]
                if len(features_max) == 0:
                    features_max = [elm for elm in features]
                else:
                    for idx, feature in enumerate(features):
                        if features_max[idx] < feature:
                            features_max[idx] = feature

                flower_data.append(features)
    
    features_max = np.array(features_max, np.float64)

    flower_data = np.divide(np.array(flower_data, np.float64), features_max)
    ##
    # expand labels (one hot vector)
    tmp = np.zeros((len(label_data), len(labels)))
    tmp[np.arange(len(label_data)), label_data] = 1
    label_data = tmp

    print('+ flowers: \n', flower_data)
    print('+ labels: \n', label_data)

    print('+ loading done!')

    if not output:
        sys.stdout = old_descriptor

    return flower_data, label_data