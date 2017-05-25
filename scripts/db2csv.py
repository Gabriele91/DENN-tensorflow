import DENN
import numpy as np
from os import path
from tqdm import tqdm



def main():
    datasets = [
        ("../datasets/bank_t75-v15-s10_40x4000_5s.gz", 8),
        ("../datasets/magic_t75-v15-s10_70x1000_5s.gz", 14),
        ("../datasets/qsar_t75-v15-s10_20x160_5s.gz", 4)

    ]
    for file_name, max_batch in datasets:
        dataset = DENN.training.Dataset(file_name)
        with open(path.basename(file_name).replace(".gz", ".csv"), "w") as outfile:
            for idx in tqdm(range(max_batch), desc="Conversion of {} batch".format(file_name)):
                data, labels = dataset[idx]
                for idx_data, value in enumerate(data):
                    new_elems = [str(elm) for elm in value.tolist()] + [str(np.argmax(labels))]
                    outfile.write("{}\n".format(",".join(new_elems)))
            
            data, labels = dataset.validation_data, dataset.validation_labels
            for idx_data in tqdm(range(len(data)), desc="Conversion of {} validation".format(file_name)):
                new_elems = [str(elm) for elm in data[idx_data].tolist()] + [str(np.argmax(labels[idx_data]))]
                outfile.write("{}\n".format(",".join(new_elems)))
            
            data, labels = dataset.test_data, dataset.test_labels
            for idx_data in tqdm(range(len(data)), desc="Conversion of {} test".format(file_name)):
                new_elems = [str(elm) for elm in data[idx_data].tolist()] + [str(np.argmax(labels[idx_data]))]
                outfile.write("{}\n".format(",".join(new_elems)))
            

if __name__ == '__main__':
    main()