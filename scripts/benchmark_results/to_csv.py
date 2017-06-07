from os import path
from os import walk
import argparse
import json
import csv

FLAGS = None

def main():
    rows = [
        [
            "TOT_GEN",
            "GEN_STEP",
            "VALIDATION_STEP",
            "d",
            "inh_type",
            "final_accuracy",
            "best_accuracy",
            "time_sec",
            "n_restart"
        ]
    ]

    for root, dirs, files in walk(FLAGS.data_dir):
        if path.isdir(root) and root != FLAGS.data_dir:
            if set(['job.json', 'test_results.json']) & set(files):
                cur_row = []
                print(path.join(root, 'job.json'))
                with open(path.join(root, 'job.json')) as json_file:
                    job = json.load(json_file)
                with open(path.join(root, 'test_results.json')) as json_file:
                    test_results = json.load(json_file)
                
                cur_row.append(job['TOT_GEN'])
                cur_row.append(job['GEN_STEP'])
                cur_row.append(job['VALIDATION_STEP'])
                cur_row.append(job['inheritance']['d'])
                cur_row.append(job['inheritance']['type'])
                cur_row.append(round(test_results['results']['rand/1/bin']['values'][-1], 3))
                cur_row.append(round(job['best']['accuracy'][-1], 3))
                cur_row.append(int(job['times']['rand/1/bin']))
                cur_row.append(test_results['results']['rand/1/bin']['reset_list'].count(True))

                rows.append(cur_row)
    
    with open('results.csv', 'w', newline='') as csvfile:
        resultwriter = csv.writer(csvfile, dialect='excel')
        resultwriter.writerows(rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Directory from where read data')
    FLAGS, unparsed = parser.parse_known_args()
    main()