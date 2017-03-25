import unittest
import DENN
import tensorflow as tf
import numpy as np
from os import path
from os import chdir
from os import getcwd


class TestDENN(unittest.TestCase):

    def setUp(self):
        if path.basename(getcwd()) != 'scripts':
            chdir('scripts')

    def __init__(self, *args, **kwargs):
        self.__de_type = [
            'rand/1/bin',
            'rand/1/exp',
            'rand/2/bin',
            'rand/2/exp',
        ]
        self.iris_all_config = path.join("config", "iris_test_ops.json")
        super(self.__class__, self).__init__(*args, **kwargs)

    def test_open_task_list(self):
        jobs = DENN.training.open_task_list(self.iris_all_config)
        self.assertIsInstance(jobs, DENN.training.tasks.DETaskList)
    
    def test_load_dataset(self):
        jobs = DENN.training.open_task_list(self.iris_all_config)
        dataset = DENN.training.Dataset(jobs[0].dataset_file)
        self.assertIsInstance(dataset, DENN.training.utils.Dataset)


if __name__ == '__main__':
    unittest.main()