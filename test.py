import unittest
import DENN
import tensorflow as tf
import numpy as np
from os import path
from os import chdir


class TestDENN(unittest.TestCase):

    def setUp(self):
        pass

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
        chdir('scripts') 
        jobs = DENN.training.open_task_list(self.iris_all_config)
        self.assertIsInstance(jobs, DENN.training.tasks.DETaskList)

if __name__ == '__main__':
    unittest.main()