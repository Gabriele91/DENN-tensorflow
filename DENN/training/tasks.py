import json
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from collections import OrderedDict
from os import path


__all__ = ["open_task_list", "task_dumps", "TaskEncoder"]


def open_task_list(name):
    """Load as a python object a task list file."""
    with open(name, 'r') as task_list:
        return json.load(task_list, cls=TaskDecoder)


def task_dumps(task, indent=4):
    """Convert the task in a proper JSON file"""
    return json.dumps(task, cls=TaskEncoder, indent=indent)


class DETaskList(object):

    """Object to manage the task list."""

    def __init__(self, task_list):
        self.tasks = [DETask(cur_task) for cur_task in task_list]

    def __iter__(self):
        return iter(self.tasks)

    def __repr__(self):
        return "".join([str(task) for task in self.tasks])


class TFFx(object):

    """Represent a TensorFlow function to be called."""

    def __init__(self, obj):
        """Prepare the function name and the base arguments.

        TensorFlow function will be called with the base
        arguments when the object is created plus the arguments
        during the calling.

        """
        self.name = obj.get('name').split('.')
        self.args = obj.get('args', [])
        self.kwargs = obj.get('kwargs', {})

    def __call__(self, *args, **kwargs):
        """Call the real TensorFlow function.

        Note that the arguments passed now have precedence
        in args list.
        """
        tmp = tf

        for string in self.name:
            tmp = getattr(tmp, string)

        cur_args = list(args) + self.args
        cur_kwargs = self.kwargs.copy()
        cur_kwargs.update(kwargs)

        return tmp(*cur_args, **cur_kwargs)

    def to_dict(self):
        """Returns the object as a dictionary.

        {
            'name': ...,
            'args': ...,
            'kwargs': ...

        }
        """
        obj = {
            'name': ".".join(self.name),
            'args': self.args,
            'kwargs': self.kwargs
        }

        return obj

    def __repr__(self):
        """A string representation of the object TFFx"""
        return """{}({}, {})""".format(
            self.name,
            self.args,
            self.kwargs
        )


class Level(object):

    """A level of the network."""

    def __init__(self, cur_lvl):
        """Parse level informations and store them.

        Params:
            cur_lvl (dict): the current level options
        """
        self.shape = cur_lvl.get("shape")
        self.preferred_device = cur_lvl.get("preferred_device", "CPU")
        self.fx = TFFx(cur_lvl.get("fx"))
        self.init = cur_lvl.get("init", [])
        ##
        # init parse
        if len(self.init) > 0 and len(self.init) != len(self.shape):
            raise Exception("Init and shape have different number of elements")
        for idx, init_fn in enumerate(self.init):
            if type(init_fn) is dict:
                self.init[idx] = TFFx(init_fn.get('fx'))
            elif type(init_fn) is list:
                self.init[idx] = np.array(init_fn)
            else:
                raise Exception(
                    "Init element '{}' is not valid".format(init_fn))

    def __flat(self, list_):
        """Flat completely an list (array) to dim 0."""
        new_list = []
        for elm in list_:
            if type(elm) is list:
                return self.__flat(elm)
            new_list.append(elm)
        return new_list

    @property
    def in_size(self):
        """Input size of the current level."""
        return self.__flat(self.shape)[0]

    @property
    def out_size(self):
        """Output size of the current level."""
        return self.__flat(self.shape)[-1]

    def __repr__(self):
        """A string representation of level."""
        return """++ {} with f(x) = {} on [{}] |-> init {}""".format(
            self.shape,
            str(self.fx),
            self.preferred_device,
            self.init
        )

    def __get_init_dict(self):
        """Returns the init dictionary as a serializable JSON obj."""
        return [
            {'fx': elm.to_dict()}
            if callable(getattr(elm, 'to_dict', None))
            else elm.tolist()
            for elm in self.init
        ]

    def to_dict(self):
        """Returns the object as a dictionary.

        {
            'shape': ...,
            'fx': ...,
            'preferred_device': ...,
            'init': ...
        }
        """
        return {
            'shape': self.shape,
            'fx': self.fx.to_dict(),
            'preferred_device': self.preferred_device,
            'init': self.__get_init_dict()
        }


class DETask(object):

    """Python object for DE tasks."""

    def __init__(self, cur_task):
        """Parse task options.

        Params:
            cur_task (dict): options of the current task
        """
        self.name = cur_task.get("name")
        ##
        # Dataset file check
        assert path.isfile(
            path.normpath(
                cur_task.get("dataset_file"))
        ), "'{}' is not a file".format(
            cur_task.get("dataset_file")
        )
        self.dataset_file = cur_task.get("dataset_file")
        self.TOT_GEN = cur_task.get("TOT_GEN")
        self.GEN_STEP = cur_task.get("GEN_STEP")
        ##
        # TYPE check
        assert cur_task.get("TYPE") in [
            'float', 'double'
        ], "'{}' is not a valid type, use 'float' or 'double'".format(
            cur_task.get("TYPE")
        )
        self.TYPE = cur_task.get("TYPE")
        self.F = cur_task.get("F")
        self.NP = cur_task.get("NP")
        self.de_types = cur_task.get("de_types")
        self.CR = cur_task.get("CR")
        self.levels = [Level(obj) for obj in cur_task.get("levels")]

        ##
        # AdaBoost
        self.ada_boost = cur_task.get("AdaBoost", None)
        if self.ada_boost is not None:
            self.ada_boost = np.array(
                self.ada_boost,
                dtype=np.float64 if self.TYPE == "double" else np.float32
            )

        self.time = None
        self.best = None
        self.accuracy = None
        self.confusionM = None
        self.stats = None

    def __repr__(self):
        """A string representation of the object TFFx"""
        string = """+++++ Task ({}) +++++
+ dataset[{}] -> {}
+ tot. gen.   -> {}
+ get. step.  -> {}
+ F  -> {}
+ NP -> {}
+ CR -> {}
+ DE types -> {}
+ AdaBoost -> {}
+ levels:\n{}
+++++""".format(
            self.name,
            self.TYPE,
            self.dataset_file,
            self.TOT_GEN,
            self.GEN_STEP,
            self.F,
            self.NP,
            self.CR,
            self.de_types,
            self.ada_boost,
            "\n".join([str(level) for level in self.levels])
        )
        return string

    def gen_F(self, shape):
        """Returns a numpy array full of F values with the given shape.

        Params:
            shape (list)
        """
        return np.full(shape, self.F)

    def get_device(self, preference):
        """Returns prefer device if available.

        Params:
            preference (string)
        """
        for dev in device_lib.list_local_devices():
            if dev.device_type == preference or\
                    dev.name.find(preference) != -1:
                return dev.name
        return "/cpu:0"

    @staticmethod
    def __equal_shape(source, target):
        """Check if source shape array values are equal to target shape.

        Params:
            source (list, iterable)
            target (list, tuple, iterable)
        """
        for idx, elm in enumerate(source):
            if elm != target[idx]:
                return False
        return True

    def gen_network(self, default_graph=False):
        """Generate the network for a DENN op.

        Params:
            default_graph (default=False): specify if you want to work with 
                                           the default graph
        Returns:
            Network object
        """

        if self.TYPE == "double":
            cur_type = tf.float64
        elif self.TYPE == "float":
            cur_type = tf.float32
        else:
            raise Exception("Not valid type_ argument: {}".format(type_))

        if default_graph:
            graph = tf.get_default_graph()
        else:
            graph = tf.Graph()

        with graph.as_default():
            levels = self.levels
            target_ref = []
            pop_ref = []
            rand_pop_ref = []
            cur_pop_VAL = tf.placeholder(cur_type, [self.NP])
            cur_gen_options = tf.placeholder(tf.int32, [2])

            if self.ada_boost is not None:
                ada_boost_ref = tf.placeholder(
                    tf.float64 if self.TYPE == "double" else tf.float32
                )
            else:
                ada_boost_ref = None

            weights = []

            input_size = levels[0].in_size
            label_size = levels[-1].out_size
            input_placeholder = tf.placeholder(cur_type,
                                               [None, input_size], name="inputs")
            label_placeholder = tf.placeholder(cur_type,
                                               [None, label_size], name="labels")

            last_input = input_placeholder

            for num, cur_level in enumerate(levels, 1):

                with tf.device(self.get_device(cur_level.preferred_device)):

                    level = cur_level.shape

                    SIZE_W, SIZE_B = level

                    ##
                    # DE W -> NN (W, B)
                    deW_nnW = self.gen_F(SIZE_W)
                    deW_nnB = self.gen_F(SIZE_B)

                    weights.append(deW_nnW)
                    weights.append(deW_nnB)

                    ##
                    # Init population
                    if len(cur_level.init) == 0:
                        create_random_population_W = tf.random_uniform(
                            [self.NP] + SIZE_W, dtype=cur_type, seed=1)
                        create_random_population_B = tf.random_uniform(
                            [self.NP] + SIZE_B, dtype=cur_type, seed=1)

                        rand_pop_ref.append(create_random_population_W)
                        rand_pop_ref.append(create_random_population_B)
                    else:
                        sizes = [
                            SIZE_W,
                            SIZE_B
                        ]
                        tmp_init = []
                        for idx, init_elm in enumerate(cur_level.init):
                            if type(init_elm) == TFFx:
                                tmp_init.append(
                                    init_elm([self.NP] + sizes[idx],
                                             dtype=cur_type))
                            elif isinstance(init_elm, (np.ndarray, np.generic)):
                                if not self.__equal_shape(init_elm.shape, sizes[idx]):
                                    raise Exception("Wrong initial shape for elm. with index {} in level {}. [{} != {}]".format(
                                        idx,
                                        num,
                                        init_elm.shape,
                                        sizes[idx]
                                    ))
                                tmp_init.append(
                                    tf.constant(init_elm,
                                                shape=[self.NP] + sizes[idx],
                                                dtype=cur_type))

                        rand_pop_ref += tmp_init

                    ##
                    # Placeholder
                    target_w = tf.placeholder(cur_type, SIZE_W, name="target_W")
                    target_b = tf.placeholder(cur_type, SIZE_B, name="target_B")

                    target_ref.append(target_w)
                    target_ref.append(target_b)

                    cur_pop_W = tf.placeholder(cur_type, [self.NP] + SIZE_W, name="cur_pop_W")
                    cur_pop_B = tf.placeholder(cur_type, [self.NP] + SIZE_B, name="cur_pop_B")

                    pop_ref.append(cur_pop_W)
                    pop_ref.append(cur_pop_B)

                    if num == len(levels):
                        ##
                        # NN TRAIN
                        y = tf.matmul(last_input, target_w) + target_b

                        if self.ada_boost is not None:
                            cross_entropy = tf.reduce_mean(
                                tf.multiply(
                                    ada_boost_ref,
                                    cur_level.fx(
                                        y, label_placeholder), name="cross_entropy")
                            )
                        else:
                            cross_entropy = tf.reduce_mean(
                                cur_level.fx(
                                    y, label_placeholder), name="cross_entropy")

                        ##
                        # NN TEST
                        y_test = tf.matmul(last_input, target_w) + target_b
                        correct_prediction = tf.equal(
                            tf.argmax(y_test, 1),
                            tf.argmax(label_placeholder, 1)
                        )
                        accuracy = tf.reduce_mean(
                            tf.cast(correct_prediction, cur_type), name="accuracy")
                    else:
                        last_input = cur_level.fx(
                            tf.matmul(last_input, target_w) + target_b)

        return Network([
            ('targets', target_ref),
            ('populations', pop_ref),
            ('rand_pop', rand_pop_ref),
            ('weights', weights),
            ('evaluated', cur_pop_VAL),
            ('y', y),
            ('y_test', y_test),
            ('cross_entropy', cross_entropy),
            ('accuracy', accuracy),
            ('graph', graph),
            ('input_placeholder', input_placeholder),
            ('label_placeholder', label_placeholder),
            ('cur_gen_options', cur_gen_options),
            ('ada_boost_vector', ada_boost_ref)
        ])


class Network(object):

    """A network container for DENN op."""

    def __init__(self, list_):
        """Insert all attributes of the network.

        List of attributes:
            - targets
            - populations
            - rand_pop
            - weights
            - evaluated
            - y
            - y_test
            - cross_entropy (with or without AdaBoost)
            - accuracy
            - graph
            - input_placeholder
            - label_placeholder
            - cur_gen_options
            - ada_boost_vector (could be None)
        """

        for name, value in list_:
            setattr(self, name, value)


class TaskEncoder(json.JSONEncoder):

    """Class encoder for DENN jobs."""

    def __init__(self, *args, **kwargs):
        super(TaskEncoder, self).__init__(*args, **kwargs)

    def default(self, obj):
        """JSON encoding function for the single task.

        Params:
            obj: the current object to serialize
        """

        if type(obj) == np.ndarray:
            return obj.tolist()
        elif type(obj) in [np.int32, np.int64]:
            return int(obj)
        elif type(obj) in [np.float32, np.float64]:
            return float(obj)

        new_obj = OrderedDict([
            ('name', obj.name),
            ('TYPE', obj.TYPE),
            ('dataset_file', obj.dataset_file),
            ('TOT_GEN', obj.TOT_GEN),
            ('GEN_STEP', obj.GEN_STEP),
            ('F', obj.F),
            ('NP', obj.NP),
            ('CR', obj.CR),
            ('de_types', obj.de_types),
            ('levels', [level.to_dict() for level in obj.levels])
        ])

        if obj.best is not None:
            new_obj['best'] = obj.best
        if obj.time is not None:
            new_obj['time'] = obj.time
        if obj.accuracy is not None:
            new_obj['accuracy'] = obj.accuracy
        if obj.confusionM is not None:
            new_obj['confusionM'] = obj.confusionM
        if obj.stats is not None:
            new_obj['stats'] = obj.stats

        return new_obj


class TaskDecoder(json.JSONDecoder):

    """Class decoder for DENN jobs."""

    def __init__(self, *args, **kwargs):
        """Decode a task object.

        Internal states:
            0 -> normal
            1 -> string
            2 -> python comment
            3 -> start C++ inline comment
            4 -> inline comment
            5 -> comment
            6 -> start end comment

        """
        super(TaskDecoder, self).__init__(*args, **kwargs)
        self.__state = 0

    def decode(self, json_string):
        """Decode properly a DE task list.

        Params:
            json_string (string): the input string to parse
        """

        # print(json_string)

        final_string = ""
        line_num = 1

        for idx, char in enumerate(json_string):
            if char == '\n':
                line_num += 1

            if self.__state == 0:
                # start string
                if char == "\"":
                    self.__state = 1
                    final_string += char
                # start python comment
                elif char == "#":
                    self.__state = 2
                # start C++ style comment
                elif char == "/":
                    self.__state = 3
                else:
                    final_string += char
            # inside string
            elif self.__state == 1:
                # string end
                if char == "\"":
                    self.__state = 0
                final_string += char
            # python comment
            elif self.__state == 2:
                # end of comment
                if char == "\n":
                    self.__state = 0
                    final_string += char
            # check inline comment
            elif self.__state == 3:
                # start inline comment
                if char == "/":
                    self.__state = 4
                # start comment
                elif char == "*":
                    self.__state = 5
                else:
                    raise json.JSONDecodeError(
                        "Wrong start of inline comment...", json_string, idx)
            # inline comment
            elif self.__state == 4:
                # end of comment
                if char == "\n":
                    self.__state = 0
                    final_string += char
            # Check end of multiline comment
            elif self.__state == 5:
                # Start ending of the comment
                if char == "*":
                    self.__state = 6
                elif char == "\n":
                    final_string += char
            # Check end of comment
            elif self.__state == 6:
                # end of comment
                if char == "/":
                    self.__state = 0
                # Go to previous check
                else:
                    self.__state = 5

        # Check ending state before
        # json deserialization
        if self.__state != 0:
            raise json.JSONDecodeError(
                "Error on closing comment...",
                json_string, len(json_string) - 1)

        # print(final_string)

        task_list = super(TaskDecoder, self).decode(final_string)

        # print(task_list)

        return DETaskList(task_list)
