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
        return json.load(task_list, cls=TaskDecoder, file_name=name)


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

        if self.name[-1] == "softmax_cross_entropy_with_logits":
            cur_kwargs = self.kwargs.copy()
            cur_kwargs.update(kwargs)
            cur_kwargs['logits'] = args[0]
            cur_kwargs['labels'] = args[1]
            cur_args = []
        else:
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
        self.start = cur_lvl.get("start", [])
        self.start_transposed = cur_lvl.get("start_transposed", [])
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
            'init': ...,
            'start': ...,
            'start_transposed': ..            
        }
        """
        return {
            'shape': self.shape,
            'fx': self.fx.to_dict(),
            'preferred_device': self.preferred_device,
            'init': self.__get_init_dict(),
            'start': self.start,
            'start_transposed': self.start_transposed
        }


class Clamp(object):

    def __init__(self, clamp):
        self.min = clamp['min'] if clamp is not None else -1.0
        self.max = clamp['max'] if clamp is not None else 1.0

    def to_dict(self):
        return {
            'min': self.min,
            'max': self.max
        }


class AdaBoost(object):

    def __init__(self, ada_boost):
        self.alpha = ada_boost['alpha'] if ada_boost is not None else .5
        self.C = ada_boost['C'] if ada_boost is not None else 1.

    def to_dict(self):
        return {
            'alpha': self.alpha,
            'C': self.C
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
        self.GEN_SAMPLES = cur_task.get("GEN_SAMPLES", 1)
        self.reinsert_best = cur_task.get("reinsert_best", False)
        ##
        # TYPE check
        assert cur_task.get("TYPE") in [
            'float', 'double'
        ], "'{}' is not a valid type, use 'float' or 'double'".format(
            cur_task.get("TYPE")
        )
        self.TYPE = cur_task.get("TYPE")
        self.F = cur_task.get("F")     
        self.CR = cur_task.get("CR")
        self.JDE = cur_task.get("JDE", 0.1)
        self.NP = cur_task.get("NP")   
        self.de_types = cur_task.get("de_types")
        self.clamp = Clamp(cur_task.get("clamp", None))
        self.training = cur_task.get("training", False)
        self.reset_every = cur_task.get("reset_every", False)
        self.levels = [Level(obj) for obj in cur_task.get("levels")]
        self.num_intra_threads = cur_task.get("NUM_INTRA_THREADS", 4)
        self.num_inter_threads = cur_task.get("NUM_INTER_THREADS", 4)
        self.smoothing = cur_task.get("smoothing", [])
        self.smoothing_n_pass = cur_task.get("smoothing_n_pass", 0)

        ##
        # AdaBoost
        tmp = cur_task.get("AdaBoost", None)
        # if self.training and tmp:
        #    raise Exception(
        #        "You can't use AdaBoost and training at the same time...")
        self.ada_boost = AdaBoost(tmp) if tmp is not None else tmp
        self.__ada_boost_cache = {}

        self.time = None
        self.times = {}
        self.best = {}
        self.accuracy = {}
        self.confusionM = {}
        self.stats = {}

    def reset_adaboost_cache(self):
        self.__ada_boost_cache = {}

    def get_adaboost_cache(self, idx, batch):
        if idx not in self.__ada_boost_cache:
            self.__ada_boost_cache[idx] = (
                ##
                # C vector: size[len(batch)]
                np.full(
                    [len(batch.data)], self.ada_boost.C, dtype=batch.data.dtype
                ),
                ##
                # EC vector: size[len(pop), len(batch)]
                np.full(
                    [self.NP, len(batch.data)], 0, dtype=np.bool
                ),
                ##
                # y vector: size[len(pop), len(batch), len(classes)]
                np.full(
                    [self.NP, len(batch.data), batch.labels.shape[-1]],
                    0.0,
                    dtype=batch.data.dtype
                ),
                True
            )
        return self.__ada_boost_cache[idx]

    def set_adaboost_cache(self, idx, C, EC, pop_y, force_evaluation=False):
        self.__ada_boost_cache[idx] = (C, EC, pop_y, force_evaluation)

    def __repr__(self):
        """A string representation of the object TFFx"""
        string = """+++++ Task ({}) +++++
+ dataset[{}]  -> {}
+ tot. gen.    -> {}
+ gen. step.   -> {}
+ gen. samples -> {}
+ F  -> {}
+ NP -> {}
+ CR -> {}
+ DE types -> {}
+ INTRA Threads -> {}
+ INTER Threads -> {}
+ AdaBoost -> {}
+ Training -> {}
+ Smooth -> {}
+ Smooth n pass -> {}
+ Clamp -> {}
+ levels:\n{}
+++++""".format(
            self.name,
            self.TYPE,
            self.dataset_file,
            self.TOT_GEN,
            self.GEN_STEP,
            self.GEN_SAMPLES,
            self.F,
            self.NP,
            self.CR,
            self.de_types,
            self.num_intra_threads,
            self.num_inter_threads,
            self.ada_boost.to_dict() if self.ada_boost is not None else None,
            self.training,
            self.smoothing,
            self.smoothing_n_pass,
            self.clamp,
            "\n".join([str(level) for level in self.levels])
        )
        return string

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
            cur_gen_options = tf.placeholder(tf.int32, [2])

            input_size = levels[0].in_size
            label_size = levels[-1].out_size
            input_placeholder = tf.placeholder(cur_type,
                                               [None, input_size],
                                               name="inputs"
                                               )
            label_placeholder = tf.placeholder(cur_type,
                                               [None, label_size],
                                               name="labels"
                                               )
            f_placeholder = tf.placeholder(cur_type,
                                           [self.NP],
                                            name="labels"
                                           )
            cr_placeholder = tf.placeholder(cur_type,
                                            [self.NP],
                                            name="labels"
                                           )
            #operation on F
            f_init  = tf.fill([self.NP], tf.cast(self.F, cur_type))
            #operation on CR
            cr_init = tf.fill([self.NP], tf.cast(self.CR, cur_type))

            if self.ada_boost is not None:
                y_placeholder = tf.placeholder(cur_type,
                                               [None, label_size],
                                               name="y_placeholder"
                                               )
                ada_C_placeholder = tf.placeholder(cur_type,
                                                   [None],
                                                   name="C"
                                                   )
                ada_EC_placeholder = tf.placeholder(tf.bool,
                                                    [self.NP, None],
                                                    name="EC"
                                                    )
                population_y_placeholder = tf.placeholder(cur_type,
                                                          [self.NP, None,
                                                              label_size],
                                                          name="population_y_placeholder"
                                                          )
                cur_pop_VAL = None
            else:
                y_placeholder = None
                ada_C_placeholder = None
                ada_EC_placeholder = None
                population_y_placeholder = None
                cur_pop_VAL = tf.placeholder(cur_type, [self.NP])

            last_input = input_placeholder

            for num, cur_level in enumerate(levels, 1):

                with tf.device(self.get_device(cur_level.preferred_device)):

                    level = cur_level.shape

                    SIZE_W, SIZE_B = level

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
                                    tf.constant(
                                        np.array([init_elm.copy()
                                                  for _ in range(self.NP)]),
                                        shape=[self.NP] + sizes[idx],
                                        dtype=cur_type
                                    )
                                )

                        rand_pop_ref += tmp_init

                    ##
                    # Placeholder
                    target_w = tf.placeholder(
                        cur_type, SIZE_W, name="target_W")
                    target_b = tf.placeholder(
                        cur_type, SIZE_B, name="target_B")

                    target_ref.append(target_w)
                    target_ref.append(target_b)

                    cur_pop_W = tf.placeholder(
                        cur_type, [self.NP] + SIZE_W, name="cur_pop_W")
                    cur_pop_B = tf.placeholder(
                        cur_type, [self.NP] + SIZE_B, name="cur_pop_B")

                    pop_ref.append(cur_pop_W)
                    pop_ref.append(cur_pop_B)

                    if num == len(levels):
                        ##
                        # NN TRAIN
                        y = tf.add(
                            tf.matmul(last_input, target_w), target_b,
                            name="nn_execution"
                        )

                        ada_label_diff = None
                        if self.ada_boost is not None:
                            ada_label_diff = tf.equal(
                                tf.argmax(y_placeholder, 1),
                                tf.argmax(label_placeholder, 1),
                                name="ada_label_diff"
                            )
                            #
                            # y   # |BATCH| x |CLASS|
                            # c   # |BATCH| x 1
                            # --------------------------
                            # y^t   # |CLASS| x |BATCH|
                            # brodcast  multiply
                            # c     #           |BATCH|
                            # out^t # |CLASS| x |BATCH|
                            # out   # |BATCH| x |CLASS|
                            # --------------------------
                            # (y^t * c )^t
                            #
                            # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
                            cross_entropy = tf.reduce_mean(
                                cur_level.fx(
                                    tf.transpose(
                                        tf.multiply(
                                            tf.transpose(y_placeholder),
                                            ada_C_placeholder
                                        )
                                    ), label_placeholder
                                ),
                                name="cross_entropy"
                            )
                        else:
                            cross_entropy = tf.reduce_mean(
                                cur_level.fx(
                                    y, label_placeholder
                                ),
                                name="cross_entropy"
                            )

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
            ('evaluated', cur_pop_VAL),
            ('nn_exec', y),
            ('y', y),
            ('y_test', y_test),
            ('cross_entropy', cross_entropy),
            ('accuracy', accuracy),
            ('graph', graph),
            ('F_init', f_init),
            ('CR_init', cr_init),
            ('F_placeholder', f_placeholder),
            ('CR_placeholder', cr_placeholder),
            ('input_placeholder', input_placeholder),
            ('label_placeholder', label_placeholder),
            ('cur_gen_options', cur_gen_options),
            ##
            # AdaBoost placeolder
            ('ada_label_diff', ada_label_diff),
            ('ada_C_placeholder', ada_C_placeholder),
            ('ada_EC_placeholder', ada_EC_placeholder),
            ('y_placeholder', y_placeholder),
            ('population_y_placeholder', population_y_placeholder)
        ])


class Network(object):

    """A network container for DENN op."""

    __slots__ = [
        'targets',
        'populations',
        'rand_pop',
        'weights',
        'evaluated',
        'nn_exec',
        'y',
        'y_test',
        'cross_entropy',
        'accuracy',
        'graph',
        'F_init',
        'CR_init',
        'F_placeholder',
        'CR_placeholder',
        'input_placeholder',
        'label_placeholder',
        'cur_gen_options',
        ##
        # AdaBoost placeolder
        'ada_label_diff',
        'ada_C_placeholder',
        'ada_EC_placeholder',
        'y_placeholder',
        'population_y_placeholder'
    ]

    def __init__(self, list_):
        """Insert all attributes of the network.

        List of attributes:
            - targets
            - populations
            - rand_pop
            - weights
            - evaluated
            - nn_exec
            - y
            - y_test
            - cross_entropy
            - accuracy
            - graph
            - F_init 
            - CR_init
            - F_placeholder
            - CR_placeholder
            - input_placeholder
            - label_placeholder
            - cur_gen_options
            - ada_label_diff (only with AdaBoost)
            - ada_C_placeholder (only with AdaBoost)
            - y_placeholder (only with AdaBoost)
            - ada_EC_placeholder (only with AdaBoost)
            - population_y_placeholder (only with AdaBoost)
        """

        for name, value in list_:
            if value is not None:
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

        # print("obj", type(obj))

        if type(obj) == np.ndarray:
            return obj.tolist()
        elif type(obj) in [np.int32, np.int64]:
            return int(obj)
        elif type(obj) in [np.float32, np.float64]:
            return float(obj)
        elif type(obj) == DETask:
            new_obj = OrderedDict([
                ('name', obj.name),
                ('TYPE', obj.TYPE),
                ('dataset_file', obj.dataset_file),
                ('TOT_GEN', obj.TOT_GEN),
                ('GEN_STEP', obj.GEN_STEP),
                ('GEN_SAMPLES', obj.GEN_SAMPLES),
                ('F', obj.F),
                ('NP', obj.NP),
                ('CR', obj.CR),
                ('JDE', obj.JDE),
                ('de_types', obj.de_types),
                ('reset_every', obj.reset_every),
                ('reinsert_best', obj.reinsert_best),
                ('NUM_INTRA_THREADS', obj.num_intra_threads),
                ('NUM_INTER_THREADS', obj.num_inter_threads),
                ('AdaBoost', obj.ada_boost.to_dict()
                 if obj.ada_boost is not None else None),
                ('training', obj.training),
                ('clamp', obj.clamp.to_dict()),
                ('smoothing', obj.smoothing),
                ('smoothing_n_pass', obj.smoothing_n_pass),
                ('levels', [level.to_dict() for level in obj.levels])
            ])

            if len(obj.best) > 0:
                new_obj['best'] = obj.best
            if len(obj.times) > 0:
                new_obj['times'] = obj.times
            if obj.time is not None:
                new_obj['time'] = obj.time
            if len(obj.accuracy) is not None:
                new_obj['accuracy'] = obj.accuracy
            if len(obj.confusionM) > 0:
                new_obj['confusionM'] = obj.confusionM
            if len(obj.stats) > 0:
                new_obj['stats'] = obj.stats
        else:
            return json.JSONEncoder.default(self, obj)

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
        self.__file_name = kwargs.get("file_name")
        del kwargs["file_name"]
        self.__base_folder = path.dirname(path.abspath(self.__file_name))
        self.__state = 0
        super(TaskDecoder, self).__init__(*args, **kwargs, object_pairs_hook=self.object_pairs_hook)

    def object_pairs_hook(self, list_):
        for idx, (key, value) in enumerate(list_):
            if type(value) == str:
                if value[:6] == "import" and value[-1] == ";":
                    file_to_import = value.split(" ")[1][:-1]
                    file_path = path.join(self.__base_folder, file_to_import)
                    with open(file_path, "r") as imported_file:
                        res = self.decode(imported_file.read(), sub_json=True)
                    list_[idx] = (key, res)
        return dict(list_)

    def decode(self, json_string, sub_json=False):
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

        decoded_string = super(TaskDecoder, self).decode(final_string)

        # print(decoded_string)

        if not sub_json:
            return DETaskList(decoded_string)
        else:
            return decoded_string
