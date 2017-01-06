import json


__all__ = ["open_task_list", "TaskEncoder"]


def open_task_list(name):
    """Load as a python object a task list file."""
    with open(name, 'r') as task_list:
        return json.load(task_list, cls=TaskDecoder)


class DETaskList(object):

    """Object to manage the task list."""

    def __init__(self, task_list):
        self.tasks = [DETask(cur_task) for cur_task in task_list]

    def __iter__(self):
        return iter(self.tasks)

    def __repr__(self):
        string = ""
        string += "".join([str(task) for task in self.tasks])
        return string


class DETask(object):

    """Python object for DE tasks."""

    def __init__(self, cur_task):
        self.name = cur_task.get("name")
        self.dataset_file = cur_task.get("dataset_file")
        self.TOT_GEN = cur_task.get("TOT_GEN")
        self.GEN_STEP = cur_task.get("GEN_STEP")
        self.TYPE = cur_task.get("TYPE")
        self.F = cur_task.get("F")
        self.NP = cur_task.get("NP")
        self.de_types = cur_task.get("de_types")
        self.CR = cur_task.get("CR")
        self.levels = cur_task.get("levels")

    def __repr__(self):
        string = """+++++ Task ({}) +++++
+ dataset[{}] -> {}
+ tot. gen.   -> {}
+ get. step.  -> {}
+ F  -> {}
+ NP -> {}
+ CR -> {}
+ DE types -> {}
+ levels -> {}
+++++""".format(
            self.name,
            self.TYPE,
            self.dataset_file,
            self. TOT_GEN,
            self.GEN_STEP,
            self.F,
            self.NP,
            self.CR,
            self.de_types,
            self.levels
        )
        return string


class TaskEncoder(json.JSONEncoder):

    def __init__(self, *args, **kwargs):
        super(TaskEncoder, self).__init__(*args, **kwargs)

    def default(self, obj):
        """JSON encoding function for the single task."""
        new_obj = {
            'name': obj.name,
            'TYPE': obj.TYPE,
            'dataset_file': obj.dataset_file,
            'TOT_GEN': obj.TOT_GEN,
            'GEN_STEP': obj.GEN_STEP,
            'F': obj.F,
            'NP': obj.NP,
            'CR': obj.CR,
            'de_types': obj.de_types,
            'levels': obj.levels
        }

        return new_obj


class TaskDecoder(json.JSONDecoder):

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
            # Check end of multiline comment
            elif self.__state == 5:
                # Start ending of the comment
                if char == "*":
                    self.__state = 6
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
