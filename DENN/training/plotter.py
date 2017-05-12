from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from math import cos, pi
import json
from os import path


def plot_3d_function(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=True)

    # Customize the z axis.
    # ax.set_zlim(0.0, 1.0)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.scatter(X, Y, Z)

    plt.show()


def NN_DE_mnist_loader(filename):
    with open(filename, "r") as input_file:
        res = json.load(input_file)
    return [np.array(res['W']), np.array(res['b'])]


def NN_DE_result_loader(filename, method='rand/1/bin'):
    with open(filename, "r") as input_file:
        res = json.load(input_file)
    return res['results'][method]['best_of']['individual']


def NN_to_image(test_result_file, shape, level=0, loader="DE", method='rand/1/bin', cmap="gray"):
    if loader == "DE":
        individual = NN_DE_result_loader(test_result_file, method)
        W = np.array(individual[0 + (level * 2)])
        b = np.array(individual[1 + (level * 2)])
    elif loader == "gradient":
        W, b = NN_DE_mnist_loader(test_result_file)
    else:
        raise Exception("Error: Unknown loader...")

    W = W + b

    columns = np.hsplit(W, W.shape[-1])

    ##
    # Reshape
    images = []
    for image in columns:
        # print(image.shape)
        images.append(image.reshape(shape))

    ##
    # Normalization
    for idx, _ in enumerate(images):
        max_ = np.max(images[idx])
        min_ = np.min(images[idx])
        images[idx] = images[idx] - min_
        images[idx] = images[idx] / (max_ - min_)
        # print(image)

    images = np.array(images)
    print(images.shape)

    fig = plt.figure()

    for num, image in enumerate(images):
        sub_plt = plt.subplot(2, 5, num + 1)
        sub_plt.set_title(str(num), fontsize=24)
        plt.imshow(image, cmap=plt.cm.Greys if cmap=="gray" else plt.cm.coolwarm,
                   interpolation='nearest')

    plt.tight_layout()
    plt.show()
    plt.close()


def my_confusion_matrix(fig, matrix):
    # print(matrix)
    plt.clf()
    norm_conf = []
    for i in matrix:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    res = plt.imshow(norm_conf, cmap=plt.cm.jet,
                     interpolation='nearest')

    width, height = matrix.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(matrix[x][y]), xy=(y, x),
                         horizontalalignment='center',
                         verticalalignment='center')

    cb = fig.colorbar(res)

    plt.xticks(range(width), [str(num) for num in range(width)])
    plt.yticks(range(height), [str(num) for num in range(height)])


def my_filter(string, rules):
    tmp = string.split("\t")

    try:
        tmp = [float(val) for val in tmp]
    except ValueError:
        return False

    return [type_(tmp[index]) for index, type_ in rules]


def extract(filename, data_filter, rules):
    values = [[] for _ in range(len(rules))]

    with open(filename, 'r') as data:
        for line in data.readlines():
            tmp = data_filter(line, rules)
            if tmp:
                for pos, value in enumerate(tmp):
                    values[pos].append(value)

    return values


def dispersion(origin, target):
    tmp = []
    for index, value in enumerate(target[0]):
        diff = value - origin[0][index]
        try:
            tmp.append((diff * 100.) / abs(origin[0][index]))
        except ZeroDivisionError:
            tmp.append(0)

    return tmp


def my_plot(fig, data):

    for points in data:
        x, y = points['values']
        plt.plot(x, y,
                 color=points['color'],
                 label=points['label'],
                 alpha=points.get('alpha', 1.0))

    plt.legend()


def to_percent(y, position):
    return "{:.02f}%".format(y * 100.)


def my_hist(fig, data, bins_, range_, colors, labels, normalized=False, max_y=None):
    plt.hist(data,
             bins=bins_, range=range_,
             color=colors, label=labels,
             alpha=0.81, normed=normalized,
             )

    plt.legend(prop={'size': 10})
    x1, x2, y1, y2 = plt.axis()

    if max_y:
        plt.axis((x1, x2, 0, max_y))

    if normalized:
        formatter = FuncFormatter(to_percent)
        plt.gca().yaxis.set_major_formatter(formatter)


def plot_results(config_file, save=False, pdf=False, latex_backend=False, show=True):
    """Plot a result graph.

    Params:
        config_file (dict or string): the config dictionary or the json file to
                                      parse

    Config example:
        config_file: {
            "title": "example",
            "x_label": "X",
            "y_label": "Y",
            "gen_step": 1000,
            "sorted": true,
            "max_step": 60000,
            "results": {
                "0": {
                    "values": "@file.json->results|rand/1/bin|best_of|accuracy;",
                    "label": "rand/1/bin",
                    "alpha": 1.0,
                    "markersize": 5,
                    "marker": "o",
                    "color": "#888888"
                },
                "1": {
                    "values": [
                        0.1,
                        0.2,
                        ...
                        0.5
                    ],
                    "label": "rand/1/bin ADA",
                    "alpha": 0.9,
                    "markersize": 5,
                    "marker": "o",
                    "color": "black",
                    "linewidth": 2
                }
            },
            "markevery": 2000,
            "legend_ancor": [
                1.0,
                0.2
            ]
        }
    """
    if latex_backend:
        mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)

    MARKERS = ['o', '^', '*', 's', '+', 'v']
    COLORS = [
        "#000000",
        "#999999",
        "#222222",
        "#555555",
        "#AAAAAA",
        "#CCCCCC"
    ]
    ALPHA = [
        0.2,
        0.4,
        0.6,
        0.8,
        1.0
    ]
    LINESTYLE = [":", "--", "-", "-.", "steps", ":"]

    if type(config_file) != dict:
        base_folder = path.dirname(path.abspath(config_file))
        with open(config_file) as result_file:
            config_file = json.load(result_file)
            for name, obj in config_file['results'].items():
                if obj['values'][0] == "@" and obj['values'][-1] == ";":
                    string_to_parse = obj['values'][1:-1]
                    file_to_import, keys = string_to_parse.split("->")
                    keys = keys.split("|")
                    file_path = path.join(base_folder, file_to_import)
                    with open(file_path) as cur_file:
                        res = json.load(cur_file)
                        for key in keys:
                            res = res[key]
                        config_file['results'][name]['values'] = res

    fig = plt.figure()
    fig.suptitle(config_file.get('title', ''), fontsize=12, fontweight='bold')

    data = config_file.get('results')
    labels = []

    if config_file.get("sorted", False):
        all_data = enumerate(sorted(data.items(), key=lambda elm: int(elm[0])))
    else:
        all_data = enumerate(data.items())

    for idx, (type_, obj) in all_data:
        gen_step = config_file.get("gen_step", 1)
        if config_file.get("max_step", False):
            _y_ = obj.get('values')[
                :int(config_file.get("max_step") / gen_step)]
        else:
            _y_ = obj.get('values')
        _x_ = range(len(_y_))
        tot_gen = (len(_y_) - 1) * gen_step

        x_real = range(tot_gen)
        y_real = []

        for _n_, val in enumerate(_y_[:-1]):
            next_ = _y_[_n_ + 1]
            y_real.append(val)
            for cur_step in range(gen_step - 1):
                ##
                # Cos interpolation
                alpha = float((cur_step + 1.) / gen_step)
                alpha2 = (1 - cos(alpha * pi)) / 2
                new_point = (val * (1 - alpha2) + next_ * alpha2)
                y_real.append(
                    new_point
                )

        ##
        # Do lines and point
        cur_plot = plt.plot(
            x_real, y_real,
            marker=obj.get('marker', MARKERS[idx % len(MARKERS)]),
            markersize=obj.get('markersize', None),
            color=obj.get('color', COLORS[idx % len(COLORS)]),
            linewidth=obj.get('linewidth', 1),
            #  ls=LINESTYLE[idx],
            alpha=obj.get('alpha', ALPHA[idx % len(ALPHA)]),
            label=obj.get('label'),
            markevery=config_file.get(
                "markevery", [int(elm * gen_step) for elm in _x_[:-1]])
        )
        labels.append(cur_plot[0])

    plt.legend(
        handler_map=dict(
            [
                (label, HandlerLine2D(numpoints=1))for label in labels
            ]
        ),
        bbox_to_anchor=config_file.get("legend_ancor", (1.0, 1.0)),
        fontsize=12
    )

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    plt.axis((0, tot_gen, 0, 1))
    plt.xlabel(config_file.get('x_label', 'Generations'), fontsize=12)
    plt.ylabel(config_file.get('y_label', 'Accuracy'), fontsize=12)
    plt.grid(True)

    if save:
        plt.savefig("{}.png".format(save), dpi=600, bbox_inches='tight')
        print("+ out file -> {}.png".format(save))
        if pdf:
            plt.savefig("{}.pdf".format(save), dpi=600, bbox_inches='tight')
            print("+ out file -> {}.pdf".format(save))
    if show:
        plt.show()
    plt.close()
