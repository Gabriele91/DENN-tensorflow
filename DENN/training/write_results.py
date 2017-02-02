from matplotlib import pyplot as plt
from . plotter import my_plot
from . plotter import my_confusion_matrix
from . tasks import TaskEncoder
from os import makedirs
from os import path
from math import pi as PI
import colorsys
import json
import numpy as np
from time import time

__all__ = ['write_all_results', 'expand_results']


def expand_results(results, gen_step, de_types):
    """Duplicate the value of the accuracy for the entire range of time.

    From:
        |
        |      x
        | x         x
        |________________

    To:
        |
        |      xxxxxx
        | xxxxxx    xxxxx
        |________________
    """
    for de_type in de_types:
        tmp = []
        for res in results[de_type].values:
            tmp.extend([res for _ in range(gen_step)])

        results[de_type].values = tmp


def norm_rgb_2_hex(colors):
    return "#{:02X}{:02X}{:02X}".format(
        *[
            int(color * 255)
            for color in colors
        ]
    )


def gen_folder_name(name, num_gen, num_batches, levels):
    return "{}_gen[{}]_batches[{}]{}".format(
        name, num_gen, num_batches,
        "".join(
            [
                "_L{}x{}+{}".format(
                    level.shape[0][0],
                    level.shape[0][1],
                    level.shape[1][0]
                )
                for level in levels
            ]
        )
    )


def gen_folder_name_by_time(name):
    return "{}_{}".format(
        name,
        int(time())
    )


def write_all_results(name, results, description, out_options, showDelimiter=False, folderByTime=True):
    color_delta = 1.0 / len(results)
    colors = [
        norm_rgb_2_hex(colorsys.hsv_to_rgb(idx * color_delta, 1, 1))
        for idx in range(len(results))
    ]

    if folderByTime:
        BASE_FOLDER = gen_folder_name_by_time(name)
    else:
        BASE_FOLDER = gen_folder_name(
            name, out_options.job.TOT_GEN, out_options.num_batches, out_options.job.levels)

    makedirs("./benchmark_results", exist_ok=True)
    makedirs(path.join("benchmark_results", BASE_FOLDER), exist_ok=True)

    with open(path.join("benchmark_results", BASE_FOLDER, "job.json"), "w") as job_file:
        json.dump(out_options.job, job_file, cls=TaskEncoder, indent=2)

    figures = []

    all_data = [
        {
            'values': [range(len(result.values)), result.values],
            'color': colors[num],
            'label': name,
            'alpha': 0.9
        }
        for num, (name, result) in enumerate(
            sorted(results.items())
        )
    ]
    figures.append(
        {
            'data': all_data,
            'title': name,
            'type': "plot",
            'axis': (0, len(all_data[0]['values'][0]), 0.0, 1.0),
            'filename': path.join("benchmark_results", BASE_FOLDER, name),
            'plot': {
                'x_label': "generation",
                'y_label': "accuracy",
            }
        }
    )

    for num, (method_name, result) in enumerate(sorted(results.items())):
        figures.append(
            {
                'data': [
                    {
                        'values': [range(len(result.values)), result.values],
                        'color': colors[0],
                        'label': method_name,
                        'alpha': 0.9
                    }
                ],
                'title': method_name,
                'type': "plot",
                'axis': (0, len(result.values), 0.0, 1.0),
                'filename': path.join("benchmark_results", BASE_FOLDER,
                                      "{}_{}".format(name, method_name.replace("/", "_"))),
                'plot': {
                    'x_label': "generation",
                    'y_label': "accuracy",
                }
            }
        )

    for figure in figures:
        fig = plt.figure()
        fig.suptitle(figure['title'], fontsize=14, fontweight='bold')

        if figure['type'] == 'plot':
            print("+ Generating {} [{}] -> {}".format(
                figure['title'],
                figure['type'],
                figure['filename']
            ))
            my_plot(fig, figure['data'])
            if 'axis' in figure:
                plt.axis(figure['axis'])
            plt.xlabel(figure['plot']['x_label'])
            plt.ylabel(figure['plot']['y_label'])
            plt.grid(True)
            plt.figtext(.39, -.02, description)

            plt.legend(bbox_to_anchor=(1.32, 1.0))

            if showDelimiter:
                delimiters = [50, 100, 200, 400]
                for delimiter in delimiters:
                    plt.axvline(delimiter, color='k')

            plt.savefig(figure['filename'], dpi=400, bbox_inches='tight')

    if out_options.job.confusionM is not None:
        fig = plt.figure()
        fig.suptitle(figure['title'], fontsize=14, fontweight='bold')
        my_confusion_matrix(fig, np.array(out_options.job.confusionM))
        plt.savefig(
            figure['filename'] + "_confusion_M", dpi=400, bbox_inches='tight')
