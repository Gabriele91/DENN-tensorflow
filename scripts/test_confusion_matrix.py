import DENN
from matplotlib import pyplot as plt
from DENN.training.plotter import my_confusion_matrix
import numpy as np

matrix = [
    [
        733,
        0,
        13,
        37,
        14,
        67,
        89,
        3,
        8,
        16
    ],
    [
        1,
        927,
        41,
        16,
        37,
        17,
        10,
        0,
        83,
        3
    ],
    [
        24,
        70,
        489,
        86,
        47,
        8,
        103,
        32,
        160,
        13
    ],
    [
        11,
        9,
        121,
        560,
        4,
        131,
        15,
        38,
        74,
        47
    ],
    [
        13,
        1,
        59,
        34,
        585,
        44,
        47,
        17,
        22,
        160
    ],
    [
        32,
        8,
        24,
        131,
        72,
        426,
        26,
        3,
        145,
        25
    ],
    [
        80,
        5,
        52,
        28,
        88,
        19,
        595,
        0,
        78,
        13
    ],
    [
        6,
        27,
        23,
        29,
        33,
        30,
        11,
        752,
        37,
        80
    ],
    [
        15,
        36,
        25,
        93,
        108,
        107,
        18,
        26,
        460,
        86
    ],
    [
        26,
        7,
        32,
        27,
        143,
        37,
        9,
        132,
        84,
        512
    ]
]

def main():
    fig = plt.figure()
    fig.suptitle("test", fontsize=14, fontweight='bold')
    my_confusion_matrix(fig, np.array(matrix))
    plt.savefig(
        "test_out", dpi=400, bbox_inches='tight')

if __name__ == '__main__':
    main()