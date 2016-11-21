import DENN
import dataset_loaders


def main():
    DEBUG = False

    datasets = [
        ("../../../minimal_dataset/data/MNIST", "load_mnist_data", "mnist")
    ]

    for dataset, loader, out_name in datasets:
        data, labels = getattr(dataset_loaders, loader)(dataset, DEBUG)
        DENN.training.create_dataset(out_name, data, labels)

if __name__ == '__main__':
    main()
