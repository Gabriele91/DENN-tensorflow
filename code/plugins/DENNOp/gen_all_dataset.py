import DENN
import dataset_loaders


def main():
    DEBUG = False

    datasets = [
        ("../../../minimal_dataset/data/bezdekIris", "load_iris_data", 3, "iris"),
        ("../../../minimal_dataset/data/letter-recognition", "load_letter_data", 20, "letter"),
        ("../../../minimal_dataset/data/MNIST", "load_mnist_data", 12, "mnist")
    ]

    for dataset, loader, batch_size, out_name in datasets:
        data, labels = getattr(dataset_loaders, loader)(dataset, DEBUG)
        DENN.training.create_dataset(out_name, data, labels, batch_size)

if __name__ == '__main__':
    main()
