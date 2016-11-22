import DENN


def main():
    DEBUG = False

    datasets = [
        ("../../../minimal_dataset/data/bezdekIris", "load_iris_data", 3, "iris"),
        ("../../../minimal_dataset/data/letter-recognition", "load_letter_data", 20, "letter"),
        ("../../../minimal_dataset/data/MNIST", "load_mnist_data", 12, "mnist")
    ]

    for dataset, loader, batch_size, out_name in datasets:
        DENN.training.create_dataset(dataset, loader, batch_size, out_name)

if __name__ == '__main__':
    main()
