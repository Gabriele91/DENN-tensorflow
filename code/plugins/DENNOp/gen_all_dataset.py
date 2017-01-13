import DENN


def main():
    DEBUG = False

    datasets = [
        ("../../../minimal_dataset/data/bezdekIris", "load_iris_data", 3, "iris", 5, True, "double"),
        ("../../../minimal_dataset/data/letter-recognition", "load_letter_data", 20, "letter", 5, True, "double"),
        ("../../../minimal_dataset/data/MNIST", "load_mnist_data", 200, "mnist", 5, False, "double"),
        ("../../../minimal_dataset/data/MNIST", "load_mnist_data", 1000, "mnist", 5, False, "double"),
        ("../../../minimal_dataset/data/MNIST", "load_mnist_data", 2000, "mnist", 5, False, "double"),
        ("../../../minimal_dataset/data/MNIST", "load_mnist_data", 4000, "mnist", 5, False, "double")
        # FLOAT
        ("../../../minimal_dataset/data/MNIST", "load_mnist_data", 2000, "mnist_f", 5, False, "float")
        ("../../../minimal_dataset/data/MNIST", "load_mnist_data", 4000, "mnist_f", 5, False, "float")
    ]

    for dataset, loader, size, out_name, n_shuffle, batch_size, type_ in datasets:
        DENN.training.create_dataset(dataset, loader, size, out_name, n_shuffle=n_shuffle, batch_size=batch_size, type_=type_)

if __name__ == '__main__':
    main()
