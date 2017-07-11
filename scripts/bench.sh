#!/bin/bash

python gd.py --dataset=../datasets/JULY_MNIST_0_540x100_1s.gz --batch_size=100 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_0.json &&
python gd.py --dataset=../datasets/JULY_MNIST_1_540x100_1s.gz --batch_size=100 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_1.json &&

python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_0_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_0.json &&
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_1.json 