#!/bin/bash

# python gd.py --dataset=../datasets/JULY_MNIST_0_540x100_1s.gz --batch_size=100 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_0.json &&

# MNIST 50
python gd.py --dataset=../datasets/JULY_MNIST_1_1080x50_1s.gz --batch_size=50 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_50_run0.json &&
python gd.py --dataset=../datasets/JULY_MNIST_1_1080x50_1s.gz --batch_size=50 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_50_run1.json &&
python gd.py --dataset=../datasets/JULY_MNIST_1_1080x50_1s.gz --batch_size=50 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_50_run2.json &&
python gd.py --dataset=../datasets/JULY_MNIST_1_1080x50_1s.gz --batch_size=50 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_50_run3.json &&
python gd.py --dataset=../datasets/JULY_MNIST_1_1080x50_1s.gz --batch_size=50 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_50_run4.json &&

# MNIST 100
python gd.py --dataset=../datasets/JULY_MNIST_1_540x100_1s.gz --batch_size=100 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_100_run0.json &&
python gd.py --dataset=../datasets/JULY_MNIST_1_540x100_1s.gz --batch_size=100 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_100_run1.json &&
python gd.py --dataset=../datasets/JULY_MNIST_1_540x100_1s.gz --batch_size=100 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_100_run2.json &&
python gd.py --dataset=../datasets/JULY_MNIST_1_540x100_1s.gz --batch_size=100 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_100_run3.json &&
python gd.py --dataset=../datasets/JULY_MNIST_1_540x100_1s.gz --batch_size=100 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_100_run4.json &&

# MNIST 200
python gd.py --dataset=../datasets/JULY_MNIST_1_270x200_1s.gz --batch_size=200 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_200_run0.json &&
python gd.py --dataset=../datasets/JULY_MNIST_1_270x200_1s.gz --batch_size=200 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_200_run1.json &&
python gd.py --dataset=../datasets/JULY_MNIST_1_270x200_1s.gz --batch_size=200 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_200_run2.json &&
python gd.py --dataset=../datasets/JULY_MNIST_1_270x200_1s.gz --batch_size=200 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_200_run3.json &&
python gd.py --dataset=../datasets/JULY_MNIST_1_270x200_1s.gz --batch_size=200 --features=784 --classes=10 --type=float --outfilename=JULY_GD_MNIST_200_run4.json &&

# python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_0_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_0.json &&

# GASS 30 h0
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_370x30_1s.gz --batch_size=30 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_30_run0.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_370x30_1s.gz --batch_size=30 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_30_run1.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_370x30_1s.gz --batch_size=30 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_30_run2.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_370x30_1s.gz --batch_size=30 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_30_run3.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_370x30_1s.gz --batch_size=30 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_30_run4.json && 

# GASS 60 h0
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_60_run0.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_60_run1.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_60_run2.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_60_run3.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_60_run4.json && 

# GASS 120 h0
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_92x120_1s.gz --batch_size=120 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_120_run0.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_92x120_1s.gz --batch_size=120 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_120_run1.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_92x120_1s.gz --batch_size=120 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_120_run2.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_92x120_1s.gz --batch_size=120 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_120_run3.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_92x120_1s.gz --batch_size=120 --features=128 --classes=6 --type=float --outfilename=JULY_GD_GASS_120_run4.json && 

# GASS 30 h1
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_370x30_1s.gz --batch_size=30 --features=128 --classes=6 --type=float --hidden=1 --steps=4000 --outfilename=JULY_GD_GASS_30_h1_run0.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_370x30_1s.gz --batch_size=30 --features=128 --classes=6 --type=float --hidden=1 --steps=4000 --outfilename=JULY_GD_GASS_30_h1_run1.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_370x30_1s.gz --batch_size=30 --features=128 --classes=6 --type=float --hidden=1 --steps=4000 --outfilename=JULY_GD_GASS_30_h1_run2.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_370x30_1s.gz --batch_size=30 --features=128 --classes=6 --type=float --hidden=1 --steps=4000 --outfilename=JULY_GD_GASS_30_h1_run3.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_370x30_1s.gz --batch_size=30 --features=128 --classes=6 --type=float --hidden=1 --steps=4000 --outfilename=JULY_GD_GASS_30_h1_run4.json && 

# GASS 60 h1
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --hidden=1 --steps=4000 --outfilename=JULY_GD_GASS_60_h1_run0.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --hidden=1 --steps=4000 --outfilename=JULY_GD_GASS_60_h1_run1.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --hidden=1 --steps=4000 --outfilename=JULY_GD_GASS_60_h1_run2.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --hidden=1 --steps=4000 --outfilename=JULY_GD_GASS_60_h1_run3.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --hidden=1 --steps=4000 --outfilename=JULY_GD_GASS_60_h1_run4.json && 

# GASS 120 h1
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_92x120_1s.gz --batch_size=120 --features=128 --classes=6 --type=float --hidden=1 --steps=4000 --outfilename=JULY_GD_GASS_120_h1_run0.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_92x120_1s.gz --batch_size=120 --features=128 --classes=6 --type=float --hidden=1 --steps=4000 --outfilename=JULY_GD_GASS_120_h1_run1.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_92x120_1s.gz --batch_size=120 --features=128 --classes=6 --type=float --hidden=1 --steps=4000 --outfilename=JULY_GD_GASS_120_h1_run2.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_92x120_1s.gz --batch_size=120 --features=128 --classes=6 --type=float --hidden=1 --steps=4000 --outfilename=JULY_GD_GASS_120_h1_run3.json &&
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_92x120_1s.gz --batch_size=120 --features=128 --classes=6 --type=float --hidden=1 --steps=4000 --outfilename=JULY_GD_GASS_120_h1_run4.json && 

# GASS 30 h2
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_370x30_1s.gz --batch_size=30 --features=128 --classes=6 --type=float --hidden=2 --steps=8000 --outfilename=JULY_GD_GASS_30_h2_run0.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_370x30_1s.gz --batch_size=30 --features=128 --classes=6 --type=float --hidden=2 --steps=8000 --outfilename=JULY_GD_GASS_30_h2_run1.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_370x30_1s.gz --batch_size=30 --features=128 --classes=6 --type=float --hidden=2 --steps=8000 --outfilename=JULY_GD_GASS_30_h2_run2.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_370x30_1s.gz --batch_size=30 --features=128 --classes=6 --type=float --hidden=2 --steps=8000 --outfilename=JULY_GD_GASS_30_h2_run3.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_370x30_1s.gz --batch_size=30 --features=128 --classes=6 --type=float --hidden=2 --steps=8000 --outfilename=JULY_GD_GASS_30_h2_run4.json &&

# GASS 60 h2
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --hidden=2 --steps=8000 --outfilename=JULY_GD_GASS_60_h2_run0.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --hidden=2 --steps=8000 --outfilename=JULY_GD_GASS_60_h2_run1.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --hidden=2 --steps=8000 --outfilename=JULY_GD_GASS_60_h2_run2.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --hidden=2 --steps=8000 --outfilename=JULY_GD_GASS_60_h2_run3.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_185x60_1s.gz --batch_size=60 --features=128 --classes=6 --type=float --hidden=2 --steps=8000 --outfilename=JULY_GD_GASS_60_h2_run4.json &&

# GASS 120 h2
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_92x120_1s.gz --batch_size=120 --features=128 --classes=6 --type=float --hidden=2 --steps=8000 --outfilename=JULY_GD_GASS_120_h2_run0.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_92x120_1s.gz --batch_size=120 --features=128 --classes=6 --type=float --hidden=2 --steps=8000 --outfilename=JULY_GD_GASS_120_h2_run1.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_92x120_1s.gz --batch_size=120 --features=128 --classes=6 --type=float --hidden=2 --steps=8000 --outfilename=JULY_GD_GASS_120_h2_run2.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_92x120_1s.gz --batch_size=120 --features=128 --classes=6 --type=float --hidden=2 --steps=8000 --outfilename=JULY_GD_GASS_120_h2_run3.json && 
python gd.py --dataset=../datasets/JULY_GasSensorArrayDrift_1_92x120_1s.gz --batch_size=120 --features=128 --classes=6 --type=float --hidden=2 --steps=8000 --outfilename=JULY_GD_GASS_120_h2_run4.json