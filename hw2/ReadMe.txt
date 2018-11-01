hw2_best.sh: bash hw2_best.sh [train_x file] [train_y file] [test_x file] [output file]
hw2_Dis.py: 根據 ./modelDiscriminative.npy 生成 [output file]

hw2.sh: bash hw2.sh [train_x file] [train_y file] [test_x file] [output file]
hw2_Gen.py: 根據 ./modelGenerative.npy 生成 [output file]

train_best.sh: bash train_best.sh [train_x file] [train_y file]
train_best.py: 生成 ./modelDiscriminative.npy

train.sh: bash train.sh [train_x file] [train_y file] [test_x file]
train.py: 生成 ./modelGenerative.npy
