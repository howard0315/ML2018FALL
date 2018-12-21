wget -O "./WordEmbedding/embedding_matrix.pickle" "https://drive.google.com/uc?export=download&id=12cT-8atp25FhC05UNFqWi06Rs-0Dw5oZ"
python3 train_RNN.py $1 $2 $3 $4
