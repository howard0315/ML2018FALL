hw4_test.sh<br />		
* 執行test_RNN.py,<br />
* 執行方式：bash hw4_test.sh <test_x file> <dict.txt.big file> <output file><br />
hw4_train.sh<br />
* 執行train_RNN.py, 但是是載入已經訓練好的embedding_matrix與word2idx<br />
* 執行方式：bash hw4_train.sh <train_x file> <train_y file> <test_x file> <dict.txt.big file><br />
hw4_train_word2vec.sh<br />
* 執行train_word2vec.py, 重新訓練embedding_matrix與word2idx，但是word vector的排序就會改變<br />
* 執行方式：bash hw4_train_word2vec.sh <train_x file> <test_x file> <dict.txt.big file><br />
test_RNN.py<br />
* 產生測試資料<br />
train_RNN.py<br />
* 訓練模式<br />
train_word2vec.py<br />
* 重新訓練embedding_matrix與word2idx<br />
