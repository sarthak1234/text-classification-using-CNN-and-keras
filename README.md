# text-classification-using-CNN-and-keras
a deep learning convolutional neural network which provides sentiment analysis of the text
used for training data scrapped from various reviews  hotel booking websites
the data scrapped was stemmed, lemmatized and key phrases were extracted (data helpers.py)
the data was then vectorized by building a vocabulary and mapping words to numbers
the training network consisted of an initial embedded layer which used skip gram model to vectorize the data
further 3 convolutional layers were used along with max pooling and l2_loss regularization to train the data

