from keras.layers import Input, Dense, Embedding, merge, Convolution2D, MaxPooling2D, Dropout
from keras.layers.core import Reshape, Flatten
from sklearn.cross_validation import train_test_split
from keras.callbacks import ModelCheckpoint
from data_helpers import load_data, text_to_vector
from keras.optimizers import Adam
from keras.models import Model
import os
from keras.models import load_model
import numpy as np
import tensorflow as tf
import warnings
import json
import datetime, time
import traceback
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

x, y, vocabulary, vocabulary_inv = load_data()

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

sequence_length = X_train.shape[1]
vocabulary_size = len(vocabulary_inv)
embedding_dim = 256
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.2
nb_epoch = 5
batch_size = 30

# this returns a tensor
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(output_dim=embedding_dim, input_dim=vocabulary_size, input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Convolution2D(num_filters, filter_sizes[0], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
conv_1 = Convolution2D(num_filters, filter_sizes[1], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
conv_2 = Convolution2D(num_filters, filter_sizes[2], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)

maxpool_0 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_0)
maxpool_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_1)
maxpool_2 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_2)

merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)
flatten = Flatten()(merged_tensor)
# reshape = Reshape((3*num_filters,))(merged_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(output_dim=2, activation='softmax')(dropout)

# this creates a model that includes

model = Model(input=inputs, output=output)
checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training

# Save this model
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

training_set_length = 'size_'+str(len(X_train))+'_epoch_'+str(nb_epoch)+'_time_'+st

try:
	# Save vocabulary file
	fp = open('./vocabulary/vocab_'+training_set_length+'.json', 'w')
	fp.write(json.dumps({'vocabulary': vocabulary, 'vocabulary_inv': vocabulary_inv}))
	fp.close()

	# Save model file
	model.save('./model/model_'+training_set_length+'.h5')
except:
	traceback.print_exc()