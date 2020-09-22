

import csv
import pandas as pd
import glob
import numpy as np
from numpy import save
from PIL import Image
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.preprocessing import image 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.models import load_model


def read_data():
	train = 'train.rotfaces/train/'
	train_df = pd.read_csv('train.rotfaces/train.truth.csv')


	encoder = LabelEncoder()
	label = encoder.fit_transform(train_df.iloc[:, -1].values)
	value = train_df.iloc[:, 1].values

	train_image = []
	for i in (train_df.iloc[:,0]):
		img = image.load_img(train+ i,target_size=(64,64,3))
		img = image.img_to_array(img)
		img = img/255
		train_image.append(img)

	test_image = []
	fn = []
	for i in glob.glob('test/*.jpg'):
		fn.append(i.split('/')[-1])
		img = image.load_img(i, target_size=(64,64,3))
		img = image.img_to_array(img)
		img = img/255
		test_image.append(img)

	X = np.array(train_image)
	x_test = np.array(test_image)
	Y = to_categorical(label)

	X_train, X_test, y_train, y_test = train_test_split(X, Y,  test_size=0.2)

	return X, X_train, X_test, y_train, y_test, x_test, fn


def cifar10_model(X, X_train, X_test, y_train, y_test):

	batch_size = 32
	epochs = 5

	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same',
					 input_shape=X.shape[1:]))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4))
	model.add(Activation('softmax'))

	# opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

	model.compile(loss='categorical_crossentropy',
				  optimizer='RMSprop',
				  metrics=['accuracy'])

	model.fit(x=X_train,y= y_train, 
				  batch_size=batch_size,
				  epochs=epochs,
				  validation_data=(X_test, y_test),
				  shuffle=True)

	scores = model.evaluate(X_test, y_test, verbose=1)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])
	model.save('cifar10_model.h5')

	model = load_model('cifar10_model.h5')

	return model


def custom_model(X, X_train, X_test, y_train, y_test):

	batch_size = 32
	epochs = 5

	model = Sequential()
	model.add(Conv2D(32, (3,3), input_shape = X.shape[1:], padding = 'same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(32, (3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))

	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dense(4))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])

	model.fit(x=X_train,y= y_train, 
			  batch_size=batch_size,
			  epochs=epochs,
			  validation_data=(X_test, y_test),
			  shuffle=True)


	scores = model.evaluate(X_test, y_test, verbose=1)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])
	model.save('customCnn_model.h5')

	model = load_model('customCnn_model.h5')

	return model


def generate_result(model, x_test, fn, outFile_name, outFolder, numpyOutput):
	y_pred = model.predict_classes(x_test)
	test_label = []
	correct_img = []

	for i in range(len(y_pred)):
		img = Image.fromarray((x_test[i]*255).astype(np.uint8))
		if y_pred[i]==0:
			test_label.append('rotated_left')
			img = img.transpose(Image.ROTATE_270)
	  
		elif y_pred[i]==1:
			test_label.append('rotated_right')
			img = img.transpose(Image.ROTATE_90)
	  
		elif y_pred[i]==2:
			test_label.append('upright')
	  
		else:
			test_label.append('upside_down')
			img = img.transpose(Image.ROTATE_180)
	  
		fl_nm = str(fn[i].split('.')[0])+'.png'
		outImg = os.path.join(outFolder, fl_nm)
		img.save(outImg)
		correct_img.append(np.array(img))

	test_pred = pd.DataFrame(index = fn, data = test_label)
	test_pred.to_csv(outFile_name)
	save(numpyOutput, correct_img)


if __name__ == '__main__':
	X, X_train, X_test, y_train, y_test, x_test, fn  = read_data()
	model_custom = custom_model(X, X_train, X_test, y_train, y_test)
	generate_result(model_custom, x_test, fn, 'custom_test.preds.csv', 'custom_output', 'custom_Imgs.npy')

	

	model_cifar10 = cifar10_model(X, X_train, X_test, y_train, y_test)
	generate_result(model_cifar10, x_test, fn, 'cifar_test.preds.csv', 'cifar10_output', 'cifar10_Imgs.npy')

