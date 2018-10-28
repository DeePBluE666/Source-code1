
#The input data without noise
#Model:DCT DeepAE CAE
#Metric:MSE & cw-ssim

import numpy as np
import pandas as pd
from PIL import Image
import os
from skimage import io
from matplotlib import pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
import ssim.ssimlib as pyssim
from skimage.measure import compare_ssim as ssim
import cv2

#-------------------------------------get data-------------------------------------#
path = '/home/dataset/AE dataset/newdata//'
files = os.listdir(path)
#files.remove('@eaDir')

def get_data():
	train_data = []
	for a in files[:10]:
		print(a)
		files_1 = os.listdir(path + a)
		try:
				files_1.remove('@eaDir')
		except:
				files_1 = files_1
		for b in files_1:
			#print(b)
			files_2 = os.listdir(path + a + '//' + b)
			try:
				files_2.remove('@eaDir')
			except:
				files_2 = files_2
			for c in files_2:
				try:
					temp = Image.open(path + a + '//' + b + '//' + c)
					try:
						train_data = train_data + [cv2.resize(np.array(temp)[:, :, 0], (50, 60))]
					except:
						print('error')
				except:
					print('ERROR')
		print(len(train_data))

	train_data = np.array(train_data)
	train_data = train_data.reshape([len(train_data), 50, 60, 1])

	val_data = []
	for a in files[10:]:
		print(a)
		files_1 = os.listdir(path + a)
		try:
				files_1.remove('@eaDir')
		except:
				files_1 = files_1
		for b in files_1:
			#print(b)
			files_2 = os.listdir(path + a + '//' + b)
			try:
				files_2.remove('@eaDir')
			except:
				files_2 = files_2
			for c in files_2:
				try:
					temp = Image.open(path + a + '//' + b + '//' + c)
					try:
						val_data = val_data + [cv2.resize(np.array(temp)[:, :, 0], (50, 60))]
					except:
						print('error')
				except:
					print('ERROR')
		print(len(val_data))

	val_data = np.array(val_data)
	val_data = val_data.reshape([len(val_data), 50, 60, 1])
	
	return train_data, val_data
	
train_data, val_data = get_data()

#---------------------------------DCT-------------------------------------#
def DCT(image, dim = 30):
    
    image1 = image.astype('float') 
    img_dct = cv2.dct(image1)
    
    if dim == 30:
        a = 5
        b = 6
    elif dim == 60:
        a = 10
        b = 6
    else:
        a = 100
        b = 100
        
    recor_temp = np.zeros(image1.shape)
    recor_temp[0:a,0:b] = img_dct[0:a,0:b]
    
    img_recor1 = cv2.idct(recor_temp)
    
    return img_recor1

#dim = 30
similarity_1 = []
MSE = []
for i in range(val_data.shape[0]):
    image1 = val_data[i].reshape(50, 60, 1)[:, :, 0]
    image1 = image1.astype('float') 
    p1 = Image.fromarray(image1).convert('L')
    p2 = Image.fromarray(DCT(image1, dim = 30)).convert('L')
    similarity_temp_1 = pyssim.SSIM(p1).cw_ssim_value(p2)
    similarity_1.append([similarity_temp_1])
    
    mse_ = mean_squared_error(image1, DCT(image1, dim = 30))
    
    MSE.append(mse_)
    
print('cw-ssim: %s'%np.mean(similarity_1))
print('MSE: %s'%np.mean(MSE))

#dim = 60
for i in range(val_data.shape[0]):
    image1 = val_data[i].reshape(50, 60, 1)[:, :, 0]
    image1 = image1.astype('float') 
    p1 = Image.fromarray(image1).convert('L')
    p2 = Image.fromarray(DCT(image1, dim = 60)).convert('L')
    similarity_temp_1 = pyssim.SSIM(p1).cw_ssim_value(p2)
    similarity_1.append([similarity_temp_1])
    
    mse_ = mean_squared_error(image1, DCT(image1, dim = 60))
    
    MSE.append(mse_)
    
print('cw-ssim: %s'%np.mean(similarity_1))
print('MSE: %s'%np.mean(MSE))

#--------------------------------DeepAE------------------------------------#
train_data = train_data.reshape([len(train_data), 3000])
test_data = test_data.reshape([len(test_data), 3000])

#dim = 30

input_img = Input(shape=(3000,))
encoded = Dense(1000, activation='sigmoid')(input_img)
encoded = Dense(500, activation='sigmoid')(encoded)
encoded = Dense(250, activation='sigmoid')(encoded)

encoded = Dense(30, activation='sigmoid')(encoded)

decoded = Dense(250, activation='sigmoid')(encoded)
decoded = Dense(500, activation='sigmoid')(decoded)
decoded = Dense(1000, activation='sigmoid')(decoded)
decoded = Dense(3000, activation='linear')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(train_data, train_data,
                epochs=200,
                batch_size=32,
                shuffle=True,
                validation_data=(test_data, test_data))
				

decoded_imgs = autoencoder.predict(test_data)


similarity_1 = []
for i in range(test_data.shape[0]):
    p1 = Image.fromarray(test_data[i].reshape(50, 60, 1)[:, :, 0]).convert('L')
    p2 = Image.fromarray(decoded_imgs[i].reshape(50, 60, 1)[:, :, 0]).convert('L')
    similarity_temp_1 = pyssim.SSIM(p1).cw_ssim_value(p2)
    similarity_1.append([similarity_temp_1])
    
print('cw-ssim: %s'%np.mean(similarity_1))


#dim = 60

input_img = Input(shape=(3000,))
encoded = Dense(1000, activation='sigmoid')(input_img)
encoded = Dense(500, activation='sigmoid')(encoded)
encoded = Dense(250, activation='sigmoid')(encoded)

encoded = Dense(60, activation='sigmoid')(encoded)

decoded = Dense(250, activation='sigmoid')(encoded)
decoded = Dense(500, activation='sigmoid')(decoded)
decoded = Dense(1000, activation='sigmoid')(decoded)
decoded = Dense(3000, activation='linear')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(train_data, train_data,
                epochs=200,
                batch_size=32,
                shuffle=True,
                validation_data=(test_data, test_data))
				

decoded_imgs = autoencoder.predict(test_data)


similarity_1 = []
for i in range(test_data.shape[0]):
    p1 = Image.fromarray(test_data[i].reshape(50, 60, 1)[:, :, 0]).convert('L')
    p2 = Image.fromarray(decoded_imgs[i].reshape(50, 60, 1)[:, :, 0]).convert('L')
    similarity_temp_1 = pyssim.SSIM(p1).cw_ssim_value(p2)
    similarity_1.append([similarity_temp_1])
    
print('cw-ssim: %s'%np.mean(similarity_1))


#-------------------------------CAE----------------------------------------#

train_data, val_date = get_data()

train_data = train_data.reshape([len(train_data), 50, 60, 1])
val_data = val_data.reshape([len(val_date), 50, 60, 1])

#dim = 1*(5,6)
input_img = Input(shape=(50, 60, 1))  

x = Conv2D(32, (8, 8), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 1), padding='same')(x)
x = Conv2D(16, (6, 6), activation='relu', padding='same')(x)
x = Conv2D(1, (4, 4), activation='relu', padding='same')(x)
encoded = MaxPooling2D((5, 5), padding='same')(x)

x = Conv2D(1, (4, 4), activation='relu', padding='same')(encoded)
x = UpSampling2D((5, 5))(x)
x = Conv2D(16, (6, 6), activation='relu', padding='same')(x)
x = Conv2D(32, (8, 8), activation='relu', padding='same')(x)
x = UpSampling2D((2, 1))(x)
decoded = Conv2D(1, (5, 5), activation='relu', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(train_data, train_data,
                epochs=200,
                batch_size=32,
                shuffle=True,
                validation_data=(val_data, val_data))
				
decoded_imgs = autoencoder.predict(val_data)
val_data = val_data.astype('float32')

similarity_1 = []
similarity_2 = []
for i in range(val_data.shape[0]):
    p1 = Image.fromarray(val_data[i].reshape(50, 60, 1)[:, :, 0]).convert('L')
    p2 = Image.fromarray(decoded_imgs[i].reshape(50, 60, 1)[:, :, 0]).convert('L')
    similarity_temp_1 = pyssim.SSIM(p1).cw_ssim_value(p2)
    similarity_1.append([similarity_temp_1])
    
print('cw-ssim: %s'%np.mean(similarity_1))


#dim = 2*(5,6)

input_img = Input(shape=(50, 60, 1))  

x = Conv2D(32, (8, 8), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 1), padding='same')(x)
x = Conv2D(16, (6, 6), activation='relu', padding='same')(x)
x = Conv2D(2, (4, 4), activation='relu', padding='same')(x)
encoded = MaxPooling2D((5, 5), padding='same')(x)

x = Conv2D(2, (4, 4), activation='relu', padding='same')(encoded)
x = UpSampling2D((5, 5))(x)
x = Conv2D(16, (6, 6), activation='relu', padding='same')(x)
x = Conv2D(32, (8, 8), activation='relu', padding='same')(x)
x = UpSampling2D((2, 1))(x)
decoded = Conv2D(1, (5, 5), activation='relu', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(train_data, train_data,
                epochs=200,
                batch_size=32,
                shuffle=True,
                validation_data=(val_data, val_data))
				
decoded_imgs = autoencoder.predict(val_data)
val_data = val_data.astype('float32')

similarity_1 = []
similarity_2 = []
for i in range(val_data.shape[0]):
    p1 = Image.fromarray(val_data[i].reshape(50, 60, 1)[:, :, 0]).convert('L')
    p2 = Image.fromarray(decoded_imgs[i].reshape(50, 60, 1)[:, :, 0]).convert('L')
    similarity_temp_1 = pyssim.SSIM(p1).cw_ssim_value(p2)
    similarity_1.append([similarity_temp_1])
    
print('cw-ssim: %s'%np.mean(similarity_1))