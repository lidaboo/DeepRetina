# A helper function to read and manipulate retianl images

# import tensorflow as tf
import numpy as np
import random
import os
import sys
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt

# wrap_counting
from wrap_counting import sampler

UNET_PATH = os.getcwd() + "/../"
sys.path.append(UNET_PATH)

DATA_PATH = os.getcwd() + "/../../RetinalDataJohn"
# DATA_PATH = os.getcwd() + "/../sampleData"
SEED = 1234

class DataProvider():

	def __init__(self, validationSize = 20, batchSize = 1):
		# super(DataProvider, self).__init__(a_min, a_max)
		# metaData dict
		self.trainData = None
		self.testData = None
		self.validData = None
		self.n_class = 2
		self.a_min = 0
		self.a_max = 255
		self.validationSize = validationSize
		self.trainSize = None
		self.batchSize = batchSize
		self.sampler = None
		self.channels = 1
		self.n_class = 2


	def __createMetaDataDict(self, path):

		files = os.listdir(path)
		# print(files)
		# metaData = {image:GT}
		metaData = {}
		images = []
		GTs = []
		# create a list of images and GTs
		for file in files:
			# print(file.split("-")[0])
			if file.split("-")[0] != "GT":
				images.append((file,file.split("_")[-1]))
			else:
				GTs.append((file,file.split("_")[-1]))

		# metaData = {image:GT}
		for img in images:
			for g in GTs:
				if g[1] == img[1]:
					metaData[img[0]] = g[0]

		return metaData

	def createAugmentedData(self, metaDataDict, dataPath):

		augDataImg = []
		augDataGt = []
		for aImg, aGt in metaDataDict.items():
			img = misc.imread(dataPath +"/" + aImg) # read image
			gt = misc.imread(dataPath +"/" + aGt) # read its GT
			augDataImg.append(img)
			augDataGt.append(gt)
			augImg, augGt = self.__augmentData(img, gt)
			for i in range(len(augImg)):
				augDataImg.append(augImg[i])
				augDataGt.append(augGt[i])

			del img, gt, augImg, augGt

		return augDataImg, augDataGt

	def __augmentData(self, img, gt):
		augImg = []
		augGt = []

		# flip up-down
		augImg.append(np.flipud(img))
		augGt.append(np.flipud(gt))

		# flip right-left
		augImg.append(np.fliplr(img))
		augGt.append(np.fliplr(gt))

		# rotate 90, 180 and 270 clockwise
		for i in range(1,4):
			augImg.append(ndimage.rotate(img, i*90))
			augGt.append(ndimage.rotate(gt, i*90))

		return augImg,augGt

	def readData(self):
		# train_path = DATA_PATH + "/test"
		train_path = DATA_PATH + "/train"
		trainMetaData = self.__createMetaDataDict(train_path)
		self.trainData = self.createAugmentedData(trainMetaData, train_path)
		print("done reading data")
		# extract validation data
		self.validData = self.__createValidationData()
		# get train size
		self.trainSize = len(self.trainData[0])
		# print("size= ", self.trainSize)
		# create sampler to get samples from train data
		self.sampler = sampler(self.batchSize, self.trainSize, seed = SEED)
		return
		# return self.trainData

	def __createValidationData(self):

		trainSize = len(self.trainData[0])
		randInt = random.sample(range(trainSize), self.validationSize)
		validImg = []
		validGT = []
		for r in randInt:
			validImg.append(self.trainData[0][r])
			validGT.append(self.trainData[1][r])
		# pop validation data from train data
		tempImg = []
		tempGT = []
		for i in range(trainSize):
			if i not in randInt:
				tempImg.append(self.trainData[0][i])
				tempGT.append(self.trainData[1][i])
		
		self.trainData = (tempImg, tempGT)
		return validImg, validGT

	def __processLabels(self, label):
		# crop
		label = self.__cropImage(label)
		# 
		nx = label.shape[1]
		ny = label.shape[0]
		# label = self.__normalize(label)
		# print(label.dtype)
		labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
		labels[..., 1] = self.__normalize(label)
		labels[..., 0] = self.__normalize(~label)
		return labels

	def __processData(self, data):
		# crop
		data = self.__cropImage(data)
        # normalization
		data = self.__normalize(data)
		return np.reshape(data, (data.shape[0], data.shape[1], self.channels))

	def __normalize(self,data):
		data = np.clip(np.fabs(data), self.a_min, self.a_max)
		data -= np.amin(data)
		data /= np.amax(data)
		return data

	def __cropImage(self,data):
		m, n = data.shape
		data = data[m/4:-m/4,n/4:-n/4]
		return data

	def __call__(self):
		
		# print(self.sampler.getOrder())
		nextIdx = self.sampler.next_inds()
		# print(nextIdx)
		# train_data, labels = self._load_data_and_label()
		nx = self.trainData[0][0].shape[0]/2
		ny = self.trainData[0][0].shape[1]/2
		X = np.zeros((self.batchSize, nx, ny, self.channels))
		Y = np.zeros((self.batchSize, nx, ny, self.n_class))
		# print("tsize= ", type(self.trainData[0][66]))
		for idx, val in enumerate(nextIdx):
			X[idx] = 	self.__processData(self.trainData[0][val])
			Y[idx] =	self.__processLabels(self.trainData[1][val])

		# print(type(X))
		return X, Y

	def getTrainSize(self):
		return self.trainSize
			
def main():
	dp = DataProvider(batchSize = 10)
	dp.readData()
	x ,y = dp()
	print(np.max(x))
	print(np.max(y))
	# sanity check
	# print(x.shape)
	# print(x.dtype)
	# print(y.shape)
	# print(y.dtype)
	# fig, ax = plt.subplots(2, 2)
	# ax[0][0].imshow(x[1,:,:,0],cmap=plt.cm.gray)
	# ax[1][0].imshow(y[1,:,:,1],cmap=plt.cm.gray)
	# ax[0][1].imshow(x[0,:,:,0],cmap=plt.cm.gray)
	# ax[1][1].imshow(y[0,:,:,1],cmap=plt.cm.gray)
	# plt.show()
	

if __name__ == '__main__':
	main()
