import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.datasets.mnist import load_data
from skimage.transform import resize
import os
import PIL

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

input1_path = "dataset/img_align_celeba/"
input2_path = "results/random_face/"


def load_image1(file_name):
    img = PIL.Image.open(file_name)
    img = img.crop([25,65,153,193])
    img = img.resize((64,64))
    data = np.asarray(img, dtype="int32" )
    return data

def load_image2(file_name):
    img = PIL.Image.open(file_name)
    img = img.resize((64,64))
    data = np.asarray(img, dtype="int32" )
    return data

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

images1 = np.array(os.listdir(input1_path))
np.random.shuffle(images1)
images1 = images1[:5000]
new_images1 = []
for file_name in images1:
	new_pic = load_image1(input1_path + file_name)
	new_images1.append(new_pic)
images1 = np.array(new_images1)

images2 = np.array(os.listdir(input2_path))
images2 = images2[:5000]
new_images2 = []
for file_name in images2:
	new_pic = load_image2(input2_path + file_name)
	new_images2.append(new_pic)
images2 = np.array(new_images2)

print('Loaded', images1.shape, images2.shape)
# convert integer to floating point values
images1 = images1.astype('float32')
images2 = images2.astype('float32')
# resize images
images1 = scale_images(images1, (299,299,3))
images2 = scale_images(images2, (299,299,3))
print('Scaled', images1.shape, images2.shape)
# pre-process images
images1 = preprocess_input(images1)
images2 = preprocess_input(images2)
# calculate fid
fid = calculate_fid(model, images1, images2)
print('FID: %.3f' % fid)
