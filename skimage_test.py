# please see the latest api document
# https://scikit-image.org/docs/stable/api/
from skimage import io, transform, data, data_dir, color
import matplotlib.pyplot as plt
import numpy as np

########################## test 1 ############################
img = io.imread('./robin.jpg')
io.imshow(img)
# use matplotlib to show image on mac
plt.show()
# please close the window to continue

# image [height,width,channel]
print("image shape is:", img.shape)
print("image dtype name is:", img.dtype.name)
# image total pixel number
print("image pixel nummber is:", img.size)
# the max pixel value
print("image max pixel value is:", img.max())
# the min pixel value
print("image min pixel value is:", img.min())
# the average pixel value
print("image average pixel value is:", img.mean())
# the pixel value of the certain position
# the shape is [int,int,int], means [R,G,B]
print("image pixel value of [0][0] is:", img[0][0]) 
# show the red single channel image
red_img = img[:,:,0]
io.imshow(red_img)
plt.show()

########################## test 2 ############################
# read the gray image
img_gray = io.imread('./robin.jpg', as_gray=True)
io.imshow(img_gray)
plt.show()
print("img_gray shape is:", img_gray.shape)
print("img_gray max pixel value is:", img_gray.max())
print("img_gray min pixel value is:", img_gray.min())
print("img_gray pixel value of [0][0] is:", img_gray[0][0])


########################## test 3 ############################
# try to construct a white image with black border
img1 = np.zeros([200, 400, 3], dtype=np.float64)
# pix value from 0.0 to 1.0
img1[10:-10, 10:-10, :] = 1.0
io.imshow(img1)
plt.show()

# add some noise to the img
rows, cols, dims = img1.shape
for i in range(5000):
    x = np.random.randint(0, rows)
    y = np.random.randint(0, cols)
    img1[x, y, :] = 0.0
io.imshow(img1)
plt.show()

# fix the noise, show the ability of the numpy array
img1_fix_flag = img1[10:-10,10:-10,:] < 1.0
print("img1_fix_flag shape is:", img1_fix_flag.shape)
# the img1_fix_flag is [180,180,3], value is True or False
# so need to change is shape
raw_pad = np.zeros([10, 380, 3], dtype=np.bool)
col_pad = np.zeros([200, 10, 3], dtype=np.bool)
pad_ret = np.vstack([raw_pad, img1_fix_flag])
pad_ret = np.vstack([pad_ret, raw_pad])
pad_ret = np.hstack([col_pad, pad_ret])
pad_ret = np.hstack([pad_ret, col_pad])
print("pad_ret shape is:", pad_ret.shape)
img1[pad_ret] = 1.0
io.imshow(img1)
plt.show()

# cut the img to a square
img1 = img1[:, 0:cols//2, :]
io.imshow(img1)
plt.show()

# save the img1 to file
io.imsave('./while_black_img.jpg', img1)

########################## test 4 ############################
# use the image from data
print("the data_dir is:", data_dir)
img_coffee = data.coffee()
io.imshow(img_coffee)
plt.show()
# change the img_coffee to 2-value image
# and make a label
img_coffee_gray = color.rgb2gray(img_coffee)
rows, cols = img_coffee_gray.shape
labels = np.zeros([rows, cols])
for i in range(rows):
    for j in range(cols):
        # the value is float64, between 0.0~1.0
        if img_coffee_gray[i,j] <= 0.4:
            img_coffee_gray[i,j] = 0.0
            labels[i,j] = 0
        elif img_coffee_gray[i,j] <= 0.7:
            img_coffee_gray[i,j] = 0.0
            labels[i,j] = 1
        else:
            img_coffee_gray[i,j] = 1.0
            labels[i,j] = 2
io.imshow(img_coffee_gray)
plt.show()
# lable the image
image_label = color.label2rgb(labels)
io.imshow(image_label)
plt.show()
# plt.show(image_label) # use this easy function

# change the color space
# dict_keys(['rgb', 'hsv', 'rgb cie', 'xyz', 'yuv', 'yiq', 'ypbpr', 'ycbcr', 'ydbdr'])
img_coffee_hsv = color.convert_colorspace(img_coffee, 'RGB', 'HSV')
io.imshow(img_coffee_hsv)
plt.show()


