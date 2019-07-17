# please see the latest api document
# https://scikit-image.org/docs/stable/api/
from skimage import io, transform, data, data_dir, color, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import os

########################## test 1 ############################
img = io.imread('./robin.jpg')
plt.imshow(img)
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
io.imshow(red_img) # different kind of show image method
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


########################## test 5 ############################
img_ast = data.astronaut()
# figure, num means id, retrieve it and active it or create a new
plt.figure(num='astronaut', figsize=(8, 8))
# split the window to 4 sub windows
# subplot(nrows, ncols, index, **kwargs)
plt.subplot(2, 2, 1)
plt.title('origin image')
plt.imshow(img_ast)

# the second window
plt.subplot(2, 2, 2)
plt.title('R channel')
# TODO:why gray?
plt.imshow(img_ast[:,:,0], plt.cm.gray)
plt.axis('off') # display the axis

# the third window
plt.subplot(2, 2, 3)
plt.title('G channel')
plt.imshow(img_ast[:,:,1], plt.cm.gray)
plt.axis('off')

# the fourth window
plt.subplot(2, 2, 4)
plt.title('B channel')
plt.imshow(img_ast[:,:,2], plt.cm.gray)
plt.axis('off')

plt.show()


########################## test 6 ############################
def convert_gray(f):
    rgb = io.imread(f)
    gray = color.rgb2gray(rgb)
    print("fname is:%s, shape is:%s, gray shape is:%s" % (f, rgb.shape, gray.shape))
    dst = transform.resize(gray, (256, 256))
    return dst
input_path = data_dir + '/*.png'
coll = io.ImageCollection(input_path, load_func=convert_gray)
output_path = "./output"
if not os.path.exists(output_path):
    os.makedirs(output_path)
for i in range(len(coll)):
    io.imsave(output_path + '/' + np.str(i) + '.jpg', coll[i])

print("has %d images" % len(coll))
io.imshow(coll[10])
# TODO: the window behave is strange
plt.figure(num='collection')
plt.subplot(1, 2, 1)
plt.title('pic 10')
plt.imshow(coll[10])

plt.subplot(1, 2, 2)
plt.title('pic 11')
plt.imshow(coll[11])

plt.show()


########################## test 7 ############################
img = data.camera()
# resize
dst = transform.resize(img, (80, 60))
plt.figure('resize shape')
plt.subplot(321)
plt.title('before resize, shape:' + str(img.shape))
plt.imshow(img)
plt.subplot(322)
plt.title('after resize, shape:' + str(dst.shape))
plt.imshow(dst, plt.cm.gray)

# rescale
dst_rescale1 = transform.rescale(img, 0.1)
plt.subplot(323)
plt.title('after rescale, shape:' + str(dst_rescale1.shape))
plt.imshow(dst_rescale1, plt.cm.gray)
dst_rescale2 = transform.rescale(img, [1, 2])
plt.subplot(324)
plt.title('after rescale, shape:' + str(dst_rescale2.shape))
plt.imshow(dst_rescale2, plt.cm.gray)

# rotate
dst_rotate1 = transform.rotate(img, 60)
plt.subplot(325)
plt.title('after rotate 60, shape:' + str(dst_rotate1.shape))
plt.imshow(dst_rotate1)
# the image will be big enough to contain the rotated image
dst_rotate2 = transform.rotate(img, 30, resize=True)
plt.subplot(326)
plt.title('after rotate 60, shape:' + str(dst_rotate2.shape))
plt.imshow(dst_rotate2, plt.cm.gray)

plt.show()


########################## test 8 ############################
img = data.astronaut()
rows, cols, dim = img.shape
print(img.shape, img.dtype.name)
# warning: the img.dtype.name is np.uint8, so the composite_image must
#          has the same dtype
pyramid = tuple(transform.pyramid_gaussian(img, downscale=2))
composite_image = np.ones((rows, cols + cols // 2, 3), dtype=np.float)
# composite the original img
composite_image[0:rows, 0:cols] = pyramid[0]

# composite the other images
i_row = 0
for p in pyramid[1:]:
    print("shape of p is:", p.shape)
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows

plt.imshow(composite_image)
plt.show()

########################## test 9 ############################
# note: data.moon() dtype is uint8
img = img_as_float(data.astronaut()) # function img_as_float is useful, see test 8
print("img shape:", img.shape)
print("img mean:", img.mean())

plt.figure(num="exposure", figsize=(8,8))
plt.subplot(4, 4, 1)
plt.title("original img")
plt.axis('off')
plt.imshow(img)
for i in range(2, 17):
    gam = exposure.adjust_gamma(img, 0.2 * i)
    plt.subplot(4, 4, i)
    plt.title("gamma " + str(0.2 * i)[:4])
    plt.axis('off')
    plt.imshow(gam)
plt.show()

plt.figure(num="exposure")
plt.subplot(4, 4, 1)
plt.title("original img")
plt.axis('off')
plt.imshow(img)
scale=[0, 0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1.0]
for i in range(2, 17):
    gam = exposure.rescale_intensity(img, in_range=(0, scale[i]))
    plt.subplot(4, 4, i)
    plt.title("inten:" + str(scale[i]) + " mean:" + str(gam.mean())[:4])
    plt.axis('off')
    plt.imshow(gam)
plt.show()

# use *1.0 to change the uint8 dtype to float
img = np.array([51, 102, 153], dtype=np.uint8)
mat = exposure.rescale_intensity(img*1.0)
print(mat) # [0. 0.5 1.]


########################## test 10 ############################
img = data.camera()
plt.figure("histogram")
arr = img.flatten()
"""
 input:
   arr: 灰度图，把二维图像转化为一维
   bins:直方图的桶数
   density:是否归一化，如果归一化，最终直方图的面积为1
          否则为像素点个数，即图像size

 output:
   n: 直方图向量，各个灰度的像素个数
   bins: range of each bin
   patches: the data list in each bin
"""
n, bins, patches = plt.hist(arr, bins=256, facecolor='red', density=1)
print(n)
print("sum of n is:%d, size of image is:%d" % (n.sum(), img.size))
print(bins.shape)
for i in range(0, min(10, len(bins))):
    print(bins[i])

for i in range(0, min(10, len(patches))):
    print(patches[i])
plt.show()

# colored histogram
img = data.astronaut()
ar = img[:,:,0].flatten()
plt.hist(ar, bins=256, facecolor='red', density=1)
ag = img[:,:,1].flatten()
plt.hist(ag, bins=256, facecolor='green', density=1)
ab = img[:,:,2].flatten()
plt.hist(ab, bins=256, facecolor='blue', density=1)
plt.show()

########################## test 11 ############################
img = data.moon()
plt.figure(num="moon")
plt.subplot(221)
plt.title('original moon')
plt.imshow(img)
plt.subplot(222)
plt.title('original hist')
n, _, _ = plt.hist(img.flatten(), bins=256, facecolor='red', density=1)
print('img size:%d, sum of n:%d' % (img.size, n.sum()))

# 直方图均衡化
img_equ = exposure.equalize_hist(img)
plt.subplot(223)
plt.title('equalize moon')
plt.imshow(img_equ)
plt.subplot(224)
plt.title('equalize hist')
n_equ, _, _ = plt.hist(img_equ.flatten(), bins=256, facecolor='red', density=1)
# TODO: 为什么n_equ.sum()不等于1了呢？
print('equalize img size:%d, sum of n:%d' % (img_equ.size, n_equ.sum()))

plt.show()

