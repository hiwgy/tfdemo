from skimage import io, color, transform, exposure, measure, draw
import numpy as np
import matplotlib.pyplot as plt

def print_img_info(name, img):
    print("image[%s] shape[%s] dtype[%s] mean[%s]" % (name, img.shape, img.dtype.name, str(img.mean())[:3]))

img=io.imread('~/Desktop/robin_talk.png')
rows, cols, channels = img.shape
print_img_info('original', img)

plt.figure("robin noise")
plt.subplot(2, 2, 1)
plt.title('origin image')
plt.imshow(img)

# change to gray image
mat = color.rgb2gray(img)
print_img_info("gray", mat)

plt.subplot(2, 2, 2)
plt.title('gray image')
plt.imshow(mat)

# padding
# warning, img[0:1, : ,:] has different shape with img[0, :, :]
pad_ret = np.vstack([mat[0:1,:], mat])
pad_ret = np.vstack([pad_ret, mat[rows-1:rows, :]])
pad_ret = np.hstack([pad_ret[:, 0:1], pad_ret])
pad_ret = np.hstack([pad_ret, pad_ret[:, cols-1:cols]])
print_img_info("pad", pad_ret)

# detect contour
contours = measure.find_contours(pad_ret, 0.5)
for n, contour in enumerate(contours):

plt.subplot(2, 2, 3)
plt.title('padding image')
plt.imshow(pad_ret)

guass = np.reshape([1,2,1,2,4,2,1,2,1], [3, 3])
# box = np.reshape([1,1,1,1,1,1,1,1,1], [3, 3])
out = np.empty((rows, cols))

for i in range(1, rows+1):
    for j in range(1, cols+1):
        conv = 0.0
        for m in range(i-1, i+2):
            for n in range(j-1, j+2):
                conv += pad_ret[m, n] * box[m-i+1, n-j+1]
        out[i-1, j-1] = conv/16.0

print_img_info("guass", out)
plt.subplot(2, 2, 4)
plt.title('guass image')
plt.imshow(out)

plt.show()

