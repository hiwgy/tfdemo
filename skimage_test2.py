from skimage import io, color, transform, exposure
import matplotlib.pyplot as plt

img=io.imread('~/Desktop/robin_talk.png')
rows, cols, _ = img.shape
print("rows:%d cols:%d" % (rows, cols))

img_gray=color.rgb2gray(img)
print("gray shape:", img_gray.shape)
print("gray dtype:", img_gray.dtype)
print(img_gray)

plt.imshow(img_gray)
plt.show()

n, bins, patches = plt.hist(img_gray.flatten(), bins=256, facecolor='red', density=1)
plt.show()

for i in range(rows):
    for j in range(cols):
        if img_gray[i,j] > 0.9:
            img[i,j] = [0,0,255,255]

img_equ = exposure.equalize_hist(img)
plt.imshow(img_equ)
plt.show()


