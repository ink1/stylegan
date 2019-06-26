import os, sys
#from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

#model_path = "results/00000-sgan-pics-1gpu/network-snapshot-005600.pkl"
model_path = sys.argv[1]
xsize = 512
ysize = 512

tflib.init_tf()
with open(model_path,"rb") as f:
        _G, _D, Gs = pickle.load(f)
fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

# generate sampled images for all dimensions

# n = Gs.input_shape[1]
n = 512  # up to 512
m = 12   # sampling
# delta -0.0002..0.0002 gives output identical to 0.0
# delta 0.01 and higher gives identical but totally different output to 0.0
delta = 0.0004

latent_vector = np.zeros((n*m, Gs.input_shape[1]), dtype=np.float32)

if m%2 != 0:
    print("sampling for odd number of points which includes 0")
    for i in range(n):
        for j in range(m):
            latent_vector[i*m+j, i] = (-(m//2)+j)*delta
else:
    print("sampling for odd number of points which excludes 0")
    for i in range(n):
        for j in range(-m+1, m+1, 2):
            j_idx = (j+m-1)//2
            latent_vector[i*m+j_idx, i] = j*(delta/2)

# generate images - 
#images = Gs.run(latent_vector, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)

# generate in batches of
batch_size = 32

num_batch = (n*m)//batch_size
images = np.zeros((n*m, ysize, xsize, 3), dtype=np.uint8)
for i in range(num_batch):
    images[i*batch_size:(i+1)*batch_size,:,:,:] = Gs.run(
        latent_vector[i*batch_size:(i+1)*batch_size,:], None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)

print(images.shape)

# images[0,0,0,0] type is np.uint8

# save images each containing k rows and m columns
k = 8
num_pics = n//k
pics = np.zeros((k*ysize, m*xsize, 3), dtype=np.uint8)
img_num = 0
for p in range(num_pics):
    for i in range(k):
        for j in range(m):
            #print('img num:', img_num, i, j)
            pics[i*ysize:(i+1)*ysize, j*xsize:(j+1)*xsize, :] = images[img_num, :,:,:]
            img_num += 1
    last_col = (img_num+1)//m
    first_col = last_col-k+1
    fname = "%03d-%03d.png" % (first_col, last_col)
    PIL.Image.fromarray(pics).save(fname)

