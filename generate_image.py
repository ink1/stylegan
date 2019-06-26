# cd  stylegan
import os, sys
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

tflib.init_tf()

#model_path = "results/00000-sgan-pics-1gpu/network-snapshot-005600.pkl"
model_path = sys.argv[1]

with open(model_path,"rb") as f:
        _G, _D, Gs = pickle.load(f)

fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

## change this number to get a different image 
#rnd = np.random.RandomState(42)
## rerun this line to get a different image
#latent_vector = rnd.randn(1, Gs.input_shape[1])
#images = Gs.run(latent_vector, None, truncation_psi=1, randomize_noise=False, output_transform=fmt)
## in notebook - display the image
#PIL.Image.fromarray(images[0])

latent_vector = np.zeros((1, Gs.input_shape[1]), dtype=np.float32)
latent_vector[0, 80] = -.0018
images = Gs.run(latent_vector, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)

# in script - save the image
PIL.Image.fromarray(images[0]).save('testimage.png')
