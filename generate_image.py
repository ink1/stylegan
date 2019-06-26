# cd  stylegan
import os, sys
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

tflib.init_tf()

model_path = "results/00000-sgan-pics-1gpu/network-snapshot-005600.pkl"
#model_path = sys.argv[1]

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

n = 48
mu    = 0.0014
sigma = 0.0004
np.random.seed(42)
latent_vector = np.zeros((n, Gs.input_shape[1]), dtype=np.float32)

# balls and slopes
#indices_neg = [366, 222]
#indices_pos = [153, 406, 497]
# houses
indices_neg = [39,66,69]
indices_pos = [1,8,24,44]
#indices_neg = [39,66,69,463,488,497,85,195,357,387,391,405,406,428]
#indices_pos = [1,8,24,44,499,504,78,122,137,163,164,177,193,217,419,431]

indices_neg = [i-1 for i in indices_neg]
indices_pos = [i-1 for i in indices_pos]

for i in range(n):
  latent_vector[i, indices_neg] =  np.random.normal(-mu, sigma, len(indices_neg))
  latent_vector[i, indices_pos] =  np.random.normal( mu, sigma, len(indices_pos))

images = Gs.run(latent_vector, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)

# in script - save the image
for i in range(n):
  fn = 'testimage%02d.png' % i
  PIL.Image.fromarray(images[i]).save(fn)
