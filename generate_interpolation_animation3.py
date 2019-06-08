# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os, sys
import pickle
import numpy as np
import PIL.Image
#import Image
import dnnlib
import dnnlib.tflib as tflib
import config


###################### HOWTO generate video from the generated frames:
###################### ffmpeg -framerate 30 -i animation_%d.png out.mp4
######################

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    #model_path = "./models/karras2019stylegan-celebahq-1024x1024.pkl"
    #!wget https://github.com/parameter-pollution/stylegan_paintings/releases/download/v0.1/network-snapshot-008040.pkl
    #model_path = "./results/00001-sgan-pics-1gpu/network-snapshot-008040.pkl"
    model_path = sys.argv[1]
    with open(model_path,"rb") as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    #rnd = np.random.RandomState(1337)
    rnd = np.random.RandomState(314)
    number_of_frames = 500
    number_of_segments = 10
    frames_per_segment = number_of_frames//number_of_segments
    frame_step = 1.0/frames_per_segment

    print('Gs shape: ', Gs.input_shape)
    latent_input = rnd.randn(1, Gs.input_shape[1])
    frame_count = 0
    x = frame_step
    for frame_count in range(1,number_of_frames):
        latent_vector = rnd.randn(1, Gs.input_shape[1])
        latent_input = latent_input + x*(latent_vector - latent_input)
        images = Gs.run(latent_input, None, truncation_psi=1, randomize_noise=False, output_transform=fmt)

        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, 'animation_'+str(frame_count)+'.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

if __name__ == "__main__":
    main()
