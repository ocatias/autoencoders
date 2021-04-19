import autoencoder.autoencoder as  autoencoder
import torch
import cv2
import numpy as np
import math
from PIL import Image
import os.path
from natsort import natsorted
import glob

# GENERAL PARAMETERS OF INPUT
picture_width = 32
picture_height = 32
channels = 3
latent_space_dims = 2

# CYCLE PARAMETERS
nr_images = 1000
radius = 2
output_path_cycle = "pictures\cycle"
width = 84
height = 84

# LATENT SPACE PARAMETERS
nr_images_per_row = 20
output_path_latent = "pictures\latent"
path_models = "autoencoder\models"

def draw_latent_space_gif():
    models_path = natsorted(glob.glob(os.path.join(path_models, "*.nn")))
    latent_space_imgs = []
    for model_path in models_path:
        model = autoencoder.Autoencoder(picture_width*picture_height*channels, latent_space_dims)
        model.load_state_dict(torch.load(model_path))
        decoder = model.decoder
        img = draw_latent_space(decoder, False)
        latent_space_imgs.append(img)

    img.save(fp="latent_space.gif", format='GIF', append_images=latent_space_imgs,  save_all=True, optimize=True,
             duration=35, loop=0)

def draw_latent_space(decoder, save_img = True):
    files = []

    start = -3.141
    end = 3.141

    image_size = 58

    l = (end - start)/nr_images_per_row

    images = []

    for i in range(nr_images_per_row):
        for j in range(nr_images_per_row):
            x = start + i*l + l/2
            y = start + j*l + l/2
            sample = torch.tensor([[x, y]])
            pic = pic_from_decoder(sample, decoder)
            images.append(pic)

    if channels == 1:
        new_im = Image.new('L', (image_size*nr_images_per_row, image_size*nr_images_per_row))
    elif channels == 3:
        new_im = Image.new('RGB', (image_size*nr_images_per_row, image_size*nr_images_per_row))


    x_offset = 0
    y_offset = 0
    for (j, im) in enumerate(images):
        if channels == 1:
            new_im.paste(Image.fromarray(im), ((j % nr_images_per_row)*image_size, int((j / nr_images_per_row))*image_size))
        elif channels == 3:
            new_im.paste(Image.fromarray(im), ((j % nr_images_per_row)*image_size, int((j / nr_images_per_row))*image_size))

    if save_img:
        new_im.save("latent_space.png")

    return new_im

def pic_from_decoder(sample, decoder):
    pic = decoder(sample)
    if channels == 3:
        pic = pic[0].view(channels,picture_width,picture_height).detach().numpy()*255
        pic = cv2.merge((pic[2], pic[1], pic[0]))
    elif  channels == 1:
        pic = pic[0].view(picture_width,picture_height).detach().numpy()*255
    pic = pic.astype(np.uint8)
    pic = cv2.resize(pic, (width, height))
    return pic

def draw_cycle(decoder):
    files = []

    for i in range(nr_images):
        angle = i / float(nr_images) * 3.141*2

        sample = torch.tensor([[radius*math.cos(angle), radius*math.sin(angle)]])
        pic = pic_from_decoder(sample, decoder)
        pic_path = os.path.join(output_path_cycle, "cycle_" + str(i) + ".png")
        cv2.imwrite(pic_path, pic)
        files.append(pic_path)

    img, *imgs = [Image.open(f) for f in files]

    img.save(fp=os.path.join(output_path_cycle, "cycle.gif"), format='GIF', append_images=imgs,  save_all=True, optimize=True,
             duration=40, loop=0)




if __name__ == "__main__":
    #draw_latent_space_gif()

    model = autoencoder.Autoencoder(picture_width*picture_height*channels, latent_space_dims)
    model.load_state_dict(torch.load("autoencoder/models/model_final.nn"))
    decoder = model.decoder
    #draw_cycle(decoder)
    draw_latent_space(decoder)
