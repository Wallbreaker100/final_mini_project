import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, rotate, iradon
from skimage.restoration import denoise_tv_chambolle
import cv2
import os

# Define input and output directories
# input_dir = './dataset'
# output_dir = './sinograms_8'
input_dir = './dataset'
output_dir1 = './sinograms_0.1'
output_dir2 = './output_0.1'

# Create the output directory if it doesn't existos.makedirs(output_dir, exist_ok=True)
# Function for forward projection
angle = 0.1

def sinogram(image,angle):
    theta = np.arange(0., 180., angle)
    sinogram = radon(image, theta=theta)
    return sinogram

def forward_projection(image, angle):    # Your forward projection code here
    theta = np.arange(0., 180., angle)
    reconstruction_img = iradon(image, theta=theta, filter_name='ramp')
    total_variation_denoised_image  = denoise_tv_chambolle(reconstruction_img, weight=0.3)  # Adjust the weight parameter as needed
    total_variation_denoised_image_normalized = (total_variation_denoised_image * 255).astype(np.uint8)    
    image_pil = Image.fromarray(total_variation_denoised_image_normalized)
    image_pil = image_pil.convert("L")    
    return image_pil

# Load images from the input directory
input_images = []
for filename in os.listdir(input_dir):    
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_dir, filename)     
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)   
        image = cv2.equalizeHist(gray_image)
        # image = np.array(Image.open(image_path))
        input_images.append(image)        

# Perform forward projection for each image with varying projection angles
for idx, image in enumerate(input_images):              
    sino = sinogram(image, angle)
    out = forward_projection(sino, angle)

    image_pil = Image.fromarray(sino)
    resize_factor = 5  # Adjust this factor as needed
    new_width = sino.shape[1] * resize_factor
    new_height = sino.shape[0]
    # Resize the image to increase its width
    image_pil = image_pil.resize((new_width, new_height), Image.BOX)    
    image_pil = image_pil.convert("L")

    output_path_sino = os.path.join(output_dir1, f'sinogram_{idx}.png')        
    image_pil.save(output_path_sino)
    output_path_recon = os.path.join(output_dir2, f'output_{idx}.png')        
    out.save(output_path_recon)    

#############################################################

