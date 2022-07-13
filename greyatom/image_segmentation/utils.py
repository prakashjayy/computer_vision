## Loading the required libraries 
import pandas as pd
import numpy as np 

from keras.preprocessing import image
from os.path import join


from keras.preprocessing import image
from os.path import join

def get_image_and_mask(img_id, data_dir = '../input'):
    """ Read the input and output images of given image_id using keras image processing library
    
    img_id: the id of the image
    data_dir: the directory in with train images and train_masks are present.
    
    output:
    output the image and its respective mask of the given id
    """
    ## Load the image
    img = image.load_img(join(data_dir, 'train', '%s.jpg' % img_id))#,
                         #target_size=(input_size, input_size))
    ## Convert into array
    img = image.img_to_array(img)
    
    ## Load the mask as grayscale image
    mask = image.load_img(join(data_dir, 'train_masks', '%s_mask.gif' % img_id), color_mode="grayscale")#,
                          #grayscale=True, target_size=(input_size, input_size))
    ## convert the image to array
    mask = image.img_to_array(mask)
    
    ## Normalize the image
    img, mask = img / 255., mask / 255.
    return img, mask