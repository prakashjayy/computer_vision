import random
import cv2
import numpy as np 
import keras
from os.path import join
from PIL import Image
import random

def aspect_ratio_calc(image, label, base_size):
    h, w = label.shape
    if h > w:
        h, w = (base_size, int(base_size*w/h))
    else:
        h, w = (int(base_size*h/w), base_size)
    return h, w

def resize_image(image, label, size):
    h, w = size
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
    label = np.asarray(label, dtype=np.int64)
    return image, label

def pad_image(image, label, required_size, mean_bgr, ignore_label):
    h, w = label.shape
    pad_h = max(required_size - h, 0)
    pad_w = max(required_size - w, 0)
    pad_kwargs = {
        "top": 0,
        "bottom": pad_h,
        "left": 0,
        "right": pad_w,
        "borderType": cv2.BORDER_CONSTANT,
    }
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, value=mean_bgr, **pad_kwargs)
        label = cv2.copyMakeBorder(label, value=ignore_label, **pad_kwargs)
    return image, label
    

def crop_image(image, label, required_size):
    h, w = label.shape
    start_h = random.randint(0, h - required_size)
    start_w = random.randint(0, w - required_size)
    end_h = start_h + required_size
    end_w = start_w + required_size
    image = image[start_h:end_h, start_w:end_w]
    label = label[start_h:end_h, start_w:end_w]
    return image, label

def flip_image(image, label, u=0.5):
    if random.random() < u:
        image = np.fliplr(image).copy()
        label = np.fliplr(label).copy()
    return image, label


def val_augmentation(image, label, resize, crop_size, mean_bgr, ignore_label=255):
    h, w = aspect_ratio_calc(image, label, resize)
    image, label = resize_image(image, label, (int(h), int(w)))
    
    # Padding to fit for crop_size
    image, label = pad_image(image, label, crop_size, mean_bgr, ignore_label)
    
    image = image[:crop_size, :crop_size]
    label = label[:crop_size, :crop_size]
    return image, label


def get_image_and_mask(root, folder_name, img_id, rgb=True):
    image_path = join(root, folder_name, img_id+ ".jpg")
    label_path = join(root, "stuff_"+folder_name, img_id+".png")
    # Load an image and label
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
    if rgb:
        image = image[:, :, ::-1]
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    return image, label
    
    

def augmentation(image, label, resize,flip,  crop_size, \
                 scales, \
                 mean_bgr,\
                ignore_label=255):
    h, w = aspect_ratio_calc(image, label, resize)
    
    scale_factor = random.choice(scales)
    h, w = (int(h * scale_factor), int(w * scale_factor))
    image, label = resize_image(image, label, (h, w))
    
    # Padding to fit for crop_size
    image, label = pad_image(image, label, crop_size, mean_bgr, ignore_label)
    
    ## randomly Crop the image to required_size
    image, label = crop_image(image, label, crop_size)
    
    if flip:
        image, label = flip_image(image, label, 0.5)
    return image, label


def random_channel_shift(x, limit, channel_axis=2):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_ch + np.random.uniform(-limit, limit), min_x, max_x) for x_ch in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return np.uint8(x)

def random_gray(img, u=0.5):
    if np.random.random() < u:
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = np.sum(img * coef, axis=2)
        img = np.dstack((gray, gray, gray))
    return np.uint8(img)


def random_contrast(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = img * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        img = alpha * img + gray
        img = np.clip(img, 0., 255.)
    return np.uint8(img)

def random_brightness(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        img = alpha * img
        img = np.clip(img, 0., 255.)
    return np.uint8(img)

def random_saturation(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])
        gray = img * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        img = alpha * img + (1. - alpha) * gray
        img = np.clip(img, 0., 255.)
    return np.uint8(img)

def random_color_augmentation(img, mask):
    img = random_channel_shift(img, limit=10)
    img = random_brightness(img, limit=(-0.5, 0.5), u=0.5)
    img = random_contrast(img, limit=(-0.5, 0.5), u=0.5)
    img = random_saturation(img, limit=(-0.5, 0.5), u=0.5)
    img = random_gray(img, u=0.2)
    return img, mask

class CocoStuffDataGenerator(keras.utils.Sequence):
    def __init__(self, list_ids, \
                 root="../input", \
                 folder_name = "train",\
                 batch_size=2, \
                 resize=512, \
                 flip = True, \
                 crop = 448,\
                 scales = (1., 1.25),\
                 mean_rgb = (0, 0, 0),\
                 pre_process = None, \
                 ignore_label = 255., \
                 rgb = True, \
                 color_transforms=False, \
                 shuffle=False):
        self.list_ids = list_ids
        self.root = root
        self.folder_name = folder_name
        self.batch_size = batch_size
        self.resize = resize 
        self.flip = flip
        self.crop = crop
        self.scales = scales 
        self.mean_rgb = mean_rgb
        self.pre_process = pre_process
        self.ignore_label = ignore_label
        self.color_transforms = color_transforms
        self.shuffle = shuffle
        self.rgb = rgb
        self.on_epoch_end()
        
    def __len__(self):
        return int((len(self.list_ids) / self.batch_size))
    
    def __getitem__(self, index):
        list_of_ids = [random.choice(self.list_ids) for i in range(self.batch_size)]
        #list_of_ids = [self.list_ids[i] for i in indexes]
        X, y = self.data_generation(list_of_ids)
        return X, y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def data_generation(self, list_ids_temp):
        imgs = []
        masks = []
        for i in list_ids_temp:
            img, mask = self.get_image_and_mask(i)
            if self.color_transforms:
                img, mask = random_color_augmentation(img, mask)
            img_aug, mask_aug = augmentation(img, mask, \
                                             resize=self.resize, \
                                             flip=self.flip, \
                                             crop_size=self.crop,\
                                             scales = self.scales,\
                                             mean_bgr = self.mean_rgb,\
                                             ignore_label = self.ignore_label)
            img_aug = np.float64(img_aug)
            if self.pre_process is not None:
                img_aug = self.pre_process(img_aug)
            
            mask_aug = np.expand_dims(mask_aug, 2)
            
            imgs.append(np.expand_dims(img_aug, 0))
            masks.append(np.expand_dims(mask_aug, 0))
        return np.concatenate(imgs), np.concatenate(masks)
        
    def get_image_and_mask(self, img_id):
        image_path = join(self.root, self.folder_name, img_id+ ".jpg")
        label_path = join(self.root, "stuff_"+self.folder_name, img_id+".png")
        # Load an image and label
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        if self.rgb:
            image = image[:, :, ::-1]
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        return image, label