import torch
import os
import numpy as np
from openslide import open_slide
from PIL import Image

def load_data(patch_size, stride, num_classes):
    '''
    Function to load in data, apply any preprocessing steps,
    extract patches, and get corresponding labels.

    Parameters:
    patch_size: dimension of patches extracted
    num_classes: number of classes to assign patches to

    Returns:
    patches: list of patches
    labels: list of corresponding patch labels
    '''

    SVS_DIR='/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/raw/svs_files/'
    MASK_DIR='/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/interim/masks/'

    svs_files = os.listdir(SVS_DIR)
    # Get file codes (IDs)
    file_codes = []
    for file in svs_files:
        name = file.replace(SVS_DIR, '').replace('.svs', '')
        file_codes.append(name)
    
    for code in file_codes:
        # LOAD SVS FILES
        file = SVS_DIR + code + '.svs'
        sld = open_slide(file)
        slide_props = sld.properties
        slide_width = int(slide_props['openslide.level[1].width']); slide_height = int(slide_props['openslide.level[1].height']) # dimensions at 10X magnification
        slide = np.array(sld.get_thumbnail(size=(slide_width, slide_height)))

        # LOAD SEGMENTATION MASKS
        mask_file = MASK_DIR + code + '.png'
        mask = np.array(Image.open(mask_file))
        mask = mask[:slide_height, :slide_width] # reshape mask file to be same size as SVS

        if slide_height != mask.shape[0] or slide_width != mask.shape[1]:
            raise Exception("Input SVS file and segmentation image do not have the same dimensions")

        # Extract patches
        patches = image_to_patches(slide, patch_size, stride)

def scale_tensor(tensor):
    '''
    Scale a tensor to the range [0, 1]
    '''
    minn = tensor.min()
    maxx = tensor.max()
    tensor = (tensor - minn)/(maxx - minn)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def image_to_patches(image, patch_size: int, stride: int):
    '''
    Function for splitting an input image into patches.

    Parameters:
    image: input image to split
    patch_size (int): dimension, patches will be square
    stride (int): controls overlap between patches

    Returns:
    Tensor of patches with shape (num_patches, im_dim (if applicable), patch_size, patch_size)
    '''
    # Convert image to PyTorch tensor
    im = torch.from_numpy(image)
    im = scale_tensor(im)
    print("image scaled")

    # Is image colour or binary?
    image_dimension = 3 if len(image.shape) == 3 else 1

    # Working with a colour image
    if image_dimension == 3:
        # Extract patches
        patches = im.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
        # Reshape tensor into tensor of shape (num_patches, 3, patch_size, patch_size)
        patches = patches.contiguous().view(-1, image_dimension, patch_size, patch_size) ###.contiguous() ensure tensor is stored in contiguous block of memory which is required for .view()
        # - Can also reshape patches into a 2D tensor, where each row is a flattened patch
        # - patches = patches.contiguous().view(-1, patch_size*patch_size)
        # todo: figure out the shaping of how to return the patches (remember still have to put all patches from all images together after each image)
    # Working with greyscale image
    else:
        # Extract patches
        patches = im.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
        # Reshape tensor into tensor of shape (num_patches, patch_size, patch_size)
        patches = patches.contiguous().view(-1, patch_size, patch_size)

    return patches

def discard_background_patches(svs_patches, mask_patches, patch_size):
    '''
    Given a set of patches, discard the patches that are determined to be
    majority background - and therefore will be ignored.
    Used to filter out patches not necessary for model training.

    Parameters:
    svs_patches: List of patches taken from svs image
    mask_patches: List of patches taken from segmentation masks
    patch_size (int): dimension of patches (square) 

    Returns:
    tissue_patches: List of patches containing tissue
    seg_patches: List of the corresponding segmentation patches
    '''
    
    # todo: complete this function

    # for patch in svs_patches:


    # return tissue_patches, seg_patches

def get_patch_labels(patches, patch_size):
    '''
    Given a set of patches, assign labels to each patch based on certain criteria for each class.

    Parameters:
    patches: List of patches from segmentation masks

    Returns:
    List of labels
    '''
    
    # todo: complete this function

    labels = []
    # Get total number of pixels
    total_px = patch_size*patch_size
    for patch in patches:
        p = np.array(patch)
        # todo: scale patch to [0, 1]
        # Number of black pixels
        nblack = patch[np.where(patch==0)].sum()
        # Number of white pixels
        nwhite = patch[np.where(patch==1)].sum()

        # Calculate % of background (black) pixels
        background_percentage = nblack/total_px

        
        # if background_percentage < 0.5:



    # return labels