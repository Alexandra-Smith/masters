import torch
import numpy as np


def image_to_patches(image, patch_size: int, stride: int):
    '''
    Function for splitting an input image into patches.

    Parameters:
    image: input image to split
    patch_size (int): dimension, patches will be square
    stride (int): controls overlap between patches

    Returns:
    List of patches
    '''
    # Convert image to PyTorch tensor
    im = torch.from_numpy(image)

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
    # Working with greyscale image
    else:
        # Extract patches
        patches = im.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
        # Reshape tensor into tensor of shape (num_patches, 3, patch_size, patch_size)
        patches = patches.contiguous().view(-1, patch_size, patch_size)

    return patches

def get_patch_labels(patches):
    '''
    Given a set of patches, assign labels to each patch based on certain criteria for each class.
    Parameters:
    patches: List of patches from segmentation masks
    Returns:
    List of labels
    '''


    return labels