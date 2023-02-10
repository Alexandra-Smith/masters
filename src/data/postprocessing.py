import torch

def patches_to_image(patches):


    return image

def check_if_background_patch(patches):
    '''
    Given a list of patches extracted from the ground truth images, 
    determine whether a patch is background or not.

    Returns:
    is_background (boolean):
    '''

    # * this needs to be done when performing predictions - if the patch is background
    # * then no prediction needs to be made and it can automatically be assigned 0 (black)
    # * else a prediction on the patch needs to be performed to determine classification as 
    # * normal or tumourous - get the probability 

    # * this function should only be needed for visualisation purposes