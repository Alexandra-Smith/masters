import torch
import os
import numpy as np
from openslide import open_slide
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def load_data(images_directory, gt_directory, patch_size: int, stride: int, num_classes):
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

    svs_files = os.listdir(images_directory)

    all_patches = torch.empty((0, 3, patch_size, patch_size))
    all_gt_patches = torch.empty((0, patch_size, patch_size))

    # Get file codes (IDs)
    file_codes = []
    for file in svs_files:
        if file.endswith('.DS_Store'):
            continue
        name = file.replace(images_directory, '').replace('.svs', '')
        file_codes.append(name)
    
    i = 1
    print("Starting processing")
    for code in file_codes:

        # LOAD SVS FILE
        file = images_directory + code + '.svs'
        sld = open_slide(file)
        slide_props = sld.properties
        slide_width = int(slide_props['openslide.level[1].width']); slide_height = int(slide_props['openslide.level[1].height']) # dimensions at 10X magnification
        slide = np.array(sld.get_thumbnail(size=(slide_width, slide_height)))
        print("Done: loading svs")

        # LOAD SEGMENTATION MASK
        mask_file = gt_directory + code + '.png'
        mask = np.array(Image.open(mask_file))
        mask = mask[:slide_height, :slide_width] # reshape mask file to be same size as SVS
        print("Done: loading mask")

        if slide_height != mask.shape[0] or slide_width != mask.shape[1]:
            raise Exception("Input SVS file and segmentation image do not have the same dimensions")

        # Extract patches
        patches = image_to_patches(slide, patch_size, stride)
        print("Done: extracting svs patches")
        # create patches for segmentation masks
        mask_patches = image_to_patches(mask, patch_size, stride)
        print("Done: extracting mask patches")
        # get rid of background patches
        tissue_patches, gt_patches = discard_background_patches(patches, mask_patches, patch_size)
        print("Done: discarding background patches")
        # concatenate all patches from all images together
        all_patches = torch.cat((all_patches, tissue_patches), dim=0); all_gt_patches = torch.cat((all_gt_patches, gt_patches), dim=0)
        
        # Keep track of processed files
        print(f"Completed processing {i}/{len(file_codes)}")
        i += 1

    # Get labels
    all_labels = get_patch_labels(all_gt_patches, patch_size)
    
    return all_patches, all_labels

def scale_tensor(tensor: torch.Tensor):
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
    # Scale image to [0, 1]
    im = scale_tensor(im)

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
        # Reshape tensor into tensor of shape (num_patches, patch_size, patch_size)
        patches = patches.contiguous().view(-1, patch_size, patch_size)

    return patches

def discard_background_patches(svs_patches, mask_patches, patch_size: int):
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

    total_num_patches = mask_patches.shape[0]
    total_px = patch_size*patch_size
    tissue_patches = torch.empty((0, 3, patch_size, patch_size))
    seg_patches = torch.empty((0, patch_size, patch_size))

    for i in range(total_num_patches):
        # Get patch (torch tensor)
        im_patch = svs_patches[i, :, :, :]
        mask_patch = mask_patches[i, :, :]
        # Convert to numpy array
        p = mask_patch.numpy()
        # Number of black pixels
        nblack_px = np.sum(p == 0)
        # Calculate % of background (black) pixels
        background_percentage = nblack_px/total_px
        # Keep patch if background < 80% of the patch (i.e patch contains > 80% tissue), otherwise discard it
        if background_percentage < 0.8:
            tissue_patches = torch.cat((tissue_patches, im_patch.unsqueeze(0)), dim=0)
            seg_patches = torch.cat((seg_patches, mask_patch.unsqueeze(0)), dim=0)

    return tissue_patches, seg_patches

def get_patch_labels(patches, patch_size):
    '''
    Given a set of patches, assign labels to each patch based on certain criteria for each class.

    Parameters:
    patches: List of patches from segmentation masks

    Returns:
    List of labels, where 0 = normal tissue and 1 = carcinoma
    '''

    total_num_patches = patches.shape[0]
    total_px = patch_size*patch_size

    labels = []
    for i in range(total_num_patches):
        # Get patch (torch tensor)
        patch = patches[i, :, :]
        # Convert to numpy array
        p = patch.numpy()
        # Calculate number of black pixels in patch
        nwhite_px = np.sum(p == 1)
        # Calculate % of white pixels == tumourous pixels
        tumour_percentage = nwhite_px/total_px
        # Patch is considered to be labelled tumourous (1) if > 80% of the tissue in the patch is tumourous, otherwise it is labelled as normal tissue (0)
        if tumour_percentage > 0.8:
            labels.append(1)
        else:
            labels.append(0)
    
    # Convert to PyTorch tensor format
    labels = torch.tensor(labels, dtype=torch.long)

    return labels