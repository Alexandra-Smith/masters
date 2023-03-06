import torch
import os
import pandas as pd
import numpy as np
from openslide import open_slide
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def LOAD(case_code, images_directory, gt_directory, patch_size: int, stride: int, num_classes):
    '''
    Function to load in ONE image, apply preprocessing steps,
    extract patches and get the corresponing patch labels.
    Also captures specific information about each slide, to be written to a database.

    Returns:
    patches: all patches extracted from the image that contain sufficient amount of tissue
    labels: corresponding class labels for all patches
    '''

    # LOAD SVS FILE
    file = images_directory + case_code + '.svs'
    sld = open_slide(file)
    slide_props = sld.properties
    slide_width = int(slide_props['openslide.level[1].width']); slide_height = int(slide_props['openslide.level[1].height']) # dimensions at 10X magnification
    slide = np.array(sld.get_thumbnail(size=(slide_width, slide_height)))

    # # LOAD SEGMENTATION MASK
    # mask_file = gt_directory + case_code + '.png'
    # mask = np.array(Image.open(mask_file))
    # mask = mask[:slide_height, :slide_width] # reshape mask file to be same size as SVS

    # if slide_height != mask.shape[0] or slide_width != mask.shape[1]:
    #     raise Exception("Input SVS file and segmentation image do not have the same dimensions")

    # extract patches
    patches = image_to_patches(slide, patch_size, stride)
    # create patches for segmentation masks
    # mask_patches = image_to_patches(mask, patch_size, stride)
    # # get rid of background patches
    # tissue_patches, gt_patches = discard_background_patches(patches, mask_patches, patch_size)

    # # Get labels
    # labels = get_patch_labels(gt_patches, patch_size)

    # create dataframe to capture data information
    # df_data = []
    # col_names = ['File_name', 'Level0_factor', 'Level1_factor', 'Level2_factor', 'Level3_factor', 'Level0_height', 'Level0_width',
    #             'Level1_height', 'Level1_width', 'Level2_height', 'Level2_width', 'Level3_height', 'Level3_width',
    #             'Patch_size', 'Stride', 'Total_num_of_patches', 'Num_of_patches_discarded', '%_of_benign_tiles','%_of_tumourous_tiles']
    # levels = len(sld.level_dimensions) # get num of levels
    # if levels > 4:
    #     raise Exception("More than 4 levels contained in this svs file")
    # factors = np.array(sld.level_downsamples)
    # # start row
    # data = [case_code.split('.')[0]] # first entry is the file name
    # for r in range(levels):
    #     data.append(factors[r]) # add all downsample factors for each level
    # if levels < 4:
    #     data.append(np.nan)
    # total_num_patches = patches.shape[0]
    # num_discarded_patches = total_num_patches - tissue_patches.shape[0]
    # percentage_benign = (torch.sum(torch.eq(labels, 0))/len(labels)).item() * 100
    # percentage_tumourous = (torch.sum(torch.eq(labels, 1))/len(labels)).item() * 100
    # if levels < 4:
    #     data.extend([slide_props['openslide.level[0].height'], slide_props['openslide.level[0].width'], 
    #             slide_props['openslide.level[1].height'], slide_props['openslide.level[1].width'],
    #             slide_props['openslide.level[2].height'], slide_props['openslide.level[2].width'],
    #             np.nan, np.nan, patch_size, stride, total_num_patches, num_discarded_patches, percentage_benign, percentage_tumourous])
    # else:
    #     data.extend([slide_props['openslide.level[0].height'], slide_props['openslide.level[0].width'], 
    #             slide_props['openslide.level[1].height'], slide_props['openslide.level[1].width'],
    #             slide_props['openslide.level[2].height'], slide_props['openslide.level[2].width'],
    #             slide_props['openslide.level[3].height'], slide_props['openslide.level[3].width'],
    #             patch_size, stride, total_num_patches, num_discarded_patches, percentage_benign, percentage_tumourous])
    # df_data.append(data)
    # df = pd.DataFrame(df_data, columns=col_names)
    
    # return tissue_patches, labels, df
    return patches

def load_indv_case(case_code, images_directory, gt_directory, patch_size: int, stride: int, num_classes):
    '''
    Function to load in ONE image, apply preprocessing steps,
    extract patches and get the corresponing patch labels.
    Also captures specific information about each slide, to be written to a database.

    Returns:
    patches: all patches extracted from the image that contain sufficient amount of tissue
    labels: corresponding class labels for all patches
    '''

    # LOAD SVS FILE
    file = images_directory + case_code + '.svs'
    sld = open_slide(file)
    slide_props = sld.properties
    slide_width = int(slide_props['openslide.level[1].width']); slide_height = int(slide_props['openslide.level[1].height']) # dimensions at 10X magnification
    slide = np.array(sld.get_thumbnail(size=(slide_width, slide_height)))

    # LOAD SEGMENTATION MASK
    mask_file = gt_directory + case_code + '.png'
    mask = np.array(Image.open(mask_file))
    mask = mask[:slide_height, :slide_width] # reshape mask file to be same size as SVS

    if slide_height != mask.shape[0] or slide_width != mask.shape[1]:
        raise Exception("Input SVS file and segmentation image do not have the same dimensions")

    # extract patches
    patches = image_to_patches(slide, patch_size, stride)
    # create patches for segmentation masks
    mask_patches = image_to_patches(mask, patch_size, stride)
    # get rid of background patches
    tissue_patches, gt_patches = discard_background_patches(patches, mask_patches, patch_size)

    # Get labels
    labels = get_patch_labels(gt_patches, patch_size)

    # create dataframe to capture data information
    df_data = []
    col_names = ['File_name', 'Level0_factor', 'Level1_factor', 'Level2_factor', 'Level3_factor', 'Level0_height', 'Level0_width',
                'Level1_height', 'Level1_width', 'Level2_height', 'Level2_width', 'Level3_height', 'Level3_width',
                'Patch_size', 'Stride', 'Total_num_of_patches', 'Num_of_patches_discarded', '%_of_benign_tiles','%_of_tumourous_tiles']
    levels = len(sld.level_dimensions) # get num of levels
    if levels > 4:
        raise Exception("More than 4 levels contained in this svs file")
    factors = np.array(sld.level_downsamples)
    # start row
    data = [case_code.split('.')[0]] # first entry is the file name
    for r in range(levels):
        data.append(factors[r]) # add all downsample factors for each level
    if levels < 4:
        data.append(np.nan)
    total_num_patches = patches.shape[0]
    num_discarded_patches = total_num_patches - tissue_patches.shape[0]
    percentage_benign = (torch.sum(torch.eq(labels, 0))/len(labels)).item() * 100
    percentage_tumourous = (torch.sum(torch.eq(labels, 1))/len(labels)).item() * 100
    if levels < 4:
        data.extend([slide_props['openslide.level[0].height'], slide_props['openslide.level[0].width'], 
                slide_props['openslide.level[1].height'], slide_props['openslide.level[1].width'],
                slide_props['openslide.level[2].height'], slide_props['openslide.level[2].width'],
                np.nan, np.nan, patch_size, stride, total_num_patches, num_discarded_patches, percentage_benign, percentage_tumourous])
    else:
        data.extend([slide_props['openslide.level[0].height'], slide_props['openslide.level[0].width'], 
                slide_props['openslide.level[1].height'], slide_props['openslide.level[1].width'],
                slide_props['openslide.level[2].height'], slide_props['openslide.level[2].width'],
                slide_props['openslide.level[3].height'], slide_props['openslide.level[3].width'],
                patch_size, stride, total_num_patches, num_discarded_patches, percentage_benign, percentage_tumourous])
    df_data.append(data)
    df = pd.DataFrame(df_data, columns=col_names)
    
    return tissue_patches, labels, df

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
    
    i = 1 # Keeping track of files completed
    print("Starting processing")
    for code in file_codes:

        # LOAD SVS FILE
        file = images_directory + code + '.svs'
        sld = open_slide(file)
        slide_props = sld.properties
        slide_width = int(slide_props['openslide.level[1].width']); slide_height = int(slide_props['openslide.level[1].height']) # dimensions at 10X magnification
        slide = np.array(sld.get_thumbnail(size=(slide_width, slide_height)))
        print(f"File: {code}")
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
    print("Extracting patches")
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

    # # Method from Coudray et al.?
    # print(f"Total patches: {total_num_patches}")
    # for i in range(total_num_patches):
    #     # Get patch (torch tensor)
    #     im_patch = svs_patches[i, :, :, :]
    #     mask_patch = mask_patches[i, :, :]
    #     # Convert to numpy array
    #     p = im_patch.numpy()
    #     # Determine slides with low amount of information (>50% covered in background)
    #     # i.e. all values below 210 in RGB colour space
    #     avg = np.average(p)*255
    #     if avg < 210:
    #         tissue_patches = torch.cat((tissue_patches, im_patch.unsqueeze(0)), dim=0)
    #         seg_patches = torch.cat((seg_patches, mask_patch.unsqueeze(0)), dim=0)

    # FIRST METHOD TRIED
    for i in range(total_num_patches):
        # Get patch (torch tensor)
        im_patch = svs_patches[i, :, :, :]
        mask_patch = mask_patches[i, :, :]
        # Convert to numpy array
        p = mask_patch.numpy()
        # Count number of pixels in different classes
        # nblack_px = np.sum(p == 0) 
        nwhite_px = np.sum(p == 1); ngrey_px = np.sum(p == 0.5)
        # Calculate % of background (black) pixels
        # background_percentage = nblack_px/total_px
        tissue_percentage = (ngrey_px + nwhite_px)/total_px
        # Keep patch if background < 70% of the patch (i.e patch contains > 70% tissue), otherwise discard it
        if tissue_percentage > 0.7:
            tissue_patches = torch.cat((tissue_patches, im_patch.unsqueeze(0)), dim=0)
            seg_patches = torch.cat((seg_patches, mask_patch.unsqueeze(0)), dim=0)
        if i%1000==0: print(f"{i}/{total_num_patches}") 

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
        # Calculate number of white pixels in patch
        nwhite_px = np.sum(p == 1)
        # Calculate % of white pixels == tumourous pixels
        tumour_percentage = nwhite_px/total_px
        # Patch is considered to be labelled tumourous (1) if > 70% of the tissue in the patch is tumourous, otherwise it is labelled as normal tissue (0)
        if tumour_percentage > 0.5:
            labels.append(1)
        else:
            labels.append(0)
    
    # Convert to PyTorch tensor format
    labels = torch.tensor(labels, dtype=torch.long)

    return labels
