import torch
import torch.nn as nn
import os
import numpy as np
import random
import json
import pandas as pd
from openslide import open_slide
from PIL import Image
from torch.utils.data import Dataset
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from skimage.transform import resize
import matplotlib.colors as colors
from torchvision import transforms
import sys
import timm
from torchvision.transforms import InterpolationMode
from PIL import Image
sys.path.append('/Users/alexandrasmith/Desktop/Workspace/Projects/masters/src')
from models.inception_model import InceptionV3

def main():

    cd, custom_grn, custom_prpl2yel = define_colours()

    PATCH_SIZE=256
    STRIDE=PATCH_SIZE
    num_classes=2
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # MODEL 1
    model_path = '/Users/alexandrasmith/Desktop/Workspace/Projects/masters/models/silver-blaze-129_model_weights.pth'
    model = load_trained_model(num_classes, model_path)
    model = model.to(device)
    # MODEL 2
    model2_path = '/Users/alexandrasmith/Desktop/Workspace/Projects/masters/models/trim-valley-173_model_weights.pth'
    model2 = timm.create_model('inception_v4', pretrained=False, num_classes=num_classes)
    model2.classif = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.7),
        nn.Linear(model2.get_classifier().in_features, num_classes)
    )
    model2.load_state_dict(torch.load(model2_path, map_location=torch.device('cpu')))
    model2.eval()
    model2 = model2.to(device)

    file_path = '/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/raw/HER2DataInfo.xlsx'
    df = pd.read_excel(file_path)

    df.drop(df.index[-2:], inplace=True)

    df['Case ID'] = df['Case ID'].str.replace('TCGA-','')
    df['Case ID'] = df['Case ID'].str.replace('-01Z-00-DX1','')

    df['Clinical.HER2.status'] = df['Clinical.HER2.status'].map({'Negative': 0, 'Positive': 1}).astype(int)

    mac_dir = '/Volumes/AlexS/MastersData/from Yale/PKG - HER2 tumor ROIs_v3/pkg_v3/Yale_HER2_cohort/SVS'
    # List all files in the directory
    cases_list = os.listdir(mac_dir)

    statuses = [img_path.split('_')[0] for img_path in cases_list]
    case_names = [img_path.split('/')[-1] for img_path in cases_list]
    print(statuses)
    df = pd.DataFrame({'Case': case_names, 'HER2.status': statuses})
    print(df)
    df['HER2.status'] = df['HER2.status'].map({'Neg': 0, 'Pos': 1}).astype(int)
    print(df)



    for case in cases_list:
        print(f"Case: {case}")
        slide_path = os.path.join(mac_dir, case)
        sld = open_slide(slide_path)
        slide_props = sld.properties
        slide_width = int(slide_props['openslide.level[1].width']); slide_height = int(slide_props['openslide.level[1].height']) # dimensions at 10X magnification
        slide = np.array(sld.get_thumbnail(size=(slide_width, slide_height)))
        image_size = slide.shape
        #### FIX 
        mask = visualise_gt_classes(case)
        patches, positions = image_to_patches_with_positions(slide, PATCH_SIZE, STRIDE)
        gt_patches, gt_positions = image_to_patches_with_positions(mask, PATCH_SIZE, STRIDE)

        labels = get_patch_labels(patches, gt_patches, PATCH_SIZE)
        patch_objects = get_patch_objects(patches, positions, labels)

        heatmap_probs, heatmap_classes = her2_inference(image_size[0:2], patch_objects, model, model2)

        prob_values = heatmap_probs[(heatmap_probs > 0) & (heatmap_probs <= 1)]
        # Calculate the average
        average = np.mean(prob_values) if len(prob_values) > 0 else 0




class Patch:
    '''
    Store properties of each patch
    '''

    def __init__(self, image, position, label, size=256):
        self.image = image
        self.position = position
        self.size = size
        self.label = label
        self.probability = None
        self.prediction = None
        self.is_background = False
        self.status_probability = None
        self.status_prediction = None
    
    def set_probability(self, probability):
        self.probability = probability

    def get_prediction(self):
        self.prediction = 1 if self.probability >= 0.5 else 0
    
    def set_patch_background(self):
        if self.label == -1:
            self.is_background = True

def define_colours():
    M_darkpurple = '#783CBB'
    M_lightpurple = '#A385DB'
    # M_green = '#479C8A'
    M_green = '#0a888a'
    M_yellow = '#FFDD99'
    M_lightpink = '#EFA9CD'
    M_darkpink = '#E953AD'

    colour_list = [M_lightpink, M_green, M_darkpurple, M_darkpink, M_lightpurple, M_yellow]
    cd = {'lightpink': M_lightpink, 'lightpurple': M_lightpurple, 'green': M_green, 'purple': M_darkpurple, 'pink': M_darkpink, 'yellow': M_yellow}
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colour_list)
    
    # Create custom gradient colourmap
    rgb_tuples =[(255, 255, 255),(249, 251, 251),(243, 247, 246),(237, 242, 242),(231, 238, 238),
                 (225, 234, 234),(219, 230, 230),(213, 226, 225),(207, 221, 221),(202, 217, 217),
                 (196, 213, 213),(190, 209, 209),(184, 205, 205),(178, 201, 201),(172, 197, 196),
                 (167, 193, 192),(161, 188, 188),(155, 184, 184),(149, 180, 180),(144, 176, 176),
                 (138, 172, 172),(132, 168, 168),(126, 164, 164),(121, 160, 160),(115, 157, 156),
                 (109, 153, 153),(103, 149, 149),(98, 145, 145),(92, 141, 141),(86, 137, 137),
                 (80, 133, 133),(74, 129, 130),(68, 125, 126),(61, 122, 122),(55, 118, 118),
                 (48, 114, 115),(41, 110, 111),(33, 106, 107),(24, 103, 104),(11, 99, 100)]
    # Normalize RGB color values to the range [0, 1]
    normalised_colours = [[r / 255, g / 255, b / 255] for r, g, b in rgb_tuples]
    custom_grn = LinearSegmentedColormap.from_list('Grn', normalised_colours, N=len(normalised_colours))

    # PURPLE TO YELLOW
    rgb_tuples =[(105, 43, 175),(121, 40, 172),(135, 38, 169),(148, 36, 166),(160, 34, 163),
                (171, 33, 159),(181, 33, 156),(190, 35, 152),(199, 37, 148),(207, 41, 144),
                (214, 45, 140),(221, 51, 136),(227, 56, 132),(233, 63, 129),(238, 69, 125),
                (242, 76, 122),(246, 84, 119),(250, 91, 115),(253, 98, 113),(255, 106, 110),
                (255, 113, 108),(255, 120, 106),(255, 128, 104),(255, 135, 103),(255, 143, 102),
                (255, 150, 102),(255, 158, 102),(255, 165, 103),(255, 172, 104),(255, 179, 106),
                (255, 186, 108),(255, 193, 111),(255, 200, 114),(255, 207, 118),(255, 214, 123),
                (255, 221, 128),(255, 228, 133),(255, 234, 139),(255, 241, 146),(255, 247, 153)]

    # Normalize RGB color values to the range [0, 1]
    normalised_colours = [[r / 255, g / 255, b / 255] for r, g, b in rgb_tuples]

    # Create separate lists for R, G, and B values
    r_values = [rgb[0] for rgb in normalised_colours]
    g_values = [rgb[1] for rgb in normalised_colours]
    b_values = [rgb[2] for rgb in normalised_colours]
    # Create a new array containing the three lists
    rgb_array = [r_values, g_values, b_values]

    custom_prpl2yel = LinearSegmentedColormap.from_list('Prple2Yel', normalised_colours, N=len(normalised_colours))
    
    return cd, custom_grn, custom_prpl2yel

def load_trained_model(num_classes, model_path): 
    model = InceptionV3(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    return model

def scale_tensor(tensor: torch.Tensor):
    '''
    Scale a tensor to the range [0, 1]
    '''
    minn = tensor.min()
    maxx = tensor.max()
    tensor = (tensor - minn)/(maxx - minn)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def image_to_patches_with_positions(image, patch_size: int, stride: int):
    '''
    Function for splitting an input image into patches.

    Parameters:
    image: input image to split
    patch_size (int): dimension, patches will be square
    stride (int): controls overlap between patches

    Returns:
    Tensor of patches with shape (num_patches, im_dim (if applicable), patch_size, patch_size) with their positions in the original image
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
    # Working with greyscale image
    else:
        # Extract patches
        patches = im.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
        # Reshape tensor into tensor of shape (num_patches, patch_size, patch_size)
        patches = patches.contiguous().view(-1, patch_size, patch_size)

    # Calculate the number of patches in each dimension
    height, width = image.shape[:2]
    num_patches_h = (height - patch_size) // stride + 1
    num_patches_w = (width - patch_size) // stride + 1

    # Generate positions of the patches
    positions = []
    for h in range(num_patches_h):
        for w in range(num_patches_w):
            # Calculate the top-left position of the current patch
            top = h * stride
            left = w * stride
            positions.append((top, left))

    return patches, positions

def get_patch_objects(patches, positions, labels):
    patch_objects = []
    for patch, position, label in zip(patches, positions, labels):
        patch_object = Patch(image=patch, position=position, label=label)
        patch_objects.append(patch_object)
    return patch_objects

def check_if_background(patch):
    '''
    Given a patch, return whether it should be classified as a background patch or not.
    '''
    im = np.array(patch) * 255
    pixels = np.ravel(im)
    mean = np.mean(pixels)
    is_background = mean >= 220
    return is_background

def choose_random_image(images, seed):
    '''
    Given list of image paths.
    Choose a SVS file randomly to perform inference and produce heatmap.

    Returns:
    random_case: image case code
    sld: SVS slide object
    slide: Level 1 image from SVS
    '''
    
    mac_dir = '/Volumes/AlexS/MastersData/SVS files/'
    # List all files in the directory
    file_list = os.listdir(mac_dir)
    
    random.seed(seed)
    cases = [img_path.split('/')[-1] for img_path in images]
    # Choose random image file
    random_case = random.choice(cases)
    
    case_id = 'TCGA-' + random_case + '-01Z-00-DX1'
    img_path = [file for file in file_list if file.startswith(case_id)][0]

    slide_path = os.path.join(mac_dir, img_path)
    sld = open_slide(slide_path)
    slide_props = sld.properties
    slide_width = int(slide_props['openslide.level[1].width']); slide_height = int(slide_props['openslide.level[1].height']) # dimensions at 10X magnification
    slide = np.array(sld.get_thumbnail(size=(slide_width, slide_height)))

    return random_case, sld, slide

def check_seg_accuracy(label_directory, case_code, patches, model):
    '''
    Determine the accuracy of the predicted labels for a specific case in comparison to ground truth patch labels.
    ** Calculate all metrics
    '''
    
    labels = torch.load(label_directory + case_code + '.pt')
    print(labels.size())
    count_tissues = 0
    for patch in patches:
        image = patch.image
        is_background = check_if_background(image)
        if not is_background:
            count_tissues += 1
    print(f"Number of tissue tiles are {count_tissues}")

    # return acc

def get_prediction(patch, output, istumour):
    # get predictions for patch
    if istumour:
        probabilities = torch.softmax(output, dim=1) # Post-process the predictions
        patch.status_probability = probabilities[0][1].item()
        predicted_class = torch.argmax(probabilities, dim=1).item() + 1 # 1 for - and 2 for + since 0 is normal
        patch.prediction = predicted_class
    else:
        probabilities = torch.softmax(output, dim=1) # Post-process the predictions
        patch.probability = probabilities[0][1].item()
        predicted_class = torch.argmax(probabilities, dim=1).item()
        patch.prediction = predicted_class

def create_heatmaps(image_size, patches):
    '''
    Generate heatmap arrays based on patch predictions.
    Returns 2D heatmaps the same dimensions as original image.

    Return: heatmap containing probabilities, and heatmap containing predicted classes (None, 0, 1) or (None, 0, 1(-), 2(+))
    '''
    heatmap_probabilities = np.full(image_size, -1, dtype=np.float64)
    heatmap_classes = np.full(image_size, None)
    true_labels_map = np.full(image_size, None)
    for patch in patches:
        i, j = patch.position
        h, w = patch.image.size()[1], patch.image.size()[2]
        heatmap_probabilities[i:i+h, j:j+w] = patch.probability
        heatmap_classes[i:i+h, j:j+w] = patch.prediction
        true_labels_map[i:i+h, j:j+w] = patch.label
    
    return heatmap_probabilities, heatmap_classes, true_labels_map

def create_her2_heatmaps(image_size, patches):
    '''
    Generate heatmap arrays based on patch predictions.
    Returns 2D heatmaps the same dimensions as original image.

    Return: heatmap containing probabilities, and heatmap containing predicted classes (None, 0, 1) or (None, 0, 1(-), 2(+))
    '''
    heatmap_probabilities = np.full(image_size, -1, dtype=np.float64)
    heatmap_classes = np.full(image_size, None)
    # true_labels_map = np.full(image_size, None)
    for patch in patches:
        i, j = patch.position
        h, w = patch.image.size()[1], patch.image.size()[2]
        heatmap_probabilities[i:i+h, j:j+w] = patch.status_probability
        heatmap_classes[i:i+h, j:j+w] = patch.prediction
        # true_labels_map[i:i+h, j:j+w] = patch.label
    
    return heatmap_probabilities, heatmap_classes

def inference(image_size, patches, model):
    '''
    Takes in Patch objects and makes predictions for each patch if it is not classified as a background patch.
    Then uses those predictions to create a heatmap.
    Returns:
    heatmap_probs: contains probabilites from each patch, and -1 for background pixels.
    heatmap_classes: contains class assignments from each patch, with None for background pixels.
    '''
    # apply transforms to patch
    # Inception
    data_transforms = transforms.Compose([
                transforms.Resize(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # inception
            ])
    for patch in patches:
        image = patch.image # tensor
        # is_background = check_if_background(image)
        patch.set_patch_background()
        is_background = patch.is_background
        t = transforms.ToPILImage()
        image = t(image)
        image = data_transforms(image)
        if not is_background:
            model.eval()
            with torch.no_grad():
                # Forward pass
                output = model(image.unsqueeze(0))
                if isinstance(output, tuple):
                    output = output[0]
                # output = model(image.unsqueeze(0))
            get_prediction(patch, output, istumour=False) 
        else:
            patch.probability = -1
            patch.prediction = None
    heatmap_probs, heatmap_classes, true_labels_map = create_heatmaps(image_size, patches)
    return heatmap_probs, heatmap_classes, true_labels_map

def her2_inference(image_size, patches, model1, model2):
    '''
    Model 1: normal vs tumour
    Model 2: HER2 status
    '''
    # apply transforms to patch
    data_transforms = transforms.Compose([
                transforms.Resize(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # inception
            ])
    for patch in patches:
        image = patch.image # tensor
        is_background = check_if_background(image)
        t = transforms.ToPILImage()
        image = t(image)
        image = data_transforms(image)
        # normal vs tumour
        if not is_background:
            model1.eval()
            with torch.no_grad():
                output = model1(image.unsqueeze(0))
                if isinstance(output, tuple):
                    output = output[0]
            get_prediction(patch, output, istumour=False)
                # total_px = patch_size*patch_size
                # nwhite_px = np.sum(p == 1); ngrey_px = np.sum(p == 0.5)
                # tissue_percentage = (ngrey_px + nwhite_px)/total_px
                # if tissue_percentage > 0.7:
            # her2 status
            # if patch.prediction == 0:
            #     patch.probability = -1 # set probability = -1 for plotting, prediction will = 0
            if patch.prediction == 1: # if tumourous
                model2.eval()
                with torch.no_grad():
                    output = model2(image.unsqueeze(0))
                    if isinstance(output, tuple):
                        output = output[0]
                get_prediction(patch, output, istumour=True)
        else:
            patch.probability = -1
            patch.prediction = None
    heatmap_probabilities, heatmap_classes= create_her2_heatmaps(image_size, patches)
    return heatmap_probabilities, heatmap_classes

def visualise_thumbnail(im, title):
    '''
    Resize an image for visualisation purposes and display it.
    '''
    new_width = 5000
    original_width, original_height = im.shape
    new_height = int(new_width * (original_width / original_height))
    thumbnail = resize(im.astype(float), (new_height, new_width))
    plt.imshow(thumbnail, cmap="gray")
    plt.title(title)
    plt.axis('off')

def visualise_classification_map(heatmap):
    '''
    Create visual of predicted classes. 
    Converts the given array containing None, 0, 1 to the values 0, 0.5, 1 needed for display purposes.
    Displays black/grey/white segmentation map, to compare to original ground truth mask.
    Plot original mask to compare.
    '''
    output = np.full(heatmap.shape, None)
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            if heatmap[i, j] == None:
                output[i, j] = 0 # background
            elif heatmap[i, j] == 0:
                output[i, j] = 0.5 # normal tissue
            elif heatmap[i, j] == 1:
                output[i, j] = 1 # malignant tissue
            else:
                raise Exception("Heatmap of predicted classes contains values other than None, 0 or 1")
    visualise_thumbnail(output, "Predicted class labels")

def visualise_gt_classes(case):

    dir = '/Volumes/AlexS/MastersData/QupathLabels/export10x/'
    # List all files in the directory
    file_list = os.listdir(dir)

    case_id = 'TCGA-' + case + '-01Z-00-DX1'
    img_path = [file for file in file_list if file.startswith(case_id)][0]
    gt_img = os.path.join(dir, img_path)
    mask = np.array(Image.open(gt_img))

    # visualise_thumbnail(mask, "Ground truth")

    return mask

def visualise_probabilities_map(heatmap_probs, colourmap):
    '''
    Display heatmap of predicted probabilities, shown only on a white background.
    '''
    values = heatmap_probs.copy()
    values = np.where(values == -1, np.nan, values)

    plt.imshow(values, cmap=colourmap, vmin=0, vmax=1)
    cbar = plt.colorbar()
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Benign', 'Malignant'])
    plt.title("Predicted tumour probabilities")
    plt.axis('off')

def visualise_heatmap_over_image(case, slide, image, heatmap_probabilities, colourmap, prob):
    '''
    Returns heatmap of probabilities overlaid on the original histopathology image.
    Given
    slide: OpenSlide slide object (full size)
    image: thumbnail image (from level 1)
    '''
    image_size = image.shape[0:2]
    # Create mask for probabilities
    mask = heatmap_probabilities < prob
    heatmap = heatmap_probabilities.copy()
    heatmap[mask] = np.nan

    # display purposes
    new_width = 5000
    original_width, original_height = image_size
    new_height = int(new_width * (original_width / original_height))
    slide_thumbnail = np.array(slide.get_thumbnail(size=(new_width, new_height)))
    thumbnail = resize(heatmap.astype(float), (new_height, new_width))
    fig=plt.figure()
    ax = plt.axes()
    im = ax.imshow(slide_thumbnail)
    im = ax.imshow(thumbnail, cmap=colourmap, vmin=0, vmax=1)
    plt.axis('off')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    plt.show()
    # plt.savefig('/Users/alexandrasmith/Downloads/' + case + '_heatmap.svg', format="svg", bbox_inches='tight', pad_inches=0.0)      

def get_patch_labels(patches, gt_patches, patch_size):
    total_num_patches = gt_patches.shape[0]
    total_px = patch_size*patch_size
    labels = []

    for i in range(total_num_patches):
        # Get patch (torch tensor)
        im_patch = patches[i, :, :, :]
        mask_patch = gt_patches[i, :, :]
        # Convert to numpy array
        p1 = mask_patch.numpy()
        p2 = im_patch.numpy()
        avg = np.average(p2)*255
        if avg < 220:
            # Calculate number of white pixels in patch
            nwhite_px = np.sum(p1 == 1)
            # Calculate % of white pixels == tumourous pixels
            tumour_percentage = nwhite_px/total_px
            # Patch is considered to be labelled tumourous (1) if > 50% of the tissue in the patch is tumourous, otherwise it is labelled as normal tissue (0)
            if tumour_percentage > 0.5:
                labels.append(1)
            else:
                labels.append(0)
        else:
            labels.append(-1)

    # Convert to PyTorch tensor format
    labels = torch.tensor(labels, dtype=torch.long)

    return labels

def TP_map(case, slide, image, pred_classes, labels):
    '''
    Given predicted classes and true labels for an instance.
    Return an image that displays TP, FN, FP, TN.
    '''
    
    image_size = image.shape[0:2]
    # print(pred_classes.shape)

    values = pred_classes.copy()
    values = np.where(values == -1, np.nan, values)

    # display purposes
    new_width = 5000
    original_width, original_height = image_size
    new_height = int(new_width * (original_width / original_height))
    slide_thumbnail = np.array(slide.get_thumbnail(size=(new_width, new_height)))
    # thumbnail = resize(values.astype(float), (new_height, new_width))
    
    for i in range(pred_classes.shape[0]):
        for j in range(pred_classes.shape[1]):
            # true positive
            if pred_classes[i, j] == 1 and labels[i, j] == 1:
                values[i, j] = 0 # TP
            # false negative
            if pred_classes[i, j] != 1 and labels[i, j] == 1:
                values[i, j] = 1 # FN
            # false positive
            if pred_classes[i, j] == 1 and labels[i, j] != 1:
                values[i, j] = 2 # FP
            # true negative
            if pred_classes[i, j] == 0 and labels[i, j] == 0:
                values[i, j] = 3 # TN

    thumbnail = resize(values.astype(float), (new_height, new_width))

    # TP, FN, FP, TN
    colours = [cd['green'], cd['pink'], cd['yellow'], cd['lightpurple']]

    cmap = ListedColormap(colours)

    fig=plt.figure()
    ax = plt.axes()
    im = ax.imshow(slide_thumbnail)
    im = ax.imshow(thumbnail, cmap=cmap)
    plt.axis('off')
    # cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    # plt.show()
    fig.savefig('/Users/alexandrasmith/Downloads/' + case + '_TP_map.svg', format="svg", bbox_inches='tight', pad_inches=0.0)


if __name__ == '__main__':
    main()