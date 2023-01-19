import os
import numpy as np
import matplotlib.pyplot as plt
import lxml.etree as ET
from geojson import Feature, Polygon, dump
from openslide import open_slide

# Get xml directory
xml_directory = '/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/raw/xml_labels'
# Save destination for geojson files
save_destination = '/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/interim/geojson_labels/'

# # Check number of files in directory
# files = [entry for entry in os.listdir(xml_directory) if os.path.isfile(os.path.join(xml_directory, entry))]
# num_files = len(files)
# # print(f'Total number of xml files found: {num_files}')

# # Keep only the .xml files within directory
# xml_files = [i for i in files if i.endswith('.xml')]
# # Verify only .xml files left (182 cases)
# print(f'Number of xml files: {len(xml_files)}')

def to_geojson(data_dest, save_dest):
    ''' 
    Function to convert given .xml files into .geojson format files.
    ---------
    Paramters:
    data_dest (str): path for folder containing xml files
    save_dest (str): path for where to save geojson files
    ---------
    Returns:
    None
    '''
    for file in os.listdir(data_dest):
        points = []
        features = []
        if os.path.isfile(os.path.join(data_dest, file)) and file.endswith('.xml'):
            # for file in xml_files:
            file_id = file.rstrip('.xml')
            # Get annotations from XML file
            tree = ET.parse(os.path.join(data_dest, file))
            root = tree.getroot()
            for Annotation in root.findall("./Annotation"): # for all annotations
                # iterate on all regions
                for Region in Annotation.findall("./*/Region"):
                    # extract region attributes
                    regionID = Region.attrib['Id']; regionArea = Region.attrib['Area']
                    # iterate all points in region
                    for Vertex in Region.findall("./*/Vertex"):
                        # get points
                        x_point = float(np.int32(np.float64(Vertex.attrib['X'])))
                        y_point = float(np.int32(np.float64(Vertex.attrib['Y'])))
                        points.append([x_point, y_point])
                    # Ensure that polygon produced is in closed form 
                    # i.e have same start and end point
                    starting_point = points[0]
                    if starting_point != points[-1]:
                        points.append(starting_point)
                    features.append(Feature(geometry=Polygon([points]), properties={"name": file_id, "region_id": regionID, "object_type": "annotation"}))
            with open(save_destination + file_id + '.geojson', 'w') as f:
                dump(features, f)

to_geojson(xml_directory, save_destination)
print("Done")