#!/usr/bin/env python
# coding: utf-8




import numpy as np
import cv2
import pandas as pd
import json
import matplotlib.pyplot as plt



def annoParse(path):
    """
    Parse the annotation file
    Args:
        path: file path to the annotation file
    Return:
        LUT: a pandas dataframe, unifying all information into a Look Up Table
        categories: a dictionary mapping category id to category name
    """
    with open(path) as fh:
        annotation = json.load(fh)
    # the annotation file has the following keys:
    # info: empty field
    # images: specifies image name, image id, height and width
    # license: empty field
    # categories: types of defects
    # annotations:

    # id-name pairs; id-dimension pairs
    id_name = {}
    id_dimension = {}
    for i in annotation['images']:
        id_name[i['id']] = i['file_name']
        id_dimension[i['id']] = (i['height'], i['width'])
    # id-category pairs; id-bbox pairs
    id_category = {}
    id_bbox = {}
    for i in annotation['annotations']:
        if i['image_id'] in id_category.keys():
            id_category[i['image_id']] = id_category[i['image_id']] + [i['category_id']]
        else:
            id_category[i['image_id']] = [i['category_id']]
        if i['image_id'] in id_bbox.keys():
            id_bbox[i['image_id']] = id_bbox[i['image_id']] + [i['bbox']]
        else:
            id_bbox[i['image_id']] = [i['bbox']]
    # merge information
    LUT = pd.DataFrame()
    LUT['id'] = sorted(id_name.keys())
    LUT['file_name'] = [id_name[x] for x in LUT['id']]
    LUT['height'] = [id_dimension[x][0] for x in LUT['id']]
    LUT['width'] = [id_dimension[x][1] for x in LUT['id']]
    LUT['category'] = [id_category[x] for x in LUT['id']]
    LUT['bbox'] = [id_bbox[x] for x in LUT['id']]
    #category mapping
    categories = {}
    for i in annotation['categories']:
        categories[i['id']] = i['name']
    return LUT, categories





def drowBbox(folderPath, annFile, outputPath):
    """
    Drow the annotation boxes.
    Args:
        folderPath: string, file path to the images path.
        annFile: string, annotation file.
        outputPath: string, output images path.
    Return:
        operation function but return last image with numpy style.
    """
    LUT, categories = annoParse(annFile)
    font = cv2.FONT_HERSHEY_TRIPLEX
    for index, row in LUT.iterrows():
        #change the condition according to your situation
        #this filter non-bottle cap images
        if index > 5:
            break
        if row['height'] > 1000:
            continue
        npimg = cv2.imread(folderPath+ row['file_name'])
        orgimg = npimg.copy()
        for i, box in enumerate(row['bbox']):
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(npimg, (x,y), (x+w, y+h), (0,255,0), 1)
            cv2.putText(npimg, str(row['category'][i]), (x+w, y+h), font, 1, (255,0,255), 2)
        npimg = np.hstack((orgimg,npimg))
        cv2.imwrite(outputPath+ row['file_name'], npimg)
    
    return npimg




images_path = './data/chongqing1_round1_train1_20191223/images/'
box_images_path = './data/test_box_image/'
annFile = './data/chongqing1_round1_train1_20191223/annotations.json'
npimg = drowBbox(images_path, annFile, box_images_path)
plt.imshow(npimg)
plt.show()

