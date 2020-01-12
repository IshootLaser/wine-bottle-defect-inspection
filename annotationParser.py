# provide method to load json file, given a file path
import json
import pandas as pd

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

if __name__ == '__main__':
    LUT, categories = annoParse('./data/chongqing1_round1_train1_20191223/annotations.json')
    print(LUT.head())
    print(categories)
