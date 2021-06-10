import os
import json
import glob
import numpy as np
import collections
import pandas as pd
import matplotlib.pyplot as plt

class compute_annot_stats():
    def __init__(self):
        ()

#computing stats for coco

    #parameters
        #data - A json file that is already called
    #returns
        #ids= A list of category ids from the json file
        #num_cat - The number of categories
        #cat_names - Teh name of categories
        #cat_ids - Id of categories used in the annotation file
        #cat_dict - A dictionary with category name as key and frequency of annotation as values
        #class_weight - A dictionary with category name as key and weights for each category as values
        #number_of_imgs - Number of images for which annotaions exist

    def coco_stat(self, data):
        ids = [annot["category_id"] for annot in data["annotations"]]
        number_of_imgs= len([image for image in data["images"]])
        cat_names = [cat["name"] for cat in data["categories"]]
        cat_ids = np.unique(ids).tolist()
        cat_dict = collections.Counter(ids)
        class_weight = self.compute_weights(cat_dict, cat_names)
        num_cat=len(cat_ids)

        return ids, num_cat, cat_names, cat_ids, cat_dict, class_weight, number_of_imgs

#computing stats for coco

    # parameters
    #   data - A json file that is already called
    # returns
        # ids= A list of category ids from the json file
        # num_cat - The number of categories
        # cat_names - Teh name of categories
        # cat_ids - Id of categories used in the annotation file
        # cat_dict - A dictionary with category name as key and frequency of annotation as values
        # class_weight - A dictionary with category name as key and weights for each category as values
        # number_of_imgs - Number of images for which annotaions exist

    def yolo_stat(self, datadir):
        os.chdir(datadir)
        ids=[]
        imgid=1
        for file in glob.glob("*" + ".txt"):
            with open(file, "r") as inputfile:
                for line in inputfile:
                    id=line.split()[0]
                    ids.append(id)
            imgid+=1
        number_of_imgs=imgid
        cat_ids = np.unique(ids).tolist()
        cat_dict = collections.Counter(ids)
        cat_names=cat_ids
        class_weight = self.compute_weights(cat_dict, cat_names)
        num_cat = len(cat_ids)
        cat_names=[]

        return ids, num_cat, cat_names, cat_ids, cat_dict, class_weight, number_of_imgs

#computing weights
    #parameters
        #cat_names - Teh name of categories
        #cat_dict - A dictionary with category name as key and frequency of annotation as values
    #returns
        #class_weight - A dictionary with category name as key and weights for each category as values

    def compute_weights(self, cat_dict, cat_names):
        max_value = float(max(cat_dict.values()))

        weight = {class_id: int(max_value / value) for class_id, value in zip(cat_names, cat_dict.values())}
        return weight

#creating dictionary that stores info for all annotation stats with the results obtained from the functions above
    def create_info_dict(self, ids, num_cat, cat_names, cat_ids, cat_dict, class_weight, number_of_imgs):
        my_dict = {}
        my_dict["# of classes"] = num_cat
        if len(cat_names)!=0:
            my_dict["class names"] = cat_names
        else:
            cat_names=cat_ids
        my_dict["class ids"] = cat_ids
        my_dict["annotation classwise"] = {class_name: value for class_name, value in zip(cat_names, cat_dict.values())}
        my_dict["annotation classwise(%)"] = {class_name: np.round(value/len(ids)*100,2) for class_name, value in zip(cat_names, cat_dict.values())}
        my_dict["total annotation #"] = np.sum([value for value in cat_dict.values()])
        my_dict["# of images"] = number_of_imgs
        my_dict["mean annotation"] = np.sum([value for value in cat_dict.values()])/ number_of_imgs
        my_dict["class_weights"] = class_weight
        return my_dict

## Below two functions are the main function the user can call for computing statistics for coco and yolo based annotations

#parameter
    #file- A path to cocojson file
# returns
    # my_dict- A dictionary that contains statistics
    # my_df- A dataframe that contains statistics
    def compute_stats_coco(self, file, plot=False):
        with open(file, encoding='utf-8', errors='ignore') as json_data:
            data = json.load(json_data, strict=False)
        my_dict=self.create_info_dict(*self.coco_stat(data))

        my_df=pd.DataFrame(my_dict.items(), columns=["parameter", "value"])
        print(my_df)

        if plot:
            ids = self.coco_stat(data)[0]

            plt.hist(ids)
            plt.show()
        return my_dict, my_df


#parameter
    #file- A path to the directory that has textfiles
# returns
    # my_dict- A dictionary that contains statistics
    # my_df- A dataframe that contains statistics
    def compute_stats_yolo(self, textdir, plot=False):
        my_dict=self.create_info_dict(*self.yolo_stat(textdir))
        my_df=pd.DataFrame(my_dict.items(), columns=["parameter", "value"])
        print(my_df)
        if plot:
            ids = self.yolo_stat(textdir)[0]
            plt.hist(ids)
            plt.show()

        return my_dict, my_df

# data_path="C:/Research/cocofile.json"
# stat=compute_annot_stats()
# stat.compute_stats_coco(data_path)
