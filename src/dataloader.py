import cv2
from random import shuffle
import os
import numpy as np

def read_test_data(dataroot, nsamples, classes_to_load=None, return_filenames=False):
    # Function to read the test data
    
    class_list = os.listdir(dataroot)
    data_list = []
    label_list = []
    
    for cl in class_list:
        if classes_to_load is not None:
            if int(cl) not in classes_to_load:
                continue
                
        class_path = os.path.join(dataroot, cl)
        filelist = os.listdir(class_path)
        filelist = [os.path.join(cl, f) for f in filelist]
        data_list.extend(filelist)
        label_list.extend(np.ones(len(filelist), dtype=np.int)*int(cl))
    
    data_list_new = [os.path.join(dataroot, f) for f in data_list]
    data_list = data_list_new
    
    index_list = np.random.permutation(len(data_list))
    data_list_new = [data_list[ind] for ind in index_list]
    label_list_new = [label_list[ind] for ind in index_list]
    data_list = data_list_new
    label_list = label_list_new

    img_list = []
    filenames_loaded = []
    for i in range(nsamples):
        filename = data_list[i]
        filenames_loaded.append(filename)
        img = cv2.imread(os.path.join(dataroot, filename), 0)
        img = cv2.resize(img, (28, 28))
        img = np.reshape(img, (1, 28*28))
        img = img.astype(np.float32)
        img = img/255.0
        img_list.append(img)
    
    if return_filenames:
        return img_list, filenames_loaded
    else:
        return img_list



