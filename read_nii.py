import os
import glob
import nibabel as nib
import cv2
import numpy as np
from scipy import ndimage 
#from ndimage import largest_connected_component
from skimage import morphology


data_type = ".nii"
data_path_012 = '/home/mio/zz_heart/result_segmentation/result_012_/20_40'

data_012= glob.glob(data_path_012+"/pre/*")
dice_012,dice_102,dice_201,dice_merge_all=0,0,0,0

for data_name in data_012:        
    name=data_name[data_name.rindex("/")+0:-4]
    images_012=nib.load(data_path_012+"/pre"+name+data_type).get_data()   
    images_gt=nib.load(data_path_012+"/gt"+name+"_gt"+data_type).get_data() 


    dice012 = 2.0*(np.sum(np.logical_and(images_012, images_gt)))/(np.sum(images_012) + np.sum(images_gt))  
    dice_012+=dice012
    

print('dice_012:'+str(dice_012/20)+'\n')




