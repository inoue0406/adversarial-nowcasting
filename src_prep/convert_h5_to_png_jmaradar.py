# Reading JMA radar data in netcdf format
# take statistics for the whole Japanese region
import numpy as np
import pandas as pd
import h5py
from matplotlib import pyplot as plt

from PIL import Image

import sys
import os.path

def convert_np_to_png(X,channels=3):
    rmax = 201.0 # max rainfall intensity
    a = 1/3.0    # "cubic root"
    Y = (X/rmax)**a # root scaling
    Y = Y*255 # to integer scale
    Y = Y.transpose((1,2,0)) # channels last for plotting
    if channels == 3:
        Y = np.concatenate([Y,Y,Y],axis=2)
    return Y

if __name__ == '__main__':
    #for year in [2015,2016,2017]:
    dir_data = "../data/data_alljapan/"
    dir_image = "../data/data_alljapan_image/"
    df = pd.read_csv("../data/train_alljapan_2yrs_JMARadar.csv")
    df = df.iloc[::-1]
    img_size = 256

    #for index, row in df.iterrows():
    for index, row in df.iterrows():
        print("processing",row)
        fname = row["fname"]
        fnext = row["fnext"]
        for fn in [fname,fnext]:
            fpath = dir_data + fn
            h5file = h5py.File(fpath,'r')
            rain = h5file['R'][()]
            rain = np.maximum(rain,0) # replace negative value with 0
            rain = rain[:,None,:,:] # add "channel" dimension as 1
            h5file.close()
            for time in range(rain.shape[0]):
                rain_img = convert_np_to_png(rain[time,:,:,:])
                im = Image.fromarray(np.uint8(rain_img))
                im = im.resize((img_size,img_size))
                str_time = "_%02d" % time
                png_fname = dir_image + fn.replace(".h5",str_time + ".png")
                im.save(png_fname)
