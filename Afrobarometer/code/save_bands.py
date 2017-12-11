import csv
import numpy as np
import matplotlib.pyplot as plt
import gdal


landsat_path = '/mnt/mounted_bucket/l8_median_afrobarometer_multiband_500x500_';

def read(tif_path, H,W):
    gdal_dataset = gdal.Open(tif_path)
    x_size, y_size = gdal_dataset.RasterXSize, gdal_dataset.RasterYSize
    gdal_result = gdal_dataset.ReadAsArray((x_size-W)//2, (y_size-H)//2, W,H)

    return np.transpose(gdal_result, (1,2,0))

i = 0

log = ""
for i in range(0, 7022):
    tifpath = landsat_path+areas[i]+'.0.tif';
    try:
        img = read(tifpath, 500, 500)
        path = 'saved_images/img_'+areas[i];
        np.save(path, img)
    except:
        log = log + 'failed: ' + areas[i] + '\n'

text_file = open("log.txt", "w");
text_file.write(log);
text_file.close;
        
