import pandas as pd
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

file=gdal.Open(r"D:\B.Tech AG\final project\data\clipped_ndvi.tif")
band=file.GetRasterBand(1)
ndvi=np.array(band.ReadAsArray(),dtype=float)

ndvi[np.where(ndvi<-1.00)]=None
ndvi[np.where(ndvi>1.00)]=None

ndvi=np.around(ndvi,3)

array=ndvi

array=array[~np.isnan(array)]

def grouping(array):
    array[np.where((array<=0) | (array==None))]=11
    array[np.where((array>0) & (array<=0.25))]=12
    array[np.where((array>0.25) & (array<=0.50))]=13
    array[np.where((array>0.50) & (array<=0.75))]=14
    array[np.where((array>0.75) & (array<=1))]=15
    
    counts={"non vegetation":0,"poor":0,"medium":0,"good":0,"excellent":0}
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i][j]==11:
                counts["non vegetation"]+=1
            if array[i][j]==12:
                counts["poor"]+=1
            if array[i][j]==13:
                counts["medium"]+=1
            if array[i][j]==14:
                counts["good"]+=1
            if array[i][j]==15:
                counts["excellent"]+=1
    #plot a classified map
    colors = ["gray", "yellow", "black", "green", "red"]
    cmap=ListedColormap(colors)
    labels=["No Vegetation","poor","medium","good","excellent"]
    legend_patches=[Patch(color=icolor,label=ilabel) for icolor,ilabel in zip(colors,labels)]
    fig,ax=plt.subplots(figsize=(10,10))
    ax.imshow(ndvi,cmap=cmap)
    ax.legend(handles=legend_patches,facecolor="white",bbox_to_anchor=(1.35,1))
    ax.set_axis_off()
    plt.show()
    return counts
    #counts=dicti
    
count=grouping(ndvi)

driver = gdal.GetDriverByName('GTiff')
file2 = driver.Create( r'D:\final project\crop_condition.tif', file.RasterXSize , file.RasterYSize , 1)
proj = file.GetProjection()
georef = file.GetGeoTransform()
file2.SetProjection(proj)
file2.SetGeoTransform(georef)
bnd=file2.GetRasterBand(1).WriteArray(ndvi)
file2.FlushCache()

def area_of_class(dicti):
    print("AREA:::::::::::::\n")
    print("Non vegetation area:",dicti['non vegetation']*10.32/100000,"sqm")
    print("Poor crop area:",dicti['poor']*10.32/100000,"sqm")
    print("Medium crop area:",dicti['medium']*10.32/100000,"sqm")
    print("Good crop area:",dicti['good']*10.32/100000,"sqm")
    print("excellent crop area:",dicti['excellent']*10.32/100000,"sqm")
    plt.bar(dicti.keys(),dicti.values())
    plt.xlabel('crop condition')
    plt.ylabel('no. of pixels')
    plt.title('Histogram')
    plt.show()

area_of_class(count)
