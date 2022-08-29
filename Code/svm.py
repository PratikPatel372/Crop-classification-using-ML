import gdal
import ogr
from sklearn import metrics
from sklearn import svm
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import sklearn.metrics as sm

inpRaster = r"D:\B.Tech AG\final project\data2\input\try.tif"
rasterDS = gdal.Open(inpRaster, gdal.GA_ReadOnly)

# Get spatial reference...........................................................................
geo_transform = rasterDS.GetGeoTransform()
projection = rasterDS.GetProjectionRef()

# Extract band's data and transform into a numpy array............................................
bandsData = []
for b in range(rasterDS.RasterCount):
    band = rasterDS.GetRasterBand(b + 1)
    band_arr = band.ReadAsArray()
    bandsData.append(band_arr)
bandsData = np.dstack(bandsData)
cols, rows, noBands = bandsData.shape

#Vector to raster convert and make a tif file from shp file.......................................

shapefile_train = r"D:\final project\data2\input\training.shp"
shapefile_test = r"D:\final project\data2\input\testing.shp"

rasterized_shp_train = r"D:\final project\data2\output\Rasterized_train.tif"
rasterized_shp_test = r"D:\final project\data2\output\Rasterized_test.tif"

def rasterizeVector(path_to_vector, cols, rows, geo_transform, projection, n_class, raster):
    lblRaster = np.zeros((rows, cols))
    inputDS = ogr.Open(path_to_vector)
    driver = gdal.GetDriverByName('MEM')
    # Define spatial reference
    for j in range(n_class):
        shpLayer = inputDS.GetLayer(0)
        class_id = j + 1
        shpLayer.SetAttributeFilter("Id = " + str(class_id))
        
        rasterDS = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
        rasterDS.SetGeoTransform(geo_transform)
        rasterDS.SetProjection(projection)
                
        gdal.RasterizeLayer(rasterDS, [1], shpLayer, burn_values=[class_id])
        
        bnd = rasterDS.GetRasterBand(1)
        bnd.FlushCache()
        arr = bnd.ReadAsArray()
        lblRaster += arr
        rasterDS = None
        # SAVE THE CREATED RASTER FILE AND 
        save_raster = gdal.GetDriverByName('GTiff').Create(raster, cols, rows, 1, gdal.GDT_UInt16)
        sband = save_raster.GetRasterBand(1)
        sband.WriteArray(lblRaster)#write a lblraster data in the band of output created file
        sband.FlushCache()
    return lblRaster

lblRaster_train = rasterizeVector(shapefile_train, rows, cols, geo_transform, projection, n_class=6, raster=rasterized_shp_train)
lblRaster_test = rasterizeVector(shapefile_test,rows,cols,geo_transform,projection,n_class=6,raster=rasterized_shp_test)

#Prepare training data (set of pixels used for training) and labels................................................
isTrain = np.nonzero(lblRaster_train)
isTest = np.nonzero(lblRaster_test)# istrain is indicate the location(in terms of tuples) of nonzero data

trainingLabels = lblRaster_train[isTrain]
testingLabels = lblRaster_test[isTest]

trainingData = bandsData[isTrain]
testingData = bandsData[isTest]

#Train SVM Classifier..............................................................................................
classifier = svm.SVC(C=1,gamma=0.1,kernel='linear',probability=True,random_state=0,shrinking=True,verbose=False)

classifier.fit(trainingData, trainingLabels)

print('Classifier fitting done!')

noSamples = rows * cols
flat_pixels = bandsData.reshape((noSamples, noBands))

#Apply this model to all the pixel...............................................................................
result = classifier.predict(flat_pixels)
p_vals = classifier.predict_proba(flat_pixels)

b_count = p_vals.shape[1]

classification = result.reshape((cols, rows, 1))
prob_array = p_vals.reshape((cols, rows, b_count))

#create a Tif file of results....................................................................................
outRaster = r"D:\final project\data2\output\SVM.tif"
out_prob = r"D:\final project\data2\output\Probability_Map.tif"

def createGeotiff(outRaster, data, geo_transform, projection, dtyp, bcount=1):
    # Create a GeoTIFF file with the given data
    driver = gdal.GetDriverByName('GTiff')
    rows, cols, _ = data.shape
    rasterDS = driver.Create(outRaster, cols, rows, bcount, dtyp)
    rasterDS.SetGeoTransform(geo_transform)
    rasterDS.SetProjection(projection)
    for i in range(bcount):
        band = rasterDS.GetRasterBand(i + 1)
        band.WriteArray(data[:, :, i])
        band.FlushCache()
    return 0


createGeotiff(outRaster, classification, geo_transform, projection, gdal.GDT_UInt16, bcount=1)
createGeotiff(out_prob, prob_array, geo_transform, projection, gdal.GDT_Float32, b_count)

#Accuracy check up.......................................................................
predicted_train_labels = classifier.predict(trainingData)
train_lbl_cnt = (np.unique(trainingLabels)).size

predicted_labels_test = classifier.predict(testingData)
test_lbl_cnt = (np.unique(testingLabels)).size

def check_accuracy(actual_labels, predicted_labels, label_count):
    error_matrix = np.zeros((label_count, label_count))
    for actual, predicted in zip(actual_labels, predicted_labels):
        error_matrix[int(actual) - 1][int(predicted) - 1] += 1
    return error_matrix

df = pd.DataFrame(check_accuracy(trainingLabels, predicted_train_labels, train_lbl_cnt))
df.to_csv(r'D:\final project\data2\output\CM_TRAIN.csv', index=False)#CM==confusion_matrix

df = pd.DataFrame(check_accuracy(testingLabels, predicted_labels_test, test_lbl_cnt))
df.to_csv(r'D:\final project\data2\output\CM_TEST.csv', index=False)

score_oa_train = classifier.score(trainingData, trainingLabels)
print('training set OA:', score_oa_train)

score_oa_test = classifier.score(testingData, testingLabels)
print('testing set OA:', score_oa_test)

Overall_train_accuracy = classifier.score(trainingData, trainingLabels)
Overall_test_accuracy = classifier.score(testingData, testingLabels)

#kappa value........................................................................
kappa_score_train = cohen_kappa_score(trainingLabels, predicted_train_labels)
print('kappa value training: ', kappa_score_train)

kappa_score_test = cohen_kappa_score(testingLabels, predicted_labels_test)
print('kappa value testing: ', kappa_score_test)

print("Classified Tiff Image created!")
img = plt.imread(r"D:\final project\data2\output\SVM.tif")
plt.imshow(img)