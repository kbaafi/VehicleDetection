
**Vehicle Detection Project**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

##Usage
```
python vehicle_detector.py inputfile outputfile
	--inputfile	:File name of input video
	--outputfile	:File name of output video 
```
---


[//]: # (Image References)
[image1]: ./output_images/v-nv_hog.png
[image2]: ./output_images/cars_cropped.png
[image3]: ./output_images/cars_hot_windows.png
[image4]: ./output_images/heatmap.png
[image5]: ./output_images/heatmap_thresh.png
[image6]: ./output_images/labeled_image.png
[image7]: ./output_images/final_detection.png
[//]:[video1]: ./proj_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

####README

### Histogram of Gradients (HOG) Features

The code for extracting features from the images are in the `FeatureExtractor.get_features()` function (lines 27 to 33). Before training the data is read from the image folders and loaded into the `FeatureExtractor` object. The object can also read pickled(`vd_data.p`) imagesets from disk and subsequently load the vehicle and non-vehicle data, after which the features are extracted 


In addition to HOG features, color histogram and spatially binned images are also extracted as features. 

In choosing a color space that works, I found YCrCb offering the best classifier training performance. The Y channel alone yielded ~84% accuracy while the entire color space yielded 99.07% accuracy on test data

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Example vehicle and non-vehicle data together with their HOG features is shown below:

![alt text][image1]

The classifier is a linear SVM classifier which receives as inputs the HOG + spatially binned + color histogram features of the the imageset. After training, the classifier and its scaler(which scales the inputs of the training data) are pickled in the file `clf.p` for later use.

###Sliding Window Search

Before searching the image I first do some preprocessing to the data. First of all I expect to search for images within the area `(320:end,400:end)` of the image. After that I use a set of multiscalar search spaces, `64,96 and 128`. The code for these can be found in lines `183 to 198` of `feature_utils.py`. The function `slide_window_multiscalar` returns a list of multiscalar sliding windows of size (64,64), (96,96) and (128,128) with overlap of 0.75 giving a total of 262 search boxes.

The cropped area used in sliding window image search is shown below:
![alt text][image2]

For each window, the HOG + color histogram + spatial bin features are extracted and served to the classifier. This task is performed in the function `search_classify_img` in the file `imsearch_utils.py`. The function classifies each window and determines if a car is found or not. After classification a heat map is generated from the search windows which returned a positive result. The heat map shows concentrations of positive results. Using a thresholding mechanism, the areas where there is the highest likelihood of finding a car/vehicle can be isolated. After thresholding we assume that areas with pixel value greater than zero represent cars

An example of positively identified windows after image search is shown below:

![alt text][image3]

An example of a heat map showing where positive car identifications were found is shown below:

![alt text][image4]

After thresholding we have:

![alt text][image5]


Using `scipy.ndimage.measurements.label()` we are then able to isolate the heatmaps into labeled pixels and then bounding box representing where the cars are in the frame can be found and drawn

The labeled images:
![alt text][image6]

Finally the located cars:
![alt text][image7]

### Dealing with False Positives and Vehicle Tracking

The following methods were used to deal with false positives and  vehicle tracking. 

1. If a labeled box is first located, if no other detected vehicles overlap with the labeled box, it is added to the list of vehicles, however, acknowledgement of the vehicle is deferred until a number of detections in the same vicinity have been found.

2. For an already detected vehicle, if a labeled box intersects with the area of the vehicle(a margin is added to this area for better detection see `verify_previously_seen` function in `Vehicle.py` ), we assume that we have found the vehicle in the new frame. The labeled box is then removed from the list of labeled boxes under consideration for other vehicles. If the intersecting labeled box is much larger or much smaller then the box is pruned or enlarged to suit the previouly detected vehicle. The labeled box is then added to the list of detected boxes for that vehicle. The average of the detected boxes form the current location of the box

3. If a vehicle is not detected after 2 frames we assume that the vehicle is not in view and thereby remove it from the list of vehicles, however, if the vehicle has been detected for a certain number of times, we keep said vehicle alive but drastically reduce the number of detections. That way we try to keep detected vehicle's alive longer, even if we momentarily cant find them.


The resulting video is shown below:
Here's a [link to the processed video](https://www.youtube.com/watch?v=cmLjH5RuYxk)

### Discussion

####Some realized mistakes
1. Taking the heatmaps over a number of frames could have yielded better results instead of relying on frame by frame heatmaps.
2. Speed up could have been obtained if HOG features and spatial information were generated once for each image instead of tardier process of calculating features for each of the 262 search windows. I tried this method on still images but was getting more false positives and false negatives and therefore decided to stick to the slower method.

####Where the solution fails
There are some areas where the solution fails. 
1. When two two cars overlap. The algorithm seems to find it difficult to distinguish between them.
2. There are still some times where I get false positives. I think this is due to the fact that the heatmaps are not integrated over a series of frames
3. The algorithm does not run in real time.
4. The algorithm may also not do well in bad lighting

####Improvements
1. With regards to HOG features, we could use GPUs to speed up the calculation of the features






