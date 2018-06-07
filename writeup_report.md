**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/data_example.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/slide_windows1.png
[image4]: ./output_images/slide_windows2.png
[image5]: ./output_images/example1.png
[image6]: ./output_images/example2.png
[image7]: ./output_images/example3.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for extracting the histogram of oriented gradients (HOG) can be found in the [`extract_hog_features` function](https://github.com/dincamihai/CarND-Vehicle-Detection/blob/master/main.py#L104-L123)


I am reading a number of `vehicle` images from the KITTI set (500 random samples) and a higher number of `non-vehicle` images (3000 random samples).
Using more non-vehicle training examples worked good in my case. It seems it cuts down the false positives while still identifying the cars in the image. This needs to be proved though.
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I have tried different combinations of parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) for the `skimage.hog()` applied to different channels of the original RGB image but also of different color spaces.
A visualisation of the HOG features for a random example of a car and a non-car images can be seen below.
Here I am using the `YUV` color space and HOG parameters of `orientations=16`, `pixels_per_cell=(4, 4)` and `cells_per_block=(4, 4)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters while looking at the HOG visualisations. I've tried to pick the parameters that produce as much structure in the visualisations of cars as possible but also tried to keep the number of the generated features manageable. (when a lot of features, fitting the data to the SVC becomes problematic)
I ran out of memory when trying to train the SVC on all the available samples in the data set while using the `pixels_per_cell=(4, 4)` and `cells_per_block=(4, 4)` for HOG. I reduced the number of the car training examples to 500 and used 3000 of non-car training examples.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I am using a SVM with an `rbf` kernel with `gamma='auto'` and `C=10.0`.

The code can be seen here: https://github.com/dincamihai/CarND-Vehicle-Detection/blob/master/main.py#L175-L214

I've trained the classifier with 500 random car samples from the KITTI set and 3000 random non-car samples.

I am also fitting a scaler and  I save both the classifier and the scaler to the disk because I need them later.

In my submission they are archived (please see NOTES.md for instructions on how to extract them)

To the HOG extracted features I am also adding the color features and the histogram features. https://github.com/dincamihai/CarND-Vehicle-Detection/blob/master/main.py#L126-L148

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In order not to waste computation time, I am only applying the sliding window search in a reduced portion of the image every 5 frames.
The frequency can be configured here: https://github.com/dincamihai/CarND-Vehicle-Detection/blob/master/project_combined.mp4
I use two sliding window sizes.

First sliding (smaller window)
![alt text][image3]

Second sliding (bigger window)
![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV channel#1 HOG features plus spatially binned color and histograms of color in the feature vector.
Here are some example images:

![alt text][image5]
![alt text][image6]
![alt text][image7]


Each identified car has it's own color for the frame.
The green circle at the top left corner of the frame can have different luminosity values and represents the confidence of the detection.
The vector starting from the center of the frame represents the trend of the objects centroids. I use that for adjusting the window position to better follow the object.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

The annotated video can be found here: https://github.com/dincamihai/CarND-Vehicle-Detection/blob/master/project_out.mp4


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I record the position of positive detection in each frame of the video and I use them to create a heatmap.
At first I implement a threshold for filtering out the false positives, but because the classifier doesn't produce false positives when tried on the project video, I've decided to disable the threshold. The implementation can be seen here: https://github.com/dincamihai/CarND-Vehicle-Detection/blob/master/main.py#L297
The filtering would work by ignoring weak positive detections overlap.
I combine all the frame heatmaps in a global heatmap that tracks also the evolution of the detection in time.
I use `cv2.findContours()` to identify the idividual blobs in the global heatmap.
For finding the contours I've also implemented a threshold that would filter out transient detections (detections that appear in a single frame for example)
The code can be seen here: https://github.com/dincamihai/CarND-Vehicle-Detection/blob/master/main.py#L312-L313
Once I have the contours/objects I keep track of the each of them for the entire length of the video.
In order to overcome the overlapping objects problem, I've implemented a mechanism that uses an euclidian distance radius to merge objects that are close to eachother. (two cars overlap) https://github.com/dincamihai/CarND-Vehicle-Detection/blob/master/main.py#L321-L336
I also remove objects that are no longer needed.

I have generated a combined video that shows both the annotated video and also the changing heatmap (greed=increase, red=decrease, gray=current value)

https://github.com/dincamihai/CarND-Vehicle-Detection/blob/master/project_combined.mp4

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In order to prevent running out of memory when training, I have trained the classifier on a small dataset (only 500 car examples). This works for the project video but  I don't expect it to work with any video input. This can be improved by training on a bigger dataset.

Even though the window is tracking the cars for the entire video, I think the tracking can be improved to be smoother.
