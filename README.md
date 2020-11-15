# SFND 3D Object Tracking
## [Rubric](https://review.udacity.com/#!/rubrics/2550/view)
By completing all the lessons, you got a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, I leant how to detect objects in an image using the YOLO deep-learning framework. And finally, I leant how to associate regions in a camera image with Lidar points in 3D space.

```2D_Feature Tracking:``` [Mid-Term_Project](https://github.com/adamanov/SFND_2D_Feature_Tracking)

<img src="images/course_code_structure.png" width="779" height="414" />

## TASKS:
### 1. Match 3D Objects
YOLO is well know Deep Learning algoirthm which can very fast produce boundingbox with live camera.
By using the YOLOv3 framework, we can extract a set of objects from a camera image that are represented by an enclosing rectangle (a "region of interest" or ROI) as well as a class label that identifies the type of object, e.g. a vehicle.

<img src="images/yolo-workflow.jpg" width="330" height="205" />  <img src="images/draggedimage-8.png" width="500" height="150" />

The objective of this task is to implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the IDs of the matched regions of interest. (i.e the boxID property)
-   Implementation overview:
1.  For each bounding ox in the current frame (currFrameBBox), iterate through all the keypoint matches.
2.  If we find a bounding box containing currFramePoint, we then consider its matched previous frame (prevFramePoint) and we try to find the associated bounding box ID (prevFrameBBOx.boxID)
3.  Store how many correspondences there are between the currFrameBBox and the PrevFrameBBox
4.  The prevFrameBBox with highest number of the keypoint correspondences is finally considered to be the best match  

``` C++ 
for (auto descr = matches.begin(); descr != matches.end(); ++descr)
  for (auto pBox = prevFrame.boundingBoxes.begin(); pBox != prevFrame.boundingBoxes.end(); ++pBox)
    if (pBox->roi.contains(prevFrame.keypoints[idxPrevious].pt) )
      for (auto cBox = currFrame.boundingBoxes.begin(); cBox != currFrame.boundingBoxes.end(); ++cBox)
        if (cBox->roi.contains(currFrame.keypoints[idxCurrent].pt))
        count.at<int>(prevBoxID,currBoxID) = count.at<int>(prevBoxID,currBoxID) + 1;
```



### 2. Compute Lidar-based TTC
Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame

<img src="images/lidar_sensor.png" width="400" height="160" />  <img src="images/lidar_measurement.png" width="400" height="160" />

The sensor in this scenario will give us the distance to the closest 3D point in the path of driving. In the figure below, the closest point is indicated by a red line emanating from a Lidar sensor on top of the CAS vehicle.
Based on the model of a constant-velocity we discussed in the last section, the velocity can be computed from two successive Lidar measurements.

Once the relative velocity is known, the time to collision can easily be computed by dividing the remaining distance between both vehicles by. So given a Lidar sensor which is able to take precise distance measurements, a system for TTC estimation can be developed based based on a CVM and on the set of equations shown above

Even though Lidar is a reliable sensor, erroneous measurements may still occur. E.g, a small number of points is located behind the tailgate, seemingly without connection to the vehicle. When searching for the closest points, such measurements will pose a problem as the estimated distance will be too small. There are ways to avoid such errors by post-processing the point cloud, but there will be no guarantee that such problems will never occur in practice.

-  PreProcessing of LidarPoints in order to get rid of random points was implemented. KD-Tree and Euclidean cluster based on distance threshold and min cluster size algorithms were integrated 
    
### 3. Associate Keypoint Correspondences with Bounding Boxes
```Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.```

### 4. Compute Camera-based TTC
```Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.```

### 5. FP.5 Performance Evaluation 1
```Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.```

### 6. FP.6 Performance Evaluation 2
```Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.```

