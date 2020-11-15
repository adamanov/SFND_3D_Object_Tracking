# SFND 3D Object Tracking
## [Rubric](https://review.udacity.com/#!/rubrics/2550/view)
By completing all the lessons, you got a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, I leant how to detect objects in an image using the YOLO deep-learning framework. And finally, I leant how to associate regions in a camera image with Lidar points in 3D space.

```2D_Feature Tracking:``` [Mid-Term_Project](https://github.com/adamanov/SFND_2D_Feature_Tracking)

<img src="images/course_code_structure.png" width="779" height="414" />

## TASKS:
### 1. Match 3D Objects
<img src="images/yolo-workflow.jpg" width="440" height="320" />

By using the YOLOv3 framework, we can extract a set of objects from a camera image that are represented by an enclosing rectangle (a "region of interest" or ROI) as well as a class label that identifies the type of object, e.g. a vehicle.

### 2. Compute Lidar-based TTC
```Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame```

### 3. Associate Keypoint Correspondences with Bounding Boxes
```Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.```

### 4. Compute Camera-based TTC
```Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.```

### 5. FP.5 Performance Evaluation 1
```Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.```

### 6. FP.6 Performance Evaluation 2
```Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.```

