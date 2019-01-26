## Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistorted]: ./output_images/undistort_calibration1.jpg "Undistorted"
[road_undist]: ./output_images/undistort_test1.jpg "Road Transformed"
[binary]: ./output_images/pipeline_test1.jpg "Binary Example"
[warped]: ./output_images/transformed_straight_lines1.jpg "Warp Example"
[fitvisual]: ./output_images/laneLine_test1.jpg "Fit Visual"
[final]: ./output_images/final_test1.jpg "Output"
[video1]: ./laneDetect_project_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is implemented in `./camera_calibrator.py`

I start by preparing "object 3d points" in real world space and "image 2d points".  I read in all the calibration images, draw the chessboard corners using by calling function `cv2.findChessboardCorners(gray, (9, 6), None)`. 

Then, I used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  
 
I applied this distortion correction to the test image "./camera_cal/calibration1.jpg" using the `cv2.undistort()` function.  

In addition, I saved the value of mtx and dist to file ./calibration_pickle.p` so that it can be reused in later steps. 

Here is the result :
![alt text][undistorted]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][road_undist]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I defined a class "LaneDetector" in file `lane_detector.py`, where the function  `self.grad_mag_dir_color_thresh(self, img)` takes undistorted image and use color and gradient thresholds to generate a binary image".  
Here's an example of my output for this step. 

![alt text][binary]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is defined in function `perspective_transformation(threshold_binary)`.  This function takes the binary image as input parameter, 
In order to call function `cv2.getPerspectiveTransform(src, dst)`, I manually identified the source and destination points using the straight line images. 

```python
    yLen = img.shape[0]
    src = np.float32([[190, yLen], [550, 480], [740, 480], [1128, yLen]])
    dst = np.float32([[190, yLen], [190, 0], [1128, 0], [1128, yLen]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 720      | 190, 720      | 
| 550, 480      | 190, 0        |
| 740, 480      | 1128, 0       |
| 1128, 720     | 1128, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

If it is the first time to identify the lane, I search the entire image with function `self.find_lane_pixels(transformed_binary)` to identify the left and right lanes
After lanes have been detected, I call function `self.search_aroundpoly()` to only search around the left/right lanes with margin 100 pixels.
A sample of the search windows for identried lane lines is shown below:

![alt text][fitvisual]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I implemented this with function `self.measure_curvature_real(ploty, left_fitx, right_fitx)`

I used the estimated figures provided in the project lectures "Measuring Curvature II", the lane is about 30 meters long and 3.7 meters wide.
```python
    self.ym_per_pix = 30 / 720
    self.xm_per_pix = 3.7 / 700
```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
I unwarp the detected lane boundaries back to the original image, which is implemented in line 81 - 83, basically following code
 
```python
    unwarped = cv2.warpPerspective(color_warp, self.Minv, (img.shape[1], img.shape[0]))
    result = cv2.addWeighted(img, 1, unwarped, 0.3, 0)
```

In addition, I calculate the offset of the car between the lane and vehicle position with respect to center using following code
```python
    def get_lane_position(self, unwarped, left_fitx, right_fitx):
        camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
        self.line_base_pos = (camera_center - unwarped.shape[1] / 2) * self.xm_per_pix
        lane_position = 'right'
        if self.line_base_pos > 0:
            lane_position = 'left'
        return lane_position
    lane_position = self.get_lane_position(unwarped, left_fitx, right_fitx)
```

Finally, I draw the text showing curvature and offset using function `self.print_lane_info(result, lane_position)`

Following is an example of the result after processing:

![alt text][final]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./laneDetect_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I tried various combination of color ranges in order to get a good result for filtering white and yellow without a really systematic apporach. 
With the help of mentor, he pointed that I can use trackerbar to test it, which helps a lot.  
However, I'm not fully confident that the color range I picked is good enough, because the results from the 2 challenge videos are not optimal.

Secondly, there is no reference data for the calculated offset of the car position and curvature of both lanes.  Without this, it is hard to validate if the implementation is good or not. 

       
