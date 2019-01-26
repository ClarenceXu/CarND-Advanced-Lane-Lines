import numpy as np
from numpy import newaxis
import cv2


class LaneDetector:

    def __init__(self, mtx, dist):
        # matrix from calibrating camera
        self.mtx = mtx
        # distortion coefficients from calibrating camera
        self.dist = dist
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the left line
        self.recent_left_xfitted = []
        # x values of the last n fits of the right line
        self.recent_right_xfitted = []
        # polynomial coefficients for left averaged over the last n iterations
        self.best_left_fit = None
        # polynomial coefficients for right averaged over the last n iterations
        self.best_right_fit = None
        # polynomial coefficients for the most recent left fit
        self.left_fit = [np.array([False])]
        # polynomial coefficients for the most recent right fit
        self.right_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # inverse perspective matrix used to unwarp to original image space
        self.Minv = None
        # Choose the number of sliding windows
        self.nwindows = 9
        # Set the width of the windows +/- margin
        self.margin = 100
        # Set minimum number of pixels found to recenter window
        self.minpix = 50
        # Define conversions in x and y from pixels space to meters
        # Fit polynomials to x,y in world space
        self.ym_per_pix = 30 / 720
        self.xm_per_pix = 3.7 / 700

    def apply_lines(self, img):
        # 1. undistort the image based on camera calibration matrix and distortion coefficients
        img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        # 2. use Sobel, direction, magnitude gradients and
        # color transforms to create a thresholded binary image.
        threshold_binary=self.grad_mag_dir_color_thresh(img)

        # 3. apply a perspective transform to rectify binary image ("birds-eye view")
        transformed_binary = self.perspective_transformation(threshold_binary)

        # 4. detect lane pixels and fit to find the lane boundary
        self.set_nonzero_x_y(transformed_binary)
        # only search the entire image if lines have not been previously detected
        if not self.detected:
            left_lane_inds, right_lane_inds, out_img = self.find_lane_pixels(transformed_binary)
        else:
            left_lane_inds, right_lane_inds = self.search_around_poly()
        self.fit_polynomial(left_lane_inds, right_lane_inds)

        # 5. draw the lane onto the warped blank image
        color_warp, ploty, left_fitx, right_fitx = self.fillPoly_image(transformed_binary)

        # 6. determine the curvature of the lane
        self.measure_curvature_real(ploty, left_fitx, right_fitx)

        # 7. warp the detected lane boundaries back onto the original image
        unwarped = cv2.warpPerspective(color_warp, self.Minv, (img.shape[1], img.shape[0]))
        result = cv2.addWeighted(img, 1, unwarped, 0.3, 0)

        # 8. calculate offset of the car between the lane and vehicle position with respect to center
        lane_position = self.get_lane_position(unwarped, left_fitx, right_fitx)

        # 9. draw the text showing curvature, offset, and speed
        self.print_lane_info(result, lane_position)
        return result

    def get_abs_sobel(self, img, orient='x', sobel_kernel=3):
        # Convert to grayscale
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        return abs_sobel

    # Define a function that takes an image, gradient orientation,
    # and threshold min / max values.
    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        abs_sobel = self.get_abs_sobel(img, orient, sobel_kernel)
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output

    # Define a function to return the magnitude of the gradient
    # for a given sobel kernel size and threshold values
    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Take both Sobel x and y gradients
        abs_sobelx = self.get_abs_sobel(img, 'x', sobel_kernel)
        abs_sobely = self.get_abs_sobel(img, 'y', sobel_kernel)

        # Calculate the gradient magnitude
        gradmag = np.sqrt(abs_sobelx ** 2 + abs_sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        return binary_output

    # Define a function that applies Sobel x and y,
    # then computes the direction of the gradient
    # and applies a threshold.
    def dir_thresh(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Take both Sobel x and y gradients
        abs_sobelx = self.get_abs_sobel(img, 'x', sobel_kernel)
        abs_sobely = self.get_abs_sobel(img, 'y', sobel_kernel)

        absgraddir = np.arctan2(abs_sobely, abs_sobelx)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        return binary_output

    # Apply all thresholding functions
    def grad_mag_dir_color_thresh(self, img):
        #gradx = self.abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 100))
        #grady = self.abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(20, 100))
        mag_binary = self.mag_thresh (img, sobel_kernel=11, mag_thresh=(80, 255))
        dir_binary = self.dir_thresh (img, sobel_kernel=15, thresh=(0.7, 1.3))
        color_binary = self.color_thresh(img, r_thresh=(220, 255), s_thresh=(150, 255))

        combined = np.zeros_like(dir_binary)
        combined[((mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1) ] = 255
        return combined

    # return a binary image, using BGR and HLS
    # set to 1 if it is in color range of white or yellow
    # otherwise set to 0
    def color_thresh(self, img, r_thresh=(0, 255), s_thresh=(0, 255)):
        # Apply a threshold to the R channel
        r_channel = img[:, :, 2]
        r_binary = np.zeros_like(img[:, :, 0])
        # Create a mask of 1's where pixel value is within the given thresholds
        r_binary[(r_channel > r_thresh[0]) & (r_channel <= r_thresh[1])] = 1

        # Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # Apply a threshold to the S channel
        s_channel = hls[:, :, 2]
        s_binary = np.zeros_like(s_channel)
        # Create a mask of 1's where pixel value is within the given thresholds
        s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # Combine two channels
        combined = np.zeros_like(img[:, :, 0])
        combined[(s_binary == 1) | (r_binary == 1)] = 1
        # Return binary output image
        return combined

    # perform perspective transformation using manually marked points
    def perspective_transformation(self, combo_binary):
        img = combo_binary
        # Define perspective transform
        img_size = (img.shape[1], img.shape[0])

        # manually identified points from the sample images
        yLen = img.shape[0]
        src = np.float32([[190, yLen], [550, 480], [740, 480], [1128, yLen]])
        dst = np.float32([[190, yLen], [190, 0], [1128, 0], [1128, yLen]])

        # Perform the transform
        M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        binary_warped = cv2.warpPerspective(combo_binary, M, img_size, flags=cv2.INTER_LINEAR)

        return binary_warped

    # using histogram to find left and right lanes
    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set the width of the windows +/- margin
        # Set minimum number of pixels found to recenter window

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # TODO: not needed for video: draw rectangles on the image
            if True:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), (0, 0, 255), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high), (0, 0, 255), 2)

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
                              (self.nonzerox >= win_xleft_low) & (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
                               (self.nonzerox >= win_xright_low) & (self.nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
            self.detected = True
        except ValueError:
            pass

        return left_lane_inds, right_lane_inds, out_img

    def set_nonzero_x_y(self, binary_warped):
        # Identify the x and y positions of all nonzero pixels in the image
        self.nonzeroy = np.array(binary_warped.nonzero()[0])
        self.nonzerox = np.array(binary_warped.nonzero()[1])

    def search_around_poly(self):
        left_lane_inds = ((self.nonzerox > (self.best_left_fit[0] * (self.nonzeroy ** 2) +
                                       self.best_left_fit[1] * self.nonzeroy +
                                       self.best_left_fit[2] - self.margin)
                           ) &
                          (self.nonzerox < (self.best_left_fit[0] * (self.nonzeroy ** 2) +
                                       self.best_left_fit[1] * self.nonzeroy +
                                       self.best_left_fit[2] + self.margin)
                           ))
        right_lane_inds = ((self.nonzerox > (self.best_right_fit[0] * (self.nonzeroy ** 2) +
                                        self.best_right_fit[1] * self.nonzeroy +
                                        self.best_right_fit[2] - self.margin)) & (
                                   self.nonzerox < (self.best_right_fit[0] * (self.nonzeroy ** 2) +
                                               self.best_right_fit[1] * self.nonzeroy +
                                               self.best_right_fit[2] + self.margin)))
        return left_lane_inds, right_lane_inds

    def fillPoly_image(self, binary_warped):
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = self.best_left_fit[0] * ploty ** 2 + self.best_left_fit[1] * ploty + self.best_left_fit[2]
        right_fitx = self.best_right_fit[0] * ploty ** 2 + self.best_right_fit[1] * ploty + self.best_right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Create an image to draw the lines
        warp_zero = np.zeros_like(binary_warped[:-20]).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        return color_warp, ploty, left_fitx, right_fitx

    def get_lane_position(self, unwarped, left_fitx, right_fitx):
        camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
        self.line_base_pos = (camera_center - unwarped.shape[1] / 2) * self.xm_per_pix
        lane_position = 'right'
        if self.line_base_pos > 0:
            lane_position = 'left'
        return lane_position

    def print_lane_info(self, result, lane_position):
        cv2.putText(result,
                    'Radius of curvature = ' + str(round(self.radius_of_curvature,0)) + '(m)',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(result,
                    'Vehicle is ' + str(abs(round(self.line_base_pos, 2))) + 'm ' + lane_position + ' of center',
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    def measure_curvature_real(self, ploty, left_fitx, right_fitx):
        # Compute the raduis of curvature of lane
        # Measure radius of curvature at y-value closest to car
        y_eval = np.max(ploty)

        left_fit_cr = np.polyfit(ploty * self.ym_per_pix, left_fitx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_pix, right_fitx * self.xm_per_pix, 2)
        # Calculate radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        self.radius_of_curvature = (left_curverad + right_curverad) / 2
        return self.xm_per_pix

    def fit_polynomial(self, left_lane_inds, right_lane_inds):
        # Extract left and right line pixel positions
        leftx = self.nonzerox[left_lane_inds]
        lefty = self.nonzeroy[left_lane_inds]
        rightx = self.nonzerox[right_lane_inds]
        righty = self.nonzeroy[right_lane_inds]
        # polynomial coefficients for the most recent fit
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        # x values of the last n fits of the left line
        self.recent_left_xfitted.append(self.left_fit)
        # x values of the last n fits of the right line
        self.recent_right_xfitted.append(self.right_fit)
        self.best_left_fit = np.mean(self.recent_left_xfitted[-min(15, len(self.recent_left_xfitted)):], axis=0)
        self.best_right_fit = np.mean(self.recent_right_xfitted[-min(15, len(self.recent_right_xfitted)):], axis=0)

