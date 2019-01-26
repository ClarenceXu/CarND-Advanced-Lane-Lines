import numpy as np
import cv2


def get_abs_sobel(img, orient='x', sobel_kernel=3):
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
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    abs_sobel = get_abs_sobel(img,orient,sobel_kernel)
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output


def grad_mag_dir_thresh(img):
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 100))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(20, 100))
    mag_binary = mag_thresh(img, sobel_kernel=3, mag_thresh=(20, 100))
    dir_binary = dir_thresh(img, sobel_kernel=15, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    abs_sobelx = get_abs_sobel(img, 'x', sobel_kernel)
    abs_sobely = get_abs_sobel(img, 'y', sobel_kernel)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(abs_sobelx**2 + abs_sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output


def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Take both Sobel x and y gradients
    abs_sobelx = get_abs_sobel(img, 'x', sobel_kernel)
    abs_sobely = get_abs_sobel(img, 'y', sobel_kernel)

    absgraddir = np.arctan2(abs_sobely, abs_sobelx)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


def color_thresh(img, r_thresh=(0, 255), s_thresh=(0, 255)):
    """
    Returns a binary image of the same size as the input image of ones where pixel values
    were in the threshold range, and zeros everywhere else.
    :param img: input image in BGR format.
    :param r_thresh: threshold (0 to 255) for determining which pixels from r_channel to include in binary output.
    :param s_thresh: threshold (0 to 255) for determining which pixels from s_channel to include in binary output.
    """
    # Apply a threshold to the R channel
    r_channel = img[:,:,2]
    r_binary = np.zeros_like(img[:,:,0])
    # Create a mask of 1's where pixel value is within the given thresholds
    r_binary[(r_channel > r_thresh[0]) & (r_channel <= r_thresh[1])] = 1

    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # Apply a threshold to the S channel
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    # Create a mask of 1's where pixel value is within the given thresholds
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine two channels
    combined = np.zeros_like(img[:,:,0])
    combined[(s_binary == 1) | (r_binary == 1)] = 1
    # Return binary output image
    return combined

