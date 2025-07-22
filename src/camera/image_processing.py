import os
from typing import Sequence

import cv2
import numpy as np
import zwoasi as asi
from cv2 import Mat
from jedi.settings import dynamic_params
from matplotlib import pyplot as plt

from src.utils.inference import show_images


def get_clean_frame(light_frame: np.ndarray,
                    master_dark: np.ndarray,
                    kernel_size: int = 3) -> np.ndarray:
    """
    Removes noise from raw light frame by subtracting dark frame and other methods

    Parameters:
        light_frame (np.ndarray): The image captured with light
        master_dark (np.ndarray): The master dark frame
        kernel_size (int): The size of the kernel for morphological opening (Defaults to 3)

    Returns:
        np.ndarray: The cleaned frame
    """
    clean_frame = cv2.subtract(light_frame, master_dark)

    # Use morphological opening to remove small noise blobs in in the background
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    clean_frame = cv2.morphologyEx(clean_frame, cv2.MORPH_OPEN, kernel, iterations=1)
    return clean_frame


def get_redness_frame(light_frame: np.ndarray) -> np.ndarray:
    """
    Calculates the redness of an image by subtracting the maximum of blue and green channels from red

    Parameters:
        light_frame (np.ndarray): A colour image in BGR format

    Returns:
        np.ndarray: A grayscale image showing where red is strongest
    """
    # Split the image into its BGR colour channels
    b, g, r = cv2.split(light_frame)

    # Subtract the strongest of blue or green from red to isolate redness
    return cv2.subtract(r, cv2.max(b, g))


def get_contours(image: np.ndarray) -> Sequence[Mat | np.ndarray]:
    """
    Finds and returns the contours in a single channel image

    Parameters:
        image (np.ndarray): A grayscale image (single channel)

    Returns:
        Sequence[Mat | np.ndarray]: A list of contours found in the image
    """
    # Convert the image to a binary mask using Otsus thresholding
    _, mask = cv2.threshold(
        src=image,
        thresh=69,  # This value is ignored because Otsus method is used
        maxval=255,
        type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Find external contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def get_contour_origin(contour: Mat | np.ndarray) -> tuple[float, float]:
    """
    Finds the origin of a contour using image moments

    Parameters:
        contour (Mat | np.ndarray): A single contour

    Returns:
        tuple[float, float]: The (x, y) coordinates of the contour's origin
    """
    # Calculate image moments (used to find shape properties like area and centre)
    M = cv2.moments(contour)
    largest_contour_area = M["m00"]

    # Make sure the area is not zero to avoid division errors
    assert largest_contour_area > 0, "Cannot find any contours"

    x = M["m10"] // largest_contour_area # M["m10"] is sum of x coordinates
    y = M["m01"] // largest_contour_area # M["m01"] is sum of x coordinates

    return x, y


def get_largest_contour(contour: Sequence[Mat | np.ndarray]) -> Mat | np.ndarray:
    """
    Returns the contour with the largest area

    Parameters:
        contour (Sequence[Mat | np.ndarray]): A list of contours

    Returns:
        Mat | np.ndarray: The largest contour found
    """
    return max(contour, key=cv2.contourArea)

def setup_kalman_filter(frame_rate: float,
                        model_uncertainty: float = 0.05,
                        measurement_uncertainty: float = 0.5
                        ) -> cv2.KalmanFilter:
    """
    Sets up a Kalman filter to track x and y position using a constant velocity model

    Parameters:
        frame_rate (float): The number of fps (used to calculate time between frames)
        model_uncertainty (float): How much uncertainty we have in the model accurately predicting the next position (default is 0.05)
        measurement_uncertainty (float): How much uncertainty we have in our x, y position measurements representing the actual position of the laser (default is 0.5)

    Returns:
        cv2.KalmanFilter: Configured Kalman filter object
    """

    # States of x position, y position, x velocity and y velocity
    # Measures the x position and y position
    kalman_filter: cv2.KalmanFilter = cv2.KalmanFilter(4, 2)

    kalman_filter.measurementMatrix = np.array([
        [1, 0, 0, 0],  # x position
        [0, 1, 0, 0]   # y position
    ], dtype=np.float32)

    # Model for constant velocity
    delta_t = 1 / frame_rate
    kalman_filter.transitionMatrix = np.array(
        [[1, 0, delta_t, 0], # vx = ∆x/∆t
                [0, 1, 0, delta_t], # vy = ∆y/∆t
                [0, 0, 1, 0], # new_vx = old_vx
                [0, 0, 0, 1]], # new_vy = old_vy
        dtype=np.float32)

    # Process noise covariance: Uncertainty in our model
    kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * model_uncertainty

    # Measurement noise covariance: Uncertainty in the measurements (from sensor/camera)
    kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_uncertainty

    return kalman_filter






