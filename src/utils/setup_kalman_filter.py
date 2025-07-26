import cv2
import numpy as np


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
        [[1, 0, delta_t, 0], # new_x = (1 * old_x) + (old_vx * ∆t)
                [0, 1, 0, delta_t], # new_y = (1 * old_y) + (old_vy * ∆t)
                [0, 0, 1, 0], # new_vx = old_vx * 1
                [0, 0, 0, 1]], # new_vy = old_vy * 1
        dtype=np.float32)

    # Process noise covariance: Uncertainty in our model
    kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * model_uncertainty

    # Measurement noise covariance: Uncertainty in the measurements (from sensor/camera)
    kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_uncertainty

    return kalman_filter