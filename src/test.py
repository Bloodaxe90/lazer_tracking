import collections
import os
import time
from collections import deque

import cv2
import numpy as np
import pandas as pd
import zwoasi as asi
import time

from src.camera.camera_calibration import get_master_dark
from src.camera.camera_setup import setup_camera
from src.camera.camera_stream import CameraStream
from src.camera.image_processing import get_clean_frame, \
    get_redness_frame, get_contours, get_contour_origin, get_largest_contour, \
    setup_kalman_filter
from src.fsm.fsm_calibration import get_amplitude_per_pixel
from src.fsm.fsm import FSM
from src.fsm.fsm_setup import setup_fsm
from src.utils.general import wait
from src.utils.inference import show_images
from src.utils.io import save_results


def test():
    BINS = 2
    GAIN = 120
    EXPOSURE = 32  # µs
    RESOLUTION = (8288, 5640)
    START_POS = (0, 0)
    FRAMES_DARK = 10
    BUFFER_CAPACITY = 1
    ITERATIONS: int = 1000
    COLOUR: bool = False
    FSM: bool = False
    KALMAN_FILTER: bool = False
    MODEL_UNCERTAINTY = 0.05
    MEASUREMENT_UNCERTAINTY = 0.5
    # Contour mode for finding center of brightest point, if false cv2.minMaxLoc will be used
    CONTOUR_MODE: bool = True
    CAMERA_ID: int = 0
    SDK_LIB_NAME: str = 'libASICamera2.dylib'
    KERNEL_SIZE: int = 3 # Kernel size of morphological openeing

    experiment_name: str = (f"square_.0A_"
                            f"G{GAIN}_"
                            f"E{EXPOSURE}_"
                            f"B{BINS}_"
                            f"R{RESOLUTION}_"
                            f"S{START_POS}_"
                            f"BC{BUFFER_CAPACITY}_"
                            f"FSM{1 if FSM else 0}_"
                            f"KF{1 if KALMAN_FILTER else 0}_"
                            f"MOU{MODEL_UNCERTAINTY}_"
                            f"MEU{MEASUREMENT_UNCERTAINTY}_"
                            f"C{1 if CONTOUR_MODE else 0}_"
                            f"K{KERNEL_SIZE}")

    root_dir = os.path.dirname(os.getcwd())

    # Setting up camera
    camera, master_dark= setup_camera(root_dir= root_dir,
                                                      sdk_lib_name= SDK_LIB_NAME,
                                                      camera_id= CAMERA_ID,
                                                      bins=BINS,
                                                      gain=GAIN,
                                                      exposure=EXPOSURE,
                                                      resolution= RESOLUTION,
                                                      start_pos= START_POS,
                                                      num_frames= FRAMES_DARK,
                                                      colour=COLOUR)

    camera_stream = CameraStream().start()

    if FSM:
        # Setting up FSM
        PORT: str = "/dev/cu.usbserial-24127109"
        BAUDRATE: int = 115200
        TIMEOUT: int = 1
        fsm: FSM = setup_fsm(PORT, BAUDRATE, TIMEOUT)
        fsm.send_command("control strategy feedforward")

        origin_x = (RESOLUTION[0] / BINS) / 2
        origin_y = (RESOLUTION[1] / BINS) / 2

        wait("Ready to calibrate amplitude per pixel.\n"
             "Please ensure the laser points to the center of the camera")

        amplitude_per_pixel_x, amplitude_per_pixel_y = get_amplitude_per_pixel(camera=camera,
                                                                               master_dark= master_dark,
                                                                               origin_pos= (origin_x, origin_y),
                                                                               fsm= fsm,
                                                                               colour= COLOUR
                                                                               )
        print(f"Amplitude Per Pixel X: {amplitude_per_pixel_x} Y: {amplitude_per_pixel_y}")

    frame_buffer: collections.deque = deque(maxlen=BUFFER_CAPACITY)

    results = pd.DataFrame(columns=["X", "Y", "Time"])

    wait("Start?")
    print("Started")

    if KALMAN_FILTER:
        kalman_filter = setup_kalman_filter(frame_rate= camera_stream.get_fps(),
                                            model_uncertainty= MODEL_UNCERTAINTY,
                                            measurement_uncertainty= MEASUREMENT_UNCERTAINTY)
    amplitude_x: float = 0
    amplitude_y: float = 0
    start_time = time.time()
    last_time = start_time

    try:
        for i in range(ITERATIONS):
            if KALMAN_FILTER:
                current_time = time.time()
                sample_time = current_time - last_time
                last_time = current_time

                kalman_filter.transitionMatrix[0, 2] = sample_time
                kalman_filter.transitionMatrix[1, 3] = sample_time

                kalman_filter.predict()

            raw_frame = camera_stream.read()
            if raw_frame is None:
                time.sleep(0.001)
                continue

            clean_frame = get_clean_frame(raw_frame, master_dark, KERNEL_SIZE)
            if COLOUR and camera.get_camera_property()["IsColorCam"]:
                clean_frame = get_redness_frame(clean_frame)

            if BUFFER_CAPACITY > 1:
                frame_buffer.append(clean_frame)

                clean_frame = (np.median(
                    np.stack(frame_buffer, axis=0), axis=0)
                .astype(np.uint8))

            if CONTOUR_MODE:
                contours = get_contours(clean_frame)
                if not contours:
                    print("Couldn't find any contours")
                    continue
                largest_contour = get_largest_contour(contours)
                measured_x, measured_y = get_contour_origin(largest_contour)
            else:
                _, _, _, max_loc = cv2.minMaxLoc(clean_frame)
                measured_x, measured_y = max_loc

            new_x = measured_x
            new_y = measured_y

            if KALMAN_FILTER:
                if i == 0:
                    kalman_filter.errorCovPost = np.eye(4, dtype=np.float32) * 1
                    kalman_filter.statePost = np.array(
                        [[measured_x], [measured_y], [0], [0]], dtype=np.float32)
                else:
                    kalman_filter.correct(np.array([[measured_x], [measured_y]], dtype=np.float32))

                predicted_state = kalman_filter.statePost
                new_x = predicted_state[0,0]
                new_y = predicted_state[1,0]

            results.loc[len(results)] = {"X" : new_x,
                                         "Y" : new_y,
                                         "Time" : time.time() - start_time}

            if (i + 1) % 100 == 0:
                print(f"Iteration: {i + 1}, FPS: {camera_stream.get_fps()}, Sample Rate: {(i + 1) / (time.time() -  start_time)}")

            if FSM:
                delta_x = origin_x - new_x
                delta_y = origin_y - new_y
                amplitude_x += delta_x * amplitude_per_pixel_x
                amplitude_y += delta_y * amplitude_per_pixel_y
                assert -1.5 <= amplitude_x <= 1.5 and -1.5 <= amplitude_y <= 1.5, f"Amplitudes ({amplitude_x}, {amplitude_y}) must be between -1.5 and 1.5"


                fsm.send_command(f"signal generate -a x -w dc -A {amplitude_x}", receive= False)
                fsm.send_command(f"signal generate -a y -w dc -A {amplitude_y}", receive= False)

    except KeyboardInterrupt:
        print("\nTracking stopped by user.")
    finally:
        print("Cleaning up")
        run_time = time.time() - start_time
        if FSM:
            fsm.send_command("control strategy off")
            fsm.disconnect()
        camera_stream.stop()
        camera.close()
        if experiment_name != "":
            save_results(root_dir, results, experiment_name)
        print("Cleanup complete.")
        print(f"Total Run Time: {run_time}")

if __name__ == "__main__":
    test()