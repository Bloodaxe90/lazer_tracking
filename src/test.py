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
    get_redness_frame, get_contours, get_contour_origin, get_largest_contour
from src.fsm.fsm_calibration import get_amplitude_per_pixel
from src.fsm.fsm import FSM
from src.fsm.fsm_setup import setup_fsm
from src.utils.live_plotter import LivePlotter
from src.utils.general import show_images, wait
from src.utils.io import save_results
from src.utils.setup_kalman_filter import setup_kalman_filter
from src.utils.spectral_analysis import get_frequencies_and_amplitudes


def test():
    # General Parameters
    ROOT_DIR = os.path.dirname(os.getcwd())
    COLOUR = False
    BUFFER_CAPACITY = 1
    KERNEL_SIZE = 3

    # Camera Parameters
    SDK_LIB_NAME: str = 'libASICamera2.dylib'
    CAMERA_ID: int = 0
    BINS = 2
    GAIN = 120
    EXPOSURE = 32
    RESOLUTION = (8288, 5640)
    START_POS = (0, 0)
    FRAMES_DARK = 10

    # FSM Parameters
    PORT: str = "/dev/cu.usbserial-24127109"
    BAUDRATE: int = 115200
    TIMEOUT: int = 1

    # Kalman Filter Parameters
    MODEL_UNCERTAINTY = 0.05
    MEASUREMENT_UNCERTAINTY = 0.5

    # testing specific paramters (Including frequency plotter)
    ITERATIONS = 1000
    KALMAN_FILTER = True
    FSM = False
    CONTOUR_MODE = True

    PLOTTER = False
    WINDOW_SIZE = 100
    PLOT_CAPACITY = ITERATIONS
    PLOT_UPDATE_RATE = 1
    PLOTTER_Y_FIELDS = (
        ("r-", "X Dominant Frequency"),
        ("b-", "Y Dominant Frequency")
    )
    PLOT_SAVE_NAME = "freq_plot"
    PLOT_TITLE = "Frequency vs Time"
    PLOT_X_LABEL = "Time (s)"
    PLOT_Y_LABEL = "Frequency (Hz)"


    EXPERIMENT_NAME: str = (f"stinky_fsm_"
                            f".003A_"
                            f"KF{1 if KALMAN_FILTER else 0}_"
                            f"G{GAIN}_"
                            f"E{EXPOSURE}_"
                            f"B{BINS}_"
                            f"R{RESOLUTION}_"
                            f"S{START_POS}_"
                            f"BC{BUFFER_CAPACITY}_"
                            f"FSM{1 if FSM else 0}_"
                            f"MOU{MODEL_UNCERTAINTY}_"
                            f"MEU{MEASUREMENT_UNCERTAINTY}_"
                            f"C{1 if CONTOUR_MODE else 0}_"
                            f"K{KERNEL_SIZE}")
    results = pd.DataFrame(columns=["X", "Y", "Time"])

    camera = None
    camera_stream = None
    fsm = None
    plotter = None
    kalman_filter = None

    amplitude_per_pixel_x = 0
    amplitude_per_pixel_y = 0

    try:
        camera, master_dark = setup_camera(root_dir=ROOT_DIR,
                                           sdk_lib_name=SDK_LIB_NAME,
                                           camera_id=CAMERA_ID,
                                           bins=BINS,
                                           gain=GAIN,
                                           exposure=EXPOSURE,
                                           resolution=RESOLUTION,
                                           start_pos=START_POS,
                                           num_frames=FRAMES_DARK,
                                           colour=COLOUR)
        camera_stream = CameraStream(camera).start()

        if KALMAN_FILTER:
            kalman_filter = setup_kalman_filter(frame_rate=camera_stream.get_fps(),
                                                model_uncertainty=MODEL_UNCERTAINTY,
                                                measurement_uncertainty=MEASUREMENT_UNCERTAINTY)

        origin_x, origin_y = ((RESOLUTION[0] / BINS) / 2, (RESOLUTION[1] / BINS) / 2)


        if FSM:
            wait("Ready to calibrate amplitude per pixel.\n"
                 "Please ensure the laser points to around the center of the camera")

            fsm: FSM = setup_fsm(PORT, BAUDRATE, TIMEOUT)
            amplitude_per_pixel_x,  amplitude_per_pixel_y= get_amplitude_per_pixel(
                camera=camera,
                master_dark=master_dark,
                origin_pos=(origin_x, origin_y),
                fsm=fsm)

            print(f"Amplitude Per Pixel X: {amplitude_per_pixel_x} Y: {amplitude_per_pixel_y}")

            fsm.send_command("control strategy feedforward")

        frame_buffer: deque = deque(maxlen=BUFFER_CAPACITY)

        if PLOTTER:
            plotter = LivePlotter(ROOT_DIR,
                                  PLOTTER_Y_FIELDS,
                                PLOT_SAVE_NAME,
                                PLOT_CAPACITY,
                                PLOT_X_LABEL,
                                PLOT_Y_LABEL,
                                PLOT_TITLE)

        amplitude_x: float = 0
        amplitude_y: float = 0

        start_time = time.time()
        last_time = start_time

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

            measured_x = None
            measured_y = None

            if CONTOUR_MODE:
                contours = get_contours(clean_frame)
                if contours:
                    largest_contour = get_largest_contour(contours)
                    measured_x, measured_y = get_contour_origin(largest_contour)
                elif not KALMAN_FILTER:
                    print("No Contours found")
                    continue
            else:
                _, _, _, max_loc = cv2.minMaxLoc(clean_frame)
                measured_x, measured_y = max_loc

            new_x = measured_x
            new_y = measured_y

            if KALMAN_FILTER:
                if measured_x is not None and measured_x is not None:
                    if i == 0:
                        kalman_filter.errorCovPost = np.eye(4, dtype=np.float32) * 1
                        kalman_filter.statePost = np.array(
                            [[measured_x], [measured_y], [0], [0]], dtype=np.float32)
                    else:
                        kalman_filter.correct(np.array([[measured_x], [measured_y]], dtype=np.float32))

                    estimated_state = kalman_filter.statePost
                    new_x = estimated_state[0, 0]
                    new_y = estimated_state[1, 0]
                else:
                    predicted_state = kalman_filter.statePre
                    new_x = predicted_state[0, 0]
                    new_y = predicted_state[1, 0]

            results.loc[len(results)] = {"X" : new_x,
                                         "Y" : new_y,
                                         "Time" : time.time() - start_time}

            if (i + 1) % 100 == 0:
                print(f"Iteration: {i + 1}, FPS: {camera_stream.get_fps()}, Sample Rate: {(i + 1) / (time.time() -  start_time)}")

            if PLOTTER:
                if (i + 1) % PLOT_UPDATE_RATE == 0 and i != 0: # Doesnt wait for buffer to fill before updating to ensure sample rate stays consistent
                    amount = (i + 1) if i < WINDOW_SIZE else WINDOW_SIZE
                    x_positions = np.array(results["X"])[-amount:]
                    y_positions = np.array(results["Y"])[-amount:]
                    times = np.array(results["Time"])[-amount:]
                    x_frequencies, x_amplitudes = get_frequencies_and_amplitudes(x_positions, times)
                    y_frequencies, y_amplitudes = get_frequencies_and_amplitudes(y_positions, times)
                    x_dominant_frequency = x_frequencies[np.argmax(x_amplitudes)]
                    y_dominant_frequency = y_frequencies[np.argmax(y_amplitudes)]
                    plotter.update((x_dominant_frequency, y_dominant_frequency), times[-1])


            if FSM:
                delta_x = origin_x - new_x
                delta_y = origin_y - new_y
                amplitude_x += delta_x * amplitude_per_pixel_x
                amplitude_y += delta_y * amplitude_per_pixel_y
                assert -1.5 <= amplitude_x <= 1.5 and -1.5 <= amplitude_y <= 1.5, f"Amplitudes ({amplitude_x}, {amplitude_y}) must be between -1.5 and 1.5"


                fsm.send_command(f"signal generate -a x -w dc -A {amplitude_x}", receive= False)
                fsm.send_command(f"signal generate -a y -w dc -A {amplitude_y}", receive= False)

    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"\nTracking stopped due to Error: {e}")

    finally:
        print("Cleaning up")
        if plotter is not None:
            plotter.close()
        if fsm is not None:
            fsm.send_command("control strategy off")
            fsm.disconnect()
        if camera_stream is not None:
            camera_stream.stop()
        if camera is not None:
            camera.close()
        save_results(ROOT_DIR, results, EXPERIMENT_NAME)
        print("Cleanup complete.")

if __name__ == "__main__":
    test()