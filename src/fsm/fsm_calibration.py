import numpy as np
import zwoasi as asi

from src.camera.image_processing import get_redness_frame, get_clean_frame, \
    get_contours, get_contour_origin, get_largest_contour
from src.fsm.fsm import FSM

def get_amplitude_per_pixel(camera: asi.Camera,
                            master_dark: np.ndarray,
                            origin_pos: tuple[float, float],
                            fsm: FSM,
                            amplitude: float = 0.5,
                            num_frames: int = 10,
                            colour: bool = False) -> tuple:
    """
    Measures how many pixels the laser dot moves per FSM amplitude step
    for both x and y axes

    Parameters:
        camera (asi.Camera): The camera capturing the laser
        master_dark (np.ndarray): A dark frame to remove sensor noise
        origin_pos (tuple): The original (x, y) position of the laser dot
        fsm (FSM): The FSM device controlling laser direction
        amplitude (float): The amplitude of the signal to send to FSM
        num_frames (int): Number of frames to average for calibration
        colour (bool): If True, uses redness method to isolate the laser dot (Defaults to False)

    Returns:
        tuple[float, float]: The amplitude per pixel for the x and y directions
    """

    amplitudes_per_pixel = []  # List to store amplitude per pixel for x and y
    moved_pos = []             # List to store measured positions after movement

    assert -1.5 < amplitude < 1.5 and amplitude != 0, \
        "Amplitude must be within the FSM's limits (-1.5, 1.5) and not zero"

    for i, axis in enumerate(["x", "y"]):

        # Turn on FSM movement
        fsm.send_command(f"control strategy feedforward")
        fsm.send_command(f"signal generate -a {axis} -w dc -A {amplitude}")

        # Capture and clean frames to find laser dot position
        calibrate_frames = [
            get_clean_frame(camera.capture(), master_dark) for _ in range(num_frames)
        ]

        # If camera is in colour mode and we are using colour, isolate the red channel
        if camera.get_camera_property()["IsColorCam"] and colour:
            calibrate_frames = [
                get_redness_frame(calibrated_frame) for calibrated_frame in calibrate_frames
            ]

        # Turn FSM movement off after frames captured
        fsm.send_command(f"control strategy off")

        # Average all captured frames to reduce noise in the result
        calibrate_image = np.median(calibrate_frames, axis=0).astype(np.uint8)

        # Find contours and calculate the new position
        contours = get_contours(calibrate_image)
        largest_contour = get_largest_contour(contours)
        calibrated_pos = get_contour_origin(largest_contour)

        moved_pos.append(calibrated_pos[i])  # Store new position for axis

        # Calculate how many pixels the dot moved
        delta_pos = abs(origin_pos[i] - calibrated_pos[i])

        assert delta_pos != 0, (
            f"Error: FSM did not move for the {axis} axis. Check connection or signal."
        )

        amplitudes_per_pixel.append(amplitude / delta_pos)

    print(f"Origin Pos X: {origin_pos[0]} Y: {origin_pos[1]}")
    print(f"Moved Pos X: {moved_pos[0]} Y: {moved_pos[1]}")
    return tuple(amplitudes_per_pixel)
