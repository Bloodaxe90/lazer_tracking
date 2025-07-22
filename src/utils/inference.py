import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def show_images(*args, title: str = ""):
    """
    Display one or more images

    Args:
        *args: One or more images to display. Each image can be either:
            - A numpy array
            - A filename (string) of an image located in 'resources/images'
        title (str): Optional title to display above all images.

    Behavior:
        - If no images are provided, prints a message and returns.
        - Loads images from disk if filenames are provided.
        - Supports grayscale (2D arrays) and color images.
    """

    num_images = len(args)  # Number of images to display
    image_dir = os.path.join(os.path.dirname(os.getcwd()), 'resources', 'images')

    assert num_images > 0, "No images provided"

    # Create subplots in one row with a column per image
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    if title != "":
        fig.suptitle(title, fontsize=16)

    # If only one image, axes is not a list by default, so make it a list
    if num_images == 1:
        axes = [axes]

    for i, img in enumerate(args):
        # If the argument is a filename, load the image from the resources directory
        if isinstance(img, str):
            img = np.array(Image.open(f"{image_dir}/{img}"))

        # Display grayscale images with a gray colormap and set value range 0-255
        if img.ndim == 2:
            axes[i].imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            axes[i].imshow(img)  # Display color images as is

        axes[i].set_title(f'Image {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def get_dominant_frequency(data: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the dominant frequency of the input signal data using a FFT
    (won't lie code was taken from stack overflow as I am not physics student)

    Args:
        data (np.ndarray): The signal data (e.g., x, y pixel location)
        times (np.ndarray): Corresponding time values for the data points

    Returns:
        dominant_frequency (float): The frequency with the highest power (excluding zero frequency)
        frequencies (np.ndarray): Array of frequencies computed by FFT.
        power_spectrum (np.ndarray): Power corresponding to each frequency.
    """
    n = len(data)

    # Remove DC component by subtracting mean, then compute FFT
    fft_values = np.fft.fft(data - data.mean())

    # Compute the normalized power spectrum (magnitude)
    power_spectrum = np.abs(fft_values) / n

    # Calculate frequencies corresponding to FFT bins
    freq_resolution = max(times) / n
    frequencies = np.fft.fftfreq(n, d=freq_resolution)

    # Keep only positive frequencies
    positive_freq_indices = np.where(frequencies >= 0)
    frequencies = frequencies[positive_freq_indices]
    power_spectrum = power_spectrum[positive_freq_indices]

    # Find the peak frequency ignoring the zero-frequency component at index 0
    if len(power_spectrum) > 1:
        peak_index = np.argmax(power_spectrum[1:]) + 1  # offset by 1 to skip DC
        dominant_frequency = frequencies[peak_index]
    else:
        dominant_frequency = 0.0  # No dominant frequency found

    return dominant_frequency, frequencies, power_spectrum

