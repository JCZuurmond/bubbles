import numpy as np


def scale_wrt_vertical_gradient(img: np.ndarray) -> np.ndarray:
    """
    The image has a vertical gradient in color; lower
    in the image it was darker. This gradient is removed
    by dividing each row with the average value of that row.

    Parameters
    ----------
    img : np.ndarray
        The image to be scaled.

    Returns
    -------
    np.ndarray : Scaled image.
    """
    img = (img.T / img.mean(axis=1)).T

    # Image should be an 8int
    img = img - img.min()
    img = img / img.max() * 255
    img = img.astype('uint8')

    return img
