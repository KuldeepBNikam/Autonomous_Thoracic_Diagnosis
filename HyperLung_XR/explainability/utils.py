import numpy as np
import cv2

def overlay_heatmap(image, cam, alpha=0.4):
    """
    image: numpy array (H, W, 3)
    cam: numpy array (H, W)
    """

    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
    return overlay
