
import numpy as np
import skimage.exposure
import cv2

from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt


def green_mask_creator(image_path: str) -> object:
    """
    This function return all green spot on an image under 5 format.
    thresh         = The adapted threshold to cut off others colors different then green.
    mask           = mid mask built by applying an exposure on the blured image.
    result         = merge of original and mask images.
    blur           = GaussianBlur on the threshold.
    mask_red_spots = final result is a color adapted image made using the mask.
    """
    blur: object
    mask: object
    img: object
    lab: object
    A: object
    thresh: object

    try:
        img = cv2.imread(image_path)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        A = lab[:, :,  2] # 2 create green circle lab # 1 create red circle lab 0 blue circle lab
        L = lab[:, :, 1]
        
        #A = cv2.bitwise_and(A, L)
        
        

        thresh = cv2.threshold(A, 3, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        blur = cv2.GaussianBlur(
            thresh, (0, 0), sigmaX=8, sigmaY=8, borderType=cv2.BORDER_DEFAULT
        )
    except BaseException as be:
        raise be.__class__

    if blur is not None:
        mask = skimage.exposure.rescale_intensity(
            blur, in_range=(0, 255), out_range=(0, 255)
        ).astype(np.uint8)
        

    else:
        raise BaseException.__class__

    return mask
        




if __name__ == "__main__":
    
    try:
    
        mask = green_mask_creator("7295761_red_green.png")
        
        background_image = cv2.imread("kaEC6_-_modified.jpg")
        background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        result = cv2.bitwise_and(background_image, background_image, mask=mask)
        plt.imshow(result)

        cv2.imwrite("result.png", result)
        print("Done")
    except BaseException as be:
        raise be.__class__
