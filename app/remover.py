import cv2
import numpy as np
import skimage.exposure
from flask import current_app as app
#from PIL import Image
#from PIL import ImageDraw


def remove_green_stars(image_path:str) -> tuple[object,
                                                object,
                                                object,
                                                object]:
    blur  : object
    mask  : object
    img   : object
    lab   : object
    A     : object
    thresh: object
    result: object

    try:
        img = cv2.imread(image_path)

        lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

        A = lab[:,:,1]

        thresh = cv2.threshold(A, 3, 255,
                               cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        blur = cv2.GaussianBlur(thresh, (0,0),
                                sigmaX=5,
                                sigmaY=5,
                                borderType = cv2.BORDER_DEFAULT)
    except BaseException as be:
        raise be.__class__

    if blur is not None:
        mask = skimage.exposure.rescale_intensity(blur,
                                                  in_range=(0,255),
                                                  out_range=(0,255)).astype(np.uint8)
        result = img.copy()
        result = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
        result[:,:,3] = mask
        mask_red_spots = skimage.exposure.rescale_intensity(thresh,
                                                            in_range=(0,255),
                                                            out_range=(0,255)).astype(np.uint8)

        a_channel = np.ones(mask_red_spots.shape, dtype=np.float)/2.0
        mask_red_spots_image = mask_red_spots*a_channel
    else:
        raise BaseException.__class__

    return thresh, mask, result, blur, mask_red_spots


def RemoveGreenStars(saving_folder, name):
    imagesPath = "app/static/images/op/" + saving_folder + "/" + name + "/"
    thresh, mask, result, blur, mask_red_spots = remove_green_stars(imagesPath + "red-green.png")
    if thresh is not None and mask is not None and result is not None and blur is not None and mask_red_spots is not None:
        cv2.imwrite(imagesPath + 'make_greenscreen_thresh.png', thresh)
        cv2.imwrite(imagesPath + 'make_greenscreen_mask.png', mask)
        cv2.imwrite(imagesPath + 'make_greenscreen_antialiased.png', result)
        cv2.imwrite(imagesPath + 'make_greenscreen_blur.png', blur)
        cv2.imwrite(imagesPath + 'make_mask_red_spots.png', mask_red_spots)
        return "ok"
    return "nok"
