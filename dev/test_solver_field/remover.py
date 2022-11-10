
import numpy as np
import skimage.exposure
import cv2

from PIL import Image
from PIL import ImageDraw

def remove_green_stars(image_path:str) -> tuple[object, 
                                                object, 
                                                object, 
                                                object]:
    """
    This function return all green spot on an image under 5 format.
    thresh         = The adapted threshold to cut off others colors different then green.
    mask           = mid mask built by applying an exposure on the blured image.
    result         = merge of original and mask images.
    blur           = GaussianBlur on the threshold.
    mask_red_spots = final result is a color adapted image made using the mask.
    """
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
  
  
def green_spot_cleaner(base_image_path:str, mask_red_spots:object) -> object:

    background_image = cv2.imread(base_image_path)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    crop_background = background_image.copy()

    crop_background[mask_red_spots != 0] = [0,0,0]

    kernel3 = np.array([[0,  -1,  0],
                        [-1,  5, -1],
                        [0, -1,  0]]) 

    plt.imshow(crop_background)
    cv2.imwrite('crop_background.png', crop_background)
    plt.show()
    return crop_background


if __name__ == "__main__":
    
    thresh, mask, result, blur, mask_red_spots = remove_green_stars("7295761_red_green.png")

    if thresh is not None and mask is not None and result is not None and blur is not None and mask_red_spots is not None:
        cv2.imwrite('make_greenscreen_thresh.png', thresh)
        cv2.imwrite('make_greenscreen_mask.png', mask)
        cv2.imwrite('make_greenscreen_antialiased.png', result)
        cv2.imwrite('make_greenscreen_blur.png', blur)
        cv2.imwrite('make_mask_red_spots.png', mask_red_spots)

    try:

        crop_background = green_spot_cleaner('kaEC6_-_modified.jpg', thresh)

        background_image = cv2.imread('kaEC6_-_modified.jpg')
        background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
        crop_background = background_image.copy()

        crop_background[mask_red_spots != 0] = [0,0,0]
        #crop_background[mask_red_spots == 0] = [255,255,255]

        cv2.imwrite('crop_background.png', crop_background)
    except BaseException as be:
        raise be.__class__

