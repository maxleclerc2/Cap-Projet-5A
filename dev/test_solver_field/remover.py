
import numpy as np
import skimage.exposure
from PIL import Image
from PIL import ImageDraw

def generate_green_mask(image_path:str) -> object:
    img = cv2.imread(image_path)
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    A = lab[:,:,1]
    thresh = cv2.threshold(A, 3, 255, 
                           cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    
    blur = cv2.GaussianBlur(thresh, (0,0), sigmaX=5, sigmaY=5, borderType = cv2.BORDER_DEFAULT)
    
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
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    cv2.imwrite('make_greenscreen_thresh.png', thresh)
    cv2.imwrite('make_greenscreen_mask.png', mask)
    cv2.imwrite('make_greenscreen_antialiased.png', result)
    cv2.imwrite('make_greenscreen_blur.png', blur)
    cv2.imwrite('make_mask_red_spots.png', mask_red_spots)
    return thresh, mask, result, blur, mask_red_spots

if __name__ == "__main__":
    
    generate_green_mask("7295761_red_green.png")
