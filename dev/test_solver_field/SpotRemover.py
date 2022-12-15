import numpy as np
import skimage.exposure
import cv2
import os

from numba import njit, prange


class SpotRemover:
    def __init__(
            self, green_red_image_path: str, original_image_path: str, mask_type: str
    ):
        """
        Initialize the class with the image path and mask type.

        Args:
            green_red_image_path (str): The path to the image that contains the green or red areas to be masked.
            original_image_path (str): The path to the original image.
            mask_type (str): The type of mask to be created, either "green" or "red".
        """
        self.green_red_image_path = green_red_image_path

        self.raw_image_path = original_image_path

        self.mask_type = mask_type

    def _load_image(self, path: str, mode: str) -> object:
        """
        Load the image from the specified file path and convert it to the specified color space.

        Args:
            path (str): The path to the image file.
            mode (str): The color space to convert the image to, either "COLOR_BGR2LAB" or "COLOR_BGR2RGB".

        Returns:
            The loaded and converted image.
        """
        img = cv2.imread(path)
        if mode == "COLOR_BGR2LAB":
            return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def merge_mask_and_orignal_image(self) -> object:
        """
        Create a mask using the specified mask type and apply it to the original image.

        Returns:
            The resulting image with the mask applied.
        """
        background_image = self._load_image(
            path=self.raw_image_path, mode="COLOR_BGR2RGB"
        )
        mask = self.create_mask()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return cv2.bitwise_and(background_image, background_image, mask=mask)

    def _get_base_file_name(self) -> str:
        """
       Extract the base file name from the path of the original image.

       Returns:
           The base file name without the file extension.
       """
        base_file_name, file_extension = os.path.splitext(self.raw_image_path)
        base_file_name_no_ext = base_file_name[: -len(file_extension)]
        return base_file_name#_no_ext

    def save(self, output_file: str, extension: str) -> None:
        """
       Save the result object as image type depending on the extension type.

       Returns:
           None
       """
        base_file_name_no_ext = self._get_base_file_name()
        # local_registration_path = base_file_name_no_ext + "_new_image." + extension
        local_registration_path = "preprocessing." + extension
        cv2.imwrite(local_registration_path, output_file)

    def save_to_tiff(self, output_file) -> None:
        """
       Save the result object as TIFF image type.

       Returns:
           None
       """
        import imageio
        base_file_name_no_ext = self._get_base_file_name()
        local_registration_path = base_file_name_no_ext + "_new_image.TIFF"
        imageio.imwrite(local_registration_path, output_file, format="TIFF")

    def create_mask(self) -> object:
        """
        thresh         = The adapted threshold to cut off others colors different then green.
        mask           = mid mask built by applying an exposure on the blured image.
        result         = merge of original and mask images.
        blur           = GaussianBlur on the threshold.
        """
        """
        Create a binary mask using the specified mask type and image.
        
        Returns:
            The resulting mask image.
        """
        blur: object
        mask: object
        _img_load: object
        lab: object
        A: object
        thresh: object

        try:
            if self.mask_type == "green":
                _img_load = self._load_image(
                    path=self.green_red_image_path, mode="COLOR_BGR2LAB"
                )
                A = _img_load[:, :, 2]
            elif self.mask_type == "red":
                _img_load = self._load_image(
                    path=self.green_red_image_path, mode="COLOR_BGR2LAB"
                )
                A = _img_load[:, :, 1]
            thresh = cv2.threshold(A, 3, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            blur = cv2.GaussianBlur(
                thresh, (0, 0), sigmaX=5, sigmaY=5, borderType=cv2.BORDER_DEFAULT
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


def remove_red_circles(img):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # Get the BGR values for the current pixel
            b, g, r = img[x, y]

            # If the pixel is green (i.e. the green value is greater than the red and blue values), set its value to 0
            if r > g and r > b:
                img[x, y] = (0, 0, 0)

            # for residuals blue spots after invert
            #if b > g and b > g:
            #    img[x, y] = (0, 0, 0)

    return img


@njit(parallel=True)
def set_mask(mask, img):
    for x in prange(img.shape[0]):
        for y in prange(img.shape[1]):
            b, g, r = img[x, y]
            if g > r and g > b:
                mask[x, y] = (255, 255, 255)


def rm_green_spot(image_file, raw_img):
    img = cv2.imread(image_file)

    mask = np.zeros_like(img)
    set_mask(mask, img)
    new_img = cv2.bitwise_and(img, mask)
    # cv2.imwrite("mask.png", mask)
    # cv2.imwrite("new_img.png", new_img)

    """
    kernel = np.ones((50, 50), np.uint8)
    invert_img_dilation = ~cv2.dilate(
        new_img, kernel, iterations=1
    )  # cv2.morphologyEx(new_img, cv2.MORPH_GRADIENT, kernel)

    cv2.imwrite("invert_img_dilation.png", invert_img_dilation)
    kernel2 = np.ones((10, 10), np.uint8)
    img_dilation = cv2.erode(invert_img_dilation, kernel2, iterations=1)
    # cv2.imwrite("img_dilation.png", img_dilation)

    bg = cv2.imread(raw_img) # background image.
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    fuz = cv2.bitwise_and(bg, img_dilation)

    cv2.imwrite("OUTPUT_IMAGE_BEFORE_POST_CLEANING.png", fuz)
    fuz_without_red_circles = remove_red_circles(fuz)

    # Save the new image in TIFF format
    cv2.imwrite("OUTPUT_IMAGE.png", fuz_without_red_circles)
    """

    bg = cv2.imread(raw_img) # background image.
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))

    A = new_img[:, :, 1]
    # cv2.imwrite("A.png", A)

    thresh = cv2.threshold(A, 3, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # cv2.imwrite("thresh.png", thresh)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite("new_thresh.png", thresh)
    thresh = cv2.bitwise_not(thresh)
    # cv2.imwrite("new_new_thresh.png", thresh)

    blur = cv2.GaussianBlur(thresh, (0, 0), sigmaX=10, sigmaY=10, borderType=cv2.BORDER_DEFAULT)
    cv2.imwrite("blur.png", blur)
    img_blur = cv2.imread("blur.png")
    grey_pixels = np.where(
        (img_blur[:, :, 0] <= 240) &
        (img_blur[:, :, 1] <= 240) &
        (img_blur[:, :, 2] <= 240)
    )
    img_blur[grey_pixels] = [0, 0, 0]
    # cv2.imwrite("new_blur.png", img_blur)

    mask = skimage.exposure.rescale_intensity(
        img_blur[:, :, 0], in_range=(0, 255), out_range=(0, 255)
    ).astype(np.uint8)
    # cv2.imwrite("new_mask.png", mask)

    truc = cv2.bitwise_and(bg, bg, mask=mask)
    cv2.imwrite("result.png", truc)


if __name__ == "__main__":

    new_object = SpotRemover(
        green_red_image_path="7295761_red_green.png",
        original_image_path="kaEC6_-_modified.jpg",
        mask_type="red",
    )

    print("Preprocessing")
    file = new_object.merge_mask_and_orignal_image()
    new_object.save(file, "PNG")
    # new_object.save_to_tiff(file)
    #remove_green_pixels2("7295761_red_green.png", "kaEC6_-_modified.jpg")

    print("Final process")
    #Build_MASK("7295761_red_green.png")
    rm_green_spot("7295761_red_green.png", "preprocessing.PNG")  # TODO nom
    #remove_green_pixels("7295761_red_green.png")
