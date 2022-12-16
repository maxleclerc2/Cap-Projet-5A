import numpy as np
import skimage.exposure
import cv2
import os
import logging

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
        return base_file_name

    def save(self, saving_path: str, output_file: object, extension: str) -> None:
        """
       Save the result object as image type depending on the extension type.

       Returns:
           None
       """
        registration_path = saving_path + "preprocessing." + extension
        cv2.imwrite(registration_path, output_file)

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


@njit(parallel=True)
def set_mask(mask, img):
    for x in prange(img.shape[0]):
        for y in prange(img.shape[1]):
            b, g, r = img[x, y]
            if g > r and g > b:
                mask[x, y] = (255, 255, 255)


def rm_green_spot(images_path, image_file_name, raw_img_name):
    image_file = images_path + image_file_name
    raw_img = images_path + raw_img_name
    img = cv2.imread(image_file)

    mask = np.zeros_like(img)
    set_mask(mask, img)
    new_img = cv2.bitwise_and(img, mask)
    # cv2.imwrite("mask.png", mask)
    # cv2.imwrite("new_img.png", new_img)

    bg = cv2.imread(raw_img)  # background image.
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
    cv2.imwrite(images_path + "blur.png", blur)
    img_blur = cv2.imread(images_path + "blur.png")
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

    result = cv2.bitwise_and(bg, bg, mask=mask)
    cv2.imwrite(images_path + "result.png", result)


def RemoveGreenStars(saving_folder, name, extension, progress):
    originalPath = "app/static/images/astrometry/" + saving_folder + "/inputs/"
    imagesPath = "app/static/images/astrometry/" + saving_folder + "/outputs/" + name + "/"

    remover = SpotRemover(
        green_red_image_path=imagesPath + "extraction.png",
        original_image_path=originalPath + name + extension,
        mask_type="red"
    )

    progress.setStatus("first removing process of " + name)
    logging.info('Astrometry Processing - First processing of ' + name)
    try:
        preprocessed = remover.merge_mask_and_orignal_image()
    except Exception as e:
        logging.error(e)
        return "nok"
    remover.save(imagesPath, preprocessed, "png")

    progress.setStatus("second removing process of " + name)
    logging.info('Astrometry Processing - Second processing of ' + name)
    rm_green_spot(imagesPath, "red-green.png", "preprocessing.png")

    return "ok"
