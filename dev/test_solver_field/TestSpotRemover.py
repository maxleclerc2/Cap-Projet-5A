#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:40:59 2022

@author: panda
"""

import unittest
import cv2
import numpy as np
import os
from SpotRemover import SpotRemover


class TestSpotRemover(unittest.TestCase):
    def test_load_image(self):
        # Create a test image and save it to a file
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite("test_image.jpg", test_image)

        # Initialize the SpotRemover class with the test image
        spot_remover = SpotRemover("test_image.jpg", "test_image.jpg", "green")

        # Test that the _load_image method correctly loads and converts the image to the correct color space
        loaded_image = spot_remover._load_image("test_image.jpg", "COLOR_BGR2LAB")
        self.assertEqual(loaded_image.shape, (100, 100, 3))
        self.assertEqual(loaded_image.dtype, np.uint8)

    def test_create_mask(self):
        # Create a test image with known green and red areas and save it to a file
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :50, 1] = 255  # Green area on the left half of the image
        test_image[:, 50:, 0] = 255  # Red area on the right half of the image
        cv2.imwrite("test_image.jpg", test_image)
    
        # Initialize the SpotRemover class with the test image
        spot_remover = SpotRemover("test_image.jpg", "test_image.jpg", "green")
    
        # Test that the create_mask method correctly creates a mask for the green area of the image
        mask = spot_remover.create_mask()
        self.assertEqual(mask.shape, (100, 100))
        self.assertTrue(np.all(mask[:, :50] == 255))  # Mask should be 255 (fully opaque) for the green area
        self.assertTrue(np.all(mask[:, 50:] == 0))  # Mask should be 0 (fully transparent) for the red area
    
        # Test that the create_mask method correctly creates a mask for the red area of the image
        spot_remover = SpotRemover("test_image.jpg", "test_image.jpg", "red")
        mask = spot_remover.create_mask()
        self.assertEqual(mask.shape, (100, 100))
        self.assertTrue(np.all(mask[:, :50] == 0))  # Mask should be 0 (fully transparent) for the green area
        self.assertTrue(np.all(mask[:, 50:] == 255))  # Mask should be 255 (fully opaque) for the red area

                               
    def test_merge_mask_and_original_image(self):
        # Create test input data
        green_red_image_path = "test_image.jpg"
        original_image_path = "tmp/test_image.jpg"
        mask_type = "green"
        
        # Create an instance of the SpotRemover class
        spot_remover = SpotRemover(green_red_image_path, original_image_path, mask_type)
        
        # Call the merge_mask_and_orignal_image method
        result = spot_remover.merge_mask_and_orignal_image()
        
        # Assert that the method returns the expected output
        self.assertEqual(result, "<expected output>")
        
    def test_get_base_file_name(self):
        # Create test input data
        green_red_image_path = "<path to test image>"
        original_image_path = "<path to test image>"
        mask_type = "green"
        
        # Create an instance of the SpotRemover class
        spot_remover = SpotRemover(green_red_image_path, original_image_path, mask_type)
        
        # Call the _get_base_file_name method
        result = spot_remover._get_base_file_name()
        
        # Assert that the method returns the expected output
        self.assertEqual(result, "<expected output>")
        
    def test_save_to_tiff(self):
        # Create test input data
        green_red_image_path = "<path to test image>"
        original_image_path = "<path to test image>"
        mask_type = "green"
        
        # Create an instance of the SpotRemover class
        spot_remover = SpotRemover(green_red_image_path, original_image_path, mask_type)
        
        # Call the save_to_tiff method
        output_file = spot_remover.merge_mask_and_orignal_image()
        spot_remover.save_to_tiff(output_file)
        
        # Assert that the file was saved correctly
        self.assertTrue(os.path.exists("<expected file path>"))
        
    def test_save(self):
        # Create test input data
        green_red_image_path = "<path to test image>"
        original_image_path = "<path to test image>"
        mask_type = "green"
        extension = "jpg"
        
        # Create an instance of the SpotRemover class
        spot_remover = SpotRemover(green_red_image_path, original_image_path, mask_type)
        
        # Call the save method
        output_file = spot_remover.merge_mask_and_orignal_image()
        spot_remover.save(output_file, extension)
        
        # Assert that the file was saved correctly
        self.assertTrue(os.path.exists("<expected file path>"))




