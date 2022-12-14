#!/usr/bin/env python
# -*- coding: utf-8 -*-
# S. Saunier (TPZ-F) 2019

import numpy as np
from osgeo import gdal
from PIL import Image

global margin_size
margin_size = 100


'''
# S.Saunier (TVUK) - EDAP Project ( Landsat Orbit based geometry assessment)
# 2019 Septembre
# Do graphics of Medicis output
# As input filtered output of Medicis, DX, DY, DC.

'''
class canvaz:
#Object for canvaz including list of image and position of each image
   def __init__(self, image_list,ql_qa_filename,fig_label = None):
       self.image_list = image_list
       self.position_x = []  #X position in Column
       self.position_y = []  #Y position in Line
       self.image_size_x = []   # Table of the image width for images in the input list
       self.image_size_y = []   # Table of the image height for images in the input list
       self.set_image_size()
       self.w_max = max(self.image_size_x) #Maximum size of image width
       self.h_max = max(self.image_size_y) #Maximum size of image height
       self.w_canvaz = []
       self.h_canvaz = []
       self.compute_canvaz_size(margin_size) #Compute size of output canvaz in px
       self.output_image_size = []
       self.set_image_position()
       self.fig_label_title = fig_label
       self.fill_canvaz(ql_qa_filename)

   def set_image_size(self):

       height_x = []
       width_y = []

       for image in self.image_list :
           dataset = gdal.Open(str(image))
           x_size = dataset.RasterXSize
           y_size = dataset.RasterYSize
           height_x.append(x_size)
           width_y.append(y_size)
           dataset = None
       self.image_size_x = height_x  # Table of the image width for images in the input list
       self.image_size_y = width_y   # Table of the image height for images in the input list

   def compute_canvaz_size(self,margin_size):
       nb_image = len(self.image_list)
       canvas_dim = [[1 , 2], #1 Image
                     [1 , 2], #2 Images
                     [2 , 2], #3 Images
                     [2 , 2], #4 Images
                     [3 , 2], #5 Images
                     [3 , 2], #6 Images
                     [3 , 3], #7 Images
                     [3 , 3], #8 Images
                     [3 , 3], #9 Images
                     [3 , 4], #10 Images
                     [3 , 4], #11 Images
                     [3 , 4], #12 Images
                     ]
       if nb_image < 13 :
           nb_image_w = canvas_dim[nb_image - 1][0]
           nb_image_h = canvas_dim[nb_image - 1][1]
       else :
          nb_image = np.divide(len(self.image_list)+1,2) * 2
          nb_image_w = np.int(np.divide(nb_image+1,2))
          nb_image_h = np.int(np.divide(nb_image,nb_image_w))

       self.w_canvaz = nb_image_w  * self.w_max + (nb_image_w + 1) * margin_size
       self.h_canvaz = nb_image_h * self.h_max + (nb_image_h + 1) * margin_size
       self.nb_image_w = nb_image_w
       self.nb_image_h = nb_image_h


   def set_image_position(self):  # set image position in the canvaz

       nb_image = len(self.image_list)
       nb_image_w = self.nb_image_w
       nb_image_h = self.nb_image_h
       print( 'Input image size width / height         : ', self.w_max, ' / ', self.h_max)
       print( 'Number of sub-images                    : ', nb_image_w, ' x ', nb_image_h)
       k = 0
       for h_i in range(1, nb_image_h + 1):
            for w_i in range(1, nb_image_w + 1):
                offset_x = margin_size * (w_i) + (w_i - 1) * self.w_max  # X position in Column
                offset_y = margin_size * (h_i) + (h_i - 1) * self.h_max  # Y position in Line
                # image_path = self.image_list[k]
                # image_name = os.path.basename(image_path)
                self.position_x.append(offset_x)  # X position in Column
                self.position_y.append(offset_y)  # Y position in Line
                k = k + 1
       print('Position of each sub-image (col, line)  : ', self.position_x, ' ', self.position_y)

# TODO:
#  Compare to oringinal fill cancaz, ql_qa_file_name absolute and remove reference to WD
   def fill_canvaz(self,ql_qa_file_name,title=None):

       out = np.ones([self.h_canvaz,self.w_canvaz,3],'uint8')*255
       tmp_file = ql_qa_file_name
       print('Canvaz processing of ql_qa_file_name : ',ql_qa_file_name)
       for k,image in enumerate(self.image_list):
           src_ds = gdal.Open(str(image))

           for ch in range(0,3):
                   array_input = src_ds.GetRasterBand(ch+1).ReadAsArray()
                   rows = len(array_input)
                   cols = len(array_input[0])
                   out[self.position_y[k]:self.position_y[k]+rows,self.position_x[k]:self.position_x[k] + cols,ch] = array_input
           src_ds = None


       img = Image.fromarray(out,'RGB')
       loc_text = np.int(out.shape[1] / 2)
       #Product Name, Path - Row, Confidence
       from PIL import ImageDraw,ImageFont
       fontsize = 30
       font = ImageFont.truetype("arial.ttf", fontsize)
       d = ImageDraw.Draw(img)

       if self.fig_label_title is None :
           title = ' '
           len_t = 0
       else :
           title = self.fig_label_title
           len_t = len(title)
       s = np.int(np.int(len_t) / 2.0)
       d.text(((loc_text - s*15), 50), title, fill=(0, 0, 0),font = font)
       img.save(tmp_file)