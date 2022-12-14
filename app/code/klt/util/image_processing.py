# -*- coding: utf-8 -*-
# init class
import os
import shutil
from osgeo import gdal
import glob
import re as r
import log
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import file_info as fi

'''
Date         :October, 02 Tuesday 2018
Author      : S . Saunier
Description :
   -- Image processing module   --
   -- Image Class for function related to image processing

03/06/2019 - Update for EDAP, add  (SS)



CLASS image :
           [_    rescaleImage(self,im1,dst_filename)
           [_    get_common_valid_mask(self,dst_rep_name)           
           [_    maskImage(self,mask,dst_rep_name)
           [_    histerysisThreshold(self,threshold,dst_rep_name)           
           [_    computeRadialError(self,dst_file_name)
			     (Consider Two Images in the list and applied root(x²+y²))
           [_    burn_image(dst,im_filename,mask_filename)			
           [_    byte_rescaling(self,dst_repo )
           [_    sigma_threshold(self,im,n_value,threshold)           
           [_    createimageQuicklook(self,ql_name,outputdir,sr_resolution)
           [_    create_rvb_ds(self,working_dir,dst_filename)           
'''

class image:

    def __init__(self,image_list):
        self.image_list = image_list
        self.image_list_valid = False
        if not (image_list == None) :
                self.image_list_valid = True

        self.quick_name = ''
        self.mask_image = ''          #BINARY MASK
        self.mask_image_valid = False
        self.masked_image = ' '       #IMAGE WITH MASK APPLIED
        self.confidence_threshold = ''
        self.image_list_output	 = ''
        self.image_list_output_valid = False
        
    def rescaleImage(self,im1,dst_filename):
#Rescale Im1 to the size of image in the list
#Return List of Images with Overlay
        input_list = self.image_list
        if (len(input_list) >  0 ):
            input_image = self.image_list[0]
            im1_rad = os.path.basename(im1).split('.')[0]
            im1_path = os.path.dirname(im1)
            im1_rescale = os.path.join(im1_path,im1_rad+'_rescale.tif')
            im1_tmp = os.path.join(im1_path,im1_rad+'_tmp.tif')

            src_ds = gdal.Open(str(input_image))
            geotransform = src_ds.GetGeoTransform()
            print( '-- Origin = (', geotransform[0], ',', geotransform[3], ')')
            print( '-- Input Pixel Size = (', geotransform[1], ',', geotransform[5], ')')

#msk_ds - land_sea_mask
            im1_ds = gdal.Open(str(im1))
#Get bands
            ref_array = src_ds.GetRasterBand(1).ReadAsArray()
            im1_array = im1_ds.GetRasterBand(1).ReadAsArray()
            nb_line = ref_array.shape[0]
            nb_col = ref_array.shape[1]

#Test Im1 and Input_Image not the same pixel_size ?

#Rescale mask to input image scale - 2 stages
            cmd = ' '.join(['gdal_translate -of GTiff','-strict -tr ',str(geotransform[1]),' ', \
                        str(geotransform[1]),
#                        '-outsize ',str(nb_line),' ',str(nb_col),
                        ' -r nearest',
                        im1,im1_tmp])
            os.system(cmd)
            cmd = ' '.join(['gdal_translate -of GTiff','-strict ',
                        '-outsize ',str(nb_col),' ',str(nb_line),
                        ' -r nearest',
                        im1_tmp,im1_rescale])
            os.system(cmd)
            os.remove(im1_tmp)
            src_ds = None
            im1_ds = None

            return True
        else :
            log.warning(' - Missing Inputs \n')
            return False

    def get_common_valid_mask(self,dst_rep_name):
    # Compute background for the two images in image_list and computed the
    # the mask of valid data

        log.infog(' - [Start] -> get Common bkg mask \n')
        log.info(' - Background exist in all images in the list')
        log.info(' - A background mask common to all input images is computed')
        input_image_1 = self.image_list[0]
        input_image_2 = self.image_list[1]
        print( "input_1 ",input_image_1)
        print( "input_2 ",input_image_2)

        bkg_mask_file = os.path.join(dst_rep_name,'bkg_mask.tif')
        if os.path.exists(bkg_mask_file):
            os.remove(glob.glob(bkg_mask_file)[0])
        in_1_ds = gdal.Open(str(input_image_1))
        in_2_ds = gdal.Open(str(input_image_2))
        in_1_array = in_1_ds.GetRasterBand(1).ReadAsArray()
        in_2_array = in_2_ds.GetRasterBand(1).ReadAsArray()
        tmp1_array = np.zeros(in_1_array.shape)
        tmp2_array = np.copy(in_2_array.shape)
        tmp1_array[(in_1_array > 0)*(in_2_array > 0) ] = 1
        msk_array = tmp1_array
        tmp_ds = gdal.GetDriverByName('MEM').CreateCopy('', in_1_ds, 0)
        tmp_ds.GetRasterBand(1).WriteArray(msk_array, 0, 0)
        # Write output
        gdal.GetDriverByName('GTiff').CreateCopy(bkg_mask_file, tmp_ds, 0)

        tmp_ds = None
        src_ds = None

        log.infog(' - [End] -> get Common bkg mask \n')
        return bkg_mask_file

    def maskImage(self,mask,dst_rep_name):
        """ Can use mask as set in the object property.
# Applied mask_to_image_list can rescale mask to fit with image size

#Only one Mask applied to the list
#For each image in image_list, if Binary mask = 1 then pixel value of image is kept
#For each image in image_list, if Binary mask = 0 then pixel value of image is set to '0'

#                        [msk       ] Name of the binary mask
#                        [dst_rep_name ] Repository where new images are stored
#
#MASK is rescalled to the input image size
        """
        log.infog(' - [Start] -> Mask Image Data with the binary mask : ')
        out = []
        poubelle = []
        input_image = self.image_list[0]

# Get image information of input mask :
        print( mask)
        in_info_mask = fi.ImageInfo(mask)
        mask_pixel_resolution = in_info_mask.xRes

# Get image information of input image :
        in_info_image = fi.ImageInfo(input_image)
        inImage_pixel_resolution = in_info_image.xRes

# Get size  input images :
        nb_col = in_info_image.ncols
        nb_line = in_info_image.nrows

# Get Projection information of input images :
        src_ds = gdal.Open(str(input_image))
        geotransform = src_ds.GetGeoTransform()
        print( '-- Origin = (', geotransform[0], ',', geotransform[3], ')')
        print( '-- Input Pixel Size = (', geotransform[1], ',', geotransform[5], ')')

# Compare pixel spacing :
        doRescale = None
        if (inImage_pixel_resolution != mask_pixel_resolution) :
            print( " Rescale Input mask")
            doRescale = True

        if doRescale :
#Set Name to Rescaled input mask (Land Sea Mask) to the same pixel size as input image
            mask_rad = os.path.basename(mask).split('.')[0]
            mask_path = os.path.dirname(mask)

            mask_rescale = os.path.join(mask_path,mask_rad+'_rescale.tif')
            self.mask_image = mask_rescale
            mask_tmp = os.path.join(mask_path,mask_rad+'_tmp.tif')

            if os.path.exists(mask_rescale):
                 os.remove(glob.glob(mask_rescale)[0])

# Apply GDAL Transformation (resize )
            cmd = ' '.join(['gdal_translate -of GTiff', '-strict -tr ', str(geotransform[1]), ' ', \
                            str(geotransform[1]),
                            ' -r nearest',
                            mask, mask_tmp])
            os.system(cmd)

            cmd = ' '.join(['gdal_translate -of GTiff', '-strict ',
                            '-outsize ', str(nb_col), ' ', str(nb_line),
                            ' -r nearest',
                            mask_tmp, mask_rescale])
            os.system(cmd)
            poubelle.append(mask_tmp)
        else :
            mask_rescale = mask

        print( 'End of data preparation')

# Loop on all input images and apply mask
        for input_image in self.image_list:

# Save input image to OLD.TIF file :
            f_name = os.path.basename(input_image)
            p = os.path.dirname(input_image)
            f_old = r.sub('.tif|.TIF' ,'_OLD.TIF',f_name)
            input_image_old = os.path.join(p,f_old)
            shutil.copy(input_image,input_image_old)
            poubelle.append(input_image)
            poubelle.append(input_image_old)

# Define the file name of output images :

            output_f_name = r.sub('.tif|.TIF' ,'_masked.TIF',f_name)
            dst_file_name = os.path.join(dst_rep_name,output_f_name)
            print( ' Out Image name (masked) : ',dst_file_name)
            self.masked_image = dst_file_name
            out.append(dst_file_name)

# Applied Land Sea Mask to Correlation Confidence Image :
            mask_ds = gdal.Open(str(mask_rescale))
            mask_array = mask_ds.GetRasterBand(1).ReadAsArray()

            input_ds = gdal.Open(input_image)
            d_array = input_ds.GetRasterBand(1).ReadAsArray()

# Test if array of same size :
            nb_line_1 = mask_array.shape[0]
            nb_col_2 = mask_array.shape[1]
            if (nb_line_1 != nb_line ) :
                log.warn('-- The image size of Mask and Confidence Different \n')
                log.warn(' '.join(['-- Masque          line_nbr x col_nbr : ',str(nb_line_1),'X',str(nb_col_2) ,' \n']))
                log.warn(' '.join(['-- Confident Image line_nbr x col_nbr : ',str(nb_line),'X',str(nb_col) ,' ',' \n']))
                return
            else :
                self.mask_image_valid = True

# Apply masking  :
            m1 = np.copy(d_array)
            m1[mask_array == 0] = 0

            tmp_ds = gdal.GetDriverByName('MEM').CreateCopy('', src_ds, 0)
            tmp_ds.GetRasterBand(1).WriteArray(m1, 0, 0)
#Write output :
            gdal.GetDriverByName('GTiff').CreateCopy(dst_file_name, tmp_ds, 0)
            mask_array = None
            d_array = None
            m1 = None

        mask_ds = None
        input_ds = None
        src_ds = None
        log.infog(' - [End] -> Mask Image Data with the following binary mask : ')
        log.info('         -->       '+ mask_rescale)
        log.infog(' - [End] -> With the following Masked Image              : ')
        log.info('         -->       '+ dst_file_name)

        for rec in poubelle :
            print( rec)
#            os.remove(rec)

        return mask_rescale,out

    def histerysisThreshold(self,threshold,dst_rep_name):

    # HISTERYSIS Thresholding -
    #
          #            Output images are "Int" type and ready for QL Generation
	  # [threshold]  : To apply to input image
	  # [dst_rep_name]   : Where to store results
	  # 			|__> maskedImage
	  # 			|__> binaryMask (pixels affected by thresholding)
      
        src_ds = gdal.Open(str(self.image_list[0])) #DC
        self.confidence_threshold = threshold
 
        array = src_ds.GetRasterBand(1).ReadAsArray() #dc_array
        log.infog(' - Start -> Threshold Input Images \n')
        output_array = np.copy(array)
#Applied Threshold :
        output_array[array < threshold] = 0
        binary_array = np.copy(array)
        binary_array[array < threshold] = 0
        binary_array[array >= threshold] = 1

#Save Masked Image :
        tmp_ds = gdal.GetDriverByName('MEM').CreateCopy('', src_ds, 0)
        tmp_ds.GetRasterBand(1).WriteArray(output_array, 0, 0)
        ##Write output
        tmp_file_name = os.path.basename(self.image_list[0]).split('.')[0]
        suff = ''.join(['_maskedImage_',str(np.int(threshold*100)),'.TIF'])
        new_name = '_'.join([tmp_file_name,suff])
        dst_file_name = os.path.join(dst_rep_name,
                                     new_name      
                                     )

        gdal.GetDriverByName('GTiff').CreateCopy(dst_file_name, tmp_ds, 0)
        self.masked_image = dst_file_name
        tmp_ds = None
#Save Binary Mask Image :
        tmp_ds = gdal.GetDriverByName('MEM').CreateCopy('', src_ds, 0)
        tmp_ds.GetRasterBand(1).WriteArray(binary_array, 0, 0)

#Write output :
        tmp_file_name = os.path.basename(self.image_list[0])
        suff = ''.join(['_binaryMask_',str(np.int(threshold*100)),'tmp.TIF'])
        new_name1 = r.sub('.tif|.TIF' ,suff,tmp_file_name)
        dst_file_name1 = os.path.join(dst_rep_name,
                                     new_name1      
                                     )
        suff = ''.join(['_binaryMask_',str(np.int(threshold*100)),'.TIF'])
        new_name2 = r.sub('.TIF' ,suff,tmp_file_name)
        dst_file_name2 = os.path.join(dst_rep_name,
                                     new_name2      
                                     )

        gdal.GetDriverByName('GTiff').CreateCopy(dst_file_name1, tmp_ds, 0)
        cmd = ' '.join(['gdal_translate -ot Byte',
                        dst_file_name1,dst_file_name2])
        os.system(cmd)
        tmp_ds = None
        src_ds = None
        os.remove(dst_file_name1)
        self.mask_image = dst_file_name2
        log.info(' -- Create : '+ self.mask_image)
        log.info(' -- Create : '+ self.masked_image+'\n')
        log.infog(' - [End] -> Hysterisis threshold  \n')


    def computeRadialError(self,dst_file_name,px_size = 1.0):

        if len(self.image_list) == 2 :
            log.infog(' - Start -> Radial Error Computation \n')
            x = self.image_list[0] #DX
            y = self.image_list[1] #DY
            log.infog(' -- File 1  -> ' + x)
            log.infog(' -- File 2  -> ' + y)
            src_x_ds = gdal.Open(str(x))
            src_y_ds = gdal.Open(str(y))
            x_array = src_x_ds.GetRasterBand(1).ReadAsArray()
            y_array = src_y_ds.GetRasterBand(1).ReadAsArray()
            r_array = px_size * np.sqrt(x_array*x_array + y_array*y_array)

            tmp_ds = gdal.GetDriverByName('MEM').CreateCopy('', src_x_ds, 0)
            tmp_ds.GetRasterBand(1).WriteArray(r_array, 0, 0)
            gdal.GetDriverByName('GTiff').CreateCopy(dst_file_name, tmp_ds, 0)
        
            src_x_ds = None
            src_y_ds = None
            log.infog(' - [End] -> Radial Error Computation \n')
            log.info(' -- Create : '+ dst_file_name+'\n')
            return True
        else :
            log.warning(' - Missing Inputs \n')
        return False
        
    def burn_image(self,dst,im_filename,mask_filename) :
        '''
        BURN INPUT IMAGES (RGB Images) with Im values, Im_msk required
                   because contain pixel to be selected
                   Output images are "Int" type and ready for QL Generation

        :param dst: List of output file names
        :param im_filename: Input image that will overlay the images
        :param mask_filename: Mask associated to Input image
        :return:
        '''

        for image in self.image_list :
            print( image)

        if len(self.image_list) == 3 :
#Load Input Dataset :
            r_filename = self.image_list[0]
            g_filename = self.image_list[1]
            b_filename = self.image_list[2]
            r_ds = gdal.OpenShared(r_filename)
            g_ds = gdal.OpenShared(g_filename)
            b_ds = gdal.OpenShared(b_filename)

            r_out_filename = dst[0]
            g_out_filename = dst[1]
            b_out_filename = dst[2]

#Load Image and mask :
            im_ds = gdal.OpenShared(im_filename)
            im_mask_ds = gdal.OpenShared(mask_filename)
            mask = im_mask_ds.GetRasterBand(1).ReadAsArray().astype(np.bool)
#Generate Image to overlay (Text ) => donne [m_text]
# Prepare headings :
            nb_line =mask.shape[0]
            nb_col = mask.shape[1]
            img = Image.new('I', (nb_col, nb_line), 0)
            d = ImageDraw.Draw(img)
# Font title :
            font_size = 31  # Titre
            font_file = os.path.join(FONT_PATH, 'arialbd.ttf')
            font_title = ImageFont.truetype(font_file, font_size)
            title = 'Radial Error'
            ch_001 = ''.join([title])
# HEADER OF THE REPORT :
            x_header_shift = 50
            y_header_shift = 50
            pas_header = 70
            d.text((y_header_shift, x_header_shift), ch_001, fill=(255), font=font_title)

# Legend RED :
            title = 'RED      : Error in pixels '
            ch = ''.join([title])
            x_header_shift = nb_line - 150
            y_header_shift = nb_col - 100
            #d.text((y_header_shift, x_header_shift), ch, fill=(255), font=font_title)

            title_map = np.array(img)
            img = None
            m_text = np.zeros((mask.shape[0], mask.shape[1]), dtype='uint8')
            m_text[title_map == 255] = 1

            array_image = r_ds.GetRasterBand(1).ReadAsArray()
            im = im_ds.GetRasterBand(1).ReadAsArray()  #Array of values
            im1 = im[mask]

            if np.size(im1) > 0 :
                mx = np.max(im1)
                mi = np.min(im1)
                moy = np.mean(im1)
                std = np.std(im1)
#RED BAND     :
                print ('Statistics for pixels after 3 sigma thresholding, min max mean std : '+str(mi)+' '+str(mx)+' '+str(moy)+' '+str(std))
                print( np.median(array_image))
                print( np.min(array_image))
                print( np.max(array_image))
                im_t = np.multiply(im, 255/(mx-mi) )
            else :
                im_t = im
            m1 = mask
            array_image[m1 == 1] = 0
            array_image = array_image + im_t
            array_image[title_map == 255] = 255

            tmp_ds = gdal.GetDriverByName('MEM').CreateCopy('', r_ds, 0)
            tmp_ds.GetRasterBand(1).WriteArray(array_image, 0, 0)
            dst_filename = r_out_filename
            gdal.GetDriverByName('GTiff').CreateCopy(dst_filename,tmp_ds,0 )
            array_image = None
            tmp_ds = None
#GREEN BAND :
            array_image = g_ds.GetRasterBand(1).ReadAsArray()
            array_image[m1 == 1] = 0
            array_image[title_map == 255] = 0

            tmp_ds = gdal.GetDriverByName('MEM').CreateCopy('', g_ds, 0)
            tmp_ds.GetRasterBand(1).WriteArray(array_image, 0, 0)
            dst_filename = g_out_filename
            gdal.GetDriverByName('GTiff').CreateCopy(dst_filename,tmp_ds,0 )
            array_image = None
            tmp_ds = None
#BLUE BAND :
            array_image = b_ds.GetRasterBand(1).ReadAsArray()
            array_image[m1 == 1] = 0
            array_image[title_map == 255] = 0
#Copy yhe file and prepare output :
            tmp_ds = gdal.GetDriverByName('MEM').CreateCopy('', b_ds, 0)
            tmp_ds.GetRasterBand(1).WriteArray(array_image, 0, 0)
            dst_filename = b_out_filename
            gdal.GetDriverByName('GTiff').CreateCopy(dst_filename,tmp_ds,0 )
            array_image = None
            tmp_ds = None
            shp_ds = None
            src_ds = None
            rgb_ds = None
            array_image = None
            return True
        else :
            log.warning(' - Three Images are needed to create the QL \n')
            return False

    def byte_rescaling(self,dst_repo ):

         output_list = []
         for image in self.image_list: 
               file_name = os.path.basename(image).replace('.TIF','_8bit.tif')
               output = os.path.join(dst_repo,file_name)
               cmd = ' '.join(['gdal_translate -ot Byte -scale -of GTiff ',image,output])
               os.system(cmd)
               output_list.append(output)
     
         return output_list
         

    def sigma_threshold(self,im,n_value,threshold):

       #REMOVE OUTLIER OF IM1 - BASED ON VALUE OF IM1 WHERE MASK VALUE > Threshold
       #im ; Input Image path
       #RETURN MODIFIED MASK & MODIFIED DC
       #RETURN DX / DY

        if len(self.image_list) == 2 :
#Load Input Dataset - DC and DC_MASK
            dc_filename = self.image_list[0] #DC
            dc_mask_filename = self.image_list[1] #DC_MASK
            dc_ds = gdal.OpenShared(dc_filename)
            dc_array = dc_ds.GetRasterBand(1).ReadAsArray()            

            dc_mask_ds = gdal.OpenShared(dc_mask_filename)
            dc_mask_array = dc_mask_ds.GetRasterBand(1).ReadAsArray().astype(np.bool)            
#Load Input Dataset - Image  (DX ou DY)
            im_ds = gdal.OpenShared(im)
            im_array = im_ds.GetRasterBand(1).ReadAsArray()
#Load Image and mask
            tmp_array = np.copy(im_array)
            #SELECT PIXELS WHERE CONFIDENCE > Threshold
# v ) Keep only values of IM where DC above the threshold :
            v = im_array[dc_array >= threshold]
            if (np.size(v)) > 0 :
                mi = np.min(v)
                mx = np.max(v)
                md = np.median(v)
                moy = np.mean(v)
                std = np.std(v)
                print (' -- Native statistics of the image im ')
                print('     for DC threshold >= '+str(threshold)+' applied')
                print ('         min     |     max     |     median     |     mean     |      std : ' )
                print ('       '+str(mi)+'    '+str(mx)+'    '+str(md)+'    '+str(moy)+'    '+str(std)+'\n')
                SUCCESS = True
            else :
                print ('No data in DC above confidence threshold ' )
                SUCCESS = False
                return
#Set to 0 - dc value where displacement exceed n_value sigma thresholding
            dc_array[np.abs(tmp_array) > moy + n_value * std ] = 0
#SET Mask Value to 0
            dc_mask_array [dc_array == 0] = 0
            v = im_array[dc_array >= threshold]
            if np.size(v) > 0 :
                mi = np.min(v)
                mx = np.max(v)
                md = np.median(v)
                moy = np.mean(v)
                std = np.std(v)
                print (' -- A posteriori statistics of im with sigma threshold applied')
                print ('         min     |     max     |     median     |     mean     |      std : ' )
                print ('       '+str(mi)+'    '+str(mx)+'    '+str(md)+'    '+str(moy)+'   '+str(std)+'\n')
                SUCCESS = True
            else :
                print (' [WARNING] No data in DC above confidence threshold ' )
                SUCCESS = False

#Update DC File and stored previously used DC as 'old' :
            new_name = r.sub('.tif|.TIF' ,'_OLD.TIF',dc_filename)
            print( ' --- Copy  : ')
            print( '        From :    '+dc_filename+' ')
            print( '         To  :    '+ new_name)

            shutil.copy(dc_filename,new_name)

            dc_ds = gdal.OpenShared(dc_filename)
            tmp_ds = gdal.GetDriverByName('MEM').CreateCopy('', dc_ds, 0)
            tmp_ds.GetRasterBand(1).WriteArray(dc_array, 0, 0)
            dc_ds = None
            gdal.GetDriverByName( 'GTIFF' ).CreateCopy(dc_filename,tmp_ds,0 )
            tmp_ds = None

#Update DC Mask File and stored previously used MASK as old:
            new_name = r.sub('.tif|.TIF' ,'_OLD.TIF',dc_mask_filename)
            print( ' --- Copy  : ')
            print( '        From :    '+dc_filename+' ')
            print( '         To  :    '+ new_name)

            shutil.copy(dc_mask_filename,new_name)
            tmp_ds = gdal.GetDriverByName('MEM').CreateCopy('', dc_mask_ds, 0)
            tmp_ds.GetRasterBand(1).WriteArray(dc_mask_array, 0, 0)
            dc_mask_ds = None
            gdal.GetDriverByName( 'GTIFF' ).CreateCopy(dc_mask_filename,tmp_ds,0 )
            tmp_ds = None

            self.mask_image = dc_mask_filename
            self.masked_image = dc_filename

            return SUCCESS


    def createimageQuicklook(self,ql_name,outputdir,sr_resolution):
#createimageQuicklook :
# Input  :        
#          - [ql_name] : Path + Filename of the quicklook
#          - [outputdir] : Path of the working directory
#          - [sr_resolution] : Pixel size of the output image

# Output :
        


        image_data_list = self.image_list
        log.info( '- Create Image Quick Looks for :')
        for rec in image_data_list:
            log.info( '-- [FILE] '+rec)

        image_rad=os.path.basename(image_data_list[0]).split('_')[0]
        output_file_list = []

        for file_name in image_data_list:
                image_data=file_name
                stret = os.path.join(outputdir,'stret.tif')
                #cmd=' '.join(['gdal_contrast_stretch -ndv 0 -outndv 0 -linear-stretch 125 50 ',image_data,stret])
                #os.system(cmd)
#TODO : Command gdal_contrast_stretch no available, skip this process,
                image_data_rs = os.path.join(outputdir,os.path.basename(file_name).replace('.tif','_rs.tif'))
                cmd=' '.join(['gdalwarp  -srcnodata 0  -r average -of GTiff -tr',
                               str(sr_resolution),str(sr_resolution),'-ot Byte ',
#                              stret, image_data_rs])
                              image_data,image_data_rs])
                os.system(cmd)
#                cmd=' '.join(['rm -f',stret])
                os.system(cmd)

                output_file_list.append(image_data_rs)

        image_string = ' '
        for file_name in output_file_list :
           image_string =  ' '.join([image_string,file_name])

        log.info( '-- Build VRT file '+'\n')
        vrt_filename=os.path.join(outputdir,''.join([image_rad,'_ql.vrt'])  )
        opt = gdal.BuildVRTOptions(xRes = sr_resolution,yRes=sr_resolution,separate= True)
        gdal.BuildVRT(vrt_filename,image_string,options = opt)

        log.info( '-- Build JPG file '+'\n')
        ext='.jpg'
        #Convertion of vrt to png
#        qlfile_name=os.path.join(outputdir,''.join([image_rad,'_ql_bd234',ext]))
        cmd=' '.join(['gdal_translate -of jpeg',vrt_filename,ql_name,'-b 3 -b 2 -b 1 -a_nodata 0'])
        os.system(cmd)

        self.quick_name = ql_name
        self.quick_name_valid = True


        log.info( '-- Clean BackLog '+'\n')
        cmd=' '.join(['rm -f' , image_string])
        os.system(cmd)

        cmd=' '.join(['rm -f' , vrt_filename])
        os.system(cmd)


    def create_single_image_ql(self,wd,im,sr_resolution):
#TODO Integrate into the class , self no use
        #im : image to be trnasform in QL.

        p = (os.path.basename(im).split('.'))[0]
        vrt_filename = os.path.join(wd, ''.join([p, '_ql.vrt']))
        ql_filename = os.path.join(wd, ''.join([p, '_ql.jpeg']))
        sr_resolution = 1000
        opt = gdal.BuildVRTOptions(xRes=sr_resolution, yRes=sr_resolution, separate=True)
        gdal.BuildVRT(vrt_filename, [im, im, im], options=opt)

        opt = gdal.TranslateOptions(format='jpeg', bandList=[3, 2, 1], noData=0,
                                    xRes=sr_resolution, yRes=sr_resolution)
        gdal.Translate(ql_filename, vrt_filename, options=opt)

    def create_rvb_ds(self,working_dir,dst_filename):
        
        #dst_filename : Name of RVB Image

        B1 = self.image_list[0]
        B2 = self.image_list[1] 
        B3 =  self.image_list[2]

        B1_8 = os.path.join(working_dir,'B1.tif')
        B2_8 = os.path.join(working_dir,'B2.tif')
        B3_8 = os.path.join(working_dir,'B3.tif')

        #Scale to 8 bit :
        cmd = ' '.join(['gdal_translate -ot Byte -scale',B1,B1_8])
        os.system(cmd)
        #Scale to 8 bit :
        cmd = ' '.join(['gdal_translate -ot Byte -scale',B2,B2_8])
        os.system(cmd)

        #Scale to 8 bit :
        cmd = ' '.join(['gdal_translate -ot Byte -scale',B3,B3_8])
        os.system(cmd)

        src_ds = gdal.OpenShared(B3_8)
        tmp_ds = gdal.GetDriverByName('MEM').CreateCopy('',src_ds,0)

        src_ds = gdal.Open(str(B2_8))
        array = src_ds.GetRasterBand(1).ReadAsArray()
        tmp_ds.AddBand(gdal.GDT_Byte)
        tmp_ds.GetRasterBand(2).WriteArray(array,0,0)

        src_ds = gdal.Open(str(B1_8))
        array = src_ds.GetRasterBand(1).ReadAsArray()
        tmp_ds.AddBand(gdal.GDT_Byte)
        tmp_ds.GetRasterBand(3).WriteArray(array,0,0)
        #http://www.gdal.org/gdal_8h.html#a22e22ce0a55036a96f652765793fb7a4
        gdal.GetDriverByName( 'GTiff' ).CreateCopy(dst_filename,tmp_ds,0 )

        src_ds = None
        tmp_ds = None

        #Scale to radiometry
        cmd = ' '.join(['gdal_translate -ot Byte -scale',dst_filename,B1_8])
        os.system(cmd)

        #REMOVE B1_8, B2_8, B3_8 :
        cmd = ' '.join(['rm -rf ',B1_8,B2_8,B3_8])
        os.system(cmd)

        return dst_filename



