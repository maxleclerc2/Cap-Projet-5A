import os,sys
import subprocess
import log
import argparse
#from matcher import matcher_variables # Default
from osgeo import gdal, osr, ogr
import file_info as fi
import tempfile
import numpy as np
import image_processing as im_p #Internal Class
import file_info as fi

'''
from shapely.wkt import loads
from collections import namedtuple
from math import ceil
from skimage.measure import block_reduce
from skimage.transform import resize as skit_resize
from skimage.transform import warp as skit_warp
from skimage.transform import SimilarityTransform
'''

global GDAL_DIR
global GDAL_DATA

#Set location of GDAL_DIR , GDAL_DATA
system = sys.platform
if (system == 'win32'):
    GDAL_DIR = 'C:\Program Files\\QGIS 3.10\\bin'
    GDAL_DATA = 'C:\Program Files\\QGIS 3.10\\share\\epsg_csv'

if (os.system == 'linux2'):
    GDAL_DIR = '//opt/anaconda2/share/bin'
    GDAL_DATA = '/opt/anaconda2/share/gdal'

if not os.path.exists(GDAL_DIR):
    print(' [Error] Check GDAL_DIR ')

if not os.path.exists(GDAL_DATA):
    print(' [Error] Check GDAL_DATA ')


def cropData_accordingSHP(pixel_resolution, input_file, output_file,
                          outEPSG=None, inSHP=None, inJson=None,
                          Resampling = None):
    """
    :param pixel_resolution:
    :param input_file:
    :param output_file:
    :param outEPSG:  EPSG of the target spatial reference
    :param inSHP:
    :param inJson:   JSON file describing the clip area
    :param Resampling:
    :return:
    """
    cmd_block = []
    resampling_block = []
    if inSHP is not None:
        cut_line = inSHP
    if inJson is not None:
        cut_line = inJson
    if outEPSG is not None:
        epsg_code = outEPSG
        cmd_block = ['-t_srs', 'EPSG:'+epsg_code]

    if Resampling is not None:
        resampling_block = ['-r', 'average']

#TAP Option should be removed from this commmand line

    args = ['gdalwarp', '-q', '-crop_to_cutline', '-cutline', cut_line,
                '-tap', '-overwrite', '-dstnodata', '0', '-of', 'GTiff',
                '-tr', str(pixel_resolution), str(pixel_resolution)]
    args += cmd_block
    args += resampling_block
    args += [input_file, output_file]

    print( " ".join(args))
    my_env = os.environ.copy()
    #gv = matcher_variables.project_variables('')
    #TODO:Set GDAL_DATA
    my_env["GDAL_DATA"] = GDAL_DATA
    child = subprocess.Popen(args, env=my_env)
    rc = child.wait()
    print( "save in : " + output_file)
    print( '--- Return code:', rc)

################################################################
def make_json_file(json_file,geojson,epsg_code):
    """

    :param json_file:
    :param geojson: geojson object
    :param epsg_code:
    :return: filled json_file
    """

    param = "{\n"
    param += "\"type\": \"FeatureCollection\",\n"
    param += "\"crs\": { \"type\": \"name\", \"properties\": { \"name\": \"urn:ogc:def:crs:EPSG::" + epsg_code + "\" } },\n"
    param += "\"features\": [\n"
    param += "{ \"type\": \"Feature\", \"properties\": { \"prop0\": null }, \"geometry\":"

    with open(json_file, 'w') as f:
        f.write(param)
        f.write(geojson + '}]}')

    log.info(' Create the JSON file : ' + json_file)

################################################################

################################################################
def prepare_data(ref, work_image, WD=None):
    """  Check the  geographical extent of the ref image and the work image
     Define a common ROI
     Output a Dictionnary (o_dic) including
        - the WD
        - the ref image to be matched
        - the working image to be matched
        - the json file
    """
    if WD is not None:
        wd = WD
    else:
        wd = tempfile.mkdtemp(prefix='alos_', dir=os.getcwd(), suffix='')

    o_dic = {}
    o_dic['WD'] = WD
    # Input image processing :
    format = 'GTiff'
    driver = gdal.GetDriverByName(format)
    print(' -----> ' + work_image)
    work = work_image
    src_ds_prj = gdal.Open(work)
    projection = src_ds_prj.GetProjection()

    # Get epsg code of input image
    srs = osr.SpatialReference(wkt=projection)
    epsg_code = srs.GetAttrValue("PROJCS|AUTHORITY", 1)
    if epsg_code:
        log.info(' Input Image (work), EPSG Code is ' + epsg_code)
    else:
        log.warn(' Empty EPSG code')

    # Get pixel spacing of input image :
    in_info = fi.ImageInfo(work)
    px_spacing_work = in_info.xRes
    log.info(' Input Image (work), pixel spacing is ' + str(px_spacing_work))

    # Extent of both images, both extents expressed in the projection of the input image
    # def getCorners(self, outWKT=None, outEPSG=None, outPROJ4=None)
    projection = in_info.projection

    # Get epsg Code of the reference Image :
    ref_ds_prj = gdal.Open(ref)
    projection_ref = ref_ds_prj.GetProjection()
    srs_ref = osr.SpatialReference(wkt=projection_ref)
    epsg_code_ref = srs_ref.GetAttrValue("PROJCS|AUTHORITY", 1)
    if epsg_code_ref:
        log.info(' Input Image (Reference), EPSG Code is ' + epsg_code_ref)
    else:
        log.warn(' Empty EPSG code')

    # Get pixel spacing of reference image :
    ref_info = fi.ImageInfo(ref)
    px_spacing_ref = ref_info.xRes
    log.info(' Input Image (work), pixel spacing is ' + str(px_spacing_ref))

    # extent  (ul_x, ul_y, ur_x, ur_y, lr_x, lr_y, ll_x, ll_y)
    extent1 = in_info.getCorners(outWKT=projection)
    # Calculer dans la geometrie de l image de travail
    l = []
    for k in range(0, 7, 2):
        l.append(str(extent1[k]) + ' ' + str(extent1[k + 1]))
    wkt1 = 'POLYGON ((' + ','.join(l) + ',' + l[0] + '))'
    log.info(' Input Image bounding box (epsg)     : ' + str(wkt1) + ' (' + str(epsg_code) + ' )')

    if epsg_code:
        # Calculer dans la geometrie de l image de reference
        # Dans la  geometrie de l image de travail (outWKT=projection)
        # Necessaire pour l intersection
        extent2 = ref_info.getCorners(outWKT=projection)
        l = []
        for k in range(0, 7, 2):
            l.append(str(extent2[k]) + ' ' + str(extent2[k + 1]))
        wkt2 = 'POLYGON ((' + ','.join(l) + ',' + l[0] + '))'
        log.info(' Input Reference bounding box (epsg) : ' + str(wkt2) + ' (' + str(epsg_code) + ' )')

        # Code pick up from :
        # https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html#calculate-intersection-between-two-geometries

        poly1 = ogr.CreateGeometryFromWkt(wkt1)
        poly2 = ogr.CreateGeometryFromWkt(wkt2)
        intersection = poly1.Intersection(poly2)

        # Si EPSG Differents :
        #          - Convertir vers EPSG de l'image de travail ?
        #          - Convertir intersection dans la geometrie de l image de reference
        # Non !!!  "gdal warp" peut utiliser en entree un json contenant des wkt avec
        # reference spatiale autres que celui de l'image .

        # Si ...epsg_code != epsg_code_ref
        # log.info(' Intersection bounding box : ' + str(intersection.ExportToWkt()))
        # wkt_input = intersection.ExportToWkt()
        # epsg_input = epsg_code
        # epsg_out = epsg_code_ref
        # poly = reproject_geometry(wkt_input, epsg_input, epsg_out)
        # geojson = poly.ExportToJson()
        # json_file_ref = os.path.join(wd, 'intersect_region_ref.json')

        # make_json_file(json_file_ref, geojson, epsg_code_ref)

        # Prepare to clip the reference / working data according to the boundary of the intersection
        # Define outfile for the reference

        # Generate json file for the boundary of intersection expressed in the projection system of the working image
        log.info(' Prepare JSON file describing the region defined as')
        log.info('            intersection between the working and the reference images')

        geojson = intersection.ExportToJson()
        json_file = os.path.join(wd, 'intersect_region.json')
        make_json_file(json_file, geojson, epsg_code)

        o_dic['JSON_file'] = json_file
    else:
        o_dic['JSON_file'] = None

    if o_dic['JSON_file'] is not None:

        # json_file = os.path.join(gv.DATA,'REFERENCE','json_test.geojson')

        # Clip data
        log.info(' Clip working data according to the json_file')
        # The output pixel resolution corresponds to the resoluton of the working image
        pixel_resolution = px_spacing_work
        input_file = work
        output_file = os.path.join(wd, 'clip_work.tif')
        o_dic['Work_Image'] = output_file

        if (px_spacing_work < px_spacing_ref):
            log.err('The reference is not a reference')
            sys.exit()
        if (px_spacing_work == px_spacing_ref):
            log.info(' Work / Ref same pixel spacing : ' + str(px_spacing_ref))
            resampling = None
        if (px_spacing_work > px_spacing_ref):
            log.info(
                ' Work / Ref different pixel spacing : ' + str(px_spacing_work) + ' m / ' + str(px_spacing_ref) + ' m ')
            log.info(' Reference data is going to be rescaled')
            resampling = "Lanczos"  # best method.

        # CLIP Working Image :
        cropData_accordingSHP(pixel_resolution, input_file, output_file,
                              outEPSG=epsg_code, inJson=json_file,
                              Resampling=resampling)
        log.info(' Clip reference data according to the json_file')

        # The output pixel resolution corresponds to the resoluton of the working image
        pixel_resolution = px_spacing_work
        input_file = ref
        output_file = os.path.join(wd, 'clip_ref.tif')
        o_dic['Reference_Image'] = output_file

        # CLIP Reference Image :
        cropData_accordingSHP(pixel_resolution, input_file, output_file,
                              outEPSG=epsg_code, inJson=json_file,
                              Resampling=resampling)

        list_image = [o_dic['Reference_Image'], o_dic['Work_Image']]
        im_st = im_p.image(list_image)
        bkg_mask = im_st.get_common_valid_mask(wd)
        mask, out = im_st.maskImage(bkg_mask, wd)

        o_dic['Work_Image'] = out[1]
        o_dic['Reference_Image'] = out[0]
    else:
        log.info(' No EPSG Code then no specific crop applied - native image as input of the processing')
        o_dic['Work_Image'] = ref
        o_dic['Reference_Image'] = work

    return o_dic

if __name__ == '__main__':

    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mon", help="path to the monitored sensor product")
    parser.add_argument("--ref", help="path to the reference sensor product")
    parser.add_argument("--wd", help="path to store working_data")

    args = parser.parse_args()

    ref = args.ref
    mon = args.mon
    wd = args.wd
    #run preprate - clip data to a common grid.
    prepare_data(ref, mon, WD=wd)





