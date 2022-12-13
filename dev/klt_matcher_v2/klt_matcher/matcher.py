#! /usr/bin/env python
# -*- coding: utf-8 -*-
# V. Debaecker (TELESPAZIO France)
# S. Saunier   (TELESPAZIO France) - November 20, 2022

import os
import sys
import numpy as np
import cv2
from skimage import io
import pandas as pd
from matplotlib import pyplot as plt
from osgeo import gdal, osr
from math import floor, ceil
from skimage.transform import warp
from mpl_toolkits.axes_grid1 import make_axes_locatable

from accuracy_analysis import plot_functions as plt_f
from accuracy_analysis import accuracy_statistics as accuracy_statistics
from accuracy_analysis import canvaz as cv

class GdalRasterImage:

    def __init__(self, filename):
        self.filepath = filename
        self.readHeader()
        self._array = None

    def readHeader(self):
        # geo information
        dst = gdal.Open(self.filepath)
        geo = dst.GetGeoTransform()
        self.xSize = dst.RasterXSize
        self.ySize = dst.RasterYSize

        # Spatial Reference System
        projection = dst.GetProjection()

        self.xRes = geo[1]
        self.yRes = geo[5]
        # self.xMin = geo[0] + 2*self.xRes
        # self.yMax = geo[3] + 2*self.yRes
        self.xMin = geo[0]
        self.yMax = geo[3]
        self.xMax = self.xMin + self.xSize * self.xRes
        self.yMin = self.yMax + self.ySize * self.yRes
        self.projection = dst.GetProjection()

        image_srs = osr.SpatialReference(wkt=self.projection)
        if "Popular Visualisation Pseudo Mercator" in self.projection:
            # bug with EO-Browser projection
            image_srs.ImportFromEPSG(3857)

        # target is 4326
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)

        if target_srs.GetAuthorityCode(None) != '4326':
            """target_srs.ImportFromEPSG(target_epsg)
            image_srs = osr.SpatialReference(wkt=projection)
            image_epsg = image_srs.GetAuthorityCode(None)"""
            print(image_srs.ExportToWkt())

            # convert UL and LR coordinates
            self.xMin, self.yMax = convert_coordinates(self.xMin, self.yMax, image_srs, target_srs)
            self.xMax, self.yMin = convert_coordinates(self.xMax, self.yMin, image_srs, target_srs)
            print(self.xMin, self.yMax)
            print(self.xMax, self.yMin)

            # recompute resolution
            self.xRes = (self.xMax - self.xMin) / self.xSize
            self.yRes = (self.yMin - self.yMax) / self.ySize
            print(self.xRes, self.yRes)

            # set prokection to 4326

            self.projection = target_srs.ExportToWkt()

        dst = None

    def read(self, band, xoff, yoff, xsize, ysize):
        dst = gdal.Open(self.filepath)
        b = dst.GetRasterBand(band)
        data = b.ReadAsArray(xoff, yoff, xsize, ysize)
        dst = None
        return data

    def get_array(self):
        if self._array is None:
            dst = gdal.Open(self.filepath)
            b = dst.GetRasterBand(1)
            self._array = b.ReadAsArray()
            dst = None
        return self._array

    array = property(get_array, doc="Access to image array (numpy array)")


def pointcheck2(x0, y0, x1, y1, score):
    dx = x1 - x0
    dy = y1 - y0
    while True:
        ind = ((np.abs(dx - dx.mean()) < 3 * dx.std())
               & (np.abs(dy - dy.mean()) < 3 * dy.std())
               & (np.abs(dx - dx.mean()) < 20)
               & (np.abs(dy - dy.mean()) < 20))
        if len(ind[ind == True]) == len(dx):
            break
        dx = dx[ind]
        dy = dy[ind]
        x0 = x0[ind]
        x1 = x1[ind]
        y0 = y0[ind]
        y1 = y1[ind]
        score = score[ind]
    return x0, y0, x1, y1, score


'''
#Parameters : 
maxCorners=20000                        # Nombre total de KP au depart. 
matching_winsize=25                     # A remonter 
minDistance=10                          # Avoir 2 points a moins de 10 pixel
blockSize=15                            # Pour la recherche des KPs - pas utiliser pour matcher.
'''


def KLT_Tracker(reference, imagedata, mask, maxCorners=20000, matching_winsize=25, outliers=False):
    # compute the initial point set
    # goodFeaturesToTrack input parameters
    feature_params = dict(maxCorners=maxCorners, qualityLevel=0.1,
                          minDistance=10, blockSize=15)
    # goodFeaturesToTrack corner extraction-ShiThomasi Feature Detector
    p0 = cv2.goodFeaturesToTrack(
        reference, mask=mask, **feature_params)
    if p0 is None:
        print("No features extracted")
        return

    # define KLT parameters-for matching
    # info("Using window of size {} for matching.".format(matching_winsize))
    lk_params = dict(winSize=(matching_winsize, matching_winsize),
                     maxLevel=1,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30,
                               0.03))  # LSM input parameters - termination criteria for corner estimation/stopping criteria

    p1, st, err = cv2.calcOpticalFlowPyrLK(reference, imagedata, p0, None,
                                           **lk_params)  # LSM image matching- KLT tracker

    p0r, st, err = cv2.calcOpticalFlowPyrLK(imagedata, reference, p1, None,
                                            **lk_params)  # LSM image matching- KLT tracker

    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
    back_threshold = 0.1
    st = d < back_threshold

    # print("Nb Bad Status: {} ".format(len(st[st == 0])))

    # filter with status
    st_valid = 1
    Ninit = len(p0)
    p0 = p0[st == st_valid]
    p1 = p1[st == st_valid]
    err = err[st == st_valid]
    d = d[st == st_valid]
    score = (1 - d / back_threshold)
    x0 = p0[:, 0, 0].reshape(len(p0))
    y0 = p0[:, 0, 1].reshape(len(p0))
    x1 = p1[:, 0, 0].reshape(len(p1))
    y1 = p1[:, 0, 1].reshape(len(p1))

    if not outliers:
        x0, y0, x1, y1, score = pointcheck2(x0, y0, x1, y1, score)

    # to dataframe
    df = pd.DataFrame.from_dict({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "dx": x1 - x0, "dy": y1 - y0, "score": score})

    d1 = df.iloc[df['dx'].argsort()[-3:]] # TODO
    # d2 = df.iloc[df['dy'].argsort()[-3:]]

    return df, Ninit


def mean_profile(val, points, N, binsize=20):
    """Compute mean values for each columns or rows.
    Possible to do stack several lines or columns (see binsize) in order
    to increase number of points that is taken into account"""
    points = np.floor_divide(points, binsize)
    dic = {"val": val, "po": points}
    df = pd.DataFrame.from_dict(dic)
    g = df.groupby('po')
    pos = np.uint16(g['po'].mean() * binsize + binsize / 2)
    npos = g['val'].count()
    meanpos = g['val'].mean()
    return (pos, meanpos, npos)


def main_KLT2(img_file, ref_file, mask_file, csv_file,
              conf, outliers=True):
    """
    """

    print("KLT...")
    # get KLT Parameters :
    grid_step = conf.klt_configuration["grid_step"]
    N = conf.klt_configuration["N"]
    xStart = conf.klt_configuration["xStart"]
    ksize = conf.klt_configuration["laplacian_kernel_size"]
    maxCorners = conf.klt_configuration["maxCorners"]
    matching_winsize = conf.klt_configuration["matching_winsize"]

    # repo:
    img = GdalRasterImage(img_file)
    ref = GdalRasterImage(ref_file)
    if mask_file is not None:
        mask = GdalRasterImage(mask_file)
    else:
        mask = None

    print(img.xSize, img.ySize)

    # equivalent grid step :
    maxCorners_init = int(N * N / grid_step / grid_step)
    print("maxCorners:", maxCorners_init)

    # outputs
    csv_init = True

    # iterate over N*N boxes
    for xOff in range(0, img.xSize, N):
        if xOff < xStart:
            continue
        for yOff in range(0, img.ySize, N):
            print('Tile: {} {} ({} {})'.format(xOff, yOff, img.xSize, img.ySize))

            # box size
            xSize = N if xOff + N < img.xSize else img.xSize - xOff
            ySize = N if yOff + N < img.ySize else img.ySize - yOff

            # read images
            ref_box = ref.read(1, xOff, yOff, xSize, ySize)
            img_box = img.read(1, xOff, yOff, xSize, ySize)

            print("mask...")
            """mask_box = np.ones((ySize, xSize), np.uint8)
            mask_box[img_box == 0] = 0
            mask_box[ref_box == 0] = 0"""
            if mask:
                mask_box = mask.read(1, xOff, yOff, xSize, ySize)
            else:
                mask_box = (img_box != 0) & (ref_box != 0) & np.isfinite(ref_box) & np.isfinite(img_box)
                mask_box = mask_box.astype(np.uint8)

            # check mask
            valid_pixels = len(mask_box[mask_box > 0])
            if valid_pixels == 0:
                print("-- No valid pixels, skipping this tile")
                print()
                continue
            print("Nb valid pixels: {}/{}".format(valid_pixels, xSize * ySize))

            # laplacian
            img_box = cv2.Laplacian(img_box, cv2.CV_8U, ksize=ksize)
            ref_box = cv2.Laplacian(ref_box, cv2.CV_8U, ksize=ksize)

            # KLT - adapt maxCorners parameters to nb of valid pixels
            maxCorners = int(maxCorners_init * valid_pixels / N ** 2)
            #  to VD: maxCorners : static or dynamic parameters ?
            points, Ninit = KLT_Tracker(ref_box, img_box,
                                        mask_box, maxCorners=maxCorners,
                                        matching_winsize=matching_winsize,
                                        outliers=outliers)
            points['x0'] = points['x0'] + xOff
            points['y0'] = points['y0'] + yOff

            print("NbPoints(init/final): {} / {}".format(Ninit, len(points.dx)))
            print("DX/DY(KLT) MEAN: {} / {}".format(points.dx.mean(), points.dy.mean()))
            print("DX/DY(KLT) STD: {} / {}".format(points.dx.std(), points.dy.std()))
            print()

            # write to csv
            if csv_init:
                points.to_csv(csv_file, sep=";")
                csv_init = False
            else:
                points.to_csv(csv_file, mode='a', sep=";", header=False)


def main_plot_mean_profile(csv_file, img_file, ref_file, outfile, conf):
    img = GdalRasterImage(img_file)
    ref = GdalRasterImage(ref_file)
    print(img.xSize, img.ySize)
    points = pd.read_csv(csv_file, sep=";")

    # Parameters :
    xStart = conf.plot_configuration["xStart"]
    factor = conf.plot_configuration["factor"]
    lim_dx = conf.plot_configuration["lim_dx"]
    lim_dy = conf.plot_configuration["lim_dy"]
    lim_nb_kp = conf.plot_configuration["lim_nb_kp"]

    fig, axes = plt.subplots(2, 5,
                             figsize=(15 * factor, 6 * factor),
                             constrained_layout=True)

    for i, param in enumerate(["Pixel (dx)", "Line (dy)"]):
        ax = axes[i, 1]
        ax.set_title(f"{param} shifts (pixels)")
        ax.set_xlim(xStart, img.xSize)
        ax.set_ylim(img.ySize, 0)

        # Create colormap based on errors
        # errors = np.sqrt(abs(points["dx"]) ** 2 + abs(points["dy"]) ** 2)
        if param == "Pixel (dx)":
            errors = points["dx"]
            lim = lim_dx
        if param == "Pixel (dy)":
            errors = points["dy"]
            lim = lim_dy

        # errors = errors.clip(-lim, lim)
        colors = errors

        # sc = ax.scatter(points["x0"], points["y0"], c=colors, s=1, vmin=-lim, vmax=lim, cmap=plt.cm.jet())
        sc = ax.scatter(points["x0"], points["y0"], c=colors,
                        cmap=plt.cm.viridis, vmin=-lim, vmax=lim, s=1)
        # sc._A = np.array(errors)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.figure.colorbar(sc, cax=cax)

        textstr = "Nb KP={}".format(len(points["x0"]))
        ax.text(40, 250, textstr,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if param == 'Line (dy)':
            (pos, meanpos, npos) = mean_profile(points["dy"], points["x0"], N=img.ySize)
            label = "dY"
            lim = lim_dy
        else:
            (pos, meanpos, npos) = mean_profile(points["dx"], points["x0"], N=img.xSize)
            label = "dX"
            lim = lim_dx

        ax = axes[i, 2]
        ax.plot(pos, meanpos, label=label)
        ax.set_ylim(-lim, lim)
        ax.set_xlim(0, img.xSize)
        ax.set_xlabel("column index")
        ax.set_title(f"{param} mean deviations (pixels)")
        ax2 = ax.twinx()
        ax2.plot(pos, npos, 'r:', label="Nb KP")
        ax2.set_ylim(0, lim_nb_kp)
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

        # profiles by rows
        if param == 'Line (dy)':
            (pos, meanpos, npos) = mean_profile(points["dy"], points["y0"], N=img.ySize)
            label = "dY"
            lim = lim_dy
        else:
            (pos, meanpos, npos) = mean_profile(points["dx"], points["y0"], N=img.xSize)
            label = "dX"
            lim = lim_dx

        ax = axes[i, 3]
        ax.plot(pos, meanpos, label=label)
        ax.set_ylim(-lim, lim)
        ax.set_xlim(0, img.ySize)
        ax.set_xlabel("row index")
        ax.set_title(f"{param} mean deviations (pixels)")
        ax2 = ax.twinx()
        ax2.plot(pos, npos, 'r:', label="Nb KP")
        ax2.set_ylim(0, lim_nb_kp)
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    # dx vs dy
    # Calculate the point density
    # data
    x = np.array(points["dx"])
    y = np.array(points["dy"])

    # reduce the number of points if too much (performance time)
    N = conf.plot_configuration["scatter_max_number_of_points"]
    if len(x) > N:
        indexes = np.random.randint(low=0, high=len(x), size=N)
        x = x[indexes]
        y = y[indexes]

    # Calculate the points density
    from scipy.stats import gaussian_kde
    x_y = np.vstack([x, y])
    density = gaussian_kde(x_y)(x_y)
    # Sort the points by density, so that the densest points are plotted last
    idx = density.argsort()
    x, y, density = x[idx], y[idx], density[idx]

    # Sort the points by density, so that the densest points are plotted last
    ax = axes[0, 4]
    ax.set_title("scatter plot dx (pixel), dy (line)")
    ax.set_xlabel("dx")
    ax.set_ylabel("dy")

    lim = np.maximum(lim_dx, lim_dy)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid()
    ax.hlines(0, -lim, lim, linestyles="dashed", colors="0.5")
    ax.vlines(0, -lim, lim, linestyles="dashed", colors="0.5")
    ax.scatter(x, y, c=density, s=5, edgecolor=None, cmap=plt.get_cmap('jet'))
    # stats
    textstr = "DX\n  mean={:.2f}\n  med={:.2f}\n  std={:.2f}".format(
        points["dx"].mean(), points["dx"].median(), points["dx"].std())
    textstr += '\n\n'
    textstr += "DY\n  mean={:.2f}\n  med={:.2f}\n  std={:.2f}".format(
        points["dy"].mean(), points["dy"].median(), points["dy"].std())

    ax.text(0.025, 0.075, textstr, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # empty figure
    ax = axes[1, 4]
    ax.set_axis_off()

    # plot image
    ax = axes[0, 0]
    ax.set_title(os.path.basename(img_file)[0:36])
    ax.imshow(img.array, cmap='gray')  # , vmin=0, vmax=800)

    # plot ref
    ax = axes[1, 0]
    ax.set_title(os.path.basename(ref_file)[0:36])
    ax.imshow(ref.array, cmap='gray')  # , vmin=0, vmax=800)

    # plt.suptitle('Inter-registration MSG4-MSG2 SEVIRI (2021-11-17 12:00/12:12)', fontsize=18)
    # plt.tight_layout()
    # plt.show()
    plt.savefig(outfile)
    print(outfile)
    plt.close()


def main_plot_histo(csv_file, img_file, ref_file,
                    output_directory, conf):
    img = GdalRasterImage(img_file)
    ref = GdalRasterImage(ref_file)
    print(img.xSize, img.ySize)
    points = pd.read_csv(csv_file, sep=";")

    # TODO: Parameters of histo analysis to be set :
    #       label of the study
    confidence_threshold = conf.accuracy_analysis["confidence_threshold"]
    n_sigma = conf.accuracy_analysis["n_sigma"]
    label = conf.accuracy_analysis["label"]

    title_label = 'X (line)  displacements - (KARIOS)'
    # Compute Histograms, Circular Error, radial error images :
    OUT = []
    #  Plot Histogram :
    dic = {'1': {'dir': 'x', 'label': ' X line displacements (KARIOS)',
                 'file_name': 'histx.png'},
           '2': {'dir': 'y', 'label': ' Y pixel displacements (KARIOS)',
                 'file_name': 'histy.png'}
           }

    #prepare st :
    st = accuracy_statistics.geometric_stat(label, points)

    #TODO : total pixel minus background
    # (compute number of bckg pixels and substract to img.xSize * img.ySize
    tt = img.xSize * img.ySize    #total pixel minus background
    st.compute_stats(confidence_threshold, n_sigma, tt)
    [ch1, chx, chy] = st.display_results()
    st.update_statistique_file([label], chx, chy,
                               os.path.join(output_directory, 'correl_res.txt'))

    #TODO : set dynamic value for px
    px = 3.125

    #Generate histogram figures and save to png :
    for rec in dic.items():

        out_image_file = os.path.join(output_directory, rec[1]['file_name'])
        plt_f.hist_vector(st, dir =rec[1]['dir'],
                                pixelResolution = px,
                                confidence = confidence_threshold,
                                title = rec[1]['label'],
                                out_image_file = out_image_file)
        OUT.append( out_image_file)

    #Generate scatter figure and save to png :
    #  Plot Scatter :
    out_image_file = os.path.join(output_directory, 'ce90.png')
    title_label = 'Circular Error Plot @ 90 percentile'
    plt_f.scatter_vector(st,
                            pixelResolution = px,
                            confidence = confidence_threshold,
                            title = title_label,
                            out_image_file = out_image_file)
    OUT.append(out_image_file)

    #Open the three images and create a canvas :
    qa_f= os.path.join(output_directory, label+'_C'+str(np.int(confidence_threshold*100))+'_qa.png')

    #Create an single image including all graphic images
    figure_title = label+'_C='+str(confidence_threshold)
    IN_IMAGES = [OUT[0], OUT[2], OUT[1]]

    cv.canvaz(IN_IMAGES,qa_f,
              fig_label= figure_title)



def print_stat(arr, name=""):
    """ print statistics of a numpy array """
    """print("Numpy Array Stats :", name)
    print("(type, shape, min, mean, max, std)")
    print("{} {} {:.2f} {:.2f} {:.2f} {:.2f}".format(arr.dtype, arr.shape, arr.min(), arr.mean(), arr.max(), arr.std()))
    print()"""
    print(name, "[type, shape, min, mean, max, std]: {} {} {:.2f} {:.2f} {:.2f} {:.2f}".format(arr.dtype, arr.shape,
                                                                                               arr.min(), arr.mean(),
                                                                                               arr.max(), arr.std()))
    print()


def writeTif(filename, data):
    # write tif with gdal
    driver = gdal.GetDriverByName("GTiff")
    if not os.path.exists(os.path.abspath(os.path.dirname(filename))):
        os.makedirs(os.path.dirname(os.path.abspath(filename)))
    creationOptions = ["COMPRESS=LZW"]
    print(data.dtype)
    if data.dtype == "uint8":
        etype = gdal.GDT_Byte
    elif data.dtype == "uint16":
        # gdal.GetDataTypeByName('UInt16')
        etype = gdal.GDT_UInt16
    elif data.dtype == "int16":
        etype = gdal.GDT_Int16
    elif data.dtype == "float64":
        etype = gdal.GDT_Float64
    print(etype)
    dataset = driver.Create(filename, xsize=data.shape[1], ysize=data.shape[0], bands=1, eType=etype,
                            options=creationOptions)
    dataset.GetRasterBand(1).WriteArray(data)
    dataset = None
    print(filename)


def orthorectify(product, band):
    # band
    obs = band[-2:]
    src_file = f"SLSTR_{product.date}_{band}.tif"
    ortho_file = src_file.replace(".tif", "_ortho.tif")

    if os.path.exists(ortho_file):
        return ortho_file

    if not os.path.exists(src_file):
        img = product.read(band)
        print_stat(img, src_file)
        writeTif(src_file, img)
    else:
        img = io.imread(src_file)
    (ysize, xsize) = img.shape
    del img

    # geo_coordinate
    lonfile = f"SLSTR_{product.date}_{obs}_lon.tif"
    latfile = f"SLSTR_{product.date}_{obs}_lat.tif"
    altfile = f"SLSTR_{product.date}_{obs}_alt.tif"
    if not os.path.exists(lonfile):
        grid_name = product.get_grid_name(band)
        print(grid_name)
        alt = product.read(grid_name, f"elevation_{obs}", autoscale=True)
        print_stat(alt, "alt")
        writeTif(altfile, alt)
        lat = product.read(grid_name, f"latitude_{obs}", autoscale=True)
        print_stat(lat, "lat")
        writeTif(latfile, lat)
        lon = product.read(grid_name, f"longitude_{obs}", autoscale=True)
        print_stat(lon, "lon")
        writeTif(lonfile, lon)

    # vrt creation
    tmpl = '''<VRTDataset RasterXSize="{xsize}" RasterYSize="{ysize}">
          <SRS>EPSG:4326</SRS>
          <VRTRasterBand band="1" dataType="Int16">
            <SimpleSource>
                <SourceFilename relativeToVRT="1">{src}</SourceFilename>
                <SourceBand>1</SourceBand>
                <SourceProperties RasterXSize="{xsize}" RasterYSize="{ysize}" DataType="Int16"/>
                <SrcRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}"/>
                <DstRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}"/>
            </SimpleSource>
          </VRTRasterBand>
          <metadata domain="GEOLOCATION">
            <mdi key="X_DATASET">{lon}</mdi>
            <mdi key="X_BAND">1</mdi>
            <mdi key="Y_DATASET">{lat}</mdi>
            <mdi key="Y_BAND">1</mdi>
            <mdi key="Z_DATASET">{alt}</mdi>
            <mdi key="Z_BAND">1</mdi>
            <mdi key="LINE_STEP">1</mdi>
            <mdi key="PIXEL_STEP">1</mdi>
            <mdi key="PIXEL_OFFSET">0</mdi>
            <mdi key="LINE_OFFSET">0</mdi>
          </metadata>
        </VRTDataset>'''.format(xsize=xsize, ysize=ysize, src=src_file,
                                alt=altfile, lat=latfile, lon=lonfile)
    vrt_file = f"SLSTR_{product.date}_{band}_vrt.tif"
    with open(vrt_file, 'w') as fid:
        fid.write(tmpl)

    # warp
    # gdalwarp -r bilinear -srcnodata 0 -dstnodata 0 Oa03_radiance.vrt Oa03_radiance_ortho.tif
    res = gdal.Warp(ortho_file, vrt_file, resampleAlg="bilinear", srcNodata=0, dstNodata=0,
                    creationOptions=["COMPRESS=LZW"])
    res = None
    return ortho_file


def reproject(ref_file, product, band_main):
    """
    REPROJECT PROGRAM
    """

    outfilename = os.path.splitext(os.path.basename(ref_file))[0] + f"_{band_main}geometry.tif"
    if os.path.exists(outfilename):
        return outfilename

    # lat/lon grids
    """alt = product.read(grid_name, "altitude", autoscale=True)
    print_stat(alt, "alt")"""
    grid_name = product.get_grid_name(band_main)
    obs = band_main[-2:]
    lat = product.read(grid_name, f"latitude_{obs}", autoscale=True)
    print_stat(lat, "lat")
    lon = product.read(grid_name, f"longitude_{obs}", autoscale=True)
    print_stat(lon, "lon")

    # load mosaic
    mosaic = GdalRasterImage(ref_file)
    band = 1
    print(mosaic.xMin, mosaic.yMax, mosaic.xRes, mosaic.yRes)
    print(mosaic.xSize, mosaic.ySize)

    # init output
    print(outfilename)
    warped_image = np.zeros((lat.shape[0], lat.shape[1]), np.int16)

    # prepare tiling
    STRIP_SIZE = 4080
    MERGIN = 8

    print('TARGET:', lon.min(), lat.min(), lon.max(), lat.max())

    xstart = floor((lon.min() - mosaic.xMin) / mosaic.xRes)
    ystart = floor((lat.max() - mosaic.yMax) / mosaic.yRes)
    target_xsize = ceil((lon.max() - lon.min()) / mosaic.xRes)
    target_ysize = ceil((lat.min() - lat.max()) / mosaic.yRes)
    print(xstart, ystart, target_xsize, target_ysize)

    # start tiling
    for yoff in range(ystart, ystart + target_ysize, STRIP_SIZE):
        for xoff in range(xstart, xstart + target_xsize, STRIP_SIZE):

            yoff = yoff - MERGIN
            ysize = STRIP_SIZE + 2 * MERGIN
            xoff = xoff - MERGIN
            xsize = STRIP_SIZE + 2 * MERGIN

            if yoff < 0:
                yoff = 0
            if yoff + ysize > mosaic.ySize:
                ysize = mosaic.ySize - yoff  # for last lines

            if xoff < 0:
                xoff = 0
            if xoff + xsize > mosaic.xSize:
                xsize = mosaic.xSize - xoff  # for last columns

            if ysize < 0 or xsize < 0:
                continue

            # compute extent of the tile
            tile_lon_min = mosaic.xMin + mosaic.xRes * xoff
            tile_lon_max = tile_lon_min + mosaic.xRes * xsize
            tile_lat_max = mosaic.yMax + mosaic.yRes * yoff
            tile_lat_min = tile_lat_max + mosaic.yRes * ysize
            # print(tile_lon_min, tile_lat_min, tile_lon_max, tile_lat_max)

            # start processing
            print("{}/{} {}/{}".format(xoff, mosaic.xSize, yoff, mosaic.ySize))
            print('TILE:', tile_lon_min, tile_lat_min, tile_lon_max, tile_lat_max)

            image_tiled = mosaic.read(band, xoff, yoff, xsize, ysize)

            # print_stat(image_tiled, "mosaic", nodata=0)

            if image_tiled is not None:
                # lon = mosaic.xRes * (col + xoff) + mosaic.xMin
                # lat = mosaic.yMax + mosaic.yRes * (row + yoff)
                # print()
                col = ((lon - mosaic.xMin) / mosaic.xRes) - xoff - 0.5
                row = ((lat - mosaic.yMax) / mosaic.yRes) - yoff - 0.5

                # print_stat(col, "col")
                # print_stat(row, "row")

                # using MSG lat/lon grids
                grid = np.ones((2, col.shape[0], col.shape[1]), dtype=np.float64)
                grid[0, :, :] = row
                grid[1, :, :] = col

                # print_stat(image_tiled, "warped_image", nodata=0)
                warped_image_tiled = warp(image_tiled, inverse_map=grid, order=3, mode='constant', cval=0.0, clip=False,
                                          preserve_range=True)
                # print_stat(warped_image_tiled, "warped_image", nodata=0)
                # print(warped_image_tiled.shape)

                warped_image_tiled = np.int16(warped_image_tiled)
                warped_image = np.where(warped_image == 0, warped_image_tiled, warped_image)

    # write image as tif with LZW compression
    writeTif(outfilename, warped_image.astype(np.int16))
    return outfilename


def match(mon, ref, conf, RESUME, mask=None):
    # prefix
    mon_name = os.path.splitext(os.path.basename(mon))[0]
    ref_name = os.path.splitext(os.path.basename(ref))[0]
    p = conf.values['output_directory']['path']
    csv_file = os.path.join(p,
                            f'KLT_matcher_{mon_name}_{ref_name}.csv')
    png_file = os.path.join(p,
                            csv_file.replace('.csv', '.png'))

    # run matcher:
    if not RESUME:
        main_KLT2(mon, ref, mask,
                  csv_file, conf,
                  outliers=True)

    # plot 1 - mean profiles:
    main_plot_mean_profile(csv_file, mon, ref,
                           png_file, conf)

    # plot 2 - mean histogram:
    #TODO: p output path or set working directory path?
    main_plot_histo(csv_file, mon, ref,
                    p, conf)
