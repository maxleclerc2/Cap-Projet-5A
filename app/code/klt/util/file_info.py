"""
Utility classes for accessing information from files,
Taken from RIOS lib.

"""
import os

import numpy
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

# List of datatype names corresponding to GDAL datatype numbers.
# The index of this list corresponds to the gdal datatype number. Not sure if this
# is a bit obscure and cryptic.....
GDALdatatypeNames = ['Unknown', 'UnsignedByte', 'UnsignedInt16', 'SignedInt16',
    'UnsignedInt32', 'SignedInt32', 'Float32', 'Float64', 'ComplexInt16', 'ComplexInt32',
    'ComplexFloat32', 'ComplexFloat64']

class ImageInfo(object):
    """
    An object with the bounds and other info for the given image,
    in GDAL conventions.

    Object contains the following fields
        * **xMin**            Map X coord of left edge of left-most pixel
        * **xMax**            Map X coord of right edge of right-most pixel
        * **yMin**            Map Y coord of bottom edge of bottom pixel
        * **yMax**            Map Y coord of top edge of top-most pixel
        * **xRes**            Map coord size of each pixel, in X direction
        * **yRes**            Map coord size of each pixel, in Y direction
        * **nrows**           Number of rows in image
        * **ncols**           Number of columns in image
        * **transform**       Transformation params to map between pixel and map coords, in GDAL form
        * **projection**      WKT string of projection
        * **rasterCount**     Number of rasters in file
        * **lnames**          Names of the layers as a list.
        * **layerType**       "thematic" or "athematic", if it is set
        * **dataType**        Data type for the first band (as a GDAL integer constant)
        * **dataTypeName**    Data type for the first band (as a human-readable string)
        * **nodataval**       Value used as the no-data indicator (per band)

    The omitPerBand argument on the constructor is provided in order to speed up the
    access of very large VRT stacks. The information which is normally extracted
    from each band will, in that case, trigger a gdal.Open() for each band, which
    can be quite slow. So, if none of that information is actually required, then
    setting omitPerBand=True will omit that information, but will return as quickly
    as for a normal single file.

    """

    def __init__(self, filename, omitPerBand=False):

        ds = gdal.Open(str(filename), gdal.GA_ReadOnly)

        geotrans = ds.GetGeoTransform()
        (ncols, nrows) = (ds.RasterXSize, ds.RasterYSize)
        self.rasterCount = ds.RasterCount

        self.xMin = geotrans[0]
        self.xRes = geotrans[1]
        self.yMax = geotrans[3]
        self.yRes = abs(geotrans[5])
        self.xMax = self.xMin + ncols * self.xRes
        self.yMin = self.yMax - nrows * self.yRes
        self.ncols = ncols
        self.nrows = nrows

        # Projection, etc.
        self.transform = geotrans
        self.projection = ds.GetProjection()

        # Per-band stuff, including layer names and no data values, and stats
        self.lnames = []
        self.nodataval = []
        if not omitPerBand:
            for band in range(ds.RasterCount):
                bandObj = ds.GetRasterBand(band + 1)
                self.lnames.append(bandObj.GetDescription())
                self.nodataval.append(bandObj.GetNoDataValue())

        gdalMeta = ds.GetRasterBand(1).GetMetadata()
        if 'LAYER_TYPE' in gdalMeta:
            self.layerType = gdalMeta['LAYER_TYPE']
        else:
            self.layerType = None

        # Pixel datatype, stored as a GDAL enum value.
        self.dataType = ds.GetRasterBand(1).DataType
        self.dataTypeName = GDALdatatypeNames[self.dataType]

        del ds

    def __str__(self):
        """
        Print a readable version of the object
        """
        lines = []
        for attribute in ['nrows', 'ncols', 'rasterCount', 'xMin', 'xMax', 'yMin', 'yMax',
                'xRes', 'yRes', 'lnames', 'layerType', 'dataType', 'dataTypeName',
                'nodataval', 'transform', 'projection']:
            value = self.__dict__[attribute]
            lines.append("%-20s%s" % (attribute, value))
        result = '\n'.join(lines)
        return result

    def getCorners(self, outWKT=None, outEPSG=None, outPROJ4=None):
        """
        Return the coordinates of the image corners, possibly reprojected.

        This is the same information as in the xMin, xMax, yMin, yMax fields,
        but with the option to reproject them into a given output projection.
        Because the output coordinate system will not in general align with the
        image coordinate system, there are separate values for all four corners.
        These are returned as::

            (ul_x, ul_y, ur_x, ur_y, lr_x, lr_y, ll_x, ll_y)

        The output projection can be given as either a WKT string, an
        EPSG number, or a PROJ4 string. If none of those is given, then
        bounds are not reprojected, but will be in the same coordinate
        system as the image corners.

        """
        if outWKT is not None:
            outSR = osr.SpatialReference(wkt=outWKT)
        elif outEPSG is not None:
            outSR = osr.SpatialReference()
            outSR.ImportFromEPSG(int(outEPSG))
        elif outPROJ4 is not None:
            outSR = osr.SpatialReference()
            outSR.ImportFromProj4(outPROJ4)
        else:
            outSR = None

        if outSR is not None:
            inSR = osr.SpatialReference(wkt=self.projection)
            t = osr.CoordinateTransformation(inSR, outSR)
            (ul_x, ul_y, z) = t.TransformPoint(self.xMin, self.yMax)
            (ll_x, ll_y, z) = t.TransformPoint(self.xMin, self.yMin)
            (ur_x, ur_y, z) = t.TransformPoint(self.xMax, self.yMax)
            (lr_x, lr_y, z) = t.TransformPoint(self.xMax, self.yMin)
        else:
            (ul_x, ul_y) = (self.xMin, self.yMax)
            (ll_x, ll_y) = (self.xMin, self.yMin)
            (ur_x, ur_y) = (self.xMax, self.yMax)
            (lr_x, lr_y) = (self.xMax, self.yMin)

        return (ul_x, ul_y, ur_x, ur_y, lr_x, lr_y, ll_x, ll_y)


class ImageLayerStats(object):
    """
    Hold the stats for a single image layer. These are as retrieved
    from the given image file, and are not calculated again. If they
    are not present in the file, they will be None.
    Typically this class is not used separately, but only instantiated
    as a part of the ImageFileStats class.

    The object contains the following fields
        * **mean**        Mean value over all non-null pixels
        * **min**         Minimum value over all non-null pixels
        * **max**         Maximum value over all non-null pixels
        * **stddev**      Standard deviation over all non-null pixels
        * **median**      Median value over all non-null pixels
        * **mode**        Mode over all non-null pixels
        * **skipfactorx** The statistics skip factor in the X direction
        * **skipfactory** The statistics skip factor in the Y direction

    There are many ways to report a histogram.
    The following attributes report it the way GDAL does.
    See GDAL doco for precise details.

        * **histoCounts**     Histogram counts (numpy array)
        * **histoMin**        Minimum edge of smallest bin
        * **histoMax**        Maximum edge of largest bin
        * **histoNumBins**    Number of histogram bins

    """

    def __init__(self, bandObj):
        metadata = bandObj.GetMetadata()
        self.mean = self.__getMetadataItem(metadata, 'STATISTICS_MEAN')
        self.stddev = self.__getMetadataItem(metadata, 'STATISTICS_STDDEV')
        self.max = self.__getMetadataItem(metadata, 'STATISTICS_MAXIMUM')
        self.min = self.__getMetadataItem(metadata, 'STATISTICS_MINIMUM')
        self.median = self.__getMetadataItem(metadata, 'STATISTICS_MEDIAN')
        self.mode = self.__getMetadataItem(metadata, 'STATISTICS_MODE')
        self.skipfactorx = self.__getMetadataItem(metadata, 'STATISTICS_SKIPFACTORX')
        self.skipfactory = self.__getMetadataItem(metadata, 'STATISTICS_SKIPFACTORY')

        self.histoMin = self.__getMetadataItem(metadata, 'STATISTICS_HISTOMIN')
        self.histoMax = self.__getMetadataItem(metadata, 'STATISTICS_HISTOMAX')
        self.histoNumBins = self.__getMetadataItem(metadata, 'STATISTICS_HISTONUMBINS')

        self.histoCounts = None
        # check the histo info - from RAT if available
        # and we are using RFC40
        rat = bandObj.GetDefaultRAT()
        if rat is not None and hasattr(rat, "ReadAsArray"):
            for col in range(rat.GetColumnCount()):
                if rat.GetUsageOfCol(col) == gdal.GFU_PixelCount:
                    self.histoCounts = rat.ReadAsArray(col)
                    break

        if self.histoCounts is None and 'STATISTICS_HISTOBINVALUES' in metadata:
            histoString = metadata['STATISTICS_HISTOBINVALUES']
            histoStringList = [c for c in histoString.split('|') if len(c) > 0]
            counts = [eval(c) for c in histoStringList]
            self.histoCounts = numpy.array(counts)

    @staticmethod
    def __getMetadataItem(metadata, key):
        "Return eval(item) by key, or None if not present"
        item = None
        if key in metadata:
            item = eval(metadata[key])
        return item

    def __str__(self):
        "Readable string representation of stats"
        fmt = "Mean: %s, Stddev: %s, Min: %s, Max: %s, Median: %s, Mode: %s"
        return (fmt % (self.mean, self.stddev, self.min, self.max, self.median, self.mode))


class ImageFileStats(object):
    """
    Hold the stats for all layers in an image file. This object can be indexed
    with the layer index, and each element is an instance of :class:`rios.fileinfo.ImageLayerStats`.

    """

    def __init__(self, filename):
        ds = gdal.Open(filename)
        self.statsList = []
        for i in range(ds.RasterCount):
            bandObj = ds.GetRasterBand(i + 1)
            self.statsList.append(ImageLayerStats(bandObj))
        del ds

    def __getitem__(self, i):
        return self.statsList[i]

    def __len__(self):
        return len(self.statsList)

    def __str__(self):
        return '\n'.join([str(s) for s in self.statsList])


class VectorFileInfo(object):
    """
    Hold useful general information about a vector file. This object
    can be indexed with the layer index, and each element is
    an instance of :class:`rios.fileinfo.VectorLayerInfo`.

    """

    def __init__(self, filename):
        ds = ogr.Open(filename)
        if ds is None:
            raise rioserrors.VectorLayerError("Unable to open vector dataset '%s'" % filename)
        layerCount = ds.GetLayerCount()
        self.layerInfo = [VectorLayerInfo(ds, i) for i in range(layerCount)]

    def __getitem__(self, i):
        return self.layerInfo[i]

    def __str__(self):
        return '\n'.join(['Layer:%s\n%s' % (i, str(self.layerInfo[i]))
                          for i in range(len(self.layerInfo))])


geometryTypeStringDict = {
    1: 'Point',
    2: 'Line',
    3: 'Polygon'
}

class VectorLayerInfo(object):
    """
    Hold useful general information about a single vector layer.

    Object contains the following fields
        * **featureCount**        Number of features in the layer
        * **xMin**                Minimum X coordinate
        * **xMax**                Maximum X coordinate
        * **yMin**                Minimum Y coordinate
        * **yMax**                Maximum Y coordinate
        * **geomType**            OGR geometry type code (integer)
        * **geomTypeStr**         Human-readable geometry type name (string)
        * **fieldCount**          Number of fields (i.e. columns) in attribute table
        * **fieldNames**          List of names of attribute table fields
        * **fieldTypes**          List of the type code numbers of each attribute table field
        * **fieldTypeNames**      List of the string names of the field types
        * **spatialRef**          osr.SpatialReference object of layer projection

    """

    def __init__(self, ds, i):
        lyr = ds.GetLayer(i)
        if lyr is None:
            raise rioserrors.VectorLayerError("Unable to open layer %s in dataset '%s'" % (i, ds.GetName()))

        self.featureCount = lyr.GetFeatureCount()
        extent = lyr.GetExtent()
        self.xMin = extent[0]
        self.xMax = extent[1]
        self.yMin = extent[2]
        self.yMax = extent[3]

        self.geomType = lyr.GetGeomType()
        if self.geomType in geometryTypeStringDict:
            self.geomTypeStr = geometryTypeStringDict[self.geomType]

        lyrDefn = lyr.GetLayerDefn()
        self.fieldCount = lyrDefn.GetFieldCount()
        fieldDefnList = [lyrDefn.GetFieldDefn(i) for i in range(self.fieldCount)]
        self.fieldNames = [fd.GetName() for fd in fieldDefnList]
        self.fieldTypes = [fd.GetType() for fd in fieldDefnList]
        self.fieldTypeNames = [fd.GetTypeName() for fd in fieldDefnList]

        self.spatialRef = lyr.GetSpatialRef()

    def __str__(self):
        valueList = []
        for valName in ['featureCount', 'xMin', 'xMax', 'yMin', 'yMax', 'geomType',
                        'geomTypeStr', 'fieldCount', 'fieldNames', 'fieldTypes', 'fieldTypeNames',
                        'spatialRef']:
            valueList.append("  %s: %s" % (valName, getattr(self, valName)))
        return '\n'.join(valueList)
