#! /usr/bin/env python

import sys
from os.path import basename
from osgeo import gdal
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


def get_string_block(array,
                     confidence,
                     scale_factor,
                     percentage_of_pixel) :
    #Output string to be included into the text box
    ch0 = ' '.join([' Conf_Value :', '%.2f' % confidence])
    ch1 = ' '.join([' %Conf Px   :', '%.2f' % percentage_of_pixel, '%'])
    ch2 = ' '.join(['Minimum    : ', '%.2f' % (np.min(array) * scale_factor), 'm'])
    ch3 = ' '.join(['Maximum   : ', '%.2f' % (np.max(array) * scale_factor), 'm'])
    ch4 = ' '.join(['Mean          : ', '%.2f' % (np.mean(array) * scale_factor), 'm'])
    ch5 = ' '.join(['Std Dev      : ', '%.2f' % (np.std(array) * scale_factor), 'm'])
    ch6 = ' '.join(['Median       : ', '%.2f' % (np.median(array) * scale_factor), 'm'])

    ch = '\n '.join([ch0,ch1, ch2, ch3, ch4, ch5, ch6])

    return ch

def pre_process_array(filename,maskfilename,confidence=None) :
    '''

    :param filename:       Input image (dx, dy)
    :param maskfilename:   Confidence image (dc)
    :param confidence:     Threshold to be applied to input image
    :return:               Condidence Thresholded input image , Condidence Thresholded input image with 3-sig filter.
    '''

    if confidence is None :
        th = 0.9
    else :
        th = confidence

    array = readTif(filename)
    array = array.flatten()
    if maskfilename is None:

        array = array[array != 0.]
    else:
        #Mask the input array with maskfilename value above th
        mask_array = (readTif(maskfilename)).flatten()
        array = array[mask_array > th]

    sig_mask = (np.abs(array) < 3 * np.std(array))
    array_f = array[sig_mask]

    return array,array_f


def readTif(filename):
    dst = gdal.Open(filename)
    array = dst.GetRasterBand(1).ReadAsArray()
    dst = None
    return array


def scatter(file1,file2):
    array1 = readTif(file1)
    array2 = readTif(file2)
    plt.figure(figsize=(10,10))
    plt.plot(array1.flatten(), array2.flatten(), ',')
    plt.plot([0,1], [0,1], 'k')
    plt.xlabel(basename(file1))
    plt.ylabel(basename(file2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    #plt.show()
    plt.savefig('scatter.png')

def hist_vector(st, dir = 'x',
                    pixelResolution = None,
                    confidence = None,
                    title = None,
                    out_image_file = None ):
    '''
    :param vect : vector np array
    :param pixelResolution:
    :param confidence: confidence value
    :param title:
    :param out_image_file:

    :return:
    '''
    import matplotlib as mpl
    from cycler import cycler

    # rcParams / Prop_cycle Not work with matplotlib 1.2.0 :
    # https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/issues/395
    #mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

    #Manage Options :
    if pixelResolution is None :
        scale_factor = 1.0
        x_label = " displacement (pixel) "
    else :
        scale_factor = pixelResolution
        x_label = " displacement (meter) "

    if confidence is None :
        confidence_factor = 0.9
    else :
        confidence_factor = confidence

    if title is None :
        title_figure = ' '
    else :
        title_figure = title


    SHOW = False
    if out_image_file is None :
        SHOW = True
    else :
        out_image_file = out_image_file

    # Get text block to be added in the figure
    ch = st.get_string_block(scale_factor, dir=dir)

    if dir == 'x' :
        vect = st.v_x

    if dir == 'y' :
        vect = st.v_y

    #The number of Bins corresponding to 0.1 pixel :
    pas = 0.1
    v = np.max([
            np.abs(np.max(vect)),
            np.abs(np.min(vect))
        ])
    print( 'value',str(2*v))
    number_of_bin = np.int(( (2*v+1) / pas))
    print( ' pixel size                      : ',scale_factor,' m')
    print( ' Bin range (m)                   : [-',v*scale_factor,' , ',v*scale_factor,']')
    print( ' Optimal bin width computation, quare root (of data size) estimator method (sqrt)  ')

    #sqrt     Square root (of data size) estimator, used by Excel
    # and other programs for its speed and simplicity.

    # bins , return the bins edge, (length(hist)+1)
    (hist, bins) = np.histogram(vect*scale_factor, bins='sqrt',
                                                   range=(-v*scale_factor,
                                                           v*scale_factor))
    #starting from bin edge compute the center of each bin
    center = (bins[:-1] + bins[1:]) / 2
    npixtotal = np.sum(hist)
    # http://www.python-simple.com/python-matplotlib/barplot.php
    h = plt.bar(center, hist,
                    width = pas/10.0,
                    color='yellow',
                    edgecolor='blue',
                    linewidth=0.5,
                    )

    #Gaussian curve
    mu = np.mean(vect*scale_factor)
    sigma = np.std(vect*scale_factor)
    n = (1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)))
    plt.plot(bins, (n/(np.sum(n))*npixtotal),
             linewidth=2, color='r')

    # Normal Test :
    k2, p = stats.normaltest(hist)
    alpha = 1e-3
    if p < alpha:  # null hypothesis: hist comes from a normal distribution
        print( "The null hypothesis can be rejected, p value : ",p)
        ch_7 = 'Normal Test : rejected'
    else:
        print( "The null hypothesis cannot be rejected, p value : ",p)
        ch_7 = 'Normal Test : not rejected'
    ch_8 = 'Total Pixels: '+str(npixtotal)
    ch_9 = 'Nbr of bins : '+str(len(hist))

    ch = '\n '.join([ch, ch_8,ch_9,ch_7 ])

    plt.ylabel(" Count ")
    plt.xlabel(x_label)

    y = np.max(hist)

    l = v*scale_factor
    plt.xlim(-l,l)

    plt.text(-v*scale_factor, y-0.5*y, ch, fontsize=10)

    plt.grid()
    plt.title(title_figure, fontsize=11)

    if SHOW :
        plt.show()
    else :
        plt.savefig(out_image_file)

    # Clean all :
        array = None
        array_f = None
        hist = None
        bins = None

    # If not close - add content in the next call of plt !!!
    plt.close()




def hist(filenames,maskfilename = None,
                    pixelResolution = None,
                    confidence = None,
                    title = None,
                    out_image_file = None,
                    sigma_threshold = None,
                    FROM_MEDICIS_TO_CARTO = True):
    '''
    :param filenames:
    :param maskfilename: Typically DC Mask
    :param pixelResolution:
    :param confidence:
    :param title:
    :param out_image_file:
    :param sigma_threshold:

    :return:
    '''
    import matplotlib as mpl
    from cycler import cycler

    # rcParams / Prop_cycle Not work with matplotlib 1.2.0 :
    # https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/issues/395
    #mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

    #Manage Options :
    if pixelResolution is None :
        scale_factor = 1.0
        x_label = " displacement (pixel) "
    else :
        scale_factor = pixelResolution
        x_label = " displacement (meter) "

    if confidence is None :
        confidence_factor = 0.9
    else :
        confidence_factor = confidence

    if title is None :
        title_figure = ' '
    else :
        title_figure = title


    SHOW = False
    if out_image_file is None :
        SHOW = True
    else :
        out_image_file = out_image_file


    #Loop on filename and clean array :
    for filename in filenames:

        array = readTif(filename)
        array = array.flatten()
        N1 = array.shape

        if maskfilename is None :
            array = array[array != 0.]
        else :
            mask_array = (readTif(maskfilename)).flatten()
            array = array[mask_array > np.float(confidence)]

        if sigma_threshold is None:
            array_f = array
        else:
            sig_mask = (np.abs(array) < 3 * np.std(array))
            array_f = array[sig_mask]

        N = array_f.shape
        percentage_of_pixel = (100 * np.double(N[0])) / np.double(N1[0])
        ch = get_string_block(array_f,
                              confidence,
                              scale_factor,
                              percentage_of_pixel)
        #Define the number of Bins corresponding to 0.1 pixel :
        pas = 0.01
        v = np.max([
            np.abs(np.max(array_f)),
            np.abs(np.min(array_f))
        ])
        number_of_bin = np.int(( v+1 / pas))
        print( ' number_of_bin : ',number_of_bin)
        print( ' min / max value of the bin (m) : ',v*scale_factor)
        (hist, bins) = np.histogram(array_f*scale_factor, bins=number_of_bin,
                                    range=(-v*scale_factor,
                                                              v*scale_factor))
        center = (bins[:-1] + bins[1:]) / 2
        npixtotal = np.sum(hist)
        # http://www.python-simple.com/python-matplotlib/barplot.php
        h = plt.bar(center, hist*100./npixtotal,
                    width = 0.1,
                    color='yellow',
                    edgecolor='blue',
                    linewidth=0.5,
                    label=basename(filename))

        plt.ylabel("% of pixels")
        plt.xlabel(x_label)

        y = np.max(hist)*100./npixtotal

        l = v*scale_factor
        plt.xlim(-l,l)
#        np.max(array)*scale_factor)

        plt.text(-v*scale_factor, y-0.5*y, ch, fontsize=10)

        plt.grid()
        plt.title(title_figure, fontsize=11)

    if SHOW :
        plt.show()
    else :
        plt.savefig(out_image_file)

    # Clean all :
        array = None
        array_f = None
        hist = None
        bins = None

    # If not close - add content in the next call of plt !!!
    plt.close()


def scatter_vector(st,
                 pixelResolution = None,
                 confidence = None,
                 percentile = None,
                 title= None,
                 out_image_file = None,
                FROM_MEDICIS_TO_CARTO = True
                 ):

        #Manage Options :
    if pixelResolution is None :
            scale_factor = 1.0
            x_label = " row displacement (pixel) "
            y_label = " line displacement (pixel) "
    else :
            scale_factor = pixelResolution
            x_label = " easting displacement (meter) "
            y_label = " northing displacement (meter) "

    if confidence is None :
        confidence_factor = 0.9
    else :
        confidence_factor = confidence

    if title is None :
        title_figure = ' '
    else :
        title_figure = title


    SHOW = False
    if out_image_file is None :
        SHOW = True
    else :
        out_image_file = out_image_file


    x = st.v_x * scale_factor
    y = st.v_y * scale_factor

    #Computation CE90 2D :
    v_s = np.sort(np.sqrt(x*x + y*y))
    perc = np.int(0.9 * v_s.shape[0])
    index_n1 = np.int(0.9 * v_s.shape[0]) - 1
    index_n2 = index_n1 + 1

    perc_frac = (0.9 * v_s.shape[0]) - perc
    ce_90 = v_s[index_n1] + (v_s[index_n2] - v_s[index_n1])*perc_frac

    # Plot all values in the graphic both flatten arrays :
    # Determine axis values :
    v1 = (np.max(np.abs(x)))
    v2 = (np.max(np.abs(y)))

    l = np.max([v1,v2])

    fig, ax = plt.subplots()

    bins = 10
    data, x_e, y_e = np.histogram2d(x, y, bins=bins)

    from scipy.interpolate import interpn
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
                data, np.vstack([x, y]).T, method="splinef2d",
                bounds_error=False)

    # Sort the points by density, so that the densest points are plotted last
    sort =True
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    if FROM_MEDICIS_TO_CARTO :
        print( 'invert axis')
        #DX:ARR1 (line)  Northing<=> -ARR1
        #DY:ARR2 (px)    Easting <=> ARR2
        sc = ax.scatter(y, -x,c = z)
    else :
        sc = ax.scatter(x, y, c = z)

# Plot in the graphic Cicrular error circle :
    u = range(0, 110, 1)
    theta = (np.array(u)/100.0)*2*np.pi
    x_ce= ce_90 * np.cos(theta)
    y_ce= ce_90 * np.sin(theta)


    ax.plot(x_ce,y_ce, '-')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim([-l,l])
    ax.set_ylim([-l,l])

    ax.grid()

    if FROM_MEDICIS_TO_CARTO :
        #DX:ARR1 (line)  Northing<=> -ARR1
        #DY:ARR2 (px)    Easting <=> ARR2
        # Add text :
        # m_x : mean easting :
        m_x = np.mean(y)
        # m_y : mean northing
        m_y = np.mean(-x)

    else :
        m_x = np.mean(x)
        # m_y : mean northing
        m_y = np.mean(y)

    #
    plt.colorbar(sc)

    print(' Statistics :  Work - ref ')
    ch1 = ' '.join([' CE @90        :', '%.2f' % ce_90,' m'])
    ch2 = ' '.join([' Confidence V  :', '%.2f' % confidence])
    ch3 = ' '.join([' Total Points  :', '%d' % x.shape[0]])
    ch4 = ' '.join([' Mean Easting  :', '%.2f' % m_x,' m'])
    ch5 = ' '.join([' Mean Northing  :', '%.2f' % m_y,' m'])

    u = 2 * l / 20

    ax.text(-l, l - 1*u, ch1, fontsize=10)
    ax.text(-l, l - 2*u, ch2, fontsize=10)
    ax.text(-l, l - 3*u, ch3, fontsize=10)
    ax.text(-l, l - 4*u, ch4, fontsize=10)
    ax.text(-l, l - 5*u, ch5, fontsize=10)

    ax.set_title(title_figure, fontsize=11)

    if SHOW :
        fig.show()
    else :
        fig.savefig(out_image_file)
        print( ' Save in ',out_image_file)


    # If not close - add content in the next call of plt !!!
    plt.close()



'''
  Function , hist_scatter
  options )
        FROM_MEDICIS_TO_CARTO, if set to True, swap axis for 
                    Northing / Easting representation
                    (it is assumed that MEDICS in input are
                    # DX Line Displacements
                    # DY Pixel Displacements 
                   For CE representation, Easting / Northing
                   DX => Northing and DY => Easting
        ORBIT_MODE, adapt CE Axis Label, if AT/AC Geometry


#file1 => dx, file2 => dy
'''
def hist_scatter(file1,file2,
                 mask_filename=None,
                 pixelResolution = None,
                 confidence = None,
                 percentile = None,
                 title= None,
                 out_image_file = None,
                 sigma_threshold = None,
                 FROM_MEDICIS_TO_CARTO = False,
                 ORBIT_MODE = False
                 ):

    if ORBIT_MODE is True :
        x_label = " Across Track Error (m) "
        y_label = " Along Track Error (m) "
    else :
        #Manage Options :
        if pixelResolution is None :
            scale_factor = 1.0
            x_label = " row displacement (pixel) "
            y_label = " line displacement (pixel) "
        else :
            scale_factor = pixelResolution
            x_label = " easting displacement (meter) "
            y_label = " northing displacement (meter) "

    if confidence is None :
        confidence_factor = 0.9
    else :
        confidence_factor = confidence

    if title is None :
        title_figure = ' '
    else :
        title_figure = title


    SHOW = False
    if out_image_file is None :
        SHOW = True
    else :
        out_image_file = out_image_file



    array1,array1_f = pre_process_array(
                    file1,mask_filename,confidence = confidence_factor)
    array2,array2_f = pre_process_array(
                    file2,mask_filename,confidence = confidence_factor)

    scale_factor = pixelResolution
    arr1 = np.multiply(array1, scale_factor)
    arr2 = np.multiply(array2,scale_factor)
    arr1_cp = np.copy(arr1)
    arr2_cp = np.copy(arr2)
    # Compute the 2D CE90 for only 3 sigma values
    # considered filtered data :
    v = (np.sqrt(arr1*arr1 + arr2*arr2))
    mask = (v < np.mean(v) + 3*np.std(v)) * (v > np.mean(v) - 3*np.std(v))
    arr1_ce = arr1_cp[mask]
    arr2_ce = arr2_cp[mask]

    print( ' Compute Circular error on filtered 3 sigma data')
    print( ' 3 Sigma Threshold Number of points Before / After : ')
    print(  '    ',arr2.shape[0],' / ',arr2_ce.shape[0])

    #Computation CE90 2D :
    v_s = np.sort(np.sqrt(arr1_ce*arr1_ce + arr2_ce*arr2_ce))
    perc = np.int(0.9 * v_s.shape[0])
    index_n1 = np.int(0.9 * v_s.shape[0]) - 1
    index_n2 = index_n1 + 1

    perc_frac = (0.9 * v_s.shape[0]) - perc
    ce_90 = v_s[index_n1] + (v_s[index_n2] - v_s[index_n1])*perc_frac

    # Plot all values in the graphic both flatten arrays :
    # Determine axis values :
    v1 = (np.max(np.abs(arr1)))
    v2 = (np.max(np.abs(arr2)))

    l = np.max([v1,v2])

    fig, ax = plt.subplots()

    z = np.ones(arr2.shape)
    bins = 10
    data, x_e, y_e = np.histogram2d(arr1, arr2, bins=bins)

    from scipy.interpolate import interpn
    x = arr1
    y = arr2
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
                data, np.vstack([x, y]).T, method="splinef2d",
                bounds_error=False)

    # Sort the points by density, so that the densest points are plotted last
    sort =True
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    if FROM_MEDICIS_TO_CARTO :
        print( 'invert axis')
        #DX:ARR1 (line)  Northing<=> -ARR1
        #DY:ARR2 (px)    Easting <=> ARR2
        sc = ax.scatter(arr2, -arr1,c = z)
    else :
        sc = ax.scatter(arr1, arr2, c = z)

# Plot in the graphic Cicrular error circle :
    u = range(0, 110, 1)
    theta = (np.array(u)/100.0)*2*np.pi
    x= ce_90 * np.cos(theta)
    y= ce_90 * np.sin(theta)


    ax.plot(x,y, '-')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim([-l,l])
    ax.set_ylim([-l,l])

    ax.grid()

    if FROM_MEDICIS_TO_CARTO :
        #DX:ARR1 (line)  Northing<=> -ARR1
        #DY:ARR2 (px)    Easting <=> ARR2
        # Add text :
        # m_x : mean easting :
        m_x = np.mean(arr2)
        # m_y : mean northing
        m_y = np.mean(-arr1)

    else :
        m_x = np.mean(arr1)
        # m_y : mean northing
        m_y = np.mean(arr2)

    #
    plt.colorbar(sc)

    print( ' Statistics :  Work - ref ')
    ch1 = ' '.join([' CE @90        :', '%.2f' % ce_90,' m'])
    ch2 = ' '.join([' Confidence V  :', '%.2f' % confidence])
    ch3 = ' '.join([' Total Points  :', '%d' % arr2.shape[0]])
    ch4 = ' '.join([' Mean Easting  :', '%.2f' % m_x,' m'])
    ch5 = ' '.join([' Mean Northing  :', '%.2f' % m_y,' m'])

    u = 2 * l / 20

    ax.text(-l, l - 1*u, ch1, fontsize=10)
    ax.text(-l, l - 2*u, ch2, fontsize=10)
    ax.text(-l, l - 3*u, ch3, fontsize=10)
    ax.text(-l, l - 4*u, ch4, fontsize=10)
    ax.text(-l, l - 5*u, ch5, fontsize=10)

    ax.set_title(title_figure, fontsize=11)

    if SHOW :
        fig.show()
    else :
        fig.savefig(out_image_file)
        print( ' Save in ',out_image_file)


    # If not close - add content in the next call of plt !!!
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        # Set Input :
        file1=sys.argv[1]
        file2=sys.argv[2]
        # Execture command :
        scatter(file1,file2)
        hist(sys.argv[1:])
        hist_scatter(file1,file2)
