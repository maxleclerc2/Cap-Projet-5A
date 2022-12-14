#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S. Saunier   (TELESPAZIO France) - November 20, 2022

import os,sys
import numpy as np

package_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(package_dir, 'util'))
import log as log

class geometric_stat :
    def __init__(self, label, points, confidence_threshold=0.9):
        self.valid = False
        self.label = label
        self.confidence = confidence_threshold
        self.n_value = ''         #threshold n_value * sigma
        self.total_pixel = ''     #Valid Pixel included no background
        self.sample_pixel = ''    #Sample of pixel used for statistics
        self.min_x = ''
        self.max_x = ''
        self.median_x = ''
        self.mean_x = ''
        self.std_x = ''
        self.min_y = ''
        self.max_y = ''
        self.median_y = ''
        self.mean_y = ''
        self.std_y = ''
        self.v_x = points["dx"] #vector of dx displacements
        self.v_y = points["dy"] #vector of dy displacements
        self.v_c = points["score"] #vector of dc displacements


    def compute_stats(self,confidence_threshold,n_value,tt):

        vx = self.v_x
        vy = self.v_y
        vc = self.v_c
        log.info (' -- Compute Final Statistics : '+'\n' )
        self.confidence = confidence_threshold
        self.n_value = n_value

        self.total_pixel = tt
        self.sample_pixel = np.size(vx)
        self.percentage_of_pixel =100*np.double(self.sample_pixel) / np.double(self.total_pixel)

        if (np.size(vx)) > 0:
            self.valid = True
            self.min_x = np.min(vx)
            self.max_x = np.max(vx)
            self.median_x = np.median(vx)
            self.mean_x = np.mean(vx)
            self.std_x = np.std(vx)

            self.min_y = np.min(vy)
            self.max_y = np.max(vy)
            self.median_y = np.median(vy)
            self.mean_y = np.mean(vy)
            self.std_y = np.std(vy)

            self.min_c = np.min(vc)
            self.max_c = np.max(vc)
            self.median_c = np.median(vc)
            self.mean_c = np.mean(vc)
            self.std_c = np.std(vc)

            self.valid = True
        else :
            log.warn ('No data in DC above confidence threshold ' )
            self.valid = False

    def display_results(self):
        st = self
        log.info ('-- DX / DY  statistics : ' )
        log.info(' Direction         : total_valid_pixel sample_pixel confidence_th   min    max    median    mean    std ' )
        ch1 = self.label
        chx = [str(st.total_pixel),str(st.sample_pixel),str(st.confidence),
                       str(st.min_x),str(st.max_x),
                       str(st.median_x),str(st.mean_x),str(st.std_x)]
        chy = [str(st.total_pixel),str(st.sample_pixel),str(st.confidence),
                       str(st.min_y),str(st.max_y),
                       str(st.median_y),str(st.mean_y),str(st.std_y)]

        log.info (ch1+': \n')
        log.info (' DX (line)        : '+' '.join(chx))
        log.info (' DY (px(column))  : '+' '.join(chy))
        return [ch1,chx,chy]

    def get_string_block(self,scale_factor,dir = 'x'):
        # Create a text block to be added in the figure of the plot
        if dir == 'x' :
            # Output string to be included into the text box
            ch0 = ' '.join([' Conf_Value :', '%.2f' % self.confidence])
            ch1 = ' '.join([' %Conf Px   :', '%.2f' % self.percentage_of_pixel, '%'])
            ch2 = ' '.join(['Minimum    : ', '%.2f' % (self.min_x * scale_factor), 'm'])
            ch3 = ' '.join(['Maximum   : ', '%.2f' % (self.max_x * scale_factor), 'm'])
            ch4 = ' '.join(['Mean          : ', '%.2f' % (self.mean_x * scale_factor), 'm'])
            ch5 = ' '.join(['Std Dev      : ', '%.2f' % (self.std_x * scale_factor), 'm'])
            ch6 = ' '.join(['Median       : ', '%.2f' % (self.median_x * scale_factor), 'm'])

        if dir == 'y' :
            # Output string to be included into the text box
            ch0 = ' '.join([' Conf_Value :', '%.2f' % self.confidence])
            ch1 = ' '.join([' %Conf Px   :', '%.2f' % self.percentage_of_pixel, '%'])
            ch2 = ' '.join(['Minimum    : ', '%.2f' % (self.min_y * scale_factor), 'm'])
            ch3 = ' '.join(['Maximum   : ', '%.2f' % (self.max_y * scale_factor), 'm'])
            ch4 = ' '.join(['Mean          : ', '%.2f' % (self.mean_y * scale_factor), 'm'])
            ch5 = ' '.join(['Std Dev      : ', '%.2f' % (self.std_y * scale_factor), 'm'])
            ch6 = ' '.join(['Median       : ', '%.2f' % (self.median_y * scale_factor), 'm'])

        ch = '\n '.join([ch0, ch1, ch2, ch3, ch4, ch5, ch6])

        return ch

    def update_statistique_file(self, ch, chx, chy, outFile=None):
        # default name
        if outFile is None:
            outFile = os.path.join(os.getcwd(),'correl_res.txt')

        # add titles if first line to write
        if not os.path.exists(outFile) :
            with open(outFile,'w' ) as txt_file:
                titles = ' '.join(['refImg', 'secImg', 'total_valid_pixel', 'sample_pixel', 'confidence_th',
                               'min_x', 'max_x', 'median_x', 'mean_x', 'std_x',
                               'min_y', 'max_y', 'median_y', 'mean_y', 'std_y'])
                txt_file.write(titles+'\n')

        # write line
        with open(outFile,'a' ) as txt_file:
            txt_file.write(' '.join(ch)+' '+' '.join(chx)+' '+' '.join(chy[3:])+'\n')
        log.info ('-- Update text file  : '+outFile)
