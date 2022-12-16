# This file is part of the Astrometry.net suite.
# Copyright 2009 Dustin Lang
# Licensed under a 3-clause BSD style license - see LICENSE
# https://github.com/dstndstn/astrometry.net

from __future__ import print_function
import os
import urllib.request

import time
import base64

from urllib.parse import urlencode, quote
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from flask import current_app as app

import json


# TODO API Key in configs
API_KEY = app.config['ASTROMETRY_API_KEY']


def json2python(data):
    try:
        return json.loads(data)
    except:
        pass
    return None


python2json = json.dumps


class MalformedResponse(Exception):
    pass


class RequestError(Exception):
    pass


class Client(object):
    default_url = 'https://nova.astrometry.net/api/'

    def __init__(self,
                 apiurl=default_url):
        self.session = None
        self.apiurl = apiurl

    def get_url(self, service):
        return self.apiurl + service

    def send_request(self, service, args={}, file_args=None):
        '''
        service: string
        args: dict
        '''
        if self.session is not None:
            args.update({'session': self.session})
        #print('Python:', args)
        json = python2json(args)
        #print('Sending json:', json)
        url = self.get_url(service)
        #print('Sending to URL:', url)

        # If we're sending a file, format a multipart/form-data
        if file_args is not None:
            import random
            boundary_key = ''.join([random.choice('0123456789') for i in range(19)])
            boundary = '===============%s==' % boundary_key
            headers = {'Content-Type':
                           'multipart/form-data; boundary="%s"' % boundary}
            data_pre = (
                    '--' + boundary + '\n' +
                    'Content-Type: text/plain\r\n' +
                    'MIME-Version: 1.0\r\n' +
                    'Content-disposition: form-data; name="request-json"\r\n' +
                    '\r\n' +
                    json + '\n' +
                    '--' + boundary + '\n' +
                    'Content-Type: application/octet-stream\r\n' +
                    'MIME-Version: 1.0\r\n' +
                    'Content-disposition: form-data; name="file"; filename="%s"' % file_args[0] +
                    '\r\n' + '\r\n')
            data_post = (
                    '\n' + '--' + boundary + '--\n')
            data = data_pre.encode() + file_args[1] + data_post.encode()

        else:
            # Else send x-www-form-encoded
            data = {'request-json': json}
            #print('Sending form data:', data)
            data = urlencode(data)
            data = data.encode('utf-8')
            #print('Sending data:', data)
            headers = {}

        request = Request(url=url, headers=headers, data=data)

        try:
            f = urlopen(request)
            #print('Got reply HTTP status code:', f.status)
            txt = f.read()
            #print('Got json:', txt)
            result = json2python(txt)
            #print('Got result:', result)
            stat = result.get('status')
            #print('Got status:', stat)
            if stat == 'error':
                errstr = result.get('errormessage', '(none)')
                raise RequestError('server error message: ' + errstr)
            return result
        except HTTPError as e:
            #print('HTTPError', e)
            return e.read()

    def login(self, apikey):
        args = {'apikey': apikey}
        result = self.send_request('login', args)
        sess = result.get('session')
        #print('Got session:', sess)
        if not sess:
            raise RequestError('no session in result')
        self.session = sess

    def _get_upload_args(self, **kwargs):
        args = {}
        for key, default, typ in [('allow_commercial_use', 'n', str),
                                  ('allow_modifications', 'n', str),
                                  ('publicly_visible', 'n', str),
                                  ('scale_units', None, str),
                                  ('scale_type', None, str),
                                  ('scale_lower', None, float),
                                  ('scale_upper', None, float),
                                  ('scale_est', None, float),
                                  ('scale_err', None, float),
                                  ('center_ra', None, float),
                                  ('center_dec', None, float),
                                  ('parity', None, int),
                                  ('radius', None, float),
                                  ('downsample_factor', None, int),
                                  ('positional_error', None, float),
                                  ('tweak_order', None, int),
                                  ('crpix_center', None, bool),
                                  ('invert', None, bool),
                                  ('image_width', None, int),
                                  ('image_height', None, int),
                                  ('x', None, list),
                                  ('y', None, list),
                                  ('album', None, str),
                                  ]:
            if key in kwargs:
                val = kwargs.pop(key)
                val = typ(val)
                args.update({key: val})
            elif default is not None:
                args.update({key: default})
        # print('Upload args:', args)
        return args

    def url_upload(self, url, **kwargs):
        args = dict(url=url)
        args.update(self._get_upload_args(**kwargs))
        result = self.send_request('url_upload', args)
        return result

    def upload(self, fn=None, **kwargs):
        args = self._get_upload_args(**kwargs)
        file_args = None
        if fn is not None:
            try:
                f = open(fn, 'rb')
                file_args = (fn, f.read())
            except IOError:
                #print('File %s does not exist' % fn)
                raise
        return self.send_request('upload', args, file_args)

    def submission_images(self, subid):
        result = self.send_request('submission_images', {'subid': subid})
        return result.get('image_ids')

    def overlay_plot(self, service, outfn, wcsfn, wcsext=0):
        from astrometry.util import util as anutil
        wcs = anutil.Tan(wcsfn, wcsext)
        params = dict(crval1=wcs.crval[0], crval2=wcs.crval[1],
                      crpix1=wcs.crpix[0], crpix2=wcs.crpix[1],
                      cd11=wcs.cd[0], cd12=wcs.cd[1],
                      cd21=wcs.cd[2], cd22=wcs.cd[3],
                      imagew=wcs.imagew, imageh=wcs.imageh)
        result = self.send_request(service, {'wcs': params})
        print('Result status:', result['status'])
        plotdata = result['plot']
        plotdata = base64.b64decode(plotdata)
        open(outfn, 'wb').write(plotdata)
        print('Wrote', outfn)

    def sdss_plot(self, outfn, wcsfn, wcsext=0):
        return self.overlay_plot('sdss_image_for_wcs', outfn,
                                 wcsfn, wcsext)

    def galex_plot(self, outfn, wcsfn, wcsext=0):
        return self.overlay_plot('galex_image_for_wcs', outfn,
                                 wcsfn, wcsext)

    def myjobs(self):
        result = self.send_request('myjobs/')
        return result['jobs']

    def job_status(self, job_id, justdict=False):
        result = self.send_request('jobs/%s' % job_id)
        if justdict:
            return result
        stat = result.get('status')
        if stat == 'success':
            result = self.send_request('jobs/%s/calibration' % job_id)
            print('Calibration:', result)
            result = self.send_request('jobs/%s/tags' % job_id)
            print('Tags:', result)
            result = self.send_request('jobs/%s/machine_tags' % job_id)
            print('Machine Tags:', result)
            result = self.send_request('jobs/%s/objects_in_field' % job_id)
            print('Objects in field:', result)
            result = self.send_request('jobs/%s/annotations' % job_id)
            print('Annotations:', result)
            result = self.send_request('jobs/%s/info' % job_id)
            print('Calibration:', result)

        return stat

    def annotate_data(self, job_id):
        """
        :param job_id: id of job
        :return: return data for annotations
        """
        result = self.send_request('jobs/%s/annotations' % job_id)
        return result

    def sub_status(self, sub_id, justdict=False):
        result = self.send_request('submissions/%s' % sub_id)
        if justdict:
            return result
        return result.get('status')

    def jobs_by_tag(self, tag, exact):
        exact_option = 'exact=yes' if exact else ''
        result = self.send_request(
            'jobs_by_tag?query=%s&%s' % (quote(tag.strip()), exact_option),
            {},
        )
        return result


def SubmitAstrometry(saving_folder, filename, file_extension, progress):
    filePath = app.config['UPLOAD_PATH'] + '/' + filename + file_extension
    saving_path = "app/static/images/astrometry/" + saving_folder + "/outputs/" + filename

    args = {'apiurl': Client.default_url,
            'allow_commercial_use': 'n',
            'allow_modifications': 'n',
            'publicly_visible': 'n'}
    c = Client(Client.default_url)
    c.login(API_KEY)

    upres = c.upload(filePath, **args)

    stat = upres['status']
    if stat != 'success':
        progress.setStatus("error")
        progress.setCause('Upload to astrometry.net failed: status ', stat)
        #print('Upload failed: status', stat)
        #print(upres)
        return "error"

    sub_id = upres['subid']

    while True:
        progress.setStatus("waiting for job to start for file " + filename)
        stat = c.sub_status(sub_id, justdict=True)
        #print('Got status:', stat)
        jobs = stat.get('jobs', [])
        if len(jobs):
            for j in jobs:
                if j is not None:
                    break
            if j is not None:
                #print('Selecting job id', j)
                solved_id = j
                break
        time.sleep(5)

    while True:
        progress.setStatus("solving file " + filename + " with astrometry.net")
        stat = c.job_status(solved_id, justdict=True)
        #print('Got job status:', stat)
        if stat.get('status', '') in ['success']:
            progress.setStatus("file " + filename + " solved")
            success = (stat['status'] == 'success')
            break
        elif stat.get('status', '') in ['failure']:
            progress.setStatus("error")
            progress.setCause("Image solving failed")
            #print("Image solving failed")
            return "error"  # TODO Error handling
        time.sleep(5)

    jobId = str(j)
    retrieve_red_green_url = "https://nova.astrometry.net/red_green_image_full/"
    retrieve_extraction_url = "https://nova.astrometry.net/extraction_image_full/"

    os.makedirs(saving_path)
    move_original_file = "app/static/images/astrometry/" + saving_folder + "/inputs"
    os.makedirs(move_original_file)
    os.rename(filePath, move_original_file + "/" + filename + file_extension)
    progress.setStatus("saving red-green pattern file of " + filename)
    urllib.request.urlretrieve(retrieve_red_green_url + jobId, saving_path + "/red-green.png")
    progress.setStatus("saving extraction pattern file of " + filename)
    urllib.request.urlretrieve(retrieve_extraction_url + jobId, saving_path + "/extraction.png")

    return "ok"
