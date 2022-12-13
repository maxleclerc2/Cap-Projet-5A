import os, sys
import satpy
import numpy as np
from skimage import io

try:
    from klt_matcher.matcher import match
except:
    package_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(os.path.join(package_dir, 'klt_matcher'))
    from matcher import match

channels = ['WV_073',
 'HRV',
 'VIS008',
 'WV_062',
 'IR_120',
 'IR_039',
 'VIS006',
 'IR_087',
 'IR_134',
 'IR_097',
 'IR_108',
 'IR_016']

# MON: MSG4
# channel = 'VIS008'
for channel in channels[:1]:
    msg4 = '/tcenas/fbf/MSG/in/MSG4/OPE4/SEVI-MSG15/MSG4-SEVI-MSG15-0100-NA-20211117121242.598000000Z-NA.nat'
    mon_file = os.path.basename(msg4).replace('.nat', f'_{channel}.tif')
    mask_mon_file = os.path.basename(msg4).replace('.nat', f'_{channel}_MASK.tif')
    if not os.path.exists(mon_file):
        scene = satpy.Scene(filenames=[msg4], reader='seviri_l1b_native')
        scene.load([channel], calibration='counts')
        mon = scene[channel].values[::-1, ::-1]
        scene = None
        mask_mon = np.bitwise_not(np.isnan(mon))
        mon = mon.astype(np.uint16)
        io.imsave(mon_file, mon)
        io.imsave(mask_mon_file, mask_mon)

    # REF: MSG2
    msg2 = '/tcenas/fbf/MSG/in/MSG2/OPE2/SEVI-MSG15/MSG2-SEVI-MSG15-0100-NA-20211117121243.357000000Z-NA.nat'
    ref_file = os.path.basename(msg2).replace('.nat', f'_{channel}.tif')
    if not os.path.exists(ref_file):
        scene = satpy.Scene(filenames=[msg2], reader='seviri_l1b_native')
        scene.load([channel], calibration='counts')
        ref = scene[channel].values[::-1, ::-1]
        scene = None
        ref = ref.astype(np.uint16)
        io.imsave(ref_file, ref)


    # run matching
    match(mon_file, ref_file, mask_mon_file)


