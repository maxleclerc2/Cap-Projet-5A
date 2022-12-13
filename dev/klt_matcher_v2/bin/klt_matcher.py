import os
import sys
import argparse

try:
    from klt_matcher.matcher import match
except:
    package_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(os.path.join(package_dir, 'klt_matcher'))
    from matcher import match

from core.configuration import Configuration

# ANCIENS ARGUMENTS :
# --mon E:\DATA\KLT_TEST_DATA\3122863_3423606_2020-02-10_2257_BGREN_Analytic_band2.tif --ref E:\DATA\KLT_TEST_DATA\3122863_3423606_2020-02-10_2257_BGREN_Analytic_band5.tif --resume True --out E:\DATA\KLT_TEST_DATA\OUT
if __name__ == '__main__':

    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mon", help="path to the monitored sensor product")
    parser.add_argument("--ref", help="path to the reference sensor product")
    parser.add_argument("--mask", help="path to the mask (optional)",
                        default=None)
    parser.add_argument("--conf", help="configuration file",
                        default=os.path.join(package_dir,
                                             "configuration/processing_configuration.json"))
    parser.add_argument("--out", help="path to save output results",
                        default=os.path.join(package_dir,
                                             "results"))
    parser.add_argument("--resume", dest="resume", type=bool,
                        help="do not apply klt, just accuracy analysis",
                        metavar="bool", default=False)

    # parser.add_argument("--no-outliers", help="display debug messages", action="store_true")

    args = parser.parse_args()

    # set up configuration :
    conf = Configuration(args)
    # run matcher :
    match(args.mon, args.ref, conf, args.resume, args.mask)

