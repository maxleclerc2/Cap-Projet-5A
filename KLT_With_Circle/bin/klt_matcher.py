import os
import sys
import argparse

try:
    from klt_matcher.matcher import match
except:
    package_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(os.path.join(package_dir, 'klt_matcher'))
    from matcher import match


if __name__ == '__main__':

    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mon", help="path to the monitored sensor product")
    parser.add_argument("--ref", help="path to the reference sensor product")
    parser.add_argument("--mask", help="path to the mask (optional)", default=None)
    # parser.add_argument("--no-outliers", help="display debug messages", action="store_true")

    args = parser.parse_args()

    # run matcher
    match(args.mon, args.ref, args.mask)
