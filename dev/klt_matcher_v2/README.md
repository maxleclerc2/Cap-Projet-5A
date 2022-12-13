# KLT_matcher

**Description:**

Image matching tool based on the KLT tracker associated to preprocessing technics and outliers filtering.

**The tool inputs:**
* monitored sensor image file
* reference sensor image file
* a mask file (optional)

_Input files shall contain only one layer of data, and the format shall recognized by gdal library._  

**The tool outputs:**
* **csv file**: list of keypoints and associated dx/dy deviations
* **png file**: visualisation of the deviations

## Requirements

the KLT_matcher is based on `python 3`, with `opencv`, `scikit-image`, `pandas` and `gdal` libraries for the matching, and 
`matplotlib` for the visualisation.

Create a new environment with conda:

```shell
conda create --name klt_matcher python=3.9 gdal opencv matplotlib scikit-image pandas -c conda-forge
conda activate klt_matcher
```

## Run time

Usage:
```shell
python bin/klt_matcher.py -h
usage: klt_matcher.py [-h] [--mon MON] [--ref REF] [--mask MASK]

optional arguments:
  -h, --help   show this help message and exit
  --mon MON    path to the monitored sensor image file
  --ref REF    path to the reference sensor image file
  --mask MASK  path to the mask (optional)
usage: klt_matcher.py [-h] [--mon MON] [--ref REF] [--mask MASK]
```

Command line example:
```shell
python bin/klt_matcher.py --mon PATH1 --ref PATH2
```



## Examples

* Example of a S2  / LS8 inter-registration accuracy (Sen2Like Products)
```shell
python bin/klt_matcher.py --mon examples/L2H_T31TFJ_20170619T103021_S2A_R108_B8A_20m_resize.TIF --ref examples/L2F_T31TFJ_20170319T102333_LS8_R196_B8A_20m_resize.TIF --mask examples/L2F_T31TFJ_20170319T102333_LS8_R196_L8_MSK_resize.TIF
```

## References

**GQA Tool study**
https://www.eumetsat.int/GQA-tool

**A comprehensive geometric quality assessment approach for MSG SEVIRI imagery,**
Sultan Kocaman, Vincent Debaecker, Sila Bas, Sebastien Saunier, Kevin Garcia, Dieter Just,
Advances in Space Research, 2021, ISSN 0273-1177,
https://doi.org/10.1016/j.asr.2021.11.018

**On the geometric accuracy and stability of MSG SEVIRI images,**
Vincent Debaecker, Sultan Kocaman, Sebastien Saunier, Kevin Garcia, Sila Bas, Dieter Just,
Atmospheric Environment, Volume 262, 2021, 118645, ISSN 1352-2310,
https://doi.org/10.1016/j.atmosenv.2021.118645

