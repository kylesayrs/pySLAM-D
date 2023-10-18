# Pyslamd #
This repo is a refactor of [pySLAM-D](https://github.com/armandok/pySLAM-D) which includes significant code cleanup, documentation, and usability improvements.

## Installation ##
1. Install pyslamd
```bash
git clone https://github.com/kylesayrs/pySLAM-D
python3 -m pip install -e .
```
2. Install [GTSAM](https://gtsam.org)
3. Install [TEASER-plusplus](https://github.com/MIT-SPARK/TEASER-plusplus)

Use cmake .. -DENABLE_DIAGNOSTIC_PRINT=OFF to turn off debug messages for teaser

## Usage ##
Edit `src/pyslamd/Settings.py` to adjust stitching settings. In the future these
settings will be documented as script arguments

```bash
pyslamd.stitch path/to/image/directory
```
Images must be geotagged with gps coordinates

## TODO ##
1. Integrate full georeferencing into pyslamd, overlap pruning
2. Implement node pruning to bound factor graph computation
3. Implement outlier reprojection and keypoint filtering
3. Test with 99% overlap vs 50% overlap
4. Implement mosiacing for quality improvement
