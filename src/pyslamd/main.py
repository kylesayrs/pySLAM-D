import os
import time
import argparse

from pyslamd.Settings import Settings, CameraSettings, PayloadSettings, MatcherSettings
from pyslamd.Tracker import Tracker
from pyslamd.utils.helpers import get_image_paths

parser = argparse.ArgumentParser("Pyslamd")
parser.add_argument("image_dir", help="Image directory path")
# TODO: add arguments for settings 


def main():
    args = parser.parse_args()

    #"""bostongc
    settings = Settings(
        camera=CameraSettings(
            fx=5223,
            fy=5223,
            cx=2038,
            cy=1558,
            width=4032,
            height=3040,
        ),
        payload=PayloadSettings(
            constant_heading=None,
            gimbal_enabled=True,
            yaw_offset=180.0,
        )
    )
    #"""
    """dahlgreen
    settings = Settings(
        camera=CameraSettings(
            fx=5223,
            fy=5223,
            cx=2038,
            cy=1558,
            width=4032,
            height=3040,
        ),
        payload=PayloadSettings(
            constant_heading=None,
            gimbal_enabled=True,
            yaw_offset=180.0,
        )
    )
    """

    tracker = Tracker(settings)

    image_paths = get_image_paths(args.image_dir)
    for image_path in image_paths:
        tracker.process_image(image_path)

    while True:
        time.sleep(1)
    

if __name__ == "__main__":
    main()
