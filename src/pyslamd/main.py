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

    #settings = Settings()

    #"""dahl green
    """
    settings = Settings(
        camera=CameraSettings(
            fx=1133.133974348766,
            fy=1129.223966507744,
            cx=1018.438140938584,
            cy=495.1870189047092,
            width=1920,
            height=1080
        ),
        payload=PayloadSettings(
            constant_heading=0.0
        )
    )
    """
    settings = Settings()

    print(settings)

    tracker = Tracker(settings)

    image_paths = get_image_paths(args.image_dir)
    for image_path in image_paths:
        tracker.process_image(image_path)
    

if __name__ == "__main__":
    main()
