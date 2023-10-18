import os
import time
import argparse

from pyslamd.Settings import Settings
from pyslamd.Tracker import Tracker
from pyslamd.helpers import get_image_paths

parser = argparse.ArgumentParser("Pyslamd")
parser.add_argument("image_dir", help="Image directory path")
# TODO: add arguments for settings 


def main():
    args = parser.parse_args()

    settings = Settings()
    tracker = Tracker(settings)

    image_paths = get_image_paths(args.image_dir)
    for image_path in image_paths:
        tracker.process_image(image_path)


if __name__ == "__main__":
    main()
