from typing import Tuple, Dict, Union

import numpy
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


CoordsType = Tuple[float, float, float]
DmsType = Tuple[float, float, float]


def get_gps_coords(image_path: str) -> numpy.ndarray:
    image = Image.open(image_path)
    exif_data = image._getexif()

    if exif_data is None:
        raise ValueError(f"{image_path} does not have exif data")

    exif_data_labeled = {
        TAGS[key]: value
        for key, value in exif_data.items()
        if key in TAGS
    }

    gps_data_labeled = {
        GPSTAGS[key]: value
        for key, value in exif_data_labeled["GPSInfo"].items()
        if key in GPSTAGS
    }

    return get_coords(gps_data_labeled)


def get_coords(gps_data: Dict[str, Union[DmsType, str]]) -> numpy.ndarray:
    lat = get_decimal_from_dms(
        gps_data["GPSLatitude"],
        gps_data["GPSLatitudeRef"],
    )

    lng = get_decimal_from_dms(
        gps_data["GPSLongitude"],
        gps_data["GPSLongitudeRef"],
    )

    alt = float(gps_data["GPSAltitude"])

    return numpy.array([lat, lng, alt])


def get_decimal_from_dms(dms: Tuple[float, float, float], ref: Tuple[float, float, float]) -> float:
    """Convert GPS in degrees, minutes, seconds to lat, long """
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0

    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return round(degrees + minutes + seconds, 9)