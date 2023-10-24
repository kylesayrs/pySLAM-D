from typing import Tuple, Dict, Union, Any

import numpy
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from pyslamd.Settings import Settings, PayloadSettings


CoordsType = Tuple[float, float, float]
DmsType = Tuple[float, float, float]


def get_exif_measurements(image_path: str, settings: Settings) -> Tuple[numpy.ndarray, numpy.ndarray]:
    exif_data = _get_labeled_exif(image_path)

    gps_coords = _get_gps_coords(exif_data)

    if settings.use_imu:
        imu_orientation = _get_imu_orientation(exif_data, settings.payload)
    else:
        imu_orientation = None

    return gps_coords, imu_orientation


def _get_labeled_exif(image_path: str) -> Dict[str, Any]:
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

    exif_data_labeled.update(gps_data_labeled)

    return exif_data_labeled


def _get_gps_coords(exif_data: Dict[str, Any]) -> numpy.ndarray:
    lat = _get_decimal_from_dms(
        exif_data["GPSLatitude"],
        exif_data["GPSLatitudeRef"],
    )

    lng = _get_decimal_from_dms(
        exif_data["GPSLongitude"],
        exif_data["GPSLongitudeRef"],
    )

    alt = float(exif_data["GPSAltitude"])

    return numpy.array([lat, lng, alt])


def _get_imu_orientation(
    exif_data: Dict[str, Union[DmsType, str]],
    settings: PayloadSettings
) -> numpy.ndarray:
    """
    Use 3-2-1 YRP standard used by Arducopter. Assume no roll or pitch if payload
    gimbal is enabled

    :param exif_data: Dictionary of labeled exif data
    :param settings: payload settings
    :return: numpy array of yaw, roll, and pitch values
    """
    if settings.gimbal_enabled:
        yaw = (
            settings.constant_heading
            if settings.constant_heading is not None
            else -1 * float(exif_data["GPSImgDirection"]) + settings.yaw_offset  # the -1 was found emperically
        )

        return numpy.array([
            yaw,
            0.0,
            0.0
        ])

    else:
        orientation_comment = exif_data["Comment"]
        data_comments = orientation_comment.split(",")
        data = {
            data_comment.split("=")[0]: data_comment.split("=")[1]
            for data_comment in data_comments
        }
        return numpy.array([
            (
                settings.constant_heading
                if settings.constant_heading is not None
                else float(data["yaw"])
            ),
            float(data["roll"]),
            float(data["pitch"])
        ])


def _get_decimal_from_dms(dms: Tuple[float, float, float], ref: Tuple[float, float, float]) -> float:
    """Convert GPS in degrees, minutes, seconds to lat, long """
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0

    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return round(degrees + minutes + seconds, 9)