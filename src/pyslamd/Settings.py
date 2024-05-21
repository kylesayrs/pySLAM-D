from typing import Union

from pydantic import BaseModel, Field
from enum import Enum

import numpy


class LogLevel(Enum):
    DEBUG = 1
    INFO = 2
    CRITICAL = 3


class CameraSettings(BaseModel):
    """ OpenSFM estimation
    fx: float = Field(default=0.8636155192377428 * 4032)  # 3482
    fy: float = Field(default=0.8636155192377428 * 4032)  # 3482
    cx: float = Field(default=4032 / 2 + 0.0054954422956500085 * 4032)  # 2038
    cy: float = Field(default=3040 / 2 + 0.012485656805473633 * 3040)  # 1557
    width: int = Field(default=4032)
    height: int = Field(default=3040)
    """

    #""" Camera Specs (5.4mm focal length, 1.55um per pixel)
    fx: float = Field(default=5.4e-3 / 1.55e-6, description="x-axis focal length in pixels")  # 3483
    fy: float = Field(default=5.4e-3 / 1.55e-6, description="y-axis focal length in pixels")  # 3483
    cx: float = Field(default=4032 / 2, description="focal center wrt top left corner")  # 2016
    cy: float = Field(default=3040 / 2, description="focal center wrt top left corner")  # 1520
    width: int = Field(default=4032, description="image width in pixels")
    height: int = Field(default=3040, description="image height in pixels")
    #"""

    """ Frist Flight (dahl green)
    fx: float = Field(default=1133.133974348766)
    fy: float = Field(default=1129.223966507744)
    cx: float = Field(default=1018.438140938584)
    cy: float = Field(default=495.1870189047092)
    width: int = Field(default=1920)
    height: int = Field(default=1080)
    """


class PayloadSettings(BaseModel):
    constant_heading: Union[float, None] = Field(default=None, description="relative to north")
    gimbal_enabled: bool = Field(default=True)
    yaw_offset: float = Field(default=180.0, description="degrees")


class KeypointSettings(BaseModel):
    num_features: int = Field(default=300, description="number of features per block")  # 1000
    scale_factor: int = Field(default=2.0, description="image size reduction per level")  # 1.2
    num_levels: int = Field(default=6, description="number of image size reductions")
    fast_threshold: int = Field(default=20)

    # blocks help ensure each area of the image has features
    num_block_rows: int = Field(default=6)  # 4
    num_block_columns: int = Field(default=6)  # 3


class MatcherSettings(BaseModel):
    keypoints: KeypointSettings = Field(default=KeypointSettings())

    # match candidates
    gps_match_bound: Union[float, None] = Field(default=75.0)  # 20
    overlap_matching: bool = Field(default=False)

    # matching thresholds
    min_num_matches: int = Field(default=100)
    max_translation_margin: int = Field(default=10, description="maximum difference between vo translation estimate and gps translation estimate in meters")
    reproject_outliers: bool = Field(default=False)

    debug_matches: bool = Field(default=False)


class FactorGraphSettings(BaseModel):
    use_vo_factor: bool = Field(default=True)
    use_gps_factor: bool = Field(default=True)
    use_imu_factor: bool = Field(default=True)

    vo_rotation_noise: float = Field(default=15.0, description="one standard deviation in degrees")
    vo_translation_noise: float = Field(default=10.0, description="one standard deviation in meters")
    imu_noise: float = Field(default=10.0, description="one standard deviation in degrees")
    gps_noise: float = Field(default=15.0, description="one standard deviation in meters")


class VisualizerSettings(BaseModel):
    render: bool = Field(default=False)
    reset_every_frame: bool = Field(default=False)
    downsample: int = Field(default=1000)


class Settings(BaseModel):
    camera: CameraSettings = Field(default=CameraSettings())
    payload: PayloadSettings = Field(default=PayloadSettings())
    matcher: MatcherSettings = Field(default=MatcherSettings())
    graph: FactorGraphSettings = Field(default=FactorGraphSettings())
    visualizer: VisualizerSettings = Field(default=VisualizerSettings())

    # TODO: image_scale: float = Field(default=1.0, description="scale image size to reduce runtime")

    use_vo: bool = Field(default=True)
    use_gps: bool = Field(default=True)
    use_imu: bool = Field(default=True)

    out_dir: Union[str, None] = Field(default="outdir")

    log_level: LogLevel = Field(default=LogLevel.DEBUG)  # not implemented yet
