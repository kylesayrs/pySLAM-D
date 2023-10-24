from typing import Union

from pydantic import BaseModel, Field
from enum import Enum

import numpy


class LogLevel(Enum):
    DEBUG = 1
    INFO = 2
    CRITICAL = 3


class CameraSettings(BaseModel):
    #""" New Payload Estimate
    fx: float = Field(default=5223, description="x-axis focal length")
    fy: float = Field(default=5223, description="y-axis focal length")
    cx: float = Field(default=2038, description="focal center wrt top left corner")
    cy: float = Field(default=1558, description="focal center wrt top left corner")
    width: int = Field(default=4032, description="image width")
    height: int = Field(default=3040, description="image height")
    #"""

    """ OpenSFM estimation
    fx: float = Field(default=0.8636155192377428 * 4032 * 1.5)  # 3482
    fy: float = Field(default=0.8636155192377428 * 4032 * 1.5)  # 3482
    cx: float = Field(default=4032 / 2 + 0.0054954422956500085 * 4032)
    cy: float = Field(default=3040 / 2 + 0.012485656805473633 * 3040)
    width: int = Field(default=4032)
    height: int = Field(default=3040)
    """

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
    num_levels: int = Field(default=8, description="number of image size reductions")
    fast_threshold: int = Field(default=20)

    num_block_rows: int = Field(default=8)  # 4
    num_block_columns: int = Field(default=8)  # 3


class MatcherSettings(BaseModel):
    keypoints: KeypointSettings = Field(default=KeypointSettings())

    # match candidates
    gps_match_bound: Union[float, None] = Field(default=None)  # 20
    overlap_matching: bool = Field(default=True)

    # matching thresholds
    min_num_matches: int = Field(default=150)
    max_translation: int = Field(default=150)

    reproject_outliers: bool = Field(default=False)  # could also do inliers

    debug_matches: bool = Field(default=False)


class FactorGraphSettings(BaseModel):
    use_vo_factor: bool = Field(default=True)
    use_gps_factor: bool = Field(default=True)
    use_imu_factor: bool = Field(default=True)

    vo_rotation_noise: float = Field(default=90.0, description="one standard deviation in degrees")
    vo_translation_noise: float = Field(default=50.0, description="one standard deviation in meters")
    imu_noise: float = Field(default=5.0, description="one standard deviation in degrees")
    gps_noise: float = Field(default=10.0, description="one standard deviation in meters")


class VisualizerSettings(BaseModel):
    render: bool = Field(default=True)
    reset_every_frame: bool = Field(default=True)
    downsample: int = Field(default=100)


class Settings(BaseModel):
    camera: CameraSettings = Field(default=CameraSettings())
    payload: PayloadSettings = Field(default=PayloadSettings())
    matcher: MatcherSettings = Field(default=MatcherSettings())
    graph: FactorGraphSettings = Field(default=FactorGraphSettings())
    visualizer: VisualizerSettings = Field(default=VisualizerSettings())

    use_vo: bool = Field(default=True)
    use_gps: bool = Field(default=True)
    use_imu: bool = Field(default=True)

    out_dir: str = Field(default="outdir")

    log_level: LogLevel = Field(default=LogLevel.DEBUG)  # not implemented yet
