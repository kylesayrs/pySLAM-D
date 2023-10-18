from typing import Union

from pydantic import BaseModel, Field
from enum import Enum

import numpy


class LogLevel(Enum):
    DEBUG = 1
    INFO = 2
    CRITICAL = 3


class CameraSettings(BaseModel):
    #""" New Payload
    fx: float = Field(default=5300)
    fy: float = Field(default=4000)
    cx: float = Field(default=4032 / 2)
    cy: float = Field(default=3040 / 2)
    width: int = Field(default=4032)
    height: int = Field(default=3040)
    #"""

    """ ???
    fx: float = Field(default=3814.31)
    fy: float = Field(default=3814.31)
    cx: float = Field(default=8.47517)
    cy: float = Field(default=49.0468)
    width: int = Field(default=4608)
    height: int = Field(default=3456)
    """

    """ Frist Flight (dahl green)
    fx: float = Field(default=1133.133974348766)
    fy: float = Field(default=1129.223966507744)
    cx: float = Field(default=1018.438140938584)
    cy: float = Field(default=495.1870189047092)
    width: int = Field(default=1920)
    height: int = Field(default=1080)
    """


class KeypointSettings(BaseModel):
    num_features: int = Field(default=1500, description="number of features per block")
    scale_factor: int = Field(default=2.0, description="image size reduction per level")
    num_levels: int = Field(default=8, description="number of image size reductions")
    fast_threshold: int = Field(default=20)

    num_block_rows: int = Field(default=4)
    num_block_columns: int = Field(default=3)


class MatcherSettings(BaseModel):
    gps_match_bound: Union[float, None] = Field(default=75)
    
    overlap: Union[float, None] = Field(default=None)
    overlap_method: Union[str, None] = Field(default=None)

    top_k_matches: Union[int, None] = Field(default=None)
    min_num_matches: int = Field(default=100)
    max_translation: int = Field(default=120)
    reproject_outliers: bool = Field(default=False)  # could also do inliers

    debug_matches: bool = Field(default=False)


class FactorGraphSettings(BaseModel):
    include_pose_factor: bool = Field(default=True)
    include_gps_factor: bool = Field(default=False)
    include_attitude_factor: bool = Field(default=False)

    rotation_noise: float = Field(default=numpy.pi * 0.08, description="one standard deviation in radians")
    translation_noise: float = Field(default=50.0, description="one standard deviation in meters")
    gps_noise: float = Field(default=50.0, description="one standard deviation in meters")
    attitude_noise: float = Field(default=30.0, description="one standard deviation in degrees")


class VisualizerSettings(BaseModel):
    render: bool = Field(default=True)
    reset_every_frame: bool = Field(default=True)
    downsample: int = Field(default=100)


class Settings(BaseModel):
    camera: CameraSettings = Field(default=CameraSettings())
    keypoints: KeypointSettings = Field(default=KeypointSettings())
    matcher: MatcherSettings = Field(default=MatcherSettings())
    graph: FactorGraphSettings = Field(default=FactorGraphSettings())
    visualizer: VisualizerSettings = Field(default=VisualizerSettings())

    log_level: LogLevel = Field(default=LogLevel.DEBUG)  # not implemented yet
