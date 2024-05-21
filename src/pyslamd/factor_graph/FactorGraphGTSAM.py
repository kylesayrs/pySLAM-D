from typing import Tuple

import gtsam
import numpy
import pymap3d

from gtsam.symbol_shorthand import X, L

from pyslamd.Frame import Frame
from pyslamd.Settings import FactorGraphSettings
from pyslamd.utils.pose import (
    get_pose,
    get_rotation,
    set_rotation,
    get_translation,
    set_translation,
    orientation_to_rotation,
)


class FactorGraphGTSAM:
    """
    Factor graph class used to track noise models, add factors, and optimize
    Rotations are relative to north, translations are relative to origin frame
    Apply translations, then rotations

    :param settings: Factor graph settings which define noise models
    """
    def __init__(self, settings: FactorGraphSettings, use_gps: bool, use_imu: bool):
        self.use_gps = use_gps
        self.use_imu = use_imu

        # construct factor graph
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()  # initial guesses used for optimization

        # define noise models
        self.pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            [numpy.radians(settings.vo_rotation_noise)] * 3 + [settings.vo_translation_noise] * 3
        )
        self.gps_noise = gtsam.noiseModel.Isotropic.Sigma(3, settings.gps_noise)
        self.imu_noise = gtsam.noiseModel.Isotropic.Sigma(3, numpy.deg2rad(settings.imu_noise))


    def add_node(self, key_frame: Frame, initial_pose_estimate: numpy.ndarray):
        """
        Add node to factor graph with initial pose guess

        :param key_frame: key frame number to be used as node number
        :param initial_pose_estimate: initial guess for pose. Better guesses lead
            to faster and more accurate optimization
        """
        self.initial.insert(
            X(key_frame.key_frame_num),
            gtsam.Pose3(initial_pose_estimate)
        )

        # fix origin
        if key_frame.key_frame_num == 0:
            rotation = key_frame.get_imu_rotation() if self.use_imu else numpy.eye(3)
            self.graph.add(
                gtsam.PoseTranslationPrior3D(
                    X(0),
                    gtsam.Pose3(
                        r=gtsam.gtsam.Rot3(rotation),
                        t=numpy.zeros(3)
                    ),
                    gtsam.noiseModel.Diagonal.Sigmas(numpy.array([0, 0, 0]))
                )
            )


    def add_between_factor(
        self,
        reference_key_frame: Frame,
        key_frame: Frame,
        relative_pose: numpy.ndarray
    ):
        """
        Add a between factor between two nodes estimated using odometry

        :param reference_key_frame: Pose destination
        :param key_frame: Pose source
        :param relative_pose: Transformation from key_frame to reference_frame
        """
        self.graph.add(
            gtsam.BetweenFactorPose3(  # src, dst
                X(reference_key_frame.key_frame_num),
                X(key_frame.key_frame_num),
                gtsam.Pose3(relative_pose),
                self.pose_noise
            )
        )


    def add_gps_factor(self, key_frame: Frame, origin_frame: Frame):
        translation = key_frame.get_gps_translation(origin_frame)
        gps_prior = gtsam.Pose3(
            r=gtsam.Rot3(numpy.eye(3)),  # not used
            t=translation
        )
        self.graph.add(
            gtsam.PoseTranslationPrior3D(
                X(key_frame.key_frame_num),
                gps_prior,
                self.gps_noise
            )
        )


    def add_imu_factor(self, key_frame: Frame):
        rotation = key_frame.get_imu_rotation()  # imu is relative to north
        orientation_prior = gtsam.Pose3(
            r=gtsam.Rot3(rotation),
            t=numpy.zeros(3)  # not used
        )

        self.graph.add(
            gtsam.PoseRotationPrior3D(
                X(key_frame.key_frame_num),
                orientation_prior,
                self.imu_noise
            )
        )


    def add_fixed_orientation_factor(self, key_frame: Frame):
        rotation = numpy.eye(3)
        orientation_prior = gtsam.Pose3(
            r=gtsam.Rot3(rotation),
            t=numpy.zeros(3)  # not used
        )

        self.graph.add(
            gtsam.PoseRotationPrior3D(
                X(key_frame.key_frame_num),
                orientation_prior,
                self.imu_noise  # use imu noise
            )
        )
        


    def optimize(self) -> gtsam.Values:
        """
        Uses Levenberg Marquard optimizer to optimize factor graph. Updates
        future initial guesses to the lastest optimized values

        :return: optimized factor graph
        """
        #print(self.graph)
        #print(dir(self.graph))
        #print(self.graph.remove.__doc__)
        #print(dir(self.graph.at(0)))
        #print(self.graph.at(0).keys())
        #print(dir(self.initial))
        #print(self.initial.keys())
        #print(X(0))
        #exit(0)

        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial)
        result = optimizer.optimize()

        self.initial = result
        
        return result


def get_result_at(result: gtsam.Values, index: int) -> numpy.ndarray:
    """
    Utility function to get node values

    :param result: factor graph
    :param index: node index
    :return: node pose at specified index
    """
    return result.atPose3(X(index)).matrix()