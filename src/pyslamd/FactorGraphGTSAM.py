from typing import Tuple

import gtsam
import numpy
import pymap3d

from gtsam.symbol_shorthand import X, L

from pyslamd.Frame import Frame
from pyslamd.Settings import FactorGraphSettings
from pyslamd.helpers import get_rotation


class FactorGraphGTSAM:
    """
    Factor graph class used to track noise models, add factors, and optimize

    :param settings: Factor graph settings which define noise models
    """
    def __init__(self, settings: FactorGraphSettings):
        # construct factor graph
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()  # initial guesses used for optimizer

        # define noise models
        self.pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            [settings.rotation_noise] * 3 + [settings.translation_noise] * 3
        )
        self.gps_noise = gtsam.noiseModel.Diagonal.Sigmas(
            [numpy.pi] * 3 + [settings.gps_noise] * 3  # TODO: combine with attitude noise
        )
        self.attitude_noise = gtsam.noiseModel.Isotropic.Sigma(3, numpy.deg2rad(settings.attitude_noise))

        # Fix the initial frame as the origin
        self.graph.add(gtsam.NonlinearEqualityPose3(X(0), gtsam.Pose3(numpy.eye(4))))
        self.gps_origin_coords = None


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

        if self.gps_origin_coords is None:
            self.gps_origin_coords = key_frame.gps_coords


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
            gtsam.BetweenFactorPose3(
                X(key_frame.key_frame_num),
                X(reference_key_frame.key_frame_num),
                gtsam.Pose3(relative_pose),
                self.pose_noise
            )
        )


    def add_gps_factor(self, key_frame: Frame, gps_coords: Tuple[float, float, float]):
        """
        TODO: combine with add_attitude_factor
        """
        relative_meters = pymap3d.geodetic2enu(*key_frame.gps_coords, *self.gps_origin_coords)
        relative_meters = (relative_meters[0], relative_meters[1], relative_meters[2])
        relative_meters = numpy.array(relative_meters, dtype=numpy.float64)

        gps_pose = numpy.eye(4)
        gps_pose[0:3, 3] = relative_meters

        self.graph.add(
            gtsam.PriorFactorPose3(
                X(key_frame.key_frame_num),
                gtsam.Pose3(gps_pose),
                self.gps_noise
            )
        )

    
    def add_attitude_factor(self, key_frame: Frame, pose_estimate: numpy.ndarray):
        """
        TODO: combine with gps_factor
        """
        rotation_estimate = get_rotation(pose_estimate)
        attitude_prior = gtsam.Pose3(
            r=gtsam.Rot3(rotation_estimate),
            t=numpy.array([0.0, 0.0, 0.0]
        ))

        self.graph.add(
            gtsam.PoseRotationPrior3D(
                X(key_frame.key_frame_num),
                attitude_prior,
                self.attitude_noise
            )
        )


    def optimize(self) -> gtsam.Values:
        """
        Uses Levenberg Marquard optimizer to optimize factor graph. Updates
        future initial guesses to the lastest optimized values

        :return: optimized factor graph
        """
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