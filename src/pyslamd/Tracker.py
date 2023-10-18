from typing import Tuple, List, Optional, Union

import os
import numpy
import warnings
import open3d as open3d

from pyslamd.Frame import Frame
from pyslamd.FrameMatcher import FrameMatcher
from pyslamd.Settings import Settings, LogLevel
from pyslamd.FactorGraphGTSAM import FactorGraphGTSAM, get_result_at
from pyslamd.helpers import (
    make_keypoint_detector,
    get_translation,
    get_rotation,
    get_pose,
    all_none
)

import time


class Tracker:
    """
    Tracker class used to process new frames, perform matching, and add positions
    to the factor graph. Responsible for orchestrating the algorithm's core operations

    :param settings: settings which describe how the stitch should be done
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.keypoint_detector = make_keypoint_detector(settings)
        self.matcher = FrameMatcher(settings)
        self.factor_graph = FactorGraphGTSAM(settings.graph)
        
        self.frames = []
        self.key_frames = []

        self.point_cloud_cache = open3d.geometry.PointCloud() # TODO: be smarter about this

        if self.settings.visualizer.render:
            self.visualizer = open3d.visualization.Visualizer()
            self.visualizer.create_window()
            self.visualizer.update_renderer()

            # used for implicitly setting the camera position
            self.bounds = numpy.array([[0, 0, 0], [0, 0, 0]])
            self.dummy_pcd = open3d.geometry.PointCloud()
        else:
            self.visualizer = None

        if not self.settings.graph.include_gps_factor:
            warnings.warn(
                "Not including GPS factors may lead to images with no VO matches, "
                "causing the stitch to fail"
            )


    def process_image(self, image_path: str):
        """
        Ingests a new image. Attempts to match new frames to neighboring key frames.
        If the frame is converted into a key fraame, then its pose and factors
        are added to the factor graph.

        :param image_path: path of image to process
        """
        frame = self._add_frame(image_path)

        # collect poses and inliers w.r.t. to key frames
        if self.settings.graph.include_pose_factor:
            relative_poses, outliers = self.matcher.match_frames(frame, self.key_frames)
        else:
            relative_poses, outliers = [], []

        # TODO: reproject outliers onto key points
        if self.settings.matcher.reproject_outliers:
            self.matcher.reproject_outliers(frame, outliers)

        # add to factor graph and key points list
        if True:#all_none(relative_poses):  # TODO: for now, all frames should be key frames
            self._add_key_frame(frame, relative_poses)

        # optimize factor graph to get new poses
        optimize_start = time.time()
        self._optimize_and_update_poses()

        # render visual
        visualize_start = time.time()
        if self.settings.visualizer.render:
            self._update_visual()


    def _optimize_and_update_poses(self):
        """
        Optimizes factor graph and updates global poses of key frames
        """
        result = self.factor_graph.optimize()

        assert result.size() == len(self.key_frames)

        for key_frame_index in range(result.size()):
            key_frame = self.key_frames[key_frame_index]
            global_pose_estimate = get_result_at(result, key_frame_index)
            key_frame.set_global_pose(global_pose_estimate)
            #print(f"{key_frame.key_frame_num}: {global_pose_estimate}")
            #corners = key_frame.get_corners(self.key_frames[0].gps_coords[:2])
            #print(f"{os.path.basename(key_frame.image_path)}: {corners}")


    def _update_visual(self):
        """
        TODO: move to Visualizer class and dedicated thread
        """
        if self.visualizer is None:
            raise ValueError("Visualizer has not been initialized")

        if self.settings.visualizer.reset_every_frame:
            self.visualizer.clear_geometries()

        for key_frame in self.key_frames:
            if self.settings.visualizer.reset_every_frame or key_frame.point_cloud_needs_add:
                self.visualizer.add_geometry(key_frame.point_cloud, reset_bounding_box=False)

                self.bounds = numpy.array([
                    numpy.min([self.bounds[0], key_frame.point_cloud.get_min_bound()], axis=0),
                    numpy.max([self.bounds[1], key_frame.point_cloud.get_max_bound()], axis=0)
                ])

                key_frame.point_cloud_needs_add = False

            self.visualizer.update_geometry(key_frame.point_cloud)

        # adds a dummy pcd with corners at the bounds of the geometry
        self.dummy_pcd.points = open3d.utility.Vector3dVector(self.bounds)
        self.dummy_pcd.colors = open3d.utility.Vector3dVector([(0, 0, 0), (0, 0, 0)])
        self.visualizer.add_geometry(self.dummy_pcd, reset_bounding_box=True)

        # remove dummy pcd
        self.dummy_pcd.points = open3d.utility.Vector3dVector([])
        self.dummy_pcd.colors = open3d.utility.Vector3dVector([])
        self.visualizer.update_geometry(self.dummy_pcd)

        self.visualizer.poll_events()

    
    def _add_frame(self, image_path: str) -> Frame:
        """
        Tracks frames
        TODO: remove this function and do not track non-keyframe frames
        """
        frame = Frame(image_path, len(self.frames), self.settings, self.keypoint_detector)
        self.frames.append(frame)

        return frame


    def _add_key_frame(self, frame: Frame, relative_poses: List[Union[numpy.ndarray, None]]):
        """
        Tracks key frames and add their factors to the factor graph

        :param frame: The frame to be converted to a key frame
        :param relative_poses: A list of relative poses calculated from VO
        """
        frame.set_key_frame_num(len(self.key_frames))
        self.key_frames.append(frame)

        # add node
        initial_pose_estimate = self._get_initial_pose_estimate(frame, relative_poses)
        self.factor_graph.add_node(frame, initial_pose_estimate)

        # add relative poses
        for reference_frame, relative_pose in zip(self.key_frames, relative_poses):
            if relative_pose is not None:
                self.factor_graph.add_between_factor(reference_frame, frame, relative_pose)

        # add gps factor
        if self.settings.graph.include_gps_factor:
            self.factor_graph.add_gps_factor(frame, frame.gps_coords)

        # add attitude factor
        if self.settings.graph.include_attitude_factor:
            self.factor_graph.add_attitude_factor(frame, initial_pose_estimate)

        # TODO: combine attitute and gps factor

        if (
            frame.frame_num != 0 and
            all([pose is None for pose in relative_poses]) and
            not self.settings.graph.include_gps_factor
        ):
            raise ValueError(f"frame {frame.frame_num} has no VO matches")


    def _get_initial_pose_estimate(
        self,
        frame: Frame,
        relative_poses: List[Union[numpy.ndarray, None]]
    ) -> numpy.ndarray:
        """
        Gets an initial pose estimate used to speed up factor graph optimization.
        The funcction first attempts to use the first viable relative pose,
        otherwise defaulting to gps estimates

        :param frame: The frame whose pose should be estimated
        :param relative_poses: List of relative poses estimated from odometry
        :return: The pose estimate for the given frame
        """
        if frame.key_frame_num == 0:
            return numpy.eye(4)

        for reference_frame, relative_pose in reversed(list(zip(self.key_frames, relative_poses))):
            if relative_pose is not None:
                return reference_frame.global_pose @ relative_pose

        # TODO: use gps position and heading, NOT last pose

        # use last frame's pose
        return self.key_frames[-2].global_pose

