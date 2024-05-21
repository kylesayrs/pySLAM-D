from typing import Tuple, List, Optional, Union

import os
import numpy
import warnings
import rasterio
import open3d as open3d
from rasterio import warp

from pyslamd.Frame import Frame
from pyslamd.odometry import OdometryMatcher
from pyslamd.Settings import Settings, LogLevel
from pyslamd.factor_graph import FactorGraphGTSAM, get_result_at
from pyslamd.utils.helpers import all_none
from pyslamd.utils.pose import get_rotation, get_translation, get_pose

import time  # used for timing
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class Tracker:
    """
    Tracker class used to process new frames, perform matching, and add positions
    to the factor graph. Responsible for orchestrating the algorithm's core operations

    :param settings: settings which describe how the stitch should be done
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.matcher = OdometryMatcher(settings.matcher, settings.use_gps, settings.use_imu)
        self.factor_graph = FactorGraphGTSAM(settings.graph, settings.use_gps, settings.use_imu)
        
        self.frames = []
        self.key_frames = []
        self.origin_frame = None

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

        if self.settings.out_dir is not None:
            for file_name in os.listdir(self.settings.out_dir):
                os.remove(os.path.join(self.settings.out_dir, file_name))


    def process_image(self, image_path: str):
        """
        Ingests a new image. Attempts to match new frames to neighboring key frames.
        If the frame is converted into a key fraame, then its pose and factors
        are added to the factor graph.

        :param image_path: path of image to process
        """
        frame = self._add_frame(image_path)

        # collect poses and inliers w.r.t. to key frames
        if self.settings.use_vo:
            relative_poses = self.matcher.match_frames(frame, self.key_frames, self.origin_frame)
        else:
            relative_poses = []

        # add to factor graph and key points list
        if True: # TODO: for now, all frames should be key frames
            self._add_key_frame(frame, relative_poses)

        # optimize factor graph to get new poses
        self._optimize_and_update_poses()

        # save pose information
        if self.settings.out_dir is not None:
            self._save_image_poses()

        # render visual
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
            key_frame.set_global_pose(global_pose_estimate)  # TODO: do not update the global pose if it does not differ from the current pose more than some threshold
            #print(f"{key_frame.key_frame_num}: {global_pose_estimate}")
            #corners = key_frame.get_latlng_corners(self.origin_frame)
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
                key_frame.point_cloud_needs_add = True

        for key_frame in self.key_frames:
            point_cloud = key_frame.get_point_cloud()
            
            if key_frame.point_cloud_needs_add:
                self.visualizer.add_geometry(point_cloud, reset_bounding_box=False)

                self.bounds = numpy.array([
                    numpy.min([self.bounds[0], point_cloud.get_min_bound()], axis=0),
                    numpy.max([self.bounds[1], point_cloud.get_max_bound()], axis=0)
                ])

                key_frame.point_cloud_needs_add = False

            self.visualizer.update_geometry(point_cloud)

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
        frame = Frame(image_path, len(self.frames), self.settings)
        self.frames.append(frame)

        return frame


    def _add_key_frame(self, frame: Frame, relative_poses: List[Union[numpy.ndarray, None]]):
        """
        Tracks key frames and add their factors to the factor graph

        :param frame: The frame to be converted to a key frame
        :param relative_poses: A list of relative poses calculated from VO
        """
        if (
            frame.frame_num != 0 and
            all_none(relative_poses) and
            not self.settings.use_gps
        ):
            warnings.warn(
                f"frame {frame.frame_num} has no VO matches and has no GPS "
                "fallback, the frame will not be tracked"
            )
            return

        # set origin frame
        if len(self.key_frames) == 0:
            self.origin_frame = frame

        # add to list of key frames
        frame.set_key_frame_num(len(self.key_frames))
        self.key_frames.append(frame)

        # add node to factor graph
        initial_pose_estimate = self._get_initial_pose_estimate(frame, relative_poses)
        self.factor_graph.add_node(frame, initial_pose_estimate)

        # add relative pose factors
        for reference_frame, relative_pose in zip(self.key_frames, relative_poses):
            if relative_pose is not None and self.settings.graph.use_vo_factor:
                self.factor_graph.add_between_factor(reference_frame, frame, relative_pose)

        # add gps factor
        if self.settings.use_gps and self.settings.graph.use_gps_factor:
            self.factor_graph.add_gps_factor(frame, self.origin_frame)
            
        # if imu is not available, assume a fixed orientation
        if self.settings.graph.use_imu_factor:
            if self.settings.use_imu:
                self.factor_graph.add_imu_factor(frame)
            else:
                self.factor_graph.add_fixed_orientation_factor(frame)

        print(f"Added {frame}")


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
        return get_pose(
            (
                frame.get_imu_rotation(self.origin_frame)
                if self.settings.use_imu
                else get_rotation(self.key_frames[-2].global_pose)
                if frame.key_frame_num != 0
                else numpy.eye(3)
            ),
            (
                frame.get_gps_translation(self.origin_frame)
                if self.settings.use_gps
                else get_translation(self.key_frames[-2].global_pose)
                if frame.key_frame_num != 0
                else numpy.zeros(3)
            )
        )


    def _save_image_poses(self):
        #for key_frame in self.key_frames:
        key_frame = self.key_frames[-1]  # for now, only write the last image
        image_corners = [
            (0, 0),
            (key_frame.settings.camera.width, 0),
            (key_frame.settings.camera.width, key_frame.settings.camera.height),
            (0, key_frame.settings.camera.height)
        ]

        geodetic_corners = [
            key_frame.image_to_geodetic_point(*corner, self.origin_frame)
            for corner in image_corners
        ]

        ground_control_points = [
            rasterio.control.GroundControlPoint(
                row=image_corner[1],
                col=image_corner[0],
                x=geodetic_corner[1],
                y=geodetic_corner[0],
                z=0.0  # assume flat projection
            )
            for image_corner, geodetic_corner in zip(image_corners, geodetic_corners)
        ]

        image_data = numpy.transpose(key_frame.get_image(), (2, 0, 1))

        crs = rasterio.CRS.from_epsg(4326)  # WGS84
        destination_data, destination_data_transform = warp.reproject(
            image_data,
            gcps=ground_control_points,
            #src_transform=src_transform,
            src_crs=crs,
            dst_crs=crs,
            resampling=rasterio.enums.Resampling.nearest,
            num_threads=1,
            warp_mem_limit=0
        )

        # TODO: move to "construct_rasterio_metadata"
        image_metadata = {
            "driver": "JPEG",
            "dtype": "uint8",
            "nodata": 0,  # pure-black pixels are no data
            "width": destination_data.shape[2],
            "height": destination_data.shape[1],
            "count": 3,
            "crs": crs,
            "transform": destination_data_transform,
        }

        destination_path = os.path.join(self.settings.out_dir, os.path.basename(key_frame.image_path))
        with rasterio.open(destination_path, "w", **image_metadata) as destination_file:
            destination_file.write(destination_data)

        #print(image_data.shape)
        #print(type(image_data))
        #print(image_metadata)
