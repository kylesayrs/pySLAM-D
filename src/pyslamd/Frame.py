from typing import Tuple, List

import cv2
import utm
import copy
import numpy
import open3d as o3d

from pyslamd.gps import get_gps_coords, CoordsType
from pyslamd.Settings import Settings, CameraSettings
from pyslamd.helpers import blocks


class Frame:
    """
    Frame object for logically grouping gps, keypoints, pose, and point clouds.
    A frame object that is assigned a key_frame_num is a keyframe

    :param image_path: path to frame image data
    :param frame_num: frame number, assigned sequentially
    :param settings: settings used for camera intrinsics and cloud sparsity
    :param keypoint_detector: detector used to detect keypoints. Passed by reference
        to avoid duplicate instatiation of the detector
    """
    def __init__(
        self,
        image_path: str,
        frame_num: int,
        settings: Settings,
        keypoint_detector: cv2.Feature2D,
    ):
        self.image_path = image_path
        self.frame_num = frame_num
        self.key_frame_num = None
        self.settings = settings

        self.gps_coords = get_gps_coords(image_path)
        self.keypoints, self.descriptors = self._get_keypoints(keypoint_detector)

        self.global_pose = None
        self.global_pose_cache = None

        self.point_cloud = None
        self.point_cloud_needs_add = True

        self.gr_point_cloud_cache = None


    def set_key_frame_num(self, key_frame_num: int):
        """
        :param key_frame_num: key frame number to assign to frame
        """
        self.key_frame_num = key_frame_num


    def set_global_pose(self, pose: numpy.ndarray):
        """
        :param pose: 4x4 pose relative to the first key frame
        """
        self.global_pose = pose

        self.update_point_cloud()

    
    def georeference_point(
        self,
        x_position: float,
        y_position: float
    ) -> Tuple[float, float, float]:
        """
        :param x_position: x position of point
        :param y_position: y position of point
        :return: east-north-down position of the point in the world
        """
        depth = self._get_pixel_depth(x_position, y_position)

        fx = self.settings.camera.fx
        fy = self.settings.camera.fy
        cx = self.settings.camera.cx
        cy = self.settings.camera.cy
        """
        """


        return numpy.array([
            (x_position - cx) * depth / fx,
            (y_position - cy) * depth / fy,
            depth
        ])


    def update_point_cloud(self):
        """
        Used to update the point cloud only when the global pose is updated

        :raises ValueError: if this function is invoked with no global pose set
        """
        if self.global_pose is None:
            raise ValueError()

        # get georeferenced cloud
        self.point_cloud = self.get_georeferenced_point_cloud()

        # transform using pose
        self.point_cloud.transform(self.global_pose)

    
    def get_georeferenced_point_cloud(self) -> o3d.geometry.PointCloud:
        """
        The georeferenced point cloud cached prior to pose transformation. This
        cache never needs to be updated

        :return: Point cloud which has been georeferenced with respect to
            global pose
        """
        if self.gr_point_cloud_cache is not None:
            return copy.deepcopy(self.gr_point_cloud_cache)

        image_rgb = o3d.io.read_image(self.image_path)
        image_depth = o3d.geometry.Image(self._get_depth_image())
        if any(image_rgb.get_max_bound() != image_depth.get_max_bound()):
            raise ValueError("Image shape does not match camera parameters")

        image_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(image_rgb, image_depth, depth_scale=1.0, depth_trunc=numpy.inf, convert_rgb_to_intensity=False)
        camera_parameters = o3d.camera.PinholeCameraIntrinsic(**self.settings.camera.dict())
        
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(image_rgbd, camera_parameters)
        point_cloud = point_cloud.uniform_down_sample(self.settings.visualizer.downsample)

        self.gr_point_cloud_cache = copy.deepcopy(point_cloud)
        return point_cloud


    def get_corners(self, first_gps_coords: numpy.ndarray) -> List[Tuple[float, float]]:
        """
        TODO: rename to 'get footprint' and standardize order

        :param first_gps_coords: used to offset translation
        :return: list of world corner positions in LL LR UR UL order
        """
        corners = [
            (0, 0),
            (self.settings.camera.width, 0),
            (self.settings.camera.width, self.settings.camera.height),
            (0, self.settings.camera.height)
        ]
        left, up, zone, letter = utm.from_latlon(*first_gps_coords)

        new_corners = []
        for corner in corners:
            corner = self.georeference_point(*corner)
            corner = self.global_pose @ numpy.array(list(corner) + [1])
            corner = numpy.array([left, up]) + corner[:2]
            corner = utm.to_latlon(*corner[:2], zone, letter)

            new_corners.append(corner)

        return new_corners
    

    def _get_keypoints(self, keypoint_detector: cv2.Feature2D) -> Tuple[List[cv2.KeyPoint], numpy.ndarray]:
        """
        Break into blocks. This ensures that each block has at least n_features,
        so the distribution of features is spread evenly throughout the image area.
        This helps matching since it ensures features exist in the overlap

        :param keypoint_detector: detector used to detect keypoints
        :return: keypoint positions and descriptors
        """
        image = cv2.imread(self.image_path)
        image_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        keypoints = []

        block_args = (self.settings.keypoints.num_block_rows, self.settings.keypoints.num_block_columns)
        for y_start, y_end, x_start, x_end in blocks(image.shape[:2], *block_args):
            block = image_greyscale[y_start: y_end, x_start: x_end]
            block_keypoints = keypoint_detector.detect(block, None)  # TODO: check if can use detectAndCompute 

            for block_keypoint in block_keypoints:
                block_keypoint.pt = (  # keypoints are x,y
                    block_keypoint.pt[0] + x_start,
                    block_keypoint.pt[1] + y_start,
                )
            
            keypoints.extend(block_keypoints)

        keypoints, descriptors = keypoint_detector.compute(image_greyscale, keypoints)

        return keypoints, descriptors


    def _get_pixel_depth(self, x_position: float, y_position: float) -> float:
        """
        Depth is defined as the gps altitude for all pixels. Future work could
        integrate depth maps or project depth onto a flat surface prior

        :param x_position: x position of the pixel
        :param y_position: y position of the pixel
        :return: depth at specified pixel value
        """
        return self.gps_coords[2]


    def _get_depth_image(self) -> numpy.ndarray:
        """
        Construct an image of pixel depth at every pixel position

        :return: depth image
        """
        image_shape = (self.settings.camera.height, self.settings.camera.width)

        depth = self._get_pixel_depth(0, 0)  # TODO
        depth_image = numpy.full(image_shape, depth, dtype=numpy.float32)
        return depth_image