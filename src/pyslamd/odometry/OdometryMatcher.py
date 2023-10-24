from typing import Tuple, Union, List

import os
import cv2
import numpy
import shapely
import pymap3d
import matplotlib.pyplot as plt

from pyslamd.Frame import Frame
from pyslamd.Settings import MatcherSettings
from pyslamd.odometry.PoseOptimizer import PoseOptimizerTeaser
from pyslamd.odometry.helpers import (
    make_keypoint_detector,
    make_keypoint_matcher,
    get_world_keypoints,
    get_matched_points,
    blocks
)
from pyslamd.odometry.overlap import get_overlap, get_overlap_masks
from pyslamd.utils.pose import get_translation, get_rotation
from pyslamd.utils.helpers import mask_list


OutliersType = Union[List[cv2.DMatch], None]


class OdometryMatcher():
    """
    Class which implements the odometry frame matching process

    :param settings: settings which define keypoint and visualization settings
    """
    def __init__(self, settings: MatcherSettings, use_gps: bool, use_imu: bool):
        self.settings = settings
        self.use_gps = use_gps
        self.use_imu = use_imu

        self.keypoint_detector =  make_keypoint_detector(settings.keypoints)
        self.keypoint_matcher = make_keypoint_matcher(settings.keypoints)


    def detect_assign_keypoints(self, frame: Frame):
        """
        Break into blocks. This ensures that each block has at least n_features,
        so the distribution of features is spread evenly throughout the image area.
        This helps matching since it ensures features exist in the overlap

        :param keypoint_detector: detector used to detect keypoints
        :return: keypoint positions and descriptors
        """
        image = cv2.imread(frame.image_path)
        image_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        keypoints = []

        block_args = (self.settings.keypoints.num_block_rows, self.settings.keypoints.num_block_columns)
        for y_start, y_end, x_start, x_end in blocks(image.shape[:2], *block_args):
            block = image_greyscale[y_start: y_end, x_start: x_end]
            block_keypoints = self.keypoint_detector.detect(block, None)  # TODO: check if can use detectAndCompute 

            for block_keypoint in block_keypoints:
                block_keypoint.pt = (  # keypoints are x,y
                    block_keypoint.pt[0] + x_start,
                    block_keypoint.pt[1] + y_start,
                )
            
            keypoints.extend(block_keypoints)

        keypoints, descriptors = self.keypoint_detector.compute(image_greyscale, keypoints)
        frame.assign_keypoints(keypoints, descriptors)
    

    def match_frames(self, frame: Frame, key_frames: List[Frame]) -> List[Union[shapely.Polygon, None]]:
        """
        :param frame: Frame which matches to key frames
        :param key_frames: List of key frames to match to
        :return: List whose ith entry is the relative pose with key_frame i or None
        """
        last_kf_num = len(key_frames) - 1

        relative_poses = []
        for key_frame in key_frames:
            is_candidate, overlap = self._is_match_candidate(key_frame, frame, last_kf_num)

            relative_poses.append(
                self._match_frame(frame, key_frame, overlap)
                if is_candidate
                else None
            )

        return relative_poses


    def _is_match_candidate(
        self,
        key_frame: Frame,
        frame: Frame,
        last_kf_num: int
    ) -> Tuple[bool, Union[shapely.Polygon, None]]:
        """
        If gps_match_bound is set, use gps_match_bound
        If gps and imu are available and overlap_matching is set, calculate overlap
        If overlap_matching is set but gps_match_bound is not, use overlap
        
        TODO: this needs a rewrite for readability (and probably correctness)

        :param key_frame: Potential match candidate
        :param frame: Frame to match from
        :param last_kf_num: last key frame index, used to check for last key frame
        :return: True if key_frame is a match candidate, false otherwise
            If the world overlap was computed, this is also returned
        """
        if self.settings.overlap_matching and not (self.use_gps and self.use_imu):
            raise ValueError("overlap_matching is set but use_gps or use_imu is not")

        if self.settings.gps_match_bound and not self.use_gps:
            raise ValueError("gps_match_bound is set but use_gps is not")


        overlap = None

        if self.settings.gps_match_bound is not None:
            relative_translation = pymap3d.geodetic2enu(*key_frame.gps_coords, *frame.gps_coords)
            distance = numpy.linalg.norm(relative_translation)
            if distance <= self.settings.gps_match_bound:
                if self.settings.overlap_matching:
                    overlap = get_overlap(frame, key_frame)

                return True, overlap

            return False, None

        if self.settings.overlap_matching:
            overlap = get_overlap(frame, key_frame)
            return overlap is not None, overlap

        if key_frame.key_frame_num == last_kf_num:
            if self.settings.overlap_matching:
                overlap = get_overlap(frame, key_frame)
            return True, overlap

        return False, None


    def _match_frame(
        self,
        frame: Frame,
        key_frame: Frame,
        overlap: Union[shapely.Polygon, None]
    ) -> Union[numpy.ndarray, None]:
        """
        TODO

        :param frame: destination frame
        :param key_frame: source frame
        :return: relative pose from source frame to destination frame
        """
        print(
            f"frame {frame.frame_num:2d} -> "
            f"key_frame {key_frame.frame_num:2d} | ",
            end=""
        )

        # detect keypoints
        if frame.keypoints is None or frame.descriptors is None:
            self.detect_assign_keypoints(frame)

        if key_frame.keypoints is None or key_frame.descriptors is None:
            self.detect_assign_keypoints(key_frame)

        # compute corresponding world points
        frame_points = get_world_keypoints(frame)
        key_frame_points = get_world_keypoints(key_frame)

        # prune points not in world overlap
        if overlap is not None:
            frame_mask, key_frame_mask = get_overlap_masks(
                frame_points, frame, key_frame_points, key_frame, overlap
            )

            frame_points = mask_list(frame_points, frame_mask)
            key_frame_points = mask_list(key_frame_points, key_frame_mask)
            frame_descriptors = frame.descriptors[frame_mask]
            key_frame_descriptors = key_frame.descriptors[key_frame_mask]
        else:
            frame_descriptors = frame.descriptors
            key_frame_descriptors = key_frame.descriptors

        if len(frame_descriptors) <= 0 or len(key_frame_descriptors) <= 0:
            print(
                "Not enough keypoints after pruning "
                f"{len(frame_descriptors)} / {self.settings.min_num_matches}"
            )
            return None

        # get matches (query, train)
        matches = self.keypoint_matcher.match(frame_descriptors, key_frame_descriptors)
        total_possible_matches = min(len(frame.descriptors), len(key_frame_descriptors))

        # print number of matches
        print(
            #f"overlap {overlap.area if overlap else 'None' } | "
            f"{len(matches):4d} / "
            f"{total_possible_matches:4d} "
            "matches | ",
            end=""
        )

        # check for minimum number of matches
        if len(matches) < self.settings.min_num_matches:
            print("min matches failure")
            return None

        # debug
        if self.settings.debug_matches:
            if overlap is not None:
                _draw_matches(frame, key_frame, matches, frame_mask, key_frame_mask)
                _draw_overlap(frame, key_frame)
            else:
                print("asdf")
                _draw_matches(frame, key_frame, matches)
            #exit(0)

        # use optimizer to find pose transform from key_frame to frame (src, dst)
        optimizer = PoseOptimizerTeaser()
        #print(numpy.mean(frame_points, axis=0))
        #print(numpy.mean(key_frame_points, axis=0))
        frame_points, key_frame_points = get_matched_points(frame_points, key_frame_points, matches)

        # starting at the keyframe position, how much do I move the frame
        # such that the keypoints match
        relative_pose = optimizer.solve(frame_points.T, key_frame_points.T)

        # check for extreme pose
        if not self.check_vo_pose(relative_pose):
            return None

        # reproject outliers onto key points
        if self.settings.reproject_outliers:
            self.reproject_outliers(frame)

        print("success")
        return relative_pose

    
    def check_vo_pose(self, pose: numpy.ndarray) -> bool:
        # check optimization
        if pose is None:
            print("optimization failure")
            return False

        # check translation threshold
        total_translation = numpy.linalg.norm(get_translation(pose))
        print(f"translation {int(total_translation):04d} | ", end="")
        if total_translation > self.settings.max_translation:
            print("translation failure")
            return False

        # TODO: print rotation in euler degrees and add a threshold

        return True


    def reproject_outliers(self, frame: Frame, key_frame_outliers: List[Union[List["cv2.DMatch"], None]]):
        """
        TODO

        :param frame: _description_
        :param key_frame_outliers: _description_
        :raises NotImplementedError: _description_
        """
        for key_frame, outliers in zip(self.key_frames, key_frame_outliers):
            if outliers is not None:
                raise NotImplementedError()


def _draw_matches(
    frame: Frame,
    key_frame: Frame,
    matches: List[cv2.DMatch],
    frame_mask = None,
    key_frame_mask = None
):
    """
    Debug function used to visualize keypoint matching
    """
    frame_image = cv2.imread(frame.image_path)
    key_frame_image = cv2.imread(key_frame.image_path)

    frame_keypoints = frame.keypoints
    key_frame_keypoints = key_frame.keypoints

    # asdfasdfasdfas
    #frame_mask = [
    #    keypoint.pt[0] < 2016# and keypoint.pt[1] < 1520
    #    for keypoint in frame.keypoints
    #]

    if frame_mask is not None:
        frame_keypoints = mask_list(frame.keypoints, frame_mask)
    if key_frame_mask is not None:
        key_frame_keypoints = mask_list(key_frame.keypoints, key_frame_mask)


    #matches_image = cv2.drawKeypoints(frame_image, frame_keypoints, 0, (255, 0, 0))

    #"""
    matches_image = cv2.drawMatches(
        frame_image,
        frame_keypoints,
        key_frame_image,
        key_frame_keypoints,
        matches,
        None
    )
    #"""

    file_name = f"{frame.frame_num}_{key_frame.frame_num}.png"
    file_path = os.path.join("matches", file_name)
    cv2.imwrite(file_path, matches_image)
    print(f"Wrote matches to {file_path}")


def _draw_overlap(frame: Frame, reference: Frame):
    from pyslamd.utils.pose import get_pose
    """
    COPY FROM overlap.py:get_overlap
    """
    corners = [
        (0, 0),
        (frame.settings.camera.width, 0),
        (frame.settings.camera.width, frame.settings.camera.height),
        (0, frame.settings.camera.height)
    ]

    extrinsic = get_pose(
        frame.get_imu_rotation(reference),
        frame.get_gps_translation(reference)
    )

    # TODO: if these are compuated with global reference, then they can be cached
    # which reduces runtime
    frame_corners = numpy.array([
        (extrinsic @ numpy.append(frame.image_to_world_point(*corner), 1))[:2]  # project to flat plane
        for corner in corners
    ])

    reference_corners = numpy.array([
        reference.image_to_world_point(*corner)[:2]  # project to flat plane
        for corner in corners
    ])

    frame_footprint = shapely.Polygon(frame_corners)
    reference_footprint = shapely.Polygon(reference_corners)

    overlap = shapely.intersection(frame_footprint, reference_footprint)

    # plot
    figure, axes = plt.subplots(1, 1)
    axes.plot(*frame_footprint.exterior.xy)
    axes.plot(*reference_footprint.exterior.xy)
    axes.plot(*overlap.exterior.xy)

    file_name = f"{frame.frame_num}_{reference.frame_num}.png"
    file_path = os.path.join("overlaps", file_name)
    figure.savefig(file_path)
    print(f"Wrote overlap to {file_path}")