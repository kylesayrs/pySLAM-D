import numpy
import teaserpp_python

from pyslamd.utils.pose import get_pose


class PoseOptimizerTeaser:
    """
    The pose optimizer class, given a list of points and correspondences,
    performs point cloud registration to estimate the pose matrix which
    describes the relative transformation between the two point clouds

    TODO: move these settings to Settings.py
    """
    def __init__(self):
        self.NOISE_BOUND = 0.1  # 0.05
        self.solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        self.solver_params.cbar2 = 0.6  # 1
        self.solver_params.noise_bound = self.NOISE_BOUND
        self.solver_params.estimate_scaling = False
        self.solver_params.rotation_estimation_algorithm = \
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        self.solver_params.rotation_gnc_factor = 1.4
        self.solver_params.rotation_max_iterations = 200
        self.solver_params.rotation_cost_threshold = 1e-15
        self.solver = teaserpp_python.RobustRegistrationSolver(self.solver_params)


    def solve(self, src: numpy.ndarray, dst: numpy.ndarray) -> numpy.ndarray:
        """
        Performs point cloud registration

        :param src: source points in corresponding order
        :param dst: destination points in corresponding order
        :return: the pose matrix which describes the transformation between the
            two point clouds
        """
        self.solver.solve(src, dst)

        solution = self.solver.getSolution()

        return get_pose(solution.rotation, solution.translation)
