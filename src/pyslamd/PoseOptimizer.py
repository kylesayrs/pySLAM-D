import numpy
import teaserpp_python


class PoseOptimizerTeaser:
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


    def solve(self, src, dst):
        self.solver.solve(src, dst)

        solution = self.solver.getSolution()

        pose = numpy.hstack((solution.rotation, numpy.expand_dims(solution.translation, axis=1)))
        pose = numpy.concatenate((pose, numpy.expand_dims(numpy.array([0, 0, 0, 1]), axis=1).T), axis=0)

        return pose