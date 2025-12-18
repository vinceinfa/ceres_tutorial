#include <random>

#include <ceres/ceres.h>
#include <gtest/gtest.h>
#include <Eigen/Dense>

/**
 * @brief Given a set of N noisy observation, retrieve an estimate x_hat of the real x 
 * trying to minimize the 3d position difference.
 * 
 */

// SizedCostFunction <number_of_residuals, parameters_dimension>
// In our case:
// 3 dim - residual
// 3 dim - parameter (x_hat)
class Pos3DFactor : public ceres::SizedCostFunction<3,3>
{
    public:

        /**
         * @brief Construct a new Pos 3D
         * 
         * @param x 
         * Observation of x
         */
        Pos3DFactor(Eigen::Vector3d x) :
            m_position(x)
        {}

        virtual bool Evaluate   (
                                    double const* const* parameters, // Pointer to the parameter to be optimized (n.b is a list of pointers)
                                    double* residuals,               // Output: residual value
                                    double** jacobians               // Output: (optional jabobian)
                                ) const
        {

            Eigen::Map<Eigen::Vector3d> current_residual(residuals);
            Eigen::Map<const Eigen::Vector3d> x_hat(parameters[0]);

            // Observation error is defined as e = (z - x_hat) where z is the observation and x_hat is the parameter
            current_residual = m_position - x_hat;

            // Compute Jacobian of the residual (for each parameter block a jacobian matrix has to be computed)
            // Jacobian has dimension: (residual_dimension x parameter_dimension)

            // Derivate of residual respect the parameter (d_e/d_xhat)
            if (jacobians != NULL)
            {
                if (jacobians[0]!= NULL)
                {
                    Eigen::Map<Eigen::Matrix3d, Eigen::RowMajor> jacobian(jacobians[0]);

                    jacobian = -1.0 * Eigen::Matrix3d::Identity();;
                }
            }

            // Residual Evaluation Done
            return true;
        }

    private:

        Eigen::Vector3d m_position;
};

// Define problem variables
// Real X value
Eigen::Vector3d x_pose {1.0, 2.0, 3.0};
// Number of observation
int num_observation  = 5000;
// Initial X estimate
Eigen::Vector3d init_x{4.0, 5.0, -2.0};

// Measurement noise distribution
// Fixed seed for random distribution
std::mt19937 gen(42); 
// Max measurement error 
double max_error = 1.0;   
// Generate an uniform error distribution 
std::uniform_real_distribution<double> noise_distribution(-max_error, max_error);

TEST(Position3D, AveragePoints)
{
    std::cout << "Test1" << std::endl;

    // Initial guess
    Eigen::Vector3d x_hat = init_x;

    // Initialize optimization problem
    ceres::Problem problem;

    // Add observations
    for (int i = 0; i < num_observation; i++)
    {
        // Generate a noisy measurement (z = x + noise) with a noise between -max_error and +max_error
        Eigen::Vector3d sample = x_pose + Eigen::Vector3d::Ones()*noise_distribution(gen);

        // For each measurement add a factor to the problem to compute the residual 
        problem.AddResidualBlock(new Pos3DFactor(sample), NULL, x_hat.data());
    }

    // Define solver options
    ceres::Solver::Options options;
    options.max_num_iterations           = 50;              // Maximum number of iteration
    options.linear_solver_type           = ceres::DENSE_QR; // Linear solver
    options.minimizer_progress_to_stdout = false;

    // Initial error
    double e0 = (x_pose - x_hat).norm();
    std::cout << "Initial Robot Position Estimate: " << x_hat.transpose() << std::endl;
    std::cout << "Inital Error: " << e0 << std::endl << std::endl;

    ceres::Solver::Summary summary;
    // Solve the problem
    ceres::Solve(options, &problem, &summary);

    // Final error
    double ef = (x_pose - x_hat).norm();
    std::cout << "Final Robot Position Estimate: : " << x_hat.transpose() << std::endl;
    std::cout << "Final Error: " << ef << std::endl << std::endl;
   
    std::cout << "Optimization tooks: " << summary.total_time_in_seconds << std::endl;

    // Gtest success condition: difference between the real x and the estimate lower than 0.01
    EXPECT_NEAR(x_hat[0], x_pose[0], 1e-2);
    EXPECT_NEAR(x_hat[1], x_pose[1], 1e-2);
    EXPECT_NEAR(x_hat[2], x_pose[2], 1e-2);
}


