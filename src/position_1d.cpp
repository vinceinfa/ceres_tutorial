#include <random>

#include <ceres/ceres.h>
#include <gtest/gtest.h>
#include <Eigen/Dense>

/**
 * @brief Given a set of N noisy observation, retrieve an estimate x_hat of the real x 
 * trying to minimize the 1d position difference.
 * 
 */

// SizedCostFunction <number_of_residuals, parameters_dimension>
// In our case:
// - 1 residual
// - 1 parameter (x_hat)
class Pos1DFactor : public ceres::SizedCostFunction<1,1>
{
    public:

        /**
         * @brief Construct a new Pos 1 D
         * 
         * @param x 
         * Observation of x
         */
        Pos1DFactor(double x) :
            m_position(x)
        {}

        virtual bool Evaluate   (
                                    double const* const* parameters, // Pointer to the parameter to be optimized (n.b is a list of pointers)
                                    double* residuals,               // Output: residual value
                                    double** jacobians               // Output: (optional jabobian)
                                ) const
        {
            // Observation error is defined as e = (z - x_hat) where z is the observation and x_hat is the parameter
            residuals[0] = m_position - parameters[0][0];

            // Compute Jacobian of the residual (for each parameter block a jacobian matrix has to be computed)
            // Jacobian has dimension: (residual_dimension x parameter_dimension)
            
            // Derivate of residual respect the parameter (d_e/d_xhat)
            if (jacobians != NULL)
            {
                if (jacobians[0]!= NULL)
                {
                    jacobians[0][0] = -1;
                }
            }

            // Residual Evaluation Done
            return true;
        }

    private:

        double m_position;
};

TEST(Position1D, AveragePoints)
{
    // Real X value
    double x             = 5.0;

    // Number of observation
    int num_observation  = 1000;

    // Initial X estimate
    double init_x        = 3.0;
    double x_hat         = init_x;

    // Initialize optimization problem
    ceres::Problem problem;

    // Add observations
    for (int i = 0; i < num_observation; i++)
    {
        // Generate a noisy measurement (z = x + noise)
        double sample = x + (rand() % 1000 - 500)/1000.0;
        // For each measurement add a factor to the problem to compute the residual 
        problem.AddResidualBlock(new Pos1DFactor(sample), NULL, &x_hat);
    }

    // Try to minimize the 

    ceres::Solver::Options options;
    options.max_num_iterations           = 25;              // Maximum number of iteration
    options.linear_solver_type           = ceres::DENSE_QR; // Linear solver
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    // Solve the problem
    ceres::Solve(options, &problem, &summary);

    std::cout << "X_hat: " << x_hat << std::endl;
    std::cout << "x: "     << x << std::endl;

    // Gtest success condition: difference between the real x and the estimate lower than 0.01
    EXPECT_NEAR(x_hat, x, 1e-2);
}