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

/**
 * @brief Define a struct template with only the residual computation without writing the jacobian of the residual.
 * It needs to AutoDiffCostFunction in test3.
 * 
 */
struct Pos1DAutoDiffFactor
{
    Pos1DAutoDiffFactor(double z) : 
    m_observation(z) 
    {}

    template <typename T>
    bool operator()(const T* const x_hat, T* residual) const
    {
        // residual = z - x_hat
        residual[0] = T(m_observation) - x_hat[0];
        return true;
    }

    private:
        double m_observation;
};

// Define problem variables
// Real X value
double x             = 5.0;
// Number of observation
int num_observation  = 1000;
// Initial X estimate
double init_x        = 3.0;

// Measurement noise distribution
// Fixed seed for random distribution
std::mt19937 gen(42); 
// Max measurement error 
double max_error = 0.5;   
// Generate an uniform error distribution 
std::uniform_real_distribution<double> noise_distribution(-max_error, max_error);

TEST(Position1D, AveragePoints)
{
    std::cout << "Test1" << std::endl;

    // Initial guess
    double x_hat = init_x;

    // Initialize optimization problem
    ceres::Problem problem;

    // Add observations
    for (int i = 0; i < num_observation; i++)
    {
        // Generate a noisy measurement (z = x + noise) with a noise between -max_error and +max_error
        double sample = x + noise_distribution(gen);
        // For each measurement add a factor to the problem to compute the residual 
        problem.AddResidualBlock(new Pos1DFactor(sample), NULL, &x_hat);
    }

    // Define solver options
    ceres::Solver::Options options;
    options.max_num_iterations           = 50;              // Maximum number of iteration
    options.linear_solver_type           = ceres::DENSE_QR; // Linear solver
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    // Solve the problem
    ceres::Solve(options, &problem, &summary);

    std::cout << "X_hat: " << x_hat << std::endl;
    std::cout << "x: "     << x << std::endl;
    std::cout << "Optimization tooks: " << summary.total_time_in_seconds << std::endl;

    // Gtest success condition: difference between the real x and the estimate lower than 0.01
    EXPECT_NEAR(x_hat, x, 1e-2);
}

TEST(Position1D, AveragePointsWithParameterBlock)
{
    std::cout << "Test2" << std::endl;

    // Initial guess
    double x_hat = init_x;

    // Initialize optimization problem
    ceres::Problem problem;
    // Add a parameter block, in this way ceres has not to declare a new parameter block inside the AddResidualBlock 
    // but it can use this parameter block alredy created
    problem.AddParameterBlock(&x_hat, 1);

    // Add observations
    for (int i = 0; i < num_observation; i++)
    {
        // Generate a noisy measurement (z = x + noise) with a noise between -max_error and +max_error
        double sample = x + noise_distribution(gen);
        // For each measurement add a factor to the problem to compute the residual 
        problem.AddResidualBlock(new Pos1DFactor(sample), NULL, &x_hat);
    }

    // Define solver options
    ceres::Solver::Options options;
    options.max_num_iterations           = 50;              // Maximum number of iteration
    options.linear_solver_type           = ceres::DENSE_QR; // Linear solver
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    // Solve the problem
    ceres::Solve(options, &problem, &summary);

    std::cout << "X_hat: " << x_hat << std::endl;
    std::cout << "x: "     << x << std::endl;
    std::cout << "Optimization tooks: " << summary.total_time_in_seconds << std::endl;

    // Gtest success condition: difference between the real x and the estimate lower than 0.01
    EXPECT_NEAR(x_hat, x, 1e-2);
}

// Try to elaborate the same test but with the ceres::AutoDiffCostFunction that doesn't need to compute jacobian
// Note: This version should be lower than the previous one

TEST(Position1D, AveragePoints_AutoDiff)
{
    std::cout << "Test3" << std::endl;

    // Initial guess
    double x_hat = init_x;

    // Initialize optimization problem
    ceres::Problem problem;

    // Add observations
    for (int i = 0; i < num_observation; i++)
    {
        // Generate a noisy measurement (z = x + noise) with a noise between -max_error and +max_error
        double sample = x + noise_distribution(gen);

        // Define the auto diff cost function
        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction< Pos1DAutoDiffFactor, // cost fun
                                                                              1,                   // number of residuals
                                                                              1                    // dimension of x_hat
                                                                            >(new Pos1DAutoDiffFactor(sample));

        problem.AddResidualBlock(cost_function, nullptr, &x_hat);
    }

    ceres::Solver::Options options;
    options.max_num_iterations           = 50;
    options.linear_solver_type           = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;

    // Solve the problem
    ceres::Solve(options, &problem, &summary);

    std::cout << "X_hat: " << x_hat << std::endl;
    std::cout << "x: "     << x << std::endl;
    std::cout << "Optimization tooks: " << summary.total_time_in_seconds << std::endl;

    EXPECT_NEAR(x_hat, x, 1e-2);
}
