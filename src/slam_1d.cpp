#include <random>

#include <ceres/ceres.h>
#include <gtest/gtest.h>
#include <Eigen/Dense>

// SizedCostFunction <<SIZE_RESIDUAL, SIZE_PARAM_BLOCK_0, SIZE_PARAM_BLOCK_1>>
// In our case:
// 1 dim - residual
// 1 dim - first parameter  (xi)
// 1 dim - second parameter (xj)
class Odometry1DFactor : public ceres::SizedCostFunction<1,1,1>
{
    public:
        Odometry1DFactor(double z, double var) :
            m_transform(z),
            m_covariance(var)
        {}

        virtual bool Evaluate   (
                                    double const* const* parameters, // Pointer to the parameter to be optimized (n.b is a list of pointers)
                                    double* residuals,               // Output: residual value
                                    double** jacobians               // Output: (optional jabobian)
                                ) const
        {
            // Define the parameters
            // Precedent node (i)
            double xi    = parameters[0][0];
            // Following node (i+1)
            double xj    = parameters[1][0];

            // Odometry residual is: (z - (x_j-x_i)) / var
            // where z is the noisy odometry measurement
            residuals[0] = (m_transform - (xj - xi))/m_covariance;

            // Compute jacobians
            if (jacobians != NULL)
            {
                if (jacobians[0]!= NULL)
                {
                    jacobians[0][0] = 1.0/m_covariance;
                }
                if (jacobians[1]!= NULL)
                {
                    jacobians[1][0] = -1.0/m_covariance;
                }
            }
            return true;
        }

    private:
        double m_transform;
        double m_covariance;
};

// SizedCostFunction <<SIZE_RESIDUAL, SIZE_PARAM_BLOCK_0, SIZE_PARAM_BLOCK_1>>
// In our case:
// 1 dim - residual
// 1 dim - first parameter  (range distance)
// 1 dim - second parameter (node pose)
class Range1DFactor : public ceres::SizedCostFunction<1,1,1>
{
    public:
        Range1DFactor(double z, double var) :
            m_range(z),
            m_covariance(var)
        {}

        virtual bool Evaluate   (
                                    double const* const* parameters, // Pointer to the parameter to be optimized (n.b is a list of pointers)
                                    double* residuals,               // Output: residual value
                                    double** jacobians               // Output: (optional jabobian)
                                ) const
        {
            // Range parameter
            double l = parameters[0][0];
            // Node parameter
            double x = parameters[1][0];
            // Range residual is: (z - (l-x))/var
            // where z is the noisy range measurement
            residuals[0] = (m_range - (l - x)) / m_covariance;

            // Compute jacobians
            if (jacobians != NULL)
            {
                if (jacobians[0]!= NULL)
                {
                    jacobians[0][0] = -1/m_covariance;
                }
                if (jacobians[1]!= NULL)
                {
                    jacobians[1][0] = 1/m_covariance;
                }
            }
            return true;
        }

    private:
        double m_range;
        double m_covariance;
};


// Measurement noise distribution
// Set random engine
std::default_random_engine gen;  
// Generate a normal gaussian error distribution with mean 0 and var 1
std::normal_distribution<double> normal_distribution(0.0,1.0);

// Odometry measurement covariance
double odometry_covariance = 1e-3;
// Range measurement covariance
double range_covariance = 1e-5;


TEST(Slam1D, SLAM)
{
    // Real Pose of the robot
    Eigen::Matrix<double, 8, 1> x_pose;
    x_pose << 0, 1, 2, 3, 4, 5, 6, 7;

    // Position of the fixed landmark in the space
    Eigen::Matrix<double, 3, 1> l_pose;
    l_pose << 10.0, 15.0, 13.0;

    // Initial estimate of the landmark
    Eigen::Matrix<double, 3, 1> l_hat;
    l_hat << 11.0, 12.0, 15.0;

    // Initial estimate of the robot pose
    Eigen::Matrix<double, 8, 1> x_hat;
    // Fix the first robot pose
    x_hat(0) = 0;

    // Set up the problem
    ceres::Problem problem;

    // Build up the graph
    for (int i = 0; i < x_pose.rows(); i++)
    {
        if (i > 0)
        {
            // Noisy odometry measurement 
            double odometry_increment = (x_pose(i) - x_pose(i-1)) + normal_distribution(gen)*std::sqrt(odometry_covariance);
            // Increment node position based on odometry
            x_hat(i) = x_hat(i-1) + odometry_increment;
            // Add to the problem an odometry constraint who link the pose of x_i-1 to x_i
            problem.AddResidualBlock(
                                        new Odometry1DFactor(odometry_increment, odometry_covariance), // cost function
                                        NULL,               // loss function
                                        x_hat.data() + i-1, // pointer to the precedent node
                                        x_hat.data() + i    // pointer to the following node
                                    );
        }

        // Landmark observation
        for (int j = 0; j < l_pose.rows(); j++)
        {
            // Noisy range measurement
            double z_bar = (l_pose[j] - x_pose[i]) + normal_distribution(gen)*std::sqrt(range_covariance);
            problem.AddResidualBlock(
                                        new Range1DFactor(z_bar, range_covariance), 
                                        NULL, 
                                        l_hat.data()+j, 
                                        x_hat.data() + i
                                    );
        }

    }

    // Define the optimization options
    ceres::Solver::Options options;
    options.max_num_iterations           = 25;
    options.linear_solver_type           = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;


    // Initial Error
    double e0 = (x_pose - x_hat).norm() + (l_pose - l_hat).norm();
     std::cout << "Initial Robot Position Estimate: "    << x_hat.transpose() << std::endl;
     std::cout << "Initial Landmark Position Estimate: " << l_hat.transpose() << std::endl;
     std::cout << "Inital Error: " << e0 << std::endl << std::endl;

    // Solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Final error
    double ef = (x_pose - x_hat).norm() + (l_pose - l_hat).norm();
     std::cout << "Final Robot Position Estimate: : " << x_hat.transpose() << std::endl;
     std::cout << "Final Landmark Position Estimate: " << l_hat.transpose() << std::endl;
     std::cout << "Final Error: " << ef << std::endl << std::endl;

    EXPECT_LE(ef, 0.17);
}