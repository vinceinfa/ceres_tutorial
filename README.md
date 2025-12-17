# ceres_tutorial

This repository contains a collection of small and focused C++ examples designed to learn and understand **Ceres Solver**.

The goal of this project is to explore the core concepts behind non-linear least squares optimization and how they are implemented in Ceres, with an emphasis on clarity and correctness rather than performance.

## Objectives

The examples in this repository aim to demonstrate:

- How to define **custom cost functions**
- How residuals and **Jacobian matrices** are structured in Ceres
- How to implement **analytical Jacobians**
- How to set up and solve optimization problems using `ceres::Problem`
- How to validate optimization results using **Google Test**

## Current Examples

- **1D Position Estimation**
  - Estimation of a scalar position from noisy observations
  - Implementation of `ceres::SizedCostFunction`
  - Manual computation of residuals and Jacobians
  - Verification of convergence using unit tests

## Project Structure

```
ceres_tutorial/
├── src/            # Source files
├── test/           # Unit tests (Google Test)
├── CMakeLists.txt
└── README.md
```

*(The structure may evolve as new examples are added.)*

## Build Instructions

```bash
mkdir build
cd build
cmake ..
make
ctest
```
