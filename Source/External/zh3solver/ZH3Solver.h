#pragma once

// The MIT License
//
// Copyright (c) 2025 Activision Publishing, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ZH3_SOLVER_PRINT_ERRORS
#define ZH3_SOLVER_PRINT_ERRORS 0
#endif

#include "Eigen/Eigen"
#include "lbfgs.hpp"

// A solver for ZH3: linear spherical harmonics with an added quadratic component for the zonal axis. 
// See https://research.activision.com/publications/2024/05/ZH3_QUADRATIC_ZONAL_HARMONICS for details.

template <typename T, size_t Columns>
struct ZH3 {
    Eigen::Matrix<T, 4, Columns> linearSH;
    Eigen::Matrix<T, 1, Columns> zh3Coefficients;
};

template <typename T>
struct ZH3<T, 1> {
    Eigen::Matrix<T, 4, 1> linearSH;
    Eigen::Matrix<T, 1, 1> zh3Coefficients;

    inline Eigen::Matrix<T, 9, 1> expanded() const;
};

template <typename T>
struct ZH3<T, 3> {
    Eigen::Matrix<T, 4, 3> linearSH;
    Eigen::Matrix<T, 1, 3> zh3Coefficients;

    inline Eigen::Matrix<T, 9, 3> expanded(
        Eigen::Matrix<T, 3, 1> luminanceCoeffs = Eigen::Matrix<T, 3, 1>(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f),
        T sharedLuminanceAxisLerp = 0.0) const;
};

struct ZH3Solver {
    static constexpr float kPi = 3.141592653589793f;

    // Convert the L1 band of the passed-in SH to a 9x1 vector to match what the
    // solver expects. The vector is in RGBRGBRGB layout (l1m-1, l1m0, l1m1)
    template <typename T, size_t Rows>
    static inline Eigen::Matrix<T, 9, 1> shExtractFlattenedL1FromRGB(
        const Eigen::Matrix<T, Rows, 3>& sh)
    {
        Eigen::Matrix<T, 9, 1> result;
        for (size_t i = 0; i < 3; i += 1) {
            result(3 * i + 0) = sh(1 + i, 0);
            result(3 * i + 1) = sh(1 + i, 1);
            result(3 * i + 2) = sh(1 + i, 2);
        }
        return result;
    }

    // Copy a flattened L1 band in RGBRGBRGB (l1m-1, l1m0, l1m1) format to a
    // matrix-formatted SH.
    template <typename T, size_t Rows>
    static void shCopyFlattenedL1ToRGB(const Eigen::Matrix<T, 9, 1>& flattenedL1,
        Eigen::Matrix<T, Rows, 3>& outSH)
    {
        outSH.row(1) = Eigen::Matrix<T, 3, 1>(flattenedL1[0], flattenedL1[1], flattenedL1[2]);
        outSH.row(2) = Eigen::Matrix<T, 3, 1>(flattenedL1[3], flattenedL1[4], flattenedL1[5]);
        outSH.row(3) = Eigen::Matrix<T, 3, 1>(flattenedL1[6], flattenedL1[7], flattenedL1[8]);
    }

    // Compute the squared error between two SH.
    template <typename T, size_t Rows, size_t Columns>
    static Eigen::Matrix<T, 1, Columns> shError(
        const Eigen::Matrix<T, Rows, Columns>& a,
        const Eigen::Matrix<T, Rows, Columns>& b)
    {
        Eigen::Matrix<T, 1, Columns> result = Eigen::Matrix<T, 1, Columns>::Zero();
        for (size_t r = 0; r < Rows; r += 1) {
            for (size_t c = 0; c < Columns; c += 1) {
                double delta = a(r, c) - b(r, c);
                result(0, c) += delta * delta;
            }
        }
        return result;
    }

    // Evaluate the l=1 band for the real spherical harmonics.
    template <typename T>
    static inline Eigen::Matrix<T, 3, 1> shEvaluateL1(
        Eigen::Matrix<T, 3, 1> dir)
    {
        const T x = dir.x();
        const T y = dir.y();
        const T z = dir.z();

        Eigen::Matrix<T, 3, 1> result;
        result[0] = -sqrt(3.0f / (4.0f * kPi)) * y;
        result[1] = sqrt(3.0f / (4.0f * kPi)) * z;
        result[2] = -sqrt(3.0f / (4.0f * kPi)) * x;
        return result;
    }

    // Evaluate the l=2 band for the real spherical harmonics.
    template <typename T>
    static inline Eigen::Matrix<T, 5, 1> shEvaluateL2(
        Eigen::Matrix<T, 3, 1> dir)
    {
        const T x = dir.x();
        const T y = dir.y();
        const T z = dir.z();

        const T x2 = x * x;
        const T y2 = y * y;
        const T z2 = z * z;

        Eigen::Matrix<T, 5, 1> result;
        result[0] = sqrt(15.0f / (4.0f * kPi)) * y * x;
        result[1] = -sqrt(15.0f / (4.0f * kPi)) * y * z;
        result[2] = sqrt(5.0f / (16.0f * kPi)) * (3.0f * z2 - 1.0f);
        result[3] = -sqrt(15.0f / (4.0f * kPi)) * x * z;
        result[4] = sqrt(15.0f / (16.0f * kPi)) * (x2 - y2);
        return result;
    }

    // Evaluate the derivatives for the l=1 band with respect to the given
    // direction.
    template <typename T>
    static Eigen::Matrix<T, 5, 3> shEvaluateL2Jacobian(
        Eigen::Matrix<T, 3, 1> dir)
    {
        T dX[5] = { sqrt(15.0f / (4.0f * kPi)) * dir.y(), 0.0f, 0.0f,
            -sqrt(15.0f / (4.0f * kPi)) * dir.z(),
            sqrt(15.0f / (4.0f * kPi)) * dir.x() };
        T dY[5] = { sqrt(15.0f / (4.0f * kPi)) * dir.x(),
            -sqrt(15.0f / (4.0f * kPi)) * dir.z(), 0.f, 0.f,
            -sqrt(15.0f / (4.0f * kPi)) * dir.y() };
        T dZ[5] = { 0.0f, -sqrt(15.0f / (4.0f * kPi)) * dir.y(),
            1.5f * sqrt(5.0f / kPi) * dir.z(),
            -sqrt(15.0f / (4.0f * kPi)) * dir.x(), 0.0f };

        Eigen::Matrix<T, 5, 3> result;
        result.col(0) = Eigen::Matrix<T, 5, 1>(dX);
        result.col(1) = Eigen::Matrix<T, 5, 1>(dY);
        result.col(2) = Eigen::Matrix<T, 5, 1>(dZ);
        return result;
    }

    // Normalize the input vector (to make it unit length).
    template <typename T>
    static inline Eigen::Matrix<T, 3, 1> normalize(Eigen::Matrix<T, 3, 1> dir)
    {
        return dir / sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    }

    // Jacobian matrix for the normalize function with respect to the input vector
    // dir.
    template <typename T>
    static inline Eigen::Matrix<T, 3, 3> normalizeJacobian(
        Eigen::Matrix<T, 3, 1> dir)
    {
        T vecLen = sqrt(dir.dot(dir));
        T lengthCubed = vecLen * vecLen * vecLen;

        Eigen::Matrix<T, 3, 3> result;
        result(0, 0) = dir.y() * dir.y() + dir.z() * dir.z();
        result(0, 1) = -dir.x() * dir.y();
        result(0, 2) = -dir.x() * dir.z();
        result(1, 0) = -dir.x() * dir.y();
        result(1, 1) = dir.x() * dir.x() + dir.z() * dir.z();
        result(1, 2) = -dir.y() * dir.z();
        result(2, 0) = -dir.x() * dir.z();
        result(2, 1) = -dir.y() * dir.z();
        result(2, 2) = dir.x() * dir.x() + dir.y() * dir.y();
        return result / lengthCubed;
    }

    // Compute an RGBRGBRGB L1 SH vector to a luminance vector containing (l1m-1,
    // l1m0, l1m1) using the provided luminance weighting.
    template <typename T>
    static inline Eigen::Matrix<T, 3, 1> luminanceSH(
        Eigen::Matrix<T, 9, 1> shL1, Eigen::Matrix<T, 3, 1> lumWeights)
    {
        T l1MNeg1 = shL1[0] * lumWeights[0] + shL1[1] * lumWeights[1] + shL1[2] * lumWeights[2];
        T l1M0 = shL1[3] * lumWeights[0] + shL1[4] * lumWeights[1] + shL1[5] * lumWeights[2];
        T l1MPos1 = shL1[6] * lumWeights[0] + shL1[7] * lumWeights[1] + shL1[8] * lumWeights[2];
        return Eigen::Matrix<T, 3, 1>(l1MNeg1, l1M0, l1MPos1);
    }

    // Jacobian of the luminanceSH function with respect to shL1.
    template <typename T>
    static inline Eigen::Matrix<T, 3, 9> luminanceSHJacobian(
        Eigen::Matrix<T, 3, 1> lumWeights)
    {
        Eigen::Matrix<T, 3, 9> result = Eigen::Matrix<T, 3, 9>::Zero();

        result.col(0) = Eigen::Matrix<T, 3, 1>(lumWeights[0], T(0), T(0));
        result.col(1) = Eigen::Matrix<T, 3, 1>(lumWeights[1], T(0), T(0));
        result.col(2) = Eigen::Matrix<T, 3, 1>(lumWeights[2], T(0), T(0));
        result.col(3) = Eigen::Matrix<T, 3, 1>(T(0), lumWeights[0], T(0));
        result.col(4) = Eigen::Matrix<T, 3, 1>(T(0), lumWeights[1], T(0));
        result.col(5) = Eigen::Matrix<T, 3, 1>(T(0), lumWeights[2], T(0));
        result.col(6) = Eigen::Matrix<T, 3, 1>(T(0), T(0), lumWeights[0]);
        result.col(7) = Eigen::Matrix<T, 3, 1>(T(0), T(0), lumWeights[1]);
        result.col(8) = Eigen::Matrix<T, 3, 1>(T(0), T(0), lumWeights[2]);

        return result;
    }

    // Optimal linear direction from the input L1 SH band.
    template <typename T>
    static inline Eigen::Matrix<T, 3, 1> shDirection(
        Eigen::Matrix<T, 3, 1> shL1)
    {
        return Eigen::Matrix<T, 3, 1>(-shL1[2], -shL1[0], shL1[1]);
    }

    // Jacobian matrix of the shDirection function.
    template <typename T>
    static inline Eigen::Matrix<T, 3, 3> shDirectionJacobian()
    {
        Eigen::Matrix<T, 3, 3> result = Eigen::Matrix<T, 3, 3>::Zero();
        result(0, 2) = -1.0f;
        result(1, 0) = -1.0f;
        result(2, 1) = 1.0f;
        return result;
    }

    // Compute the normalized axis/zonal direction from the input L1 SH band.
    template <typename T>
    static inline Eigen::Matrix<T, 3, 1> axis(Eigen::Matrix<T, 3, 1> shL1)
    {
        return normalize(shDirection(shL1));
    }

    // Jacobian matrix of the axis function.
    template <typename T>
    static inline Eigen::Matrix<T, 3, 3> axisJacobian(
        Eigen::Matrix<T, 3, 1> shL1)
    {
        Eigen::Matrix<T, 3, 1> shDir = ZH3Solver::shDirection(shL1);
        Eigen::Matrix<T, 3, 3> shDirJacobian = ZH3Solver::shDirectionJacobian<T>();
        Eigen::Matrix<T, 3, 3> normalizeJacobian = ZH3Solver::normalizeJacobian(shDir);

        return normalizeJacobian * shDirJacobian;
    }

    // Compute the normalized shared luminance axis/zonal direction from the input
    // L1 SH band.
    template <typename T>
    static inline Eigen::Matrix<T, 3, 1> sharedLuminanceAxis(
        Eigen::Matrix<T, 9, 1> shL1, Eigen::Matrix<T, 3, 1> lumWeights)
    {
        return normalize(shDirection(luminanceSH(shL1, lumWeights)));
    }

    // Jacobian matrix of the sharedLuminanceAxis function with respect to the
    // input L1 SH band.
    template <typename T>
    static inline Eigen::Matrix<T, 3, 9> sharedLuminanceAxisJacobian(
        Eigen::Matrix<T, 9, 1> shL1, Eigen::Matrix<T, 3, 1> lumWeights)
    {
        Eigen::Matrix<T, 3, 1> lumSH = ZH3Solver::luminanceSH(shL1, lumWeights);
        Eigen::Matrix<T, 3, 1> shDir = ZH3Solver::shDirection(lumSH);
        Eigen::Matrix<T, 3, 9> luminanceSHJacobian = ZH3Solver::luminanceSHJacobian(lumWeights);
        Eigen::Matrix<T, 3, 3> shDirJacobian = ZH3Solver::shDirectionJacobian<T>();
        Eigen::Matrix<T, 3, 3> normalizeJacobian = ZH3Solver::normalizeJacobian(shDir);

        return normalizeJacobian * shDirJacobian * luminanceSHJacobian;
    }
};

struct ZH3PerChannelSolver : ZH3Solver {
    // Compute the ZH3 coefficient that represents targetL2 with the
    // lowest least-squares error, given that y2 is the l=2 band evaluated
    // in the zonal direction.
    // The approximation of targetL2 is given by zh3Coefficient * y2.
    template <typename T>
    static inline T zh3Coefficient(Eigen::Matrix<T, 5, 1> y2,
        Eigen::Matrix<T, 5, 1> targetL2)
    {
        return 4.0f * ZH3Solver::kPi / 5.0f * (y2.dot(targetL2));
    }

    // Compute the reconstruction error for target where the DC term is given by
    // target[0], the linear SH is given by shL1, and the zonal term is given by
    // the least-squares fit of the zonal L2 band to the direction given by shL1.
    template <typename T>
    static inline T zh3Error(Eigen::Matrix<T, 3, 1> shL1,
        Eigen::Matrix<T, 9, 1> target)
    {
        Eigen::Matrix<T, 3, 1> targetL1(target[1], target[2], target[3]);
        T l2Array[5] = { target[4], target[5], target[6], target[7], target[8] };
        Eigen::Matrix<T, 5, 1> targetL2(l2Array);

        T result = T(0.0f);

        Eigen::Matrix<T, 3, 1> deltaL1 = shL1 - targetL1;
        result += deltaL1.dot(deltaL1);

        Eigen::Matrix<T, 3, 1> axis = ZH3Solver::axis(shL1);
        Eigen::Matrix<T, 5, 1> y2 = ZH3Solver::shEvaluateL2(axis);
        Eigen::Matrix<T, 5, 1> fitL2 = zh3Coefficient(y2, targetL2) * y2;
        Eigen::Matrix<T, 5, 1> deltaL2 = fitL2 - targetL2;
        result += deltaL2.dot(deltaL2);

        return result;
    }

    // Jacobian matrix of zh3Error with respect to shL1.
    template <typename T>
    static inline Eigen::Matrix<T, 3, 1> zh3ErrorDerivative(
        Eigen::Matrix<T, 3, 1> shL1, Eigen::Matrix<T, 9, 1> target)
    {
        Eigen::Matrix<T, 3, 1> targetL1(target[1], target[2], target[3]);
        Eigen::Matrix<T, 5, 1> targetL2;
        targetL2[0] = target[4];
        targetL2[1] = target[5];
        targetL2[2] = target[6];
        targetL2[3] = target[7];
        targetL2[4] = target[8];

        Eigen::Matrix<T, 3, 1> result = Eigen::Matrix<T, 3, 1>::Zero();

        Eigen::Matrix<T, 3, 1> deltaL1 = (shL1 - targetL1);
        Eigen::Matrix<T, 3, 1> deltaL1Derivative = 2.0f * deltaL1;
        result += deltaL1Derivative;

        Eigen::Matrix<T, 3, 1> axis = ZH3Solver::axis(shL1);
        Eigen::Matrix<T, 5, 1> y2 = ZH3Solver::shEvaluateL2(axis);

        Eigen::Matrix<T, 5, 1> fitL2 = zh3Coefficient(y2, targetL2) * y2;
        Eigen::Matrix<T, 5, 1> deltaL2 = fitL2 - targetL2;

        Eigen::Matrix<T, 5, 3> y2Jacobian = ZH3Solver::shEvaluateL2Jacobian(axis);
        Eigen::Matrix<T, 3, 3> axisJacobian = ZH3Solver::axisJacobian(shL1);

        Eigen::Matrix<T, 5, 3> y2OfAxisJacobian = y2Jacobian * axisJacobian;

        // Product rule.
        Eigen::Matrix<T, 5, 5> deltaL2DerivativePartA = y2 * targetL2.transpose();
        T deltaL2DerivativePartB = targetL2.transpose() * y2;
        Eigen::Matrix<T, 5, 3> deltaL2Derivative = T(4.0f * ZH3Solver::kPi / 5.0f) * (deltaL2DerivativePartA + deltaL2DerivativePartB * Eigen::Matrix<T, 5, 5>::Identity()) * y2OfAxisJacobian;

        result += 2.0f * (deltaL2.transpose() * deltaL2Derivative);
        return result;
    }

    // Solver:
    typedef Eigen::Matrix<double, 4, 1> SH2;
    typedef Eigen::Matrix<double, 9, 1> SH3;

    static inline double zh3PerChannelCostFunction(void* targetPtr,
        const Eigen::VectorXd& x,
        Eigen::VectorXd& g)
    {
        SH3 target = *((SH3*)targetPtr);

        Eigen::Matrix<double, 3, 1> shL1 = x;

        double error = ZH3PerChannelSolver::zh3Error(shL1, target);
        g = ZH3PerChannelSolver::zh3ErrorDerivative(shL1, target);

        return error;
    }

    // Solve for the ZH3 parameters (linear SH and ZH3 coefficients)
    // that represent target with the lowest least-squares error.
    static inline ZH3<double, 1> solve(SH3 target, double* outError = nullptr)
    {
        SH2 fittedLinearSH;
        fittedLinearSH.row(0) = target.row(0);
        fittedLinearSH.row(1) = target.row(1);
        fittedLinearSH.row(2) = target.row(2);
        fittedLinearSH.row(3) = target.row(3);

        // Tetrahedron vertices
        Eigen::Vector3d testAxes[4] = { Eigen::Vector3d(1, 1, 1).normalized(),
            Eigen::Vector3d(1, -1, -1).normalized(),
            Eigen::Vector3d(-1, 1, -1).normalized(),
            Eigen::Vector3d(-1, -1, 1).normalized() };

        Eigen::VectorXd x = Eigen::Vector3d(target[1], target[2], target[3]);

        double bestError = zh3Error(Eigen::Vector3d(x[0], x[1], x[2]), target);

        for (const Eigen::Vector3d& axis : testAxes) {
            Eigen::Vector3d linearSHDir = ZH3Solver::shEvaluateL1(axis);
            SH2 linearSH = {};
            linearSH.row(0) = target.row(0);

            Eigen::Vector3d l1Vec = Eigen::Vector3d(-target(3), -target(1), target(2));
            double zonalScale = axis.dot(l1Vec) / sqrt(0.75f / ZH3Solver::kPi);

            linearSH(1) = linearSHDir[0] * zonalScale;
            linearSH(2) = linearSHDir[1] * zonalScale;
            linearSH(3) = linearSHDir[2] * zonalScale;

            x = Eigen::Vector3d(linearSH[1], linearSH[2], linearSH[3]);

            double error = zh3Error(Eigen::Vector3d(x[0], x[1], x[2]), target);

            lbfgs::lbfgs_parameter_t param;

            int result = lbfgs::lbfgs_optimize(x, error, zh3PerChannelCostFunction,
                nullptr, nullptr, &target, param);

#if ZH3_SOLVER_PRINT_ERRORS
            if (result < 0) {
                printf("LBFGS failed to converge: %s. Error is %lf.\n",
                    lbfgs::lbfgs_strerror(result), error);
            }
#else
            (void)result;
#endif

            if (error < bestError) {
                fittedLinearSH[1] = x[0];
                fittedLinearSH[2] = x[1];
                fittedLinearSH[3] = x[2];
                bestError = error;
            }
        }

        if (outError) {
            *outError = bestError;
        }

        ZH3<double, 1> fittedZH3;
        fittedZH3.linearSH = fittedLinearSH;

        Eigen::Vector3d axis = ZH3Solver::axis(Eigen::Vector3d(
            fittedZH3.linearSH[1], fittedZH3.linearSH[2], fittedZH3.linearSH[3]));
        Eigen::Matrix<double, 5, 1> y2 = ZH3Solver::shEvaluateL2(axis);
        double targetL2[5] = { target(4), target(5), target(6), target(7),
            target(8) };
        fittedZH3.zh3Coefficients(0) = ZH3PerChannelSolver::zh3Coefficient(
            y2, Eigen::Matrix<double, 5, 1>(targetL2));

        return fittedZH3;
    }

    // Solve for the ZH3 parameters (linear SH and ZH3 coefficients)
    // that respresent target with the lowest least-squares error.
    static inline ZH3<float, 1> solve(Eigen::Matrix<float, 9, 1> target,
        double* outError = nullptr)
    {
        SH3 targetDouble = target.cast<double>();

        ZH3<double, 1> resultDouble = solve(targetDouble, outError);

        ZH3<float, 1> result;
        result.linearSH = resultDouble.linearSH.cast<float>();
        result.zh3Coefficients = resultDouble.zh3Coefficients.cast<float>();
        return result;
    }

    // Solve for the ZH3 parameters (linear SH and ZH3 coefficients)
    // that respresent target with the lowest least-squares error.
    template <typename T>
    static inline ZH3<T, 3> solve(
        Eigen::Matrix<T, 9, 3> target,
        Eigen::Matrix<T, 3, 1> luminanceWeightingCoeffs = Eigen::Matrix<T, 3, 1>(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f),
        double* outError = nullptr)
    {
        double error = 0.0;

        ZH3<T, 3> result;
        for (size_t c = 0; c < 3; c += 1) {
            double channelError = 0.0;
            SH3 targetDouble = target.col(c).template cast<double>();
            ZH3<double, 1> resultDouble = solve(targetDouble, &channelError);

            result.linearSH.col(c) = resultDouble.linearSH.template cast<T>();
            result.zh3Coefficients(c) = T(resultDouble.zh3Coefficients(0));

            error += double(luminanceWeightingCoeffs[c]) * channelError;
        }

        if (outError) {
            *outError = error;
        }

        return result;
    }
};

struct ZH3SharedLuminanceSolver : ZH3Solver {
    // Compute the ZH3 coefficient that represents targetL2 with the
    // lowest least-squares error, given that y2 is the l=2 band evaluated
    // in the zonal direction.
    // The approximation of targetL2 is given by zh3Coefficient * y2.
    template <typename T>
    static inline T zh3Coefficient(Eigen::Matrix<T, 5, 1> y2,
        Eigen::Matrix<T, 5, 1> targetL2)
    {
        return 4.0f * ZH3Solver::kPi / 5.0f * (y2.dot(targetL2));
    }

    // Compute the reconstruction error for target where the DC term is given by
    // target[0], the linear SH is given by shL1, and the zonal term is given by
    // the least-squares fit of the zonal L2 band to the direction given by
    // the shared luminance axis from shL1.
    // channelWeights gives the weight for the channel currently being evaluated
    // (so e.g. (1, 0, 0) for red), and lumWeights gives the weighting for each L1
    // channel in the shared luminance axis.
    template <typename T>
    static inline T zh3Error(Eigen::Matrix<T, 9, 1> shL1,
        Eigen::Matrix<T, 9, 1> target,
        Eigen::Matrix<T, 3, 1> channelWeights,
        Eigen::Matrix<T, 3, 1> lumWeights)
    {
        Eigen::Matrix<T, 3, 1> targetL1(target[1], target[2], target[3]);
        T l2Array[5] = { target[4], target[5], target[6], target[7], target[8] };
        Eigen::Matrix<T, 5, 1> targetL2(l2Array);

        T result = T(0.0f);

        Eigen::Matrix<T, 3, 1> deltaL1 = ZH3Solver::luminanceSH(shL1, channelWeights) - targetL1;
        result += deltaL1.dot(deltaL1);

        Eigen::Matrix<T, 3, 1> axis = ZH3Solver::sharedLuminanceAxis(shL1, lumWeights);
        Eigen::Matrix<T, 5, 1> y2 = ZH3Solver::shEvaluateL2(axis);
        Eigen::Matrix<T, 5, 1> fitL2 = zh3Coefficient(y2, targetL2) * y2;
        Eigen::Matrix<T, 5, 1> deltaL2 = fitL2 - targetL2;
        result += deltaL2.dot(deltaL2);

        return result;
    }

    // Jacobian matrix of zh3Error with respect to shL1.
    template <typename T>
    static inline Eigen::Matrix<T, 9, 1> zh3ErrorDerivative(
        Eigen::Matrix<T, 9, 1> shL1, Eigen::Matrix<T, 9, 1> target,
        Eigen::Matrix<T, 3, 1> channelWeights,
        Eigen::Matrix<T, 3, 1> lumWeights)
    {
        Eigen::Matrix<T, 3, 1> targetL1(target[1], target[2], target[3]);
        Eigen::Matrix<T, 5, 1> targetL2;
        targetL2[0] = target[4];
        targetL2[1] = target[5];
        targetL2[2] = target[6];
        targetL2[3] = target[7];
        targetL2[4] = target[8];

        Eigen::Matrix<T, 9, 1> result = Eigen::Matrix<T, 9, 1>::Zero();

        Eigen::Matrix<T, 3, 1> deltaL1 = (ZH3Solver::luminanceSH(shL1, channelWeights) - targetL1);
        Eigen::Matrix<T, 9, 1> deltaL1Derivative = ZH3Solver::luminanceSHJacobian(channelWeights).transpose() * deltaL1;
        result += 2.0f * deltaL1Derivative;

        Eigen::Matrix<T, 3, 1> axis = ZH3Solver::sharedLuminanceAxis(shL1, lumWeights);
        Eigen::Matrix<T, 5, 1> y2 = ZH3Solver::shEvaluateL2(axis);

        Eigen::Matrix<T, 5, 1> fitL2 = zh3Coefficient(y2, targetL2) * y2;
        Eigen::Matrix<T, 5, 1> deltaL2 = fitL2 - targetL2;

        Eigen::Matrix<T, 5, 3> y2Jacobian = ZH3Solver::shEvaluateL2Jacobian(axis);
        Eigen::Matrix<T, 3, 9> axisJacobian = ZH3Solver::sharedLuminanceAxisJacobian(shL1, lumWeights);

        Eigen::Matrix<T, 5, 9> y2OfAxisJacobian = y2Jacobian * axisJacobian;

        // Product rule.
        Eigen::Matrix<T, 5, 5> deltaL2DerivativePartA = y2 * targetL2.transpose();
        T deltaL2DerivativePartB = targetL2.transpose() * y2;
        Eigen::Matrix<T, 5, 9> deltaL2Derivative = T(4.0f * ZH3Solver::kPi / 5.0f) * (deltaL2DerivativePartA + deltaL2DerivativePartB * Eigen::Matrix<T, 5, 5>::Identity()) * y2OfAxisJacobian;

        result += 2.0f * (deltaL2.transpose() * deltaL2Derivative);

        return result;
    }

    // Solver:
    typedef Eigen::Matrix<double, 4, 1> SH2;
    typedef Eigen::Matrix<double, 4, 3> SH2RGB;
    typedef Eigen::Matrix<double, 9, 1> SH3;
    typedef Eigen::Matrix<double, 9, 3> SH3RGB;

    struct SharedLuminanceCostFunctionParams {
        SH3RGB target;
        Eigen::Vector3d
            luminanceCoeffs; // relative weighting for R/G/B; should sum to 1.
        double axisLuminanceLerp;
    };

    static inline double zh3SharedLuminanceError(Eigen::Matrix<double, 9, 1> shL1,
        SH3RGB target,
        Eigen::Vector3d luminanceCoeffs,
        double axisLuminanceLerp)
    {
        SH3 targetR;
        SH3 targetG;
        SH3 targetB;

        for (size_t i = 0; i < 9; i += 1) {
            targetR[i] = target(i, 0);
            targetG[i] = target(i, 1);
            targetB[i] = target(i, 2);
        }

        Eigen::Vector3d rLumCoeffs = (1.0 - axisLuminanceLerp) * Eigen::Vector3d(1.0, 0.0, 0.0) + axisLuminanceLerp * luminanceCoeffs;
        Eigen::Vector3d gLumCoeffs = (1.0 - axisLuminanceLerp) * Eigen::Vector3d(0.0, 1.0, 0.0) + axisLuminanceLerp * luminanceCoeffs;
        Eigen::Vector3d bLumCoeffs = (1.0 - axisLuminanceLerp) * Eigen::Vector3d(0.0, 0.0, 1.0) + axisLuminanceLerp * luminanceCoeffs;

        return Eigen::Vector3d(
            ZH3SharedLuminanceSolver::zh3Error(
                shL1, targetR, Eigen::Vector3d(1.0, 0.0, 0.0), rLumCoeffs),
            ZH3SharedLuminanceSolver::zh3Error(
                shL1, targetG, Eigen::Vector3d(0.0, 1.0, 0.0), gLumCoeffs),
            ZH3SharedLuminanceSolver::zh3Error(
                shL1, targetB, Eigen::Vector3d(0.0, 0.0, 1.0), bLumCoeffs))
            .dot(luminanceCoeffs);
    }

    static inline double zh3SharedLuminanceCostFunction(void* targetPtr,
        const Eigen::VectorXd& x,
        Eigen::VectorXd& g)
    {
        SharedLuminanceCostFunctionParams params = *((SharedLuminanceCostFunctionParams*)targetPtr);

        Eigen::Matrix<double, 9, 1> shL1 = x;

        Eigen::Matrix<double, 9, 1> targetR;
        Eigen::Matrix<double, 9, 1> targetG;
        Eigen::Matrix<double, 9, 1> targetB;

        for (size_t i = 0; i < 9; i += 1) {
            targetR[i] = params.target(i, 0);
            targetG[i] = params.target(i, 1);
            targetB[i] = params.target(i, 2);
        }

        Eigen::Vector3d rLumCoeffs = (1.0 - params.axisLuminanceLerp) * Eigen::Vector3d(1.0, 0.0, 0.0) + params.axisLuminanceLerp * params.luminanceCoeffs;
        Eigen::Vector3d gLumCoeffs = (1.0 - params.axisLuminanceLerp) * Eigen::Vector3d(0.0, 1.0, 0.0) + params.axisLuminanceLerp * params.luminanceCoeffs;
        Eigen::Vector3d bLumCoeffs = (1.0 - params.axisLuminanceLerp) * Eigen::Vector3d(0.0, 0.0, 1.0) + params.axisLuminanceLerp * params.luminanceCoeffs;

        double error = Eigen::Vector3d(
            ZH3SharedLuminanceSolver::zh3Error(
                shL1, targetR, Eigen::Vector3d(1.0, 0.0, 0.0), rLumCoeffs),
            ZH3SharedLuminanceSolver::zh3Error(
                shL1, targetG, Eigen::Vector3d(0.0, 1.0, 0.0), gLumCoeffs),
            ZH3SharedLuminanceSolver::zh3Error(
                shL1, targetB, Eigen::Vector3d(0.0, 0.0, 1.0), bLumCoeffs))
                           .dot(params.luminanceCoeffs);

        g = params.luminanceCoeffs[0] * ZH3SharedLuminanceSolver::zh3ErrorDerivative(shL1, targetR, Eigen::Vector3d(1.0, 0.0, 0.0), rLumCoeffs);
        g += params.luminanceCoeffs[1] * ZH3SharedLuminanceSolver::zh3ErrorDerivative(shL1, targetG, Eigen::Vector3d(0.0, 1.0, 0.0), gLumCoeffs);
        g += params.luminanceCoeffs[2] * ZH3SharedLuminanceSolver::zh3ErrorDerivative(shL1, targetB, Eigen::Vector3d(0.0, 0.0, 1.0), bLumCoeffs);

        return error;
    }

    // Solve for the ZH3 parameters (linear SH and ZH3 coefficients)
    // that respresent target with the lowest least-squares error,
    // where the zonal axis for the L2 band is given for each channel by
    // lerp(optimalLinearAxis(channel), optimalLinearAxis(luminance(l1SH)),
    // axisLuminanceLerp). luminanceCoeffs gives the relative weighting for R/G/B,
    // and should add to one.
    static inline ZH3<double, 3> solve(SH3RGB target,
        Eigen::Vector3d luminanceCoeffs,
        double axisLuminanceLerp,
        double* outError = nullptr)
    {
        SH2RGB fittedLinearSH;
        fittedLinearSH.row(0) = target.row(0);
        fittedLinearSH.row(1) = target.row(1);
        fittedLinearSH.row(2) = target.row(2);
        fittedLinearSH.row(3) = target.row(3);

        // Tetrahedron vertices
        Eigen::Vector3d testAxes[4] = { Eigen::Vector3d(1, 1, 1).normalized(),
            Eigen::Vector3d(1, -1, -1).normalized(),
            Eigen::Vector3d(-1, 1, -1).normalized(),
            Eigen::Vector3d(-1, -1, 1).normalized() };

        Eigen::VectorXd x = ZH3Solver::shExtractFlattenedL1FromRGB<double, 9>(target);

        double bestError = zh3SharedLuminanceError(x, target, luminanceCoeffs, axisLuminanceLerp);

        for (const Eigen::Vector3d& axis : testAxes) {
            Eigen::Vector3d linearSHDir = ZH3Solver::shEvaluateL1(axis);
            SH2RGB linearSHRGB = {};
            linearSHRGB.row(0) = target.row(0);
            for (size_t c = 0; c < 3; c += 1) {
                Eigen::Vector3d l1Vec = Eigen::Vector3d(-target(3, c), -target(1, c), target(2, c));
                double zonalScale = axis.dot(l1Vec) / sqrt(0.75f / ZH3Solver::kPi);

                linearSHRGB(1, c) = linearSHDir[0] * zonalScale;
                linearSHRGB(2, c) = linearSHDir[1] * zonalScale;
                linearSHRGB(3, c) = linearSHDir[2] * zonalScale;
            }

            x = ZH3Solver::shExtractFlattenedL1FromRGB<double, 4>(linearSHRGB);

            double error = zh3SharedLuminanceError(x, target, luminanceCoeffs,
                axisLuminanceLerp);

            lbfgs::lbfgs_parameter_t param;
            SharedLuminanceCostFunctionParams costFunctionParams = {
                target, luminanceCoeffs, axisLuminanceLerp
            };

            int result = lbfgs::lbfgs_optimize(x, error, zh3SharedLuminanceCostFunction,
                nullptr, nullptr, &costFunctionParams, param);

#if ZH3_SOLVER_PRINT_ERRORS
            if (result < 0) {
                printf("LBFGS failed to converge: %s. Error is %lf.\n",
                    lbfgs::lbfgs_strerror(result), error);
            }

#else
            (void)result;
#endif

            if (error < bestError) {
                ZH3Solver::shCopyFlattenedL1ToRGB<double, 4>(x, fittedLinearSH);
                bestError = error;
            }
        }

        if (outError) {
            *outError = bestError;
        }

        ZH3<double, 3> fittedZH3;
        fittedZH3.linearSH = fittedLinearSH;

        Eigen::Matrix<double, 9, 1> flattenedFittedLinearSH = ZH3Solver::shExtractFlattenedL1FromRGB<double, 4>(fittedLinearSH);

        for (size_t c = 0; c < 3; c += 1) {
            Eigen::Vector3d channelCoeff = Eigen::Vector3d(
                c == 0 ? 1.0 : 0.0, c == 1 ? 1.0 : 0.0, c == 2 ? 1.0 : 0.0);
            Eigen::Vector3d lumCoeffs = (1.0 - axisLuminanceLerp) * channelCoeff + axisLuminanceLerp * luminanceCoeffs;
            Eigen::Vector3d axis = ZH3Solver::sharedLuminanceAxis(flattenedFittedLinearSH, lumCoeffs);
            Eigen::Matrix<double, 5, 1> y2 = ZH3Solver::shEvaluateL2(axis);
            double targetL2[5] = { target(4, c), target(5, c), target(6, c),
                target(7, c), target(8, c) };
            fittedZH3.zh3Coefficients[c] = ZH3SharedLuminanceSolver::zh3Coefficient(
                y2, Eigen::Matrix<double, 5, 1>(targetL2));
        }

        return fittedZH3;
    }

    // Solve for the ZH3 parameters (linear SH and ZH3 coefficients)
    // that respresent target with the lowest least-squares error,
    // where the zonal axis for the L2 band is given for each channel by
    // lerp(optimalLinearAxis(channel), optimalLinearAxis(luminance(l1SH)),
    // axisLuminanceLerp). luminanceCoeffs gives the relative weighting for R/G/B,
    // and should add to one.
    static inline ZH3<float, 3> solve(Eigen::Matrix<float, 9, 3> target,
        Eigen::Vector3f luminanceCoeffs,
        float axisLuminanceLerp,
        double* outError = nullptr)
    {
        SH3RGB targetDouble = target.cast<double>();

        ZH3<double, 3> resultDouble = solve(targetDouble, luminanceCoeffs.cast<double>(),
            double(axisLuminanceLerp), outError);

        ZH3<float, 3> result;
        result.linearSH = resultDouble.linearSH.cast<float>();
        result.zh3Coefficients = resultDouble.zh3Coefficients.cast<float>();
        return result;
    }
};

struct ZH3HallucinateSolver : ZH3Solver {
    typedef Eigen::Matrix<double, 4, 1> SH2;
    typedef Eigen::Matrix<double, 4, 3> SH2RGB;
    typedef Eigen::Matrix<double, 9, 1> SH3;
    typedef Eigen::Matrix<double, 9, 3> SH3RGB;

    // Compute the ratio of the L1 band projected onto axis
    // to the DC/L0 coefficient.
    template <typename T>
    static inline T zh3Ratio(Eigen::Matrix<T, 3, 1> shL1,
        Eigen::Matrix<T, 3, 1> axis, T shL0)
    {
        return ZH3Solver::shDirection(shL1).dot(axis) / shL0;
    }

    // Jacobian matrix of zh3Ratio with respect to shL1.
    template <typename T>
    static inline Eigen::Matrix<T, 1, 3> zh3RatioJacobianDSHL1(
        Eigen::Matrix<T, 3, 1> axis, T shL0)
    {
        T dMNeg1 = -axis[1] / shL0;
        T dM0 = axis[2] / shL0;
        T dMPos1 = -axis[0] / shL0;
        return Eigen::Matrix<T, 1, 3>(dMNeg1, dM0, dMPos1);
    }

    // Jacobian matrix of zh3Ratio with respect to axis.
    template <typename T>
    static inline Eigen::Matrix<T, 1, 3> zh3RatioJacobianDsharedLuminanceAxis(
        Eigen::Matrix<T, 3, 1> shL1, T shL0)
    {
        T dX = -shL1[2] / shL0;
        T dY = -shL1[0] / shL0;
        T dZ = shL1[1] / shL0;
        return Eigen::Matrix<T, 1, 3>(dX, dY, dZ);
    }

    // Estimate the ZH3 coefficient from the provided ratio.
    // irradiance should be true if the ratio and output are for
    // an SH representing irradiance.
    // The equation is given by a quadratic curve fit to production
    // SH; if your data is different, you can replace this with a different
    // approximation.
    template <typename T>
    static inline T zh3Coefficient(T ratio, T shL0, bool irradiance)
    {
        if (irradiance) {
            return T(0.25) * zh3Coefficient(ratio * T(1.5), shL0, false);
        } else {
            return shL0 * T(sqrt(4.0 * ZH3Solver::kPi / 5.0)) * (ratio * (T(0.08f) + T(0.6f) * ratio));
        }
    }

    // Derivative of zh3Coefficient with respect to ratio.
    template <typename T>
    static inline T zh3CoefficientDerivative(T ratio, T shL0, bool irradiance)
    {
        if (irradiance) {
            return T(0.25) * zh3CoefficientDerivative(ratio * T(1.5), shL0, false);
        } else {
            return shL0 * T(sqrt(4.0 * ZH3Solver::kPi / 5.0)) * (T(0.08f) + T(0.6f * 2.0f) * ratio);
        }
    }

    template <typename T>
    static inline ZH3<T, 1> hallucinateZH3(Eigen::Matrix<T, 4, 1> linearSH,
        bool isIrradiance)
    {
        ZH3<T, 1> result;
        result.linearSH = linearSH;

        // Compute the hallucinated ZH3 coefficient.
        Eigen::Vector3d shL1 = Eigen::Vector3d(
            double(linearSH(1, 0)), double(linearSH(2, 0)), double(linearSH(3, 0)));
        Eigen::Vector3d axis = ZH3Solver::axis(shL1);

        double ratio = ZH3HallucinateSolver::zh3Ratio(shL1, axis, linearSH(0, c));
        double fittedZH3Coeff = ZH3HallucinateSolver::zh3Coefficient(
            ratio, linearSH(0, c), isIrradiance);
        result.zh3Coefficients(0) = T(fittedZH3Coeff);

        return result;
    }

    template <typename T>
    static inline ZH3<T, 3> hallucinateZH3(
        Eigen::Matrix<T, 4, 3> linearSH, bool isIrradiance,
        Eigen::Matrix<T, 3, 1> luminanceWeightingCoeffs = Eigen::Matrix<T, 3, 1>(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f),
        T sharedLuminanceAxisLerp = T(0.0))
    {
        ZH3<T, 1> result;
        result.linearSH = linearSH;

        // Compute the hallucinated ZH3 coefficients.
        Eigen::Matrix<double, 9, 1> shL1Flat = ZH3Solver::shExtractFlattenedL1FromRGB(linearSH.cast<double>());
        for (size_t c = 0; c < 3; c += 1) {
            Eigen::Vector3d channelWeights = Eigen::Vector3d(
                c == 0 ? 1.0 : 0.0, c == 1 ? 1.0 : 0.0, c == 2 ? 1.0 : 0.0);
            Eigen::Vector3d luminanceWeights = (1.0 - sharedLuminanceAxisLerp) * channelWeights + luminanceWeightingCoeffs * sharedLuminanceAxisLerp;

            Eigen::Vector3d axis = ZH3Solver::sharedLuminanceAxis(shL1Flat, luminanceWeights);
            Eigen::Vector3d solvedSHL1 = ZH3Solver::luminanceSH(shL1Flat, channelWeights);

            double ratio = ZH3HallucinateSolver::zh3Ratio(solvedSHL1, axis, linearSH(0, c));
            double fittedZH3Coeff = ZH3HallucinateSolver::zh3Coefficient(
                ratio, linearSH(0, c), isIrradiance);
            result.zh3Coefficients(c) = T(fittedZH3Coeff);
        }
        return result;
    }

    // Compute the SH3 expansion of the linear SH using a hallucinated ZH3
    // coefficient.
    template <typename T>
    static inline Eigen::Matrix<T, 9, 1> expandSH(Eigen::Matrix<T, 4, 1> shL1,
        bool isIrradiance)
    {
        Eigen::Matrix<T, 9, 1> result;
        result.row(0) = shL1.row(0);
        result.row(1) = shL1.row(1);
        result.row(2) = shL1.row(2);
        result.row(3) = shL1.row(3);

        Eigen::Matrix<T, 3, 1> shL1Band = Eigen::Matrix<T, 3, 1>(shL1[1], shL1[2], shL1[3]);
        Eigen::Matrix<T, 3, 1> axis = ZH3Solver::axis(shL1Band);
        Eigen::Matrix<T, 5, 1> y2 = ZH3Solver::shEvaluateL2(axis);

        T ratio = ZH3HallucinateSolver::zh3Ratio(shL1Band, axis, shL1(0));
        T fittedZH3Coeff = ZH3HallucinateSolver::zh3Coefficient(ratio, shL1(0), isIrradiance);

        result(4) = y2[0] * fittedZH3Coeff;
        result(5) = y2[1] * fittedZH3Coeff;
        result(6) = y2[2] * fittedZH3Coeff;
        result(7) = y2[3] * fittedZH3Coeff;
        result(8) = y2[4] * fittedZH3Coeff;
        return result;
    }

    // Compute the SH3 expansion of the linear SH using hallucinated ZH3
    // coefficients. luminanceCoeffs gives the weighting of R/G/B for the shared
    // luminance axis, and axisLuminanceLerp blends between using a separate axis
    // per-channel (0) and a shared luminance axis (1).
    template <typename T>
    static inline Eigen::Matrix<T, 9, 3> expandSH(Eigen::Matrix<T, 4, 3> shL1,
        bool isIrradiance,
        Eigen::Vector3d luminanceCoeffs,
        double axisLuminanceLerp)
    {
        Eigen::Matrix<T, 9, 3> result;
        result.row(0) = shL1.row(0);
        result.row(1) = shL1.row(1);
        result.row(2) = shL1.row(2);
        result.row(3) = shL1.row(3);

        Eigen::Matrix<T, 9, 1> shL1Flat = ZH3Solver::shExtractFlattenedL1FromRGB(shL1);
        for (size_t c = 0; c < 3; c += 1) {
            Eigen::Matrix<T, 3, 1> channelWeights = Eigen::Matrix<T, 3, 1>(
                c == 0 ? 1.0 : 0.0, c == 1 ? 1.0 : 0.0, c == 2 ? 1.0 : 0.0);
            Eigen::Matrix<T, 3, 1> luminanceWeights = (1.0 - axisLuminanceLerp) * channelWeights + luminanceCoeffs * axisLuminanceLerp;

            Eigen::Matrix<T, 3, 1> axis = ZH3Solver::sharedLuminanceAxis(shL1Flat, luminanceWeights);
            Eigen::Matrix<T, 5, 1> y2 = ZH3Solver::shEvaluateL2(axis);
            Eigen::Matrix<T, 3, 1> solvedSHL1 = ZH3Solver::luminanceSH(shL1Flat, channelWeights);

            T ratio = ZH3HallucinateSolver::zh3Ratio(solvedSHL1, axis, shL1(0, c));
            T fittedZH3Coeff = ZH3HallucinateSolver::zh3Coefficient(ratio, shL1(0, c), isIrradiance);

            result(4, c) = y2[0] * fittedZH3Coeff;
            result(5, c) = y2[1] * fittedZH3Coeff;
            result(6, c) = y2[2] * fittedZH3Coeff;
            result(7, c) = y2[3] * fittedZH3Coeff;
            result(8, c) = y2[4] * fittedZH3Coeff;
        }
        return result;
    }
};

struct ZH3HallucinatePerChannelSolver : ZH3HallucinateSolver {
    // Compute the reconstruction error for target where the DC term is given by
    // target[0], the linear SH is given by shL1, and the zonal term is given by
    // the least-squares fit of the zonal L2 band to the direction given by shL1.
    // targetIsIrradiance should be true if target is an SH representing
    // irradiance and false otherwise.
    template <typename T>
    static inline T zh3Error(Eigen::Matrix<T, 3, 1> shL1,
        Eigen::Matrix<T, 9, 1> target,
        bool targetIsIrradiance)
    {
        Eigen::Matrix<T, 3, 1> targetL1(target[1], target[2], target[3]);
        T l2Array[5] = { target[4], target[5], target[6], target[7], target[8] };
        Eigen::Matrix<T, 5, 1> targetL2(l2Array);

        T result = T(0.0f);

        Eigen::Matrix<T, 3, 1> deltaL1 = shL1 - targetL1;
        result += deltaL1.dot(deltaL1);

        Eigen::Matrix<T, 3, 1> axis = ZH3Solver::axis(shL1);
        Eigen::Matrix<T, 5, 1> y2 = ZH3Solver::shEvaluateL2(axis);

        T ratio = ZH3HallucinateSolver::zh3Ratio(shL1, axis, target[0]);
        T fittedZH3Coeff = ZH3HallucinateSolver::zh3Coefficient(ratio, target[0],
            targetIsIrradiance);
        Eigen::Matrix<T, 5, 1> fitL2 = fittedZH3Coeff * y2;
        Eigen::Matrix<T, 5, 1> deltaL2 = fitL2 - targetL2;
        result += deltaL2.dot(deltaL2);

        return result;
    }

    // Jacobian matrix of zh3Error with respect to shL1.
    template <typename T>
    static inline Eigen::Matrix<T, 3, 1> zh3ErrorDerivative(
        Eigen::Matrix<T, 3, 1> shL1, Eigen::Matrix<T, 9, 1> target,
        bool targetIsIrradiance)
    {
        Eigen::Matrix<T, 3, 1> targetL1(target[1], target[2], target[3]);
        Eigen::Matrix<T, 5, 1> targetL2;
        targetL2[0] = target[4];
        targetL2[1] = target[5];
        targetL2[2] = target[6];
        targetL2[3] = target[7];
        targetL2[4] = target[8];

        Eigen::Matrix<T, 3, 1> result = Eigen::Matrix<T, 3, 1>::Zero();

        Eigen::Matrix<T, 3, 1> deltaL1 = (shL1 - targetL1);
        Eigen::Matrix<T, 3, 1> deltaL1Derivative = deltaL1;
        result += 2.0f * deltaL1Derivative;

        Eigen::Matrix<T, 3, 1> axis = ZH3Solver::axis(shL1);
        Eigen::Matrix<T, 5, 1> y2 = ZH3Solver::shEvaluateL2(axis);

        T ratio = ZH3HallucinateSolver::zh3Ratio(shL1, axis, target[0]);
        T fittedZH3Coeff = ZH3HallucinateSolver::zh3Coefficient(ratio, target[0],
            targetIsIrradiance);

        Eigen::Matrix<T, 5, 1> fitL2 = fittedZH3Coeff * y2;
        Eigen::Matrix<T, 5, 1> deltaL2 = fitL2 - targetL2;

        Eigen::Matrix<T, 5, 3> y2Jacobian = ZH3Solver::shEvaluateL2Jacobian(axis);
        Eigen::Matrix<T, 3, 3> axisJacobian = ZH3Solver::axisJacobian(shL1);

        Eigen::Matrix<T, 5, 3> y2OfAxisJacobian = y2Jacobian * axisJacobian;

        Eigen::Matrix<T, 1, 3> ratioJacobian = zh3RatioJacobianDSHL1(axis, target[0]) + zh3RatioJacobianDsharedLuminanceAxis(shL1, target[0]) * axisJacobian;
        Eigen::Matrix<T, 1, 3> coeffDerivative = zh3CoefficientDerivative(ratio, target[0], targetIsIrradiance) * ratioJacobian;

        // Product rule.
        Eigen::Matrix<T, 5, 3> deltaL2DerivativePartA = y2 * coeffDerivative;
        Eigen::Matrix<T, 5, 3> deltaL2DerivativePartB = fittedZH3Coeff * y2OfAxisJacobian;
        Eigen::Matrix<T, 5, 3> deltaL2Derivative = (deltaL2DerivativePartA + deltaL2DerivativePartB);

        result += 2.0f * (deltaL2.transpose() * deltaL2Derivative);

        return result;
    }

    struct HallucinateCostFunctionParams {
        SH3 target;
        bool targetIsIrradiance;
    };

    static double zh3HallucinateCostFunction(void* targetPtr,
        const Eigen::VectorXd& x,
        Eigen::VectorXd& g)
    {
        HallucinateCostFunctionParams params = *((HallucinateCostFunctionParams*)targetPtr);
        bool targetIsIrradiance = params.targetIsIrradiance;

        Eigen::Matrix<double, 3, 1> shL1 = x;

        double error = ZH3HallucinatePerChannelSolver::zh3Error(shL1, params.target,
            targetIsIrradiance);
        g = ZH3HallucinatePerChannelSolver::zh3ErrorDerivative(shL1, params.target,
            targetIsIrradiance);

        return error;
    }

    // Solve for the linear SH that best represents target when used with
    // hallucinated ZH3, where the zonal axis for the L2 band is given by
    // axis(l1SH) and the ZH3 coefficient by zh3Coefficient. targetIsIrradiance
    // should be true if target is an SH representing irradiance and false
    // otherwise. A ZH3 is returned for convenience, but the zh3Coefficient in the
    // result can be trivially recomputed from the linearSH (see
    // ZH3HallucinateSolver::hallucinateZH3).
    static inline ZH3<double, 1> solve(SH3 target, bool targetIsIrradiance,
        double* outError = nullptr)
    {
        // Tetrahedron vertices
        Eigen::Vector3d testAxes[4] = { Eigen::Vector3d(1, 1, 1).normalized(),
            Eigen::Vector3d(1, -1, -1).normalized(),
            Eigen::Vector3d(-1, 1, -1).normalized(),
            Eigen::Vector3d(-1, -1, 1).normalized() };

        SH2 fittedSH;
        fittedSH.row(0) = target.row(0);
        fittedSH.row(1) = target.row(1);
        fittedSH.row(2) = target.row(2);
        fittedSH.row(3) = target.row(3);

        Eigen::VectorXd x = Eigen::Matrix<double, 3, 1>(target(1), target(2), target(3));

        double bestError = zh3Error(Eigen::Matrix<double, 3, 1>(x), target, targetIsIrradiance);

        for (const Eigen::Vector3d& axis : testAxes) {
            Eigen::Vector3d linearSHDir = ZH3Solver::shEvaluateL1(axis);
            SH2 linearSH = {};
            linearSH.row(0) = target.row(0);
            Eigen::Vector3d l1Vec = Eigen::Vector3d(-target(3), -target(1), target(2));
            double zonalScale = axis.dot(l1Vec) / sqrt(0.75f / ZH3HallucinateSolver::kPi);
            linearSH(1) = linearSHDir[0] * zonalScale;
            linearSH(2) = linearSHDir[1] * zonalScale;
            linearSH(3) = linearSHDir[2] * zonalScale;

            x = Eigen::Matrix<double, 3, 1>(linearSH(1), linearSH(2), linearSH(3));

            double error = zh3Error(Eigen::Matrix<double, 3, 1>(x), target, targetIsIrradiance);

            lbfgs::lbfgs_parameter_t param;
            HallucinateCostFunctionParams costFunctionParams = { target,
                targetIsIrradiance };

            int result = lbfgs::lbfgs_optimize(x, error, zh3HallucinateCostFunction, nullptr,
                nullptr, &costFunctionParams, param);
#if ZH3_SOLVER_PRINT_ERRORS
            if (result < 0) {
                printf("LBFGS failed to converge: %s. Error is %lf.\n",
                    lbfgs::lbfgs_strerror(result), error);
            }

#else
            (void)result;
#endif

            if (error < bestError) {
                fittedSH.row(0) = target.row(0);
                fittedSH.row(1) = x.row(0);
                fittedSH.row(2) = x.row(1);
                fittedSH.row(3) = x.row(2);

                bestError = error;
            }
        }

        if (outError) {
            *outError = bestError;
        }

        // Compute the hallucinated ZH3 coefficient.
        ZH3<double, 1> result;
        result.linearSH = fittedSH;

        Eigen::Vector3d solvedL1 = Eigen::Vector3d(fittedSH(1), fittedSH(2), fittedSH(3));
        Eigen::Vector3d axis = ZH3Solver::axis(solvedL1);

        double ratio = ZH3HallucinateSolver::zh3Ratio(solvedL1, axis, fittedSH(0));
        double fittedZH3Coeff = ZH3HallucinateSolver::zh3Coefficient(
            ratio, fittedSH(0), targetIsIrradiance);
        result.zh3Coefficients(0) = fittedZH3Coeff;
        return result;
    }

    // Solve for the ZH3 parameters (linear SH and ZH3 coefficients)
    // that represent target with the lowest least-squares error.
    // A ZH3 is returned for convenience, but the zh3Coefficient in the result can
    // be trivially recomputed from the linearSH (see
    // ZH3HallucinateSolver::hallucinateZH3).
    static inline ZH3<float, 1> solve(Eigen::Matrix<float, 9, 1> target,
        bool targetIsIrradiance,
        double* outError = nullptr)
    {
        ZH3<double, 1> doubleResult = solve(SH3(target.cast<double>()), targetIsIrradiance, outError);

        ZH3<float, 1> result;
        result.linearSH = doubleResult.linearSH.cast<float>();
        result.zh3Coefficients = doubleResult.zh3Coefficients.cast<float>();
        return result;
    }

    // Solve for the ZH3 parameters (linear SH and ZH3 coefficients)
    // that represent target with the lowest least-squares error.
    // A ZH3 is returned for convenience, but the zh3Coefficient in the result can
    // be trivially recomputed from the linearSH (see
    // ZH3HallucinateSolver::hallucinateZH3).
    template <typename T>
    static inline ZH3<T, 3> solve(Eigen::Matrix<T, 9, 3> target,
        bool targetIsIrradiance,
        Eigen::Matrix<T, 3, 1>* outError = nullptr)
    {
        Eigen::Vector3d error = Eigen::Vector3d::Zero();

        ZH3<T, 3> result;
        for (size_t c = 0; c < 3; c += 1) {
            double channelError = 0.0;
            SH3 targetDouble = target.col(c).template cast<double>();
            ZH3<double, 1> resultDouble = solve(targetDouble, targetIsIrradiance, &channelError);

            result.linearSH.col(c) = resultDouble.linearSH.template cast<T>();
            result.zh3Coefficients.col(c) = resultDouble.zh3Coefficients.template cast<T>();
            error(c, 0) += channelError;
        }

        if (outError) {
            *outError = error.template cast<T>();
        }

        return result;
    }
};

struct ZH3HallucinateSharedLuminanceSolver : ZH3HallucinateSolver {
    // Compute the reconstruction error for target where the DC term is given by
    // target[0], the linear SH is given by shL1, and the zonal term is given by
    // the least-squares fit of the zonal L2 band to the direction given by
    // the shared luminance axis from shL1.
    // targetIsIrradiance should be true if target is an SH representing
    // irradiance and false otherwise. channelWeights gives the weight for the
    // channel currently being evaluated (so e.g. (1, 0, 0) for red), and
    // lumWeights gives the weighting for each L1 channel in the shared luminance
    // axis.
    template <typename T>
    static inline T zh3Error(Eigen::Matrix<T, 9, 1> shL1,
        Eigen::Matrix<T, 9, 1> target,
        bool targetIsIrradiance,
        Eigen::Matrix<T, 3, 1> channelWeights,
        Eigen::Matrix<T, 3, 1> lumWeights)
    {
        Eigen::Matrix<T, 3, 1> targetL1(target[1], target[2], target[3]);
        T l2Array[5] = { target[4], target[5], target[6], target[7], target[8] };
        Eigen::Matrix<T, 5, 1> targetL2(l2Array);

        T result = T(0.0f);

        Eigen::Matrix<T, 3, 1> solvedSHL1 = ZH3Solver::luminanceSH(shL1, channelWeights);
        Eigen::Matrix<T, 3, 1> deltaL1 = solvedSHL1 - targetL1;
        result += deltaL1.dot(deltaL1);

        Eigen::Matrix<T, 3, 1> axis = ZH3Solver::sharedLuminanceAxis(shL1, lumWeights);
        Eigen::Matrix<T, 5, 1> y2 = ZH3Solver::shEvaluateL2(axis);

        T ratio = ZH3HallucinateSolver::zh3Ratio(solvedSHL1, axis, target[0]);
        T fittedZH3Coeff = ZH3HallucinateSolver::zh3Coefficient(ratio, target[0],
            targetIsIrradiance);
        Eigen::Matrix<T, 5, 1> fitL2 = fittedZH3Coeff * y2;
        Eigen::Matrix<T, 5, 1> deltaL2 = fitL2 - targetL2;
        result += deltaL2.dot(deltaL2);

        return result;
    }

    // Jacobian matrix of zh3Error with respect to shL1.
    template <typename T>
    static inline Eigen::Matrix<T, 9, 1> zh3ErrorDerivative(
        Eigen::Matrix<T, 9, 1> shL1, Eigen::Matrix<T, 9, 1> target,
        bool targetIsIrradiance, Eigen::Matrix<T, 3, 1> channelWeights,
        Eigen::Matrix<T, 3, 1> lumWeights)
    {
        Eigen::Matrix<T, 3, 1> targetL1(target[1], target[2], target[3]);
        Eigen::Matrix<T, 5, 1> targetL2;
        targetL2[0] = target[4];
        targetL2[1] = target[5];
        targetL2[2] = target[6];
        targetL2[3] = target[7];
        targetL2[4] = target[8];

        Eigen::Matrix<T, 9, 1> result = Eigen::Matrix<T, 9, 1>::Zero();

        Eigen::Matrix<T, 3, 9> lumSHJacobian = ZH3Solver::luminanceSHJacobian(channelWeights);

        Eigen::Matrix<T, 3, 1> solvedSHL1 = ZH3Solver::luminanceSH(shL1, channelWeights);
        Eigen::Matrix<T, 3, 1> deltaL1 = (solvedSHL1 - targetL1);
        Eigen::Matrix<T, 9, 1> deltaL1Derivative = lumSHJacobian.transpose() * deltaL1;
        result += 2.0f * deltaL1Derivative;

        Eigen::Matrix<T, 3, 1> axis = ZH3Solver::sharedLuminanceAxis(shL1, lumWeights);
        Eigen::Matrix<T, 5, 1> y2 = ZH3Solver::shEvaluateL2(axis);

        T ratio = ZH3HallucinateSolver::zh3Ratio(solvedSHL1, axis, target[0]);
        T fittedZH3Coeff = ZH3HallucinateSolver::zh3Coefficient(ratio, target[0],
            targetIsIrradiance);

        Eigen::Matrix<T, 5, 1> fitL2 = fittedZH3Coeff * y2;
        Eigen::Matrix<T, 5, 1> deltaL2 = fitL2 - targetL2;

        Eigen::Matrix<T, 5, 3> y2Jacobian = ZH3Solver::shEvaluateL2Jacobian(axis);
        Eigen::Matrix<T, 3, 9> axisJacobian = ZH3Solver::sharedLuminanceAxisJacobian(shL1, lumWeights);

        Eigen::Matrix<T, 5, 9> y2OfAxisJacobian = y2Jacobian * axisJacobian;

        Eigen::Matrix<T, 1, 9> ratioJacobian = zh3RatioJacobianDSHL1(axis, target[0]) * lumSHJacobian + zh3RatioJacobianDsharedLuminanceAxis(solvedSHL1, target[0]) * axisJacobian;
        Eigen::Matrix<T, 1, 9> coeffDerivative = zh3CoefficientDerivative(ratio, target[0], targetIsIrradiance) * ratioJacobian;

        // Product rule.
        Eigen::Matrix<T, 5, 9> deltaL2DerivativePartA = y2 * coeffDerivative;
        Eigen::Matrix<T, 5, 9> deltaL2DerivativePartB = fittedZH3Coeff * y2OfAxisJacobian;
        Eigen::Matrix<T, 5, 9> deltaL2Derivative = (deltaL2DerivativePartA + deltaL2DerivativePartB);

        result += 2.0f * (deltaL2.transpose() * deltaL2Derivative);

        return result;
    }

    // Solver:
    static double zh3HallucinateError(Eigen::Matrix<double, 9, 1> shL1,
        SH3RGB target, bool targetIsIrradiance,
        Eigen::Vector3d luminanceCoeffs,
        double axisLuminanceLerp)
    {
        SH3 targetR;
        SH3 targetG;
        SH3 targetB;

        for (size_t i = 0; i < 9; i += 1) {
            targetR[i] = target(i, 0);
            targetG[i] = target(i, 1);
            targetB[i] = target(i, 2);
        }

        Eigen::Vector3d rLumCoeffs = (1.0 - axisLuminanceLerp) * Eigen::Vector3d(1.0, 0.0, 0.0) + axisLuminanceLerp * luminanceCoeffs;
        Eigen::Vector3d gLumCoeffs = (1.0 - axisLuminanceLerp) * Eigen::Vector3d(0.0, 1.0, 0.0) + axisLuminanceLerp * luminanceCoeffs;
        Eigen::Vector3d bLumCoeffs = (1.0 - axisLuminanceLerp) * Eigen::Vector3d(0.0, 0.0, 1.0) + axisLuminanceLerp * luminanceCoeffs;

        return Eigen::Vector3d(ZH3HallucinateSharedLuminanceSolver::zh3Error(
                                   shL1, targetR, targetIsIrradiance,
                                   Eigen::Vector3d(1.0, 0.0, 0.0), rLumCoeffs),
            ZH3HallucinateSharedLuminanceSolver::zh3Error(
                shL1, targetG, targetIsIrradiance,
                Eigen::Vector3d(0.0, 1.0, 0.0), gLumCoeffs),
            ZH3HallucinateSharedLuminanceSolver::zh3Error(
                shL1, targetB, targetIsIrradiance,
                Eigen::Vector3d(0.0, 0.0, 1.0), bLumCoeffs))
            .dot(luminanceCoeffs);
    }

    struct HallucinateSharedLumCostFunctionParams {
        SH3RGB target;
        Eigen::Vector3d luminanceCoeffs;
        double axisLuminanceLerp;
        bool targetIsIrradiance;
    };

    static double zh3HallucinateCostFunction(void* targetPtr,
        const Eigen::VectorXd& x,
        Eigen::VectorXd& g)
    {
        HallucinateSharedLumCostFunctionParams params = *((HallucinateSharedLumCostFunctionParams*)targetPtr);
        bool targetIsIrradiance = params.targetIsIrradiance;

        Eigen::Matrix<double, 9, 1> shL1 = x;

        Eigen::Matrix<double, 9, 1> targetR;
        Eigen::Matrix<double, 9, 1> targetG;
        Eigen::Matrix<double, 9, 1> targetB;

        for (size_t i = 0; i < 9; i += 1) {
            targetR[i] = params.target(i, 0);
            targetG[i] = params.target(i, 1);
            targetB[i] = params.target(i, 2);
        }

        Eigen::Vector3d rLumCoeffs = (1.0 - params.axisLuminanceLerp) * Eigen::Vector3d(1.0, 0.0, 0.0) + params.axisLuminanceLerp * params.luminanceCoeffs;
        Eigen::Vector3d gLumCoeffs = (1.0 - params.axisLuminanceLerp) * Eigen::Vector3d(0.0, 1.0, 0.0) + params.axisLuminanceLerp * params.luminanceCoeffs;
        Eigen::Vector3d bLumCoeffs = (1.0 - params.axisLuminanceLerp) * Eigen::Vector3d(0.0, 0.0, 1.0) + params.axisLuminanceLerp * params.luminanceCoeffs;

        double error = Eigen::Vector3d(ZH3HallucinateSharedLuminanceSolver::zh3Error(
                                           shL1, targetR, targetIsIrradiance,
                                           Eigen::Vector3d(1.0, 0.0, 0.0), rLumCoeffs),
            ZH3HallucinateSharedLuminanceSolver::zh3Error(
                shL1, targetG, targetIsIrradiance,
                Eigen::Vector3d(0.0, 1.0, 0.0), gLumCoeffs),
            ZH3HallucinateSharedLuminanceSolver::zh3Error(
                shL1, targetB, targetIsIrradiance,
                Eigen::Vector3d(0.0, 0.0, 1.0), bLumCoeffs))
                           .dot(params.luminanceCoeffs);

        g = params.luminanceCoeffs[0] * ZH3HallucinateSharedLuminanceSolver::zh3ErrorDerivative(shL1, targetR, targetIsIrradiance, Eigen::Vector3d(1.0, 0.0, 0.0), rLumCoeffs);
        g += params.luminanceCoeffs[1] * ZH3HallucinateSharedLuminanceSolver::zh3ErrorDerivative(shL1, targetG, targetIsIrradiance, Eigen::Vector3d(0.0, 1.0, 0.0), gLumCoeffs);
        g += params.luminanceCoeffs[2] * ZH3HallucinateSharedLuminanceSolver::zh3ErrorDerivative(shL1, targetB, targetIsIrradiance, Eigen::Vector3d(0.0, 0.0, 1.0), bLumCoeffs);

        return error;
    }

    // Solve for the linear SH that best represents target when used with
    // hallucinated ZH3, where the zonal axis for the L2 band is given for each
    // channel by lerp(optimalLinearAxis(channel),
    // optimalLinearAxis(luminance(l1SH)), axisLuminanceLerp), and the ZH3
    // coefficient by zh3Coefficient. luminanceCoeffs gives the relative weighting
    // for R/G/B, and should add to one. targetIsIrradiance should be true if
    // target is an SH representing irradiance and false otherwise. A ZH3 is
    // returned for convenience, but the zh3Coefficient in the result can be
    // trivially recomputed from the linearSH (see
    // ZH3HallucinateSolver::hallucinateZH3).
    static inline ZH3<double, 3> solve(
        SH3RGB target, bool targetIsIrradiance,
        Eigen::Vector3d luminanceWeightingCoeffs = Eigen::Vector3d(1.0f / 3.0f,
            1.0f / 3.0f,
            1.0f / 3.0f),
        double axisLuminanceLerp = 0.0, double* outError = nullptr)
    {
        // Tetrahedron vertices
        Eigen::Vector3d testAxes[4] = { Eigen::Vector3d(1, 1, 1).normalized(),
            Eigen::Vector3d(1, -1, -1).normalized(),
            Eigen::Vector3d(-1, 1, -1).normalized(),
            Eigen::Vector3d(-1, -1, 1).normalized() };

        SH2RGB fittedSH;
        fittedSH.row(0) = target.row(0);
        fittedSH.row(1) = target.row(1);
        fittedSH.row(2) = target.row(2);
        fittedSH.row(3) = target.row(3);

        Eigen::VectorXd x = ZH3Solver::shExtractFlattenedL1FromRGB<double, 9>(target);

        double bestError = zh3HallucinateError(x, target, targetIsIrradiance,
            luminanceWeightingCoeffs, axisLuminanceLerp);

        for (const Eigen::Vector3d& axis : testAxes) {
            Eigen::Vector3d linearSHDir = ZH3Solver::shEvaluateL1(axis);
            SH2RGB linearSHRGB = {};
            linearSHRGB.row(0) = target.row(0);
            for (size_t c = 0; c < 3; c += 1) {
                Eigen::Vector3d l1Vec = Eigen::Vector3d(-target(3, c), -target(1, c), target(2, c));
                double zonalScale = axis.dot(l1Vec) / sqrt(0.75f / ZH3HallucinateSolver::kPi);

                linearSHRGB(1, c) = linearSHDir[0] * zonalScale;
                linearSHRGB(2, c) = linearSHDir[1] * zonalScale;
                linearSHRGB(3, c) = linearSHDir[2] * zonalScale;
            }

            x = ZH3Solver::shExtractFlattenedL1FromRGB<double, 4>(linearSHRGB);

            double error = zh3HallucinateError(x, target, targetIsIrradiance,
                luminanceWeightingCoeffs, axisLuminanceLerp);

            lbfgs::lbfgs_parameter_t param;
            HallucinateSharedLumCostFunctionParams costFunctionParams = {
                target, luminanceWeightingCoeffs, axisLuminanceLerp,
                targetIsIrradiance
            };

            int result = lbfgs::lbfgs_optimize(x, error, zh3HallucinateCostFunction, nullptr,
                nullptr, &costFunctionParams, param);

#if ZH3_SOLVER_PRINT_ERRORS
            if (result < 0) {
                printf("LBFGS failed to converge: %s. Error is %lf.\n",
                    lbfgs::lbfgs_strerror(result), error);
            }

#else
            (void)result;
#endif

            if (error < bestError) {
                SH2RGB linearSH;
                linearSH.row(0) = target.row(0);
                ZH3Solver::shCopyFlattenedL1ToRGB<double, 4>(x, linearSH);

                bestError = error;
                fittedSH = linearSH;
            }
        }

        if (outError) {
            *outError = bestError;
        }

        ZH3<double, 3> result;
        result.linearSH = fittedSH;

        // Compute the hallucinated ZH3 coefficients.
        Eigen::Matrix<double, 9, 1> shL1Flat = ZH3Solver::shExtractFlattenedL1FromRGB(fittedSH);
        for (size_t c = 0; c < 3; c += 1) {
            Eigen::Vector3d channelWeights = Eigen::Vector3d(
                c == 0 ? 1.0 : 0.0, c == 1 ? 1.0 : 0.0, c == 2 ? 1.0 : 0.0);
            Eigen::Vector3d luminanceWeights = (1.0 - axisLuminanceLerp) * channelWeights + luminanceWeightingCoeffs * axisLuminanceLerp;

            Eigen::Vector3d axis = ZH3Solver::sharedLuminanceAxis(shL1Flat, luminanceWeights);
            Eigen::Vector3d solvedSHL1 = ZH3Solver::luminanceSH(shL1Flat, channelWeights);

            double ratio = ZH3HallucinateSolver::zh3Ratio(solvedSHL1, axis, fittedSH(0, c));
            double fittedZH3Coeff = ZH3HallucinateSolver::zh3Coefficient(
                ratio, fittedSH(0, c), targetIsIrradiance);
            result.zh3Coefficients(c) = fittedZH3Coeff;
        }
        return result;
    }

    // Solve for the linear SH that best represents target when used with
    // hallucinated ZH3, where the zonal axis for the L2 band is given for each
    // channel by lerp(optimalLinearAxis(channel),
    // optimalLinearAxis(luminance(l1SH)), axisLuminanceLerp), and the ZH3
    // coefficient by ZH3HallucinateSolver::zh3Coefficient. luminanceCoeffs gives
    // the relative weighting for R/G/B, and should add to one. targetIsIrradiance
    // should be true if target is an SH representing irradiance and false
    // otherwise. A ZH3 is returned for convenience, but the zh3Coefficient in the
    // result can be trivially recomputed from the linearSH (see
    // ZH3HallucinateSolver::hallucinateZH3).
    static inline ZH3<float, 3> solve(Eigen::Matrix<float, 9, 3> target,
        bool targetIsIrradiance,
        Eigen::Vector3f luminanceCoeffs,
        float axisLuminanceLerp,
        double* outError = nullptr)
    {
        ZH3<double, 3> resultDouble = solve(
            target.cast<double>(), targetIsIrradiance,
            luminanceCoeffs.cast<double>(), double(axisLuminanceLerp), outError);

        ZH3<float, 3> result;
        result.linearSH = resultDouble.linearSH.cast<float>();
        result.zh3Coefficients = resultDouble.zh3Coefficients.cast<float>();
        return result;
    }
};

template <typename T>
inline Eigen::Matrix<T, 9, 1> ZH3<T, 1>::expanded() const
{
    Eigen::Matrix<T, 9, 1> result;
    for (size_t i = 0; i < 4; i += 1) {
        result.row(i) = this->linearSH.row(i);
    }
    Eigen::Matrix<T, 3, 1> axis = ZH3Solver::axis(Eigen::Matrix<T, 3, 1>(result(1), result(2), result(3)));
    Eigen::Matrix<T, 5, 1> y2 = ZH3Solver::shEvaluateL2(axis);
    for (size_t i = 0; i < 5; i += 1) {
        result(i + 4) = y2(i) * this->zh3Coefficients(0);
    }
    return result;
}

template <typename T>
inline Eigen::Matrix<T, 9, 3> ZH3<T, 3>::expanded(
    Eigen::Matrix<T, 3, 1> luminanceCoeffs, T sharedLuminanceAxisLerp) const
{
    Eigen::Matrix<T, 9, 3> result;
    for (size_t i = 0; i < 4; i += 1) {
        result.row(i) = this->linearSH.row(i);
    }

    Eigen::Matrix<T, 9, 1> flattenedL1 = ZH3Solver::shExtractFlattenedL1FromRGB<T, 4>(this->linearSH);

    for (size_t c = 0; c < 3; c += 1) {
        Eigen::Matrix<T, 3, 1> lumCoeffs = (T(1.0) - sharedLuminanceAxisLerp) * Eigen::Matrix<T, 3, 1>(c == 0 ? T(1) : T(0), c == 1 ? T(1) : T(0), c == 2 ? T(1) : T(0)) + sharedLuminanceAxisLerp * luminanceCoeffs;
        Eigen::Matrix<T, 3, 1> axis = ZH3Solver::sharedLuminanceAxis(flattenedL1, lumCoeffs);
        Eigen::Matrix<T, 5, 1> y2 = ZH3Solver::shEvaluateL2(axis);
        for (size_t i = 0; i < 5; i += 1) {
            result(i + 4, c) = y2(i) * this->zh3Coefficients(c);
        }
    }
    return result;
}
