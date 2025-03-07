// ThinPlateSplineInterpolator.h
#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <limits>

namespace _2D {

    /**
     * Custom implementation of Thin Plate Spline interpolation
     * that mimics the interface of libInterpolate's ThinPlateSplineInterpolator.
     * Uses only standard C++ libraries with no external dependencies.
     */
    template<typename T>
    class ThinPlateSplineInterpolator {
    public:
        // Define vector and matrix types
        typedef std::vector<T> VectorType;
        typedef std::vector<std::vector<T>> MatrixType;

    private:
        // Data members
        VectorType* X;         // x-coordinates of points
        VectorType* Y;         // y-coordinates of points
        MatrixType* Z;         // Function values at points
        VectorType weights;    // TPS weights
        VectorType coeffs;     // Polynomial coefficients [a, b, c]
        T smoothingFactor;     // Regularization parameter
        bool isInitialized;    // Flag to check if interpolator is ready
        int numPoints;         // Number of data points

        /**
         * Radial basis function for TPS: r^2 * log(r)
         */
        T radiusFunction(T r) const {
            if (r < std::numeric_limits<T>::epsilon()) return 0;
            return r * r * std::log(r);
        }

        /**
         * Compute Euclidean distance between two points
         */
        T distance(T x1, T y1, T x2, T y2) const {
            T dx = x1 - x2;
            T dy = y1 - y2;
            return std::sqrt(dx * dx + dy * dy);
        }

        /**
         * Solve a linear system Ax = b using Gaussian elimination with partial pivoting
         * @param A - Matrix of coefficients (will be modified)
         * @param b - Right-hand side vector (will be modified)
         * @param x - Output solution vector
         * @return True if successful, false if singular system
         */
        bool solveLinearSystem(MatrixType& A, VectorType& b, VectorType& x) const {
            int n = b.size();
            x.resize(n, 0);

            // Gaussian elimination with partial pivoting
            for (int i = 0; i < n - 1; i++) {
                // Find pivot row
                int maxRow = i;
                T maxVal = std::abs(A[i][i]);
                for (int k = i + 1; k < n; k++) {
                    if (std::abs(A[k][i]) > maxVal) {
                        maxVal = std::abs(A[k][i]);
                        maxRow = k;
                    }
                }

                // Check for singularity
                if (maxVal < std::numeric_limits<T>::epsilon()) {
                    return false;  // Singular matrix
                }

                // Swap rows if needed
                if (maxRow != i) {
                    std::swap(A[i], A[maxRow]);
                    std::swap(b[i], b[maxRow]);
                }

                // Eliminate below
                for (int k = i + 1; k < n; k++) {
                    T factor = A[k][i] / A[i][i];
                    b[k] -= factor * b[i];
                    for (int j = i; j < n; j++) {
                        A[k][j] -= factor * A[i][j];
                    }
                }
            }

            // Check if the system is singular
            if (std::abs(A[n - 1][n - 1]) < std::numeric_limits<T>::epsilon()) {
                return false;
            }

            // Back substitution
            for (int i = n - 1; i >= 0; i--) {
                T sum = 0;
                for (int j = i + 1; j < n; j++) {
                    sum += A[i][j] * x[j];
                }
                x[i] = (b[i] - sum) / A[i][i];
            }

            return true;
        }

        /**
         * Set up and solve the TPS interpolation system
         */
        bool setupInterpolation() {
            if (!X || !Y || !Z || X->empty() || Y->empty() || Z->empty() ||
                X->size() != Y->size() || X->size() != Z->at(0).size()) {
                return false;
            }

            numPoints = X->size();
            int n = numPoints + 3;  // Total system size (points + 3 polynomial terms)

            // Create system matrix and right-hand side
            MatrixType A(n, VectorType(n, 0));
            VectorType b(n, 0);

            // Fill in the radial basis function part of the matrix
            for (int i = 0; i < numPoints; i++) {
                for (int j = 0; j < numPoints; j++) {
                    T dist = distance(X->at(i), Y->at(i), X->at(j), Y->at(j));
                    A[i][j] = radiusFunction(dist);

                    // Add regularization on the diagonal
                    if (i == j) {
                        A[i][j] += smoothingFactor;
                    }
                }
            }

            // Fill in the polynomial part
            for (int i = 0; i < numPoints; i++) {
                // P part
                A[i][numPoints] = X->at(i);
                A[i][numPoints + 1] = Y->at(i);
                A[i][numPoints + 2] = 1;

                // P^T part
                A[numPoints][i] = X->at(i);
                A[numPoints + 1][i] = Y->at(i);
                A[numPoints + 2][i] = 1;
            }

            // Fill in right-hand side with z values
            for (int i = 0; i < numPoints; i++) {
                b[i] = Z->at(0)[i];
            }

            // Zeros for the additional constraint equations
            // These ensure polynomial terms are orthogonal to the weights
            b[numPoints] = 0;
            b[numPoints + 1] = 0;
            b[numPoints + 2] = 0;

            // Solve the system
            VectorType solution(n);
            if (!solveLinearSystem(A, b, solution)) {
                return false;
            }

            // Extract weights and coefficients
            weights.resize(numPoints);
            coeffs.resize(3);

            for (int i = 0; i < numPoints; i++) {
                weights[i] = solution[i];
            }

            coeffs[0] = solution[numPoints];       // x coefficient
            coeffs[1] = solution[numPoints + 1];   // y coefficient
            coeffs[2] = solution[numPoints + 2];   // constant term

            isInitialized = true;
            return true;
        }

    public:
        /**
         * Constructor
         */
        ThinPlateSplineInterpolator() :
            X(new VectorType()), Y(new VectorType()), Z(new MatrixType(1)),
            smoothingFactor(0), isInitialized(false), numPoints(0) {
        }

        /**
         * Destructor
         */
        ~ThinPlateSplineInterpolator() {
            delete X;
            delete Y;
            delete Z;
        }

        /**
         * Set the smoothing factor for regularization
         * @param lambda Smoothing factor (higher values give smoother interpolation)
         */
        void setSmoothing(T lambda) {
            smoothingFactor = lambda;
            if (isInitialized) {
                setupInterpolation();  // Re-setup if already initialized
            }
        }

        /**
         * Set the data points for interpolation
         * @param x X-coordinates of data points
         * @param y Y-coordinates of data points
         * @param z Function values at data points
         */
        void setData(const VectorType& x, const VectorType& y, const VectorType& z) {
            if (x.size() < 3 || x.size() != y.size() || x.size() != z.size()) {
                throw std::invalid_argument("TPS requires at least 3 points and all vectors must be the same size");
            }

            // Copy data
            *X = x;
            *Y = y;
            Z->clear();
            Z->push_back(z);

            isInitialized = setupInterpolation();
            if (!isInitialized) {
                throw std::runtime_error("Failed to initialize TPS interpolation system");
            }
        }

        /**
         * Perform interpolation at a point
         * @param x X-coordinate
         * @param y Y-coordinate
         * @return Interpolated value
         */
        T operator()(T x, T y) const {
            if (!isInitialized) {
                throw std::runtime_error("TPS interpolator not initialized");
            }

            // Start with the polynomial part: a*x + b*y + c
            T value = coeffs[0] * x + coeffs[1] * y + coeffs[2];

            // Add the weighted radial basis functions
            for (int i = 0; i < numPoints; i++) {
                T dist = distance(x, y, X->at(i), Y->at(i));
                value += weights[i] * radiusFunction(dist);
            }

            return value;
        }

        /**
         * Update internal structures (included for API compatibility)
         */
        void updateStructures() {
            // Not needed in this implementation, but included for API compatibility
        }

        /**
         * Accessors for internal vectors (for compatibility with original code)
         */
        VectorType getX() const { return *X; }
        VectorType getY() const { return *Y; }
        MatrixType getZ() const { return *Z; }
    };

    /**
     * Class that matches the specific interface used in the original code
     */
    class ThinPlateSplineInter : public ThinPlateSplineInterpolator<double> {
    public:
        VectorType getX() { return ThinPlateSplineInterpolator<double>::getX(); }
        VectorType getY() { return ThinPlateSplineInterpolator<double>::getY(); }
        MatrixType getZ() { return ThinPlateSplineInterpolator<double>::getZ(); }
    };

} // namespace _2D