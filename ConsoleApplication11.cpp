#include <iostream>
#include <chrono>
#include <tbb/tbb.h>

double* gauss(double** A, double* Y, int n) {
    double* X = new double[n];
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            double m = A[j][k] / A[k][k];
            for (int i = k; i < n; i++) {
                A[j][i] = A[j][i] - m * A[k][i];
            }
            Y[j] = Y[j] - m * Y[k];
        }
    }
    for (int k = n - 1; k >= 0; k--) {
        X[k] = Y[k];
        for (int i = k + 1; i < n; i++) {
            X[k] = X[k] - A[k][i] * X[i];
        }
        X[k] = X[k] / A[k][k];
    }
    return X;
}

double* gauss_tbb(double** A, double* Y, int n) {
    double* X = new double[n];

    tbb::parallel_for(0, n, [&](int k) {
        for (int j = k + 1; j < n; j++) {
            double m = A[j][k] / A[k][k];
            for (int i = k; i < n; i++) {
                A[j][i] = A[j][i] - m * A[k][i];
            }
            Y[j] = Y[j] - m * Y[k];
        }
        });

    for (int k = n - 1; k >= 0; k--) {
        X[k] = Y[k];
        for (int i = k + 1; i < n; i++) {
            X[k] = X[k] - A[k][i] * X[i];
        }
        X[k] = X[k] / A[k][k];
    }

    return X;
}

int main() {
    int n = 500;

    double** A = new double* [n];
    for (int i = 0; i < n; i++) {
        A[i] = new double[n];
    }
    double* Y = new double[n];

    auto start = std::chrono::steady_clock::now();
    double* solution = gauss(A, Y, n);
    auto end = std::chrono::steady_clock::now();
    auto res = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "sequential: " << res.count() << std::endl;
    delete[] solution;

    solution = new double[n];

    start = std::chrono::steady_clock::now();
    solution = gauss_tbb(A, Y, n);
    end = std::chrono::steady_clock::now();
    res = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "parallel: " << res.count();

    delete[] solution;
    delete[] Y;
    delete[] A;

    return 0;
}