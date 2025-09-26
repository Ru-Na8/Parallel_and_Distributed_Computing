#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>

void matmul_blocked(const double *A, const double *B, double *C, int N, int BLOCK_SIZE)
{
    for (int iBlock = 0; iBlock < N; iBlock += BLOCK_SIZE)
    {
        for (int jBlock = 0; jBlock < N; jBlock += BLOCK_SIZE)
        {
            for (int kBlock = 0; kBlock < N; kBlock += BLOCK_SIZE)
            {
                const int iBlockMax = std::min(iBlock + BLOCK_SIZE, N);
                const int jBlockMax = std::min(jBlock + BLOCK_SIZE, N);
                const int kBlockMax = std::min(kBlock + BLOCK_SIZE, N);

                for (int i = iBlock; i < iBlockMax; i++)
                {
                    for (int j = jBlock; j < jBlockMax; j++)
                    {
                        double sum = C[i * N + j];
                        for (int k = kBlock; k < kBlockMax; k++)
                        {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

// Blockedikj order
void matmul_blocked_omp_ikj(const double *A, const double *B, double *C, int N, int BLOCK_SIZE)
{
    for (int iBlock = 0; iBlock < N; iBlock += BLOCK_SIZE)
    {
        for (int kBlock = 0; kBlock < N; kBlock += BLOCK_SIZE)
        {
            for (int jBlock = 0; jBlock < N; jBlock += BLOCK_SIZE)
            {
                int iBlockMax = std::min(iBlock + BLOCK_SIZE, N);
                int kBlockMax = std::min(kBlock + BLOCK_SIZE, N);
                int jBlockMax = std::min(jBlock + BLOCK_SIZE, N);

                for (int i = iBlock; i < iBlockMax; i++)
                {
                    for (int k = kBlock; k < kBlockMax; k++)
                    {
                        double r = A[i * N + k]; // Store A[i][k] in a register
                        for (int j = jBlock; j < jBlockMax; j++)
                        {
                            C[i * N + j] += r * B[k * N + j]; 
                        }
                    }
                }
            }
        }
    }
}

int main()
{
    int N = 1024;       
    int BLOCK_SIZE = 64; 

    std::vector<double> A(N * N), B(N * N);
    std::vector<double> C_naive(N * N, 0.0);
    std::vector<double> C_ikj(N * N, 0.0);

    for (int i = 0; i < N * N; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    auto start_naive = std::chrono::high_resolution_clock::now();
    matmul_blocked(A.data(), B.data(), C_naive.data(), N, BLOCK_SIZE);
    auto end_naive = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_naive = end_naive - start_naive;

    std::cout << "Elapsed time (Blocked Naive ijk): " << elapsed_naive.count() << " s\n";

    auto start_ikj = std::chrono::high_resolution_clock::now();
    matmul_blocked_omp_ikj(A.data(), B.data(), C_ikj.data(), N, BLOCK_SIZE);
    auto end_ikj = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_ikj = end_ikj - start_ikj;

    std::cout << "Elapsed time (Blocked Optimized ikj): " << elapsed_ikj.count() << " s\n";

    double eps = 1e-9;
    for (int i = 0; i < N * N; i++)
    {
        double expected = 2.0 * N;
        if (std::fabs(C_naive[i] - expected) > eps || std::fabs(C_ikj[i] - expected) > eps)
        {
            std::cerr << "Error: result mismatch at index " << i
                      << ", got C_naive = " << C_naive[i] << ", C_ikj = " << C_ikj[i]
                      << ", expected " << expected << "\n";
            return 1;
        }
    }

    std::cout << "Validation passed: Both methods produce the same correct result.\n";

    return 0;
}
