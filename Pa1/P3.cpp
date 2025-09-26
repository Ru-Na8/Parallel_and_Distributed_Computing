#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm> 
#include <cmath>     
#include <cassert>   
#ifdef _OPENMP
#include <omp.h>
#endif

void matmul_blocked_omp(const double *A, const double *B, double *C, int N, int BLOCK_SIZE)
{
#pragma omp parallel for collapse(2) schedule(static)
    for (int iBlock = 0; iBlock < N; iBlock += BLOCK_SIZE)
    {
        for (int jBlock = 0; jBlock < N; jBlock += BLOCK_SIZE)
        {
            for (int kBlock = 0; kBlock < N; kBlock += BLOCK_SIZE)
            {
                int iBlockMax = std::min(iBlock + BLOCK_SIZE, N);
                int jBlockMax = std::min(jBlock + BLOCK_SIZE, N);
                int kBlockMax = std::min(kBlock + BLOCK_SIZE, N);

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

void matmul_blocked_omp_ikj(const double *A, const double *B, double *C, int N, int BLOCK_SIZE)
{
#pragma omp parallel for collapse(2) schedule(static)
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
                        double r = A[i * N + k]; // Store A[i][k] in register
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

    std::vector<int> block_sizes = {8, 16, 32, 64};

    std::vector<int> thread_counts = {1, 2, 4, 8};

    std::vector<double> A(N * N), B(N * N);
    for (int i = 0; i < N * N; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0;
    }

#ifdef _OPENMP
    std::cout << "OpenMP is enabled.\n";
#else
    std::cout << "OpenMP not enabled: everything will be single-threaded.\n";
#endif

    std::cout << "\nComparison of matrix multiplication performance:\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "Threads | Block Size | Time (ijk)  | Time (ikj)\n";
    std::cout << "---------------------------------------------------\n";

    for (int t : thread_counts)
    {
#ifdef _OPENMP
        omp_set_num_threads(t);
#endif
        for (int bs : block_sizes)
        {
            // Create fresh output matrices for each method
            std::vector<double> C_ijk(N * N, 0.0);
            std::vector<double> C_ikj(N * N, 0.0);

            // Measure time for `ijk` order
            auto start_ijk = std::chrono::high_resolution_clock::now();
            matmul_blocked_omp(A.data(), B.data(), C_ijk.data(), N, bs);
            auto end_ijk = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_ijk = end_ijk - start_ijk;

            // Measure time for `ikj` order
            auto start_ikj = std::chrono::high_resolution_clock::now();
            matmul_blocked_omp_ikj(A.data(), B.data(), C_ikj.data(), N, bs);
            auto end_ikj = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_ikj = end_ikj - start_ikj;

            // Validate correctness
            double eps = 1e-9;
            for (int i = 0; i < N * N; i++)
            {
                double expected = 2.0 * N;
                if (std::fabs(C_ijk[i] - expected) > eps || std::fabs(C_ikj[i] - expected) > eps)
                {
                    std::cerr << "Error: result mismatch at index "
                              << i << ", got C_ijk = " << C_ijk[i] << ", C_ikj = " << C_ikj[i]
                              << ", expected " << expected << "\n";
                    return 1;
                }
            }

            // Print comparison results
            std::cout << t << "       | " << bs << "         | "
                      << elapsed_ijk.count() << " s  | " << elapsed_ikj.count() << " s\n";
        }
    }

    return 0;
}