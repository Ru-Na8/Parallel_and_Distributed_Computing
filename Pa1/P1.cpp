#include <vector>
#include <iostream>
#include <chrono>

void matmul_naive(const double *A, const double *B, double *C, int N)
{
    for (int i = 0; i < N; i++) //rows
    {
        for (int j = 0; j < N; j++) //columns
        {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
void matmul_optimized(const double *A, const double *B, double *C, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < N; k++)
        {
            double r = A[i * N + k]; // Store A[i][k] in a register
            for (int j = 0; j < N; j++)
            {
                C[i * N + j] += r * B[k * N + j]; // Access B row-wise
            }
        }
    }
}

int main()
{
    int N = 1024; // size 
    std::vector<double> A(N * N), B(N * N), C(N * N, 0.0);

    for (int i = 0; i < N * N; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    auto start = std::chrono::high_resolution_clock::now();


    matmul_naive(A.data(), B.data(), C.data(), N);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time for naive: " << elapsed.count() << " s\n";

    auto start_opt = std::chrono::high_resolution_clock::now();

    // matmul_naive(A.data(), B.data(), C.data(), N);

    matmul_optimized(A.data(), B.data(), C.data(), N);

    auto end_opt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_op = end_opt - start_opt;
    std::cout << "Elapsed time for optimized: " << elapsed_op.count() << " s\n";

    return 0;
}
