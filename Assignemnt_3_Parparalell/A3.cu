#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>

// Increased coarsening factor and tile width for better performance
#define COARSENING 8    // Increased from 1 to 4
#define TILE_WIDTH 16   // Increased from 16 to 32

// --- 1. Privatization Only (Baseline) ---
__global__ void kernelPrivatizationOnly(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// --- 2. Improved Memory Coalescing ---
__global__ void kernelMemoryCoalescing(const double* A, const double* B_T, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        
        // Process in chunks for better cache behavior
        const int CHUNK_SIZE = 32;
        for (int i = 0; i < N; i += CHUNK_SIZE) {
            for (int k = i; k < min(N, i + CHUNK_SIZE); ++k) {
                sum += A[row * N + k] * B_T[col * N + k];
            }
        }
        
        C[row * N + col] = sum;
    }
}

// --- 3. Improved Thread Coarsening ---
__global__ void kernelThreadCoarsening(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int colStart = (blockIdx.x * blockDim.x + threadIdx.x) * COARSENING;

    if (row < N) {
        // Preload a row of A into registers for reuse
        double rowA[COARSENING];
        double results[COARSENING] = {0.0};
        
        for (int k = 0; k < N; ++k) {
            double ak = A[row * N + k];
            
            // Update all columns handled by this thread
            for (int offset = 0; offset < COARSENING && (colStart + offset) < N; ++offset) {
                results[offset] += ak * B[k * N + (colStart + offset)];
            }
        }
        
        // Write results back to global memory
        for (int offset = 0; offset < COARSENING && (colStart + offset) < N; ++offset) {
            C[row * N + colStart + offset] = results[offset];
        }
    }
}

// --- 4. Improved Full Optimized ---
__global__ void kernelFullOptimized(const double* A, const double* B_T, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int colStart = (blockIdx.x * blockDim.x + threadIdx.x) * COARSENING;

    if (row < N) {
        // Precompute results in registers for better performance
        double results[COARSENING] = {0.0};
        
        // Process in chunks for better cache behavior
        const int CHUNK_SIZE = 32;
        for (int i = 0; i < N; i += CHUNK_SIZE) {
            for (int k = i; k < min(N, i + CHUNK_SIZE); ++k) {
                double a = A[row * N + k];
                
                // Update all columns handled by this thread
                for (int offset = 0; offset < COARSENING && (colStart + offset) < N; ++offset) {
                    double b = B_T[(colStart + offset) * N + k];
                    results[offset] += a * b;
                }
            }
        }
        
        // Write results back to global memory
        for (int offset = 0; offset < COARSENING && (colStart + offset) < N; ++offset) {
            C[row * N + colStart + offset] = results[offset];
        }
    }
}

// --- 5. Improved Tiled Shared Memory ---
__global__ void kernelTiledSharedMemory(const double* A, const double* B, double* C, int N) {
    __shared__ double tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tileB[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    double sum = 0.0;

    // Cache blocking - process the matrix in tiles
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        // Collaborative loading with coalesced access pattern
        if (row < N && m * TILE_WIDTH + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + m * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0;

        if (col < N && m * TILE_WIDTH + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(m * TILE_WIDTH + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        // Compute using shared memory tiles
        #pragma unroll 
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }
    
    if (row < N && col < N)
        C[row * N + col] = sum;
}

__global__ void kernelTiledCoarsening(const double* A, const double* B, double* C, int N) {
    __shared__ double tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tileB[TILE_WIDTH][TILE_WIDTH * COARSENING];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int colBase = blockIdx.x * TILE_WIDTH * COARSENING;
    
    double results[COARSENING] = {0.0};
    
    // Process the matrix in tiles
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        // Load tile of A
        if (row < N && m * TILE_WIDTH + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + m * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0;
            
        // Load coarsened tile of B (each thread loads multiple elements)
        for (int c = 0; c < COARSENING; c++) {
            int col = colBase + c * TILE_WIDTH + threadIdx.x;
            if (col < N && m * TILE_WIDTH + threadIdx.y < N)
                tileB[threadIdx.y][c * TILE_WIDTH + threadIdx.x] = 
                    B[(m * TILE_WIDTH + threadIdx.y) * N + col];
            else
                tileB[threadIdx.y][c * TILE_WIDTH + threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        // Compute using shared memory tiles with coarsening
        for (int k = 0; k < TILE_WIDTH; ++k) {
            double aVal = tileA[threadIdx.y][k];
            for (int c = 0; c < COARSENING; c++) {
                results[c] += aVal * tileB[k][c * TILE_WIDTH + threadIdx.x];
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    for (int c = 0; c < COARSENING; c++) {
        int col = colBase + c * TILE_WIDTH + threadIdx.x;
        if (row < N && col < N)
            C[row * N + col] = results[c];
    }
}

// --- Validation ---
bool validate(const std::vector<double>& C, int N, double expected) {
    for (int i = 0; i < N * N; ++i)
        if (fabs(C[i] - expected) > 1e-5)
            return false;
    return true;
}

// --- Benchmark Wrapper ---
template <typename KernelFunc>
void runKernel(const char* name, KernelFunc kernel, const double* d_A, const double* d_B_or_BT,
               double* d_C, std::vector<double>& h_C, int N, dim3 grid, dim3 block) {
    size_t bytes = N * N * sizeof(double);
    cudaMemset(d_C, 0, bytes);

    // Warm-up run
    kernel<<<grid, block>>>(d_A, d_B_or_BT, d_C, N);
    cudaDeviceSynchronize();
    
    // Timing run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    kernel<<<grid, block>>>(d_A, d_B_or_BT, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Calculate performance metrics
    double seconds = milliseconds / 1000.0;
    double gflops = (2.0 * N * N * N) / (seconds * 1e9);  // FMA operations for matmul
    
    std::cout << "[" << name << "] Time: " << seconds << " s | "
              << "GFLOPS: " << gflops << " | "
              << (validate(h_C, N, N * 2.0) ? "Correct" : "Incorrect") << "\n";
              
    // Calculate memory bandwidth
    double memory_bytes = 3.0 * N * N * sizeof(double);  // Read A, read B, write C
    double bandwidth = memory_bytes / (seconds * 1e9);  // GB/s
    std::cout << "  Bandwidth: " << bandwidth << " GB/s\n";
}

// --- Main ---
int main() {
    int N = 1024; // Matrix size
    size_t bytes = N * N * sizeof(double);

    std::vector<double> h_A(N * N, 1.0), h_B(N * N, 2.0), h_B_T(N * N, 0.0), h_C(N * N, 0.0);

    // Transpose B to B_T
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            h_B_T[j * N + i] = h_B[i * N + j];

    double *d_A, *d_B, *d_B_T, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_B_T, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_T, h_B_T.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_basic((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    dim3 grid_coarsened((N + (block.x * COARSENING) - 1) / (block.x * COARSENING), (N + block.y - 1) / block.y);
    dim3 grid_tiled((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 grid_tiled_coarse((N + TILE_WIDTH * COARSENING - 1) / (TILE_WIDTH * COARSENING), (N + TILE_WIDTH - 1) / TILE_WIDTH);

    std::cout << "Matrix size: " << N << "x" << N << ", ";
    std::cout << "COARSENING = " << COARSENING << ", TILE_WIDTH = " << TILE_WIDTH << "\n\n";

    runKernel("1. Privatization Only", kernelPrivatizationOnly, d_A, d_B, d_C, h_C, N, grid_basic, block);
    runKernel("2. Memory Coalescing", kernelMemoryCoalescing, d_A, d_B_T, d_C, h_C, N, grid_basic, block);
    runKernel("3. Thread Coarsening", kernelThreadCoarsening, d_A, d_B, d_C, h_C, N, grid_coarsened, block);
    runKernel("4. Full Optimized", kernelFullOptimized, d_A, d_B_T, d_C, h_C, N, grid_coarsened, block);
    runKernel("5. Tiled Shared Memory", kernelTiledSharedMemory, d_A, d_B, d_C, h_C, N, grid_tiled, block);
    runKernel("6. Tiled + Coarsening", kernelTiledCoarsening, d_A, d_B, d_C, h_C, N, grid_tiled_coarse, block);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_B_T);
    cudaFree(d_C);
    return 0;
}