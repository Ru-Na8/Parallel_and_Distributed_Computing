#  Parallel and Distributed Computing (4DT906)

This repository contains my work for the course **Parallel and Distributed Computing (4DT906)** at Linnaeus University.  
The assignments explore how performance of **matrix multiplication (C = A × B)** can be dramatically improved by applying different optimization techniques — first on **multi-core CPUs** with OpenMP, and later on **GPUs** with CUDA.

---

## Why Matrix Multiplication?
Matrix multiplication is one of the most fundamental operations in computing, with applications in:
- Machine learning  
- Graphics  
- Physics simulations  
- Scientific computing  

Naively, it runs in **O(N³)**, but with the right optimizations, we can unleash the full power of modern hardware.

---

## CPU Journey – from naive to parallel
### 🔹 Step 1: Naive implementation
- Triple loop (`ijk` order).  
- Very poor cache locality → lots of cache misses.  
📄 [`P1.cpp`](./P1.cpp)

### 🔹 Step 2: Loop reordering
- Reordered loops (`ikj` order).  
- Keeps `A[i][k]` in registers.  
- Huge improvement in cache usage.  
📄 [`P1.cpp`](./P1.cpp)

### 🔹 Step 3: Blocking (tiling)
- Divide matrices into smaller `Block_Size × Block_Size` tiles.  
- Improves cache reuse.  
- Best performance at block size 8–16.  
📄 [`P2.cpp`](./P2.cpp)

### 🔹 Step 4: OpenMP parallelization
- Add multi-threading to blocked multiplication.  
- Scales across CPU cores.  
- Best with 4–8 threads.  
📄 [`P3.cpp`](./P3.cpp)

💡 **Takeaway**: Small algorithmic tweaks + parallelism = **massive performance gains**.  

---

## GPU Journey – unleashing CUDA
After optimizing on CPUs, the challenge was to port matrix multiplication to **CUDA GPUs** and exploit massive parallelism.

### 🔹 Privatization
- Store partial sums in registers.  
- Consistently fastest (~0.0166 s).  

### 🔹 Memory coalescing
- Align memory accesses for efficiency.  
- Poor alone, since values still fetched repeatedly.  

### 🔹 Thread coarsening
- Each thread computes multiple results.  
- Best performance at coarsening factor 32 (~0.0147 s).  

### 🔹 Tiled shared memory
- Cache tiles of A and B in fast on-chip memory.  
- Reduces global memory access.  

### 🔹 Combined (tiling + coarsening)
- Balanced reuse and workload.  
- Stable performance (~0.018–0.019 s).  

📄 CUDA Implementation: [`A3.cu`](./A3.cu)  

---

## Experiments & Results
- **CPU**: Blocking + OpenMP gave **5–10× speedup** compared to naive.  
- **GPU**: CUDA optimizations achieved **20–30× speedup** compared to CPU naive.  
- Performance depends on cache size, number of threads, and tile width.  

📄 Full results: [Experiments.md](./Experiments.md)  
📄 Reports: [Assignment 1](./Programming%20assignment%201%20-%20Google%20Dokument.pdf), [Assignment 3](./Assignment-3_Parallel4DT906.pdf)  

---

##  How to Run
### CPU (C++)
```bash
# Problem 1 (Naive & Loop Reorder)
clang++ -std=c++17 -O3 P1.cpp -o P1 && ./P1

# Problem 2 (Blocking)
clang++ -std=c++17 -O3 P2.cpp -o P2 && ./P2

# Problem 3 (Blocking + OpenMP)
clang++ -std=c++17 -O3 -fopenmp P3.cpp -o P3 -lomp && ./P3

### GPU (CUDA)
nvcc -O3 A3.cu -o A3
./A3
