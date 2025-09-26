### Programming Assignment 1 
## Matrix Multiplication Optimization
==============================================
## This assignment focus on matrix multiplication and optimization techniques.
**- Naive implementation ijk order then optimized by loop reordering using ikj order. File name **P1.cpp**
****- Blocking.  File name **P2.cpp**
**- Blocked + Parallelization using `OpenMP`. File name **P3.cpp**


## How to Compile & Run
------------------------
###  Problem 1

```bash
clang++ -std=c++17 -O3 P1.cpp -o P1
./P1
```

### Problem 2

```bash
clang++ -std=c++17 -O3 P2.cpp -o P2
./P2
```

```python
**Note:** Edit `BLOCK_SIZE` inside the code to test 8, 16, 32, or 64.

```


### Problem 3

```bash
clang++ -std=c++17 -O3 -fopenmp \
    -I/usr/local/opt/libomp/include \
    -L/usr/local/opt/libomp/lib -lomp \
    P3.cpp -o P3

./P3
```