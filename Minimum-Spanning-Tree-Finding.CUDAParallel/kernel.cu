#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>

#define FULL_MASK 0xffffffff

using namespace std;
using namespace cooperative_groups;

__inline__ __device__ int smallWarpReduceSum(int val);
__inline__ __device__ int warpReduceSum(int val, unsigned mask);
__inline__ __device__ int blockReduceSum(int val);
__inline__ __device__ int deviceReduceSum(int val, int* sumArr);
void quickSort(int* B, int* A, int n);
void quickSort1(int* A, int n);
__global__ void kernel(int* V, int* VRoot, int* TreeSize, int* TreeNeigh,
    int* local_min, int* tree_min_edge, int* local_weight, int* VTree,
    const int N, const int M, const int* Neigh, const int* E, const int* Weight,
    int& NumOfIterations, int* reductionArr);
cudaError_t MSTBoruvka(int N, int M, int* Neigh, int* E, int* Weight, float gpuTime,
    int numOfIterations);
void generateRandomFullConnectedCSRGraph(int minWeight, int maxWeight, int numOfVertices,
    int* dests, int* neighDisps, int* weights) {
    neighDisps[0] = 0;
    int randomWeight;
    for (int i = 0; i < numOfVertices; i++)
    {
        int j;
        for (j = 0; j < i; j++) {
            randomWeight = rand() % (maxWeight - minWeight + 1) + minWeight;
            dests[i * (numOfVertices - 1) + j] = j;
            dests[j * (numOfVertices - 1) + i - 1] = i;
            weights[i * (numOfVertices - 1) + j] = randomWeight;
            weights[j * (numOfVertices - 1) + i - 1] = randomWeight;
        }
        neighDisps[i + 1] = neighDisps[i] + (numOfVertices - 1);
    }
}

int main() {
    int numOfVertices, numOfEdges;
    int* dests;
    int* weights;
    int* neighDisps;
    srand((unsigned)time(NULL));
    bool printMode = 0;
    int endPrintIndex = 100;
    int numOfIterations = 0;
    float gpuTime = 0.0;
    numOfVertices = 12000;
    numOfEdges = (numOfVertices - 1) * numOfVertices;
    int minWeight = -1000, maxWeight = 1000;
    neighDisps = new int[numOfVertices + 1];
    dests = new int[numOfEdges];
    weights = new int[numOfEdges];
    generateRandomFullConnectedCSRGraph(minWeight, maxWeight,
        numOfVertices, dests, neighDisps, weights);
    cout << "\nNumOfVertices: " << numOfVertices << "\n";
    cout << "\nNumOfEdges: " << numOfEdges << "\n";
    if (printMode) {
        cout << "\nGraph before sort:\n";
        for (int i = 0; i < numOfVertices; i++) {
            cout << i << " d -> ";
            for (int j = neighDisps[i]; (j < neighDisps[i + 1]) && (j <= j + endPrintIndex); j++)
            {
                cout << dests[j] << " ";
            }
            cout << "\n";
            cout << i << " w -> ";
            for (int j = neighDisps[i]; (j < neighDisps[i + 1]) && (j <= j + endPrintIndex); j++)
            {
                cout << weights[j] << " ";
            }
            cout << "\n";
        }
        cout << "\nNeigh disps:\n";
        for (int i = 0; (i < numOfVertices + 1) && (i <= endPrintIndex); i++) {
            cout << neighDisps[i] << " ";
        }
        cout << "\n";
    }
    for (int i = 0; i < numOfVertices; i++)
        quickSort(dests + neighDisps[i], weights + neighDisps[i], neighDisps[i + 1] - neighDisps[i]);
    if (printMode) {
        cout << "\nGraph after sort:\n";
        for (int i = 0; i < numOfVertices; i++) {
            cout << i << " d -> ";
            for (int j = neighDisps[i]; (j < neighDisps[i + 1]) && (j <= j + endPrintIndex); j++)
            {
                cout << dests[j] << " ";
            }
            cout << "\n";
            cout << i << " w -> ";
            for (int j = neighDisps[i]; (j < neighDisps[i + 1]) && (j <= j + endPrintIndex); j++)
            {
                cout << weights[j] << " ";
            }
            cout << "\n";
        }
        cout << "\nNeigh disps:\n";
        for (int i = 0; (i < numOfVertices + 1) && (i <= endPrintIndex); i++) {
            cout << neighDisps[i] << " ";
        }
        cout << "\n";
    }
    MSTBoruvka(numOfVertices, numOfEdges, neighDisps, dests, weights, gpuTime, numOfIterations);
    delete[] neighDisps;
    delete[] dests;
    delete[] weights;
    return 0;
}

void quickSort(int* B, int* A, int n) {
    int p;
    int i, j, temp, Btemp;
    i = 0, j = n - 1;
    p = A[n >> 1];
    do {
        while (A[i] < p) i++;
        while (A[j] > p) j--;
        if (i <= j) {
            temp = A[i];
            A[i] = A[j];
            A[j] = temp;
            Btemp = B[i];
            B[i] = B[j];
            B[j] = Btemp;
            i++;
            j--;
        }
    } while (i <= j);
    if (j > 0)
        quickSort(B, A, j + 1);
    if (n > i)
        quickSort(B + i, A + i, n - i);
    int s = 1;
    for (i = 0; i < n; ++i)
        if (A[i] != A[i + 1] || i == n - 1) {
            if (s == 2)
                if (B[i - 1] > B[i]) {
                    temp = B[i];
                    B[i] = B[i - 1];
                    B[i - 1] = temp;
                }
            if (s > 2)
                quickSort1(B + (i - s + 1), s);
            s = 1;
        }
        else if (A[i] == A[i + 1])
            ++s;
}

void quickSort1(int* A, int n) {
    int p;
    int i, j, temp;
    i = 0, j = n - 1;
    p = A[n >> 1];
    do {
        while (A[i] < p) i++;
        while (A[j] > p) j--;
        if (i <= j) {
            temp = A[i];
            A[i] = A[j];
            A[j] = temp;
            i++;
            j--;
        }
    } while (i <= j);
    if (j > 0)
        quickSort1(A, j + 1);
    if (n > i)
        quickSort1(A + i, n - i);
}

cudaError_t MSTBoruvka(int N, int M, int* Neigh, int* E, int* Weight, float gpuTime,
    int numOfIterations)
{
    int* dev_V, * dev_VRoot, * dev_TreeSize, * dev_TreeNeigh;
    int* dev_local_min, * dev_tree_min_edge, * dev_local_weight;
    int* dev_VTree;
    int* dev_Neigh, * dev_E, * dev_Weight;
    int* dev_NumOfIterations;
    int* dev_reductionArr;
    cudaError_t cudaStatus;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    int cudaVersion;
    cudaError_t cudaError = cudaRuntimeGetVersion(&cudaVersion);
    if (cudaError != cudaSuccess) {
        std::cerr << "Failed to get CUDA version: " << cudaGetErrorString(cudaError) << std::endl;
    }
    int major = cudaVersion / 1000;
    int minor = (cudaVersion % 1000) / 10;
    int blockSize;
    int gridSize;
    if (N <= 256) {
        blockSize = 10;
        gridSize = 1;
    }
    else {
        blockSize = 512;
        gridSize = min(N / 512, 16);
    }
    printf("\nblockSize: %d\n", blockSize);
    printf("gridSize: %d\n", gridSize);
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, 0);
    if (supportsCoopLaunch != 1) throw runtime_error("Cooperative Launch is not supported on this machine configuration.");
    cudaStatus = cudaMalloc((void**)&dev_V, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_VRoot, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_TreeSize, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_TreeNeigh, (N + 1) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_VTree, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_local_min, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_local_weight, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_tree_min_edge, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_Neigh, (N + 1) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_E, M * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_Weight, M * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_NumOfIterations, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_reductionArr, gridSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    int* V, * VRoot, * TreeSize, * TreeNeigh;
    V = (int*)malloc(N * sizeof(int));
    VRoot = (int*)malloc(N * sizeof(int));
    TreeSize = (int*)malloc(N * sizeof(int));
    TreeNeigh = (int*)malloc((N + 1) * sizeof(int));
    for (int i = 0; i < N; ++i) {
        V[i] = i;
        VRoot[i] = i;
        TreeSize[i] = 1;
        TreeNeigh[i] = i;
    }
    TreeNeigh[N] = N;
    cudaStatus = cudaMemcpy(dev_V, &V[0], N * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_VRoot, &VRoot[0], N * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_TreeSize, &TreeSize[0], N * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_TreeNeigh, &TreeNeigh[0], (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_Neigh, &Neigh[0], (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_E, &E[0], M * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_Weight, &Weight[0], M * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    dim3 blocks(gridSize);
    dim3 threads(blockSize);
    void* args[] = {
        &dev_V, &dev_VRoot, &dev_TreeSize, &dev_TreeNeigh, &dev_local_min,
        &dev_tree_min_edge, &dev_local_weight,
        &dev_VTree,
        &N, &M, &dev_Neigh, &dev_E, &dev_Weight,
        &dev_NumOfIterations,
        &dev_reductionArr
    };
    cudaEvent_t startTime, stopTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaEventRecord(startTime, 0);
    cudaLaunchCooperativeKernel((void*)kernel, blocks, threads, args);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
        goto Error;
    }
    cudaEventRecord(stopTime, 0);
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&gpuTime, startTime, stopTime);
    cudaEventDestroy(startTime);
    cudaEventDestroy(stopTime);
    cudaStatus = cudaMemcpy(&numOfIterations, dev_NumOfIterations, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cout << "\nnumOfIterations:" << numOfIterations;
    cout << "\ngpuTime: " << gpuTime << " ms";
Error:
    cudaFree(dev_V);
    cudaFree(dev_VRoot);
    cudaFree(dev_TreeSize);
    cudaFree(dev_TreeNeigh);
    cudaFree(dev_local_min);
    cudaFree(dev_tree_min_edge);
    cudaFree(dev_local_weight);
    cudaFree(dev_VTree);
    cudaFree(dev_Neigh);
    cudaFree(dev_E);
    cudaFree(dev_Weight);
    cudaFree(dev_NumOfIterations);
    cudaFree(dev_reductionArr);
    return cudaStatus;
}

__global__ void kernel(int* V, int* VRoot, int* TreeSize, int* TreeNeigh,
    int* local_min, int* tree_min_edge, int* local_weight, int* VTree,
    const int N, const int M, const int* Neigh, const int* E, const int* Weight,
    int& NumOfIterations, int* reductionArr)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int endPrintIndex = 150;
    int j, s, temp;
    int weight;
    int iteration = 0;
    __shared__ int Tnum;
    if (threadIdx.x == 0) {
        Tnum = N;
    }
    __syncthreads();
    int T = 0;
    while (Tnum != 1) {
        for (int i = threadId; i < Tnum; i += blockDim.x * gridDim.x) {
            VTree[i] = -1;
        }
        this_grid().sync();
        for (int i = threadId; i < N; i += blockDim.x * gridDim.x) {
            local_min[i] = 1000000; local_weight[i] = 100000;
            for (int k = Neigh[i]; k < Neigh[i + 1]; ++k) {
                if (VRoot[E[k]] != VRoot[i]) {
                    local_min[i] = E[k];
                    local_weight[i] = Weight[k];
                    while (Weight[++k] == local_weight[i]) {
                        if (VRoot[E[k]] < VRoot[local_min[i]] && VRoot[E[k]] != VRoot[i]) {
                            local_min[i] = E[k];
                        }
                    }
                    break;
                }
            }
        }
        this_grid().sync();
        for (int i = threadId; i < Tnum; i += blockDim.x * gridDim.x) {
            tree_min_edge[i] = -1;
            for (int k = TreeNeigh[i]; k < TreeNeigh[i + 1]; ++k) {
                if (tree_min_edge[i] == -1) {
                    if (local_min[V[k]] != 1000000) {
                        tree_min_edge[i] = local_min[V[k]];
                        weight = local_weight[V[k]];
                    }
                }
                else if (local_weight[V[k]] < weight && local_weight[V[k]] != 100000) {
                    tree_min_edge[i] = local_min[V[k]];
                    weight = local_weight[V[k]];
                }
                else if (local_weight[V[k]] == weight) {
                    if (VRoot[local_min[V[k]]] < VRoot[tree_min_edge[i]]) {
                        tree_min_edge[i] = local_min[V[k]];
                        weight = local_weight[V[k]];
                    }
                }
            }
            tree_min_edge[i] = VRoot[tree_min_edge[i]];
        }
        this_grid().sync();
        for (int i = threadId; i < Tnum; i += blockDim.x * gridDim.x) {
            temp = tree_min_edge[tree_min_edge[i]];
            if (temp == i) {
                if (i < tree_min_edge[i]) {
                    tree_min_edge[i] = i;
                    VTree[i] = i;
                }
            }
            if (i != tree_min_edge[i]) {
                ++T;
            }
        }
        T = deviceReduceSum(T, reductionArr);
        for (int i = threadId; i < Tnum; i += blockDim.x * gridDim.x) {
            if (tree_min_edge[i] != i) {
                temp = i;
                while (temp != tree_min_edge[temp]) {
                    temp = tree_min_edge[temp];
                }
                for (int k = TreeNeigh[i]; k < TreeNeigh[i + 1]; ++k)
                    VRoot[V[k]] = temp;
                atomicAdd((TreeSize + temp), (TreeNeigh[i + 1] - TreeNeigh[i]));
                TreeSize[i] = 0;
            }
        }
        this_grid().sync();
        if (threadId == 1) {
            //printf("\n5.1. VRoot:\n");
            //for (int i = 0; (i < Tnum) && (i <= endPrintIndex); i++) {
            //    printf("%d ", VRoot[i]);
            //}
            //printf("\n");
            //printf("\n5.2. V:\n");
            //for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
            //    printf("%d ", V[i]);
            //}
            //printf("\n");
            //printf("\n5.3. VTree:\n");
            //for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
            //    printf("%d ", VTree[i]);
            //}
            //printf("\n");
            //printf("\n5.3. TreeSize:\n");
            //for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
            //    printf("%d ", TreeSize[i]);
            //}
            //printf("\n");
            //printf("\n5.4. TreeNeigh:\n");
            //for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
            //    printf("%d ", TreeNeigh[i]);
            //}
            //printf("\n");
            //printf("\n5.5. TreeMinEdge:\n");
            //for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
            //    printf("%d ", tree_min_edge[i]);
            //}
            //printf("\n");
        }
        if (threadId == 0) {
            s = 0;
            TreeNeigh[0] = 0;
            for (int i = 0; i < Tnum; ++i) {
                if (VTree[i] != -1) {
                    VTree[i] = s;
                    TreeSize[s] = TreeSize[i];
                    if (i != s)
                        TreeSize[i] = 0;
                    ++s;
                    TreeNeigh[s] = TreeNeigh[s - 1] + TreeSize[s - 1];
                }
            }
        }
        this_grid().sync();
        for (int i = threadId; i < Tnum; i += blockDim.x * gridDim.x) {
            if (VTree[i] != -1) {
                j = TreeNeigh[VTree[i]];
                s = 0;
                for (int k = 0; k < N; ++k) {
                    if (VRoot[k] == i) {
                        V[j++] = k;
                        ++s;
                        if (s == TreeSize[VTree[i]])
                            break;
                    }
                }
            }
        }
        if (threadIdx.x == 0) {
            Tnum -= T;
        }
        T = 0;
        __syncthreads();
        for (int i = threadId; i < Tnum; i += blockDim.x * gridDim.x)
            for (int k = TreeNeigh[i]; k < TreeNeigh[i + 1]; ++k)
                VRoot[V[k]] = i;
        if (threadId == 0) ++NumOfIterations;
    }
}

__inline__ __device__ int smallWarpReduceSum(int val)
{
    if ((blockDim.x & (blockDim.x - 1)) == 0) {
        for (int offset = (blockDim.x % warpSize) / 2; offset > 0; offset /= 2)
            val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    else {
        for (int i = (blockDim.x % warpSize) - 1; i > 0; i -= 1) {
            if (threadIdx.x % warpSize == 0) {
                val += __shfl_down_sync(FULL_MASK, val, 1);
            }
            else {
                val = __shfl_down_sync(FULL_MASK, val, 1);
            }
        }
    }
    return val;
}

__inline__ __device__ int warpReduceSum(int val, unsigned mask = FULL_MASK)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}

__inline__ __device__ int blockReduceSum(int val)
{
    static __shared__ int shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    int numOfSumInBlock = blockDim.x / warpSize;
    if (blockDim.x % warpSize != 0) {
        numOfSumInBlock++;
        if ((wid + 1) != numOfSumInBlock) {
            val = warpReduceSum(val);
        }
        else {
            val = smallWarpReduceSum(val);
        }
    }
    else {
        val = warpReduceSum(val);
    }
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < numOfSumInBlock) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__inline__ __device__ int deviceReduceSum(int val, int* sumArr)
{
    int sum = 0;
    if (blockDim.x > warpSize) {
        sum = blockReduceSum(val);
    }
    else {
        unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < blockDim.x);
        sum = warpReduceSum(val, mask);
    }
    if (threadIdx.x == 0) {
        sumArr[blockIdx.x] = sum;
    }
    this_grid().sync();
    int result = 0;
    if (threadIdx.x == 0) {
        for (int i = 0; i < gridDim.x; i++) {
            result += sumArr[i];
        }
    }
    return result;
}

__inline__ __device__ void deviceBellmanFord(int threadId, int indexForArr1, int indexForArr2, int indexForGlobalMem, int numOfIterations,
    int minIterations, int* shortestPathsToVertices, int* shortestTempPaths, int* previousVertices, int startVertex, const int* edges, const int* weights,
    const int* rangesOfAdjacentVertices, int numOfVertices, int* sumArr, int* numOfGlobalIterations)
{
    int INF = 2000000000;
    int i, k, j;
    for (i = 0; i < minIterations; i += blockDim.x) {
        __syncthreads();
        shortestTempPaths[indexForArr1 + i] = INF;
        previousVertices[indexForArr2 + i] = -1;
        shortestPathsToVertices[indexForGlobalMem + i] = INF;
    }
    for (; i < numOfIterations; i += blockDim.x) {
        shortestTempPaths[indexForArr1 + i] = INF;
        previousVertices[indexForArr2 + i] = -1;
        shortestPathsToVertices[indexForGlobalMem + i] = INF;
    }
    __shared__ bool completionOfTheMainLoop;
    int localCompletionOfTheMainLoop;
    if (threadId % numOfVertices == startVertex) {
        shortestPathsToVertices[startVertex] = 0;
        shortestTempPaths[startVertex] = 0;
    }
    if (threadIdx.x == 0) {
        completionOfTheMainLoop = true;
    }
    int newShortestPath, edge;
    int maxNumOfGlobalIterations = numOfVertices - 1, numOfJCycleIterations;
    this_grid().sync();
    for (i = 0; (i < maxNumOfGlobalIterations) && completionOfTheMainLoop; ++i) {
        localCompletionOfTheMainLoop = 0;
        for (k = 0; k < minIterations; k += blockDim.x) {
            __syncthreads();
            numOfJCycleIterations = rangesOfAdjacentVertices[indexForGlobalMem + k + 1];
            for (j = rangesOfAdjacentVertices[indexForGlobalMem + k]; j < numOfJCycleIterations; ++j) {
                edge = edges[j];
                newShortestPath = shortestPathsToVertices[edge] + weights[j];
                if (shortestTempPaths[indexForArr1 + k] > newShortestPath) {
                    shortestTempPaths[indexForArr1 + k] = newShortestPath;
                    previousVertices[indexForArr2 + k] = edge;
                    ++localCompletionOfTheMainLoop;
                }
            }
        }
        for (; k < numOfIterations; k += blockDim.x) {
            numOfJCycleIterations = rangesOfAdjacentVertices[indexForGlobalMem + k + 1];
            for (j = rangesOfAdjacentVertices[indexForGlobalMem + k]; j < numOfJCycleIterations; ++j) {
                edge = edges[j];
                newShortestPath = shortestPathsToVertices[edge] + weights[j];
                if (shortestTempPaths[indexForArr1 + k] > newShortestPath) {
                    shortestTempPaths[indexForArr1 + k] = newShortestPath;
                    previousVertices[indexForArr2 + k] = edge;
                    ++localCompletionOfTheMainLoop;
                }
            }
        }
        for (k = 0; k < minIterations; k += blockDim.x) {
            __syncthreads();
            shortestPathsToVertices[indexForGlobalMem + k] = shortestTempPaths[indexForArr1 + k];
        }
        for (; k < numOfIterations; k += blockDim.x) {
            shortestPathsToVertices[indexForGlobalMem + k] = shortestTempPaths[indexForArr1 + k];
        }
        if (threadIdx.x == 0) {
            if (localCompletionOfTheMainLoop == 0) completionOfTheMainLoop = false;
        }
        __syncthreads();
    }
    if (threadId == 1) *numOfGlobalIterations = i;
}