#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include <unordered_set>
#include <queue>
#include <iostream>

using namespace std;

void generateRandomFullConnectedCSRGraph(
	int minWeight,
	int maxWeight,
	int numOfVertices,
	int* dests,
	int* neighDisps,
	int* weights
	)
{
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

void Boruvka(
	int N,
	int M,
	int* Neigh,
	int* E,
	int* Weight,
	int endPrintIndex = 5
	)
{
	int i, j, k, s, temp;
	int* V, * VRoot, * TreeSize, * TreeNeigh;
	int* local_min, * tree_min_edge, * local_weight;
	int* VTree;
	int weight;
	double ompstart, ompend;
	int* res, * sourceRes, * finalDestRes, * finalSourceRes;
	V = (int*)malloc(N * sizeof(int));
	VRoot = (int*)malloc(N * sizeof(int));
	TreeSize = (int*)malloc(N * sizeof(int));
	TreeNeigh = (int*)malloc((N + 1) * sizeof(int));
	VTree = (int*)malloc(N * sizeof(int));
	local_min = (int*)malloc(N * sizeof(int));
	local_weight = (int*)malloc(N * sizeof(int));
	tree_min_edge = (int*)malloc(N * sizeof(int));
	res = (int*)malloc(N * sizeof(int));
	sourceRes = (int*)malloc(N * sizeof(int));
	finalDestRes = (int*)malloc(N * sizeof(int));
	finalSourceRes = (int*)malloc(N * sizeof(int));
	for (i = 0; i < N; ++i) {
		VRoot[i] = i;
		V[i] = i;
		TreeNeigh[i] = i;
		TreeSize[i] = 1;
	}
	TreeNeigh[i] = i;
	int iteration = 0;
	int Tnum = N;
	int T = Tnum; 
	int numOfEdges;
	int finalResIterator = 0;
	ompstart = omp_get_wtime();
	while (Tnum != 1) {
#pragma omp parallel for
		for (i = 0; i < Tnum; ++i)
			VTree[i] = -1;
		//cout << "\n1. Search min edge for every vertex:\n";
#pragma omp parallel for private (k)
		for (i = 0; i < N; ++i) {
			local_min[i] = 1000000;
			local_weight[i] = 100000;
			for (k = Neigh[i]; k < Neigh[i + 1]; ++k) {
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
		/*cout << "\n1.1. Local min:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << local_min[i] << " ";
		}
		cout << "\n";
		cout << "\n1.2. Local weight:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << local_weight[i] << " ";
		}
		cout << "\n";
		cout << "\n2. Search min edge for every tree:\n";*/
#pragma omp parallel for private (k, weight)
		for (i = 0; i < Tnum; ++i) {
			res[i] = -1;
			tree_min_edge[i] = -1;
			for (k = TreeNeigh[i]; k < TreeNeigh[i + 1]; ++k) { 
				if (tree_min_edge[i] == -1) {
					if (local_min[V[k]] != 1000000) {
						sourceRes[i] = V[k];
						tree_min_edge[i] = local_min[V[k]];
						weight = local_weight[V[k]];
					}
				}
				else if (local_weight[V[k]] < weight && local_weight[V[k]] != 100000) { // 2000000
					sourceRes[i] = V[k];
					tree_min_edge[i] = local_min[V[k]];
					weight = local_weight[V[k]];
				}
				else if (local_weight[V[k]] == weight) {
					if (VRoot[local_min[V[k]]] < VRoot[tree_min_edge[i]]) {
						sourceRes[i] = V[k];
						tree_min_edge[i] = local_min[V[k]];
						weight = local_weight[V[k]];
					}
				}
			}
			res[i] = tree_min_edge[i];
			tree_min_edge[i] = VRoot[tree_min_edge[i]];
		}
		/*cout << "\n2.1. Tree min edge:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << tree_min_edge[i] << " ";
		}
		cout << "(Tnum)\n";
		cout << "\n2.2. sourceRes:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << sourceRes[i] << " ";
		}
		cout << "(Tnum)\n";
		cout << "\n2.3. res:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << res[i] << " ";
		}
		cout << "(Tnum)\n";
		cout << "\n3. Delete duplicate edges:\n";*/
#pragma omp parallel for private (temp) reduction(-:T)
		for (i = 0; i < Tnum; ++i) {
			temp = tree_min_edge[tree_min_edge[i]];
			if (temp == i) {
				if (i < tree_min_edge[i]) {
					res[i] = i;
					sourceRes[i] = i;
					tree_min_edge[i] = i; 
					VTree[i] = i; 
				}
			}
			if (i != tree_min_edge[i]) { 
				--T;
			}
		}
		/*cout << "\n3.1. Tree min edge:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << tree_min_edge[i] << " ";
		}
		cout << "(Tnum)\n";
		cout << "\n3.2. sourceRes:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << sourceRes[i] << " ";
		}
		cout << "(Tnum)\n";
		cout << "\n3.3. res:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << res[i] << " ";
		}
		cout << "(Tnum)\n";
		cout << "\n4. Build MST:\n";*/
		numOfEdges = 0;
		vector<int> threadResSizes(8, 0);
#pragma omp parallel private(k, temp)
		{
			vector<int> localDestResQueue;
			vector<int> localSourceResQueue;
			int threadNum = omp_get_thread_num();
			int numThreads = omp_get_num_threads();
#pragma omp for schedule(dynamic,32)
			for (i = 0; i < Tnum; ++i) {
				if (tree_min_edge[i] != i) { 
					localDestResQueue.push_back(res[i]);
					localSourceResQueue.push_back(sourceRes[i]);
					threadResSizes[threadNum]++;
					//cout << "threadResSizes: " << threadNum;
					temp = i;
					while ((temp != tree_min_edge[temp])) { 
						temp = tree_min_edge[temp];
					}
					for (k = TreeNeigh[i]; k < TreeNeigh[i + 1]; ++k) 
						VRoot[V[k]] = temp; 
#pragma omp atomic
					TreeSize[temp] += (TreeNeigh[i + 1] - TreeNeigh[i]);
					TreeSize[i] = 0;
				}
			}
			int startPosition = finalResIterator;
			for (int j = 0; j < threadNum; ++j) { 
				startPosition += threadResSizes[j];
			}
			//cout << "startPosition " << threadNum << ": " << startPosition << " ";
			for (int j = 0; j < localDestResQueue.size(); ++j) {
				finalDestRes[startPosition] = localDestResQueue[j];
				finalSourceRes[startPosition] = localSourceResQueue[j];
				startPosition++;
			}
		}
		for (int j = 0; j < threadResSizes.size(); ++j)
			finalResIterator += threadResSizes[j];
		/*cout << "\n4.1. VRoot:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << VRoot[i] << " ";
		}
		cout << "\n";
		cout << "\n5.1. VRoot:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << VRoot[i] << " ";
		}
		cout << "\n";
		cout << "\n5.2. V:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << V[i] << " ";
		}
		cout << "\n";
		cout << "\n5.3. VTree:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << VTree[i] << " ";
		}
		cout << "(Tnum)\n";
		cout << "\n5.4. TreeSize:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << TreeSize[i] << " ";
		}
		cout << "\n";
		cout << "\n5.5. TreeNeigh:\n";
		for (int i = 0; (i < N + 1) && (i <= endPrintIndex); i++) {
			cout << TreeNeigh[i] << " ";
		}
		cout << "(Tnum)\n";
		cout << "\n5.6. TreeMinEdge:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << tree_min_edge[i] << " ";
		}
		cout << "(Tnum)\n";
		cout << "\n5. Renumering:\n";*/
		s = 0;
		TreeNeigh[0] = 0;
		for (i = 0; i < Tnum; ++i) {
			if (VTree[i] != -1) {
				VTree[i] = s;
				TreeSize[s] = TreeSize[i];
				if (i != s)
					TreeSize[i] = 0;
				++s;
				TreeNeigh[s] = TreeNeigh[s - 1] + TreeSize[s - 1];
			}
		}
#pragma omp parallel for private (k, j, s) schedule(dynamic,32)
		for (i = 0; i < Tnum; ++i) {
			if (VTree[i] != -1) {
				j = TreeNeigh[VTree[i]]; 
				s = 0;
				for (k = 0; k < N; ++k) {
					if (VRoot[k] == i) { 
						V[j++] = k;
						++s;
						if (s == TreeSize[VTree[i]]) 
							break;
					}
				}
			}
		}
		Tnum = T;
#pragma omp parallel for private (k) schedule(dynamic,32)
		for (i = 0; i < Tnum; ++i)
			for (k = TreeNeigh[i]; k < TreeNeigh[i + 1]; ++k) 
				VRoot[V[k]] = i;
		/*cout << "\n6.1. VRoot:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << VRoot[i] << " ";
		}
		cout << "\n";
		cout << "\n6.2. V:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << V[i] << " ";
		}
		cout << "\n";
		cout << "\n6.3. VTree:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << VTree[i] << " ";
		}
		cout << "(Tnum)\n";
		cout << "\n6.4. TreeSize:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << TreeSize[i] << " ";
		}
		cout << "\n";
		cout << "\n6.5. TreeNeigh:\n";
		for (int i = 0; (i < N + 1) && (i <= endPrintIndex); i++) {
			cout << TreeNeigh[i] << " ";
		}
		cout << "(Tnum)\n";
		cout << "\n6.6. TreeMinEdge:\n";
		for (int i = 0; (i < N) && (i <= endPrintIndex); i++) {
			cout << tree_min_edge[i] << " ";
		}
		cout << "(Tnum)\n";*/
		++iteration;
	}
	/*cout << "\n7.1 finalDestRes:\n";
	for (int i = 0; (i < (N - 1)) && (i <= endPrintIndex); i++) {
		cout << finalDestRes[i] << " ";
	}
	cout << "(N-1)\n";
	cout << "\n7.2 finalSourceRes:\n";
	for (int i = 0; (i < (N - 1)) && (i <= endPrintIndex); i++) {
		cout << finalSourceRes[i] << " ";
	}
	cout << "(N-1)\n";*/
	ompend = omp_get_wtime();
	printf("iteration=%d\n", iteration);
	printf("Time = %f\n", ompend - ompstart);
	free(TreeSize);
	free(TreeNeigh);
	free(VRoot);
	free(V);
	free(VTree);
	free(local_min);
	free(tree_min_edge);
	free(res);
	free(sourceRes);
	free(finalSourceRes);
	free(finalDestRes);
}

int quickSortPartition(
	vector <int>& arr,
	vector <int>& arr2,
	int start,
	int end
	)
{
	int pivot = arr[end];
	int pIndex = start;
	for (int i = start; i < end; i++)
	{
		if (arr[i] <= pivot)
		{
			swap(arr[i], arr[pIndex]);
			swap(arr2[i], arr2[pIndex]);
			pIndex++;
		}
	}
	swap(arr[pIndex], arr[end]);
	swap(arr2[pIndex], arr2[end]);
	return pIndex;
}

void tailRecursiveQuickSort(
	vector <int>& arr,
	vector <int>& arr2,
	int start,
	int end
	)
{
	while (start < end)
	{
		int pivot = quickSortPartition(arr, arr2, start, end);
		if (pivot - start < end - pivot)
		{
			tailRecursiveQuickSort(arr, arr2, start, pivot - 1);
			start = pivot + 1;
		}
		else
		{
			tailRecursiveQuickSort(arr, arr2, pivot + 1, end);
			end = pivot - 1;
		}
	}
}

void quickSort1(int* A, int n)
{
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

void quickSort(int* B, int* A, int n)
{
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

int main() {
	int numOfVertices, numOfEdges;
	int* dests;
	int* weights;
	int* neighDisps;
	srand((unsigned)time(NULL));
	bool printMode = 0;
	int endPrintIndex = 100;
	omp_set_num_threads(4); // 8
	
	/*numOfVertices = 6;
	numOfEdges = 20;
	int neighDispsC[] = { 0,3,6,11,14,17,20 };
	int destsC[] = { 1,2,3, 0,2,4, 0,1,3,4,5, 0,2,5, 1,2,5, 2,3,4 };
	int weightsC[] = { 6,1,5, 6,5,3, 1,5,5,6,4, 5,5,2, 3,6,6, 4,2,6 };
	neighDisps = new int[numOfVertices + 1];
	dests = new int[numOfEdges];
	weights = new int[numOfEdges];
	for (int i = 0; i < numOfVertices + 1; i++) {
		neighDisps[i] = neighDispsC[i];
	}
	for (int i = 0; i < numOfEdges; i++) {
		dests[i] = destsC[i];
		weights[i] = weightsC[i];
	}*/

	numOfVertices = 12000;
	numOfEdges = (numOfVertices-1) * numOfVertices;
	int minWeight = -1000, maxWeight = 1000;
	//int minWeight = 1, maxWeight = 20;
	neighDisps = new int[numOfVertices + 1];
	dests = new int[numOfEdges];
	weights = new int[numOfEdges];
	generateRandomFullConnectedCSRGraph(
		minWeight,
		maxWeight,
		numOfVertices,
		dests,
		neighDisps,
		weights);
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
	//int startPosition, endPosition;
	//for (int i = 0; i < numOfVertices; i++) {
	//	startPosition = neighDisps[i];
	//	endPosition = neighDisps[i + 1] - 1;
	//	if (startPosition != endPosition)
	//		tailRecursiveQuickSort(weights, dests, startPosition, endPosition);
	//}
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
	Boruvka(numOfVertices, numOfEdges, neighDisps, dests, weights, endPrintIndex);
	delete[] neighDisps;
	delete[] dests;
	delete[] weights;
	return 0;
}
