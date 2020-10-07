#include <iostream>
#include <cmath>
#include <ctime>
#include <chrono>
#include <random>


#define N 100000000
#define TIMING
#ifdef TIMING
#define INIT_TIMER auto start = std::chrono::high_resolution_clock::now();
#define START_TIMER start = std::chrono::high_resolution_clock::now();
#define STOP_TIMER std::cout << "Runtime of " << N << ": " << \
	std::chrono::duration_cast<std::chrono::milliseconds>( \
		std::chrono::high_resolution_clock::now()-start \
	).count() << " ms " << std:: endl;
#else
#define INIT_TIMER
#define START_TIMER
#define START_TIMER()
#endif

__global__ void buildPoint(double *a, double *b, double *c) {
	if(pow(a[blockIdx.x], 2) + pow(b[blockIdx.x],2) <= 0.25) {
		c[blockIdx.x] = 1.0;
	}
}

int main(void) {
	INIT_TIMER
	START_TIMER
	std::random_device rd;
	std::mt19937 eng(rd());
	std::uniform_real_distribution<> distr(-0.5, 0.5);
	double *a, *b, *c;
	double *d_a, *d_b, *d_c;
	int size = N * sizeof(double);
	
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	
	a = (double *)malloc(size);
	b = (double *)malloc(size);
	c = (double *)malloc(size);
	
	for(int i = 0; i < N; ++i) {
		a[i] = distr(eng);
		b[i] = distr(eng);
	}
	
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	
	buildPoint<<<N,1>>>(d_a, d_b, d_c);
	
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	
	double pointsIn = 0;
	for(int i = 0; i < N; ++i)
		pointsIn += c[i];
		
	STOP_TIMER
	
	std::cout.precision(9);
	std::cout << (4 * pointsIn) / N << std::endl;
	
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}