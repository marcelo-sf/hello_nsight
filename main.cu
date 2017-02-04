#include <stdio.h>
#include <math.h>
#include <cuComplex.h>

#include "book.h"

__constant__ double pi;

__device__ void DoubleComplexExp(cuDoubleComplex arg, cuDoubleComplex* res) {
	double s, c;
	double e = exp(arg.x);
	sincos(arg.y, &s, &c);
	res->x = c * e;
	res->y = s * e;
}

__global__ void GenerateComplexSinusoid(const double frequency,
		const unsigned int sampleCount, cuDoubleComplex* sinusoid) {
	int sampleIndex = blockIdx.x;

	cuDoubleComplex dftExponent = make_cuDoubleComplex(0.0,
			-2.0 * pi * frequency * sampleIndex / sampleCount);
	DoubleComplexExp(dftExponent, &(sinusoid[sampleIndex]));
}

__device__ void DeviceGenerateComplexSinusoid(const double frequency,
		const unsigned int sampleCount, cuDoubleComplex* sinusoid) {
	int sampleIndex = blockIdx.x;

	cuDoubleComplex dftExponent = make_cuDoubleComplex(0.0,
			-2.0 * pi * frequency * sampleIndex / sampleCount);
	DoubleComplexExp(dftExponent, &(sinusoid[sampleIndex]));
}

__global__ void GenerateDftTerms(const unsigned int sampleCount, cuDoubleComplex* signal, cuDoubleComplex* dftTerms) {
	int sampleIndex = blockIdx.x;
	double frequency = blockIdx.y;

	cuDoubleComplex dftExponent = make_cuDoubleComplex(0.0,-2.0 * pi * frequency * sampleIndex / sampleCount);

	cuDoubleComplex sinusoidValue;
	DoubleComplexExp(dftExponent, &sinusoidValue);
	dftTerms[sampleIndex + sampleCount * blockIdx.y] = cuCmul(signal[(int)frequency], sinusoidValue);

}

__global__ void Dft(const unsigned int sampleCount, cuDoubleComplex* dftTerms, cuDoubleComplex* dftResult) {
	int sampleIndex = blockIdx.x;

	cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
	for (unsigned int index = 0; index < sampleCount; index++) {
		sum.x += dftTerms[sampleIndex + sampleCount * index].x;
		sum.y += dftTerms[sampleIndex + sampleCount * index].y;
	}

	dftResult[sampleIndex] = sum;
}

void examineDftTerms(cuDoubleComplex* dev_dftTerms, unsigned int sampleCount) {
	cuDoubleComplex* dftTerms = (cuDoubleComplex*)malloc(sampleCount*sampleCount*sizeof(cuDoubleComplex));

	HANDLE_ERROR(cudaMemcpy(dftTerms, dev_dftTerms, sampleCount*sampleCount*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

	for(unsigned int y = 0 ; y < sampleCount; y++) {
		for (unsigned int x = 0; x < sampleCount; x++ ) {
			printf("(%8.2f %8.2fj) ", dftTerms[x + y*sampleCount].x, dftTerms[x + y*sampleCount].y);
		}
		printf("\n");
	}

	free(dftTerms);
}

void runDft(const double* realSignal, const unsigned int sampleCount) {

	cuDoubleComplex* dev_dftTerms;
	cuDoubleComplex* dev_complexSignal;
	cuDoubleComplex* complexSignal = (cuDoubleComplex*) malloc(
			sampleCount * sizeof(cuDoubleComplex));
	cuDoubleComplex* dev_dftResult;
	cuDoubleComplex* dftResult = (cuDoubleComplex*) malloc(
			sampleCount * sizeof(cuDoubleComplex));
	;

	// convert real to complex signal
	for (unsigned int index = 0; index < sampleCount; index++) {
		complexSignal[index].x = realSignal[index];
		complexSignal[index].y = 0.0;
	}

	HANDLE_ERROR(
			cudaMalloc(&dev_dftTerms,
					sampleCount * sampleCount * sizeof(cuDoubleComplex)));
	HANDLE_ERROR(
			cudaMalloc(&dev_complexSignal,
					sampleCount * sizeof(cuDoubleComplex)));
	HANDLE_ERROR(
			cudaMalloc(&dev_dftResult, sampleCount * sizeof(cuDoubleComplex)));

	// copy signal from host to device memory
	HANDLE_ERROR(
			cudaMemcpy(dev_complexSignal, complexSignal,
					sampleCount * sizeof(cuDoubleComplex),
					cudaMemcpyHostToDevice));

	dim3 grid(sampleCount, sampleCount);

	GenerateDftTerms<<<grid, 1>>>(sampleCount, dev_complexSignal, dev_dftTerms);

	//examineDftTerms(dev_dftTerms, sampleCount);

	Dft<<<sampleCount, 1>>>(sampleCount, dev_dftTerms, dev_dftResult);

	// copy dft from device memory to host memory
	HANDLE_ERROR(
			cudaMemcpy(dftResult, dev_dftResult,
					sampleCount * sizeof(cuDoubleComplex),
					cudaMemcpyDeviceToHost));

	printf("DFT\n");
	for (unsigned int index = 0; index < sampleCount; index++) {
		printf("(%8.2f, %8.2fj) ", dftResult[index].x, dftResult[index].y);
	}

	cudaFree(dev_dftResult);
	cudaFree(dev_complexSignal);
	cudaFree(dev_dftTerms);

	free(dftResult);
	free(complexSignal);
}

void printDeviceProp(cudaDeviceProp* pProp) {
	printf("Device Name: %.256s\n", pProp->name);
	printf("Device Major: %d\n", pProp->major);
	printf("Device Minor: %d\n", pProp->minor);
	printf("Compute Mode: %d\n", pProp->computeMode);

	printf("Global memory available on device in bytes: %ld\n",
			pProp->totalGlobalMem);
	printf("Shared memory available per block in bytes: %ld\n",
			pProp->sharedMemPerBlock);
	printf("32-bit registers available per block: %d\n", pProp->regsPerBlock);
	printf("Warp size in threads: %d\n", pProp->warpSize);
	printf("Maximum pitch in bytes allowed by memory copies: %ld\n",
			pProp->memPitch);
	printf("Maximum number of threads per block: %d\n",
			pProp->maxThreadsPerBlock);
	printf("Maximum size of each dimension of a block: (%d,%d,%d)\n",
			pProp->maxThreadsDim[0], pProp->maxThreadsDim[1],
			pProp->maxThreadsDim[2]);

}

int main(void) {
	//int c;
	//int* dev_c;
	cudaDeviceProp prop;

	int deviceCount;
	HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
	for (int currentDeviceIndex = 0; currentDeviceIndex < deviceCount;
			currentDeviceIndex++) {
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, currentDeviceIndex));

		printDeviceProp(&prop);
	}

	double cpuPI = 3.14159265358979323846;
	cudaMemcpyToSymbol(pi, &cpuPI, sizeof(double));

	//double signal[] = { 1, 2, 3, 4 };
	unsigned int sampleCount = 1024;
	double* signal = (double*)malloc(sampleCount*sizeof(double));

	double freq = 2;
	for(unsigned int index = 0 ; index < sampleCount; index++) {
		signal[index] = cos(2*cpuPI*freq*index/sampleCount) + cos(2*cpuPI*3*freq*index/sampleCount) /*+ (double)rand() / (double)((unsigned)RAND_MAX + 1) */ ;
	}
	runDft(signal, sampleCount);
	free(signal);

	return 0;
}
