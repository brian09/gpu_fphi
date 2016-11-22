#include "cudafphi.cuh"
#include <stdio.h>
#include <math_functions.h>
#include <iostream>

static __global__ void calculate_means(const float * d_sy, float * mean, const size_t n_voxels, const size_t n_subjects){

		unsigned int tIdx = threadIdx.x;

		unsigned int rowIdx = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int vIdx = blockIdx.y*blockDim.y + threadIdx.y;


		extern __shared__ float shared_mean[];

		if(rowIdx < n_subjects && vIdx < n_voxels)
			shared_mean[tIdx] = d_sy[vIdx*n_subjects + rowIdx];
		else
			shared_mean[tIdx] = 0.f;




		__syncthreads();

		if(blockDim.x >= 1024 && tIdx < 512) shared_mean[tIdx] += shared_mean[tIdx + 512];

		__syncthreads();


		if(blockDim.x >= 512 && tIdx < 256) shared_mean[tIdx] += shared_mean[tIdx + 256];

		__syncthreads();



		if(blockDim.x >= 256 && tIdx < 128) shared_mean[tIdx] += shared_mean[tIdx + 128];

		__syncthreads();

		if(blockDim.x >= 128 && tIdx < 64) shared_mean[tIdx] += shared_mean[tIdx + 64];

		__syncthreads();

		if(blockDim.x >= 64 && tIdx < 32) shared_mean[tIdx] += shared_mean[tIdx + 32];

		__syncthreads();

		if(blockDim.x >= 32 && tIdx < 16) shared_mean[tIdx] += shared_mean[tIdx + 16];

		__syncthreads();

		if(blockDim.x >= 16 && tIdx < 8) shared_mean[tIdx] += shared_mean[tIdx + 8];

		__syncthreads();

		if(blockDim.x >= 8 && tIdx < 4) shared_mean[tIdx] += shared_mean[tIdx + 4];

		__syncthreads();

		if(blockDim.x >= 4 && tIdx < 2) shared_mean[tIdx] += shared_mean[tIdx + 2];

		__syncthreads();

		if(blockDim.x >= 2 && tIdx < 1) shared_mean[tIdx] += shared_mean[tIdx + 1];

		__syncthreads();

		if(tIdx == 0){
			atomicAdd(&mean[vIdx], shared_mean[0]);
		}





}

static __global__ void demean_columns(float * d_Y, const float * d_sy, float * mean, size_t n_voxels, size_t n_subjects){

  const size_t rowIdx = threadIdx.x +blockDim.x*blockIdx.x;
  const size_t voxel = threadIdx.y + blockDim.y*blockIdx.y;


  if(rowIdx < n_subjects && voxel < n_voxels){
	  const float value  =  d_sy[voxel*n_subjects + rowIdx] - mean[voxel]/float(n_subjects);
      d_Y[rowIdx + voxel*n_subjects] = value;
  }
}

static __global__ void calculate_sigma(const float * d_sy, float * sigma,  size_t n_voxels,size_t n_subjects){

	const unsigned int tIdx = threadIdx.x;

	const unsigned int rowIdx = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int vIdx = blockIdx.y*blockDim.y + threadIdx.y;


	extern __shared__ float shared_sigma[];

	if(rowIdx < n_subjects && vIdx < n_voxels){
		float const value = d_sy[vIdx*n_subjects + rowIdx];
		shared_sigma[tIdx] = value*value;
	}else{
		shared_sigma[tIdx] = 0.f;
	}




	__syncthreads();

	if(blockDim.x >= 1024 && tIdx < 512) shared_sigma[tIdx] += shared_sigma[tIdx + 512];

	__syncthreads();


	if(blockDim.x >= 512 && tIdx < 256) shared_sigma[tIdx] += shared_sigma[tIdx + 256];

	__syncthreads();



	if(blockDim.x >= 256 && tIdx < 128) shared_sigma[tIdx] += shared_sigma[tIdx + 128];

	__syncthreads();

	if(blockDim.x >= 128 && tIdx < 64) shared_sigma[tIdx] += shared_sigma[tIdx + 64];

	__syncthreads();

	if(blockDim.x >= 64 && tIdx < 32) shared_sigma[tIdx] += shared_sigma[tIdx + 32];

	__syncthreads();

	if(blockDim.x >= 32 && tIdx < 16) shared_sigma[tIdx] += shared_sigma[tIdx + 16];

	__syncthreads();

	if(blockDim.x >= 16 && tIdx < 8) shared_sigma[tIdx] += shared_sigma[tIdx + 8];

	__syncthreads();

	if(blockDim.x >= 8 && tIdx < 4) shared_sigma[tIdx] += shared_sigma[tIdx + 4];

	__syncthreads();

	if(blockDim.x >= 4 && tIdx < 2) shared_sigma[tIdx] += shared_sigma[tIdx + 2];

	__syncthreads();

	if(blockDim.x >= 2 && tIdx < 1) shared_sigma[tIdx] += shared_sigma[tIdx + 1];

	__syncthreads();


	if(tIdx == 0){
		atomicAdd(&sigma[vIdx], shared_sigma[0]);

	}



}





static __global__ void calculate_inverse_normal(float * d_Y,  const float * d_sigma, size_t n_voxels, size_t n_subjects){

  size_t rowIdx = threadIdx.x +blockDim.x*blockIdx.x;
  size_t voxel = threadIdx.y + blockDim.y*blockIdx.y;


  if(rowIdx < n_subjects && voxel < n_voxels){
	  float value = d_Y[rowIdx + voxel*n_subjects]/sqrt(d_sigma[voxel]/(n_subjects - 1.f));
      d_Y[rowIdx + voxel*n_subjects] = value;
  }
}



int compute_F(const float * d_hat, float* d_sy, const float *d_evectors, float * d_F,
              compute_F_variables vars, bool covariates, size_t n_subjects,
               size_t n_voxels, cudaStream_t stream, cublasHandle_t handle) {





	int blockSize_n_subjects = 32;

	if(n_subjects >= 64){
		blockSize_n_subjects = 64;
	}
	if(n_subjects >= 128){
		blockSize_n_subjects = 128;
	}
	if(n_subjects >= 256){
		blockSize_n_subjects = 256;
	}
	if(n_subjects >= 512){
		blockSize_n_subjects = 512;
	}
	if(n_subjects >= 1024){

		if(n_subjects % 1024 <= n_subjects % 512){
			blockSize_n_subjects = 1024;
		}else{
			blockSize_n_subjects = 512;
		}


	}

	dim3 blockSize_mean(blockSize_n_subjects, 1, 1);
	dim3 gridSize_mean(ceil(float(n_subjects)/float(blockSize_n_subjects)), n_voxels, 1);

	dim3 blockSize_set(blockSize_n_subjects, 1024/blockSize_n_subjects, 1);

	dim3 gridSize_set(ceil(float(n_subjects)/float(blockSize_n_subjects)), ceil(float(n_voxels)/float(1024.f/blockSize_n_subjects)), 1);

    gpuErrchk(cudaMemsetAsync(vars.mean_or_sigma, 0, sizeof(float)*n_voxels, stream ));

    calculate_means<<<gridSize_mean, blockSize_mean, sizeof(float)*blockSize_n_subjects, stream>>>(d_sy, vars.mean_or_sigma, n_voxels, n_subjects);
    gpuErrchk(cudaPeekAtLastError());
	demean_columns<<<gridSize_set, blockSize_set, 0, stream>>>(vars.d_Y, d_sy, vars.mean_or_sigma, n_voxels, n_subjects);
    gpuErrchk(cudaPeekAtLastError());

	float alpha = 1.f;
	float beta = 0.f;
    gpuErrchk(cudaMemsetAsync(vars.mean_or_sigma, 0, sizeof(float)*n_voxels, stream ));

	calculate_sigma<<<gridSize_mean, blockSize_mean, sizeof(float)*blockSize_n_subjects, stream>>>(vars.d_Y, vars.mean_or_sigma, n_voxels, n_subjects);
	gpuErrchk(cudaPeekAtLastError());
	calculate_inverse_normal<<<gridSize_set, blockSize_set, 0, stream>>>(vars.d_Y, vars.mean_or_sigma, n_voxels, n_subjects);
	gpuErrchk(cudaPeekAtLastError());
	cublasErrchk(cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N, n_subjects,n_voxels, n_subjects, &alpha, d_evectors,
			n_subjects, vars.d_Y, n_subjects, &beta, d_sy, n_subjects));

   if(covariates){

	    cublasErrchk(cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_subjects,n_voxels, n_subjects, &alpha, d_hat, n_subjects, d_sy, n_subjects, &beta, d_F, n_subjects));

   }else{
	   gpuErrchk(cudaMemcpyAsync(d_F, d_sy, sizeof(float)*n_subjects*n_voxels, cudaMemcpyDeviceToDevice, stream));
   }


   return 1;
}
