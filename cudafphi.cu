/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
/*
 * cudafphi.cu
 * Primary Author: Brian Donohue
 * Date: 12/05/2015
 * Email: bdono09@gmail.edu
 */

/*
 * To whoever wishes to modify and redistribute this file please write your name the date
 * and an explanation of the modification.
 * Modifications:
 *
 */

#include<iostream>

#include<cstdlib>
#include<time.h>
#include<stdio.h>
#include "cudafphi.cuh"
#include <cuda_runtime.h>

extern size_t iterations;
extern size_t batch_size;
extern size_t free_voxels;

static  void cudafphi(float * h2, float * indicator, float * pvals,float * d_sy,float * d_F, const float * d_hat,
              const float * d_evectors, float * d_h2, float * d_indicator, float * d_pvals,    compute_F_variables compute_F_vars,
             aux_variables aux_vars, compute_h2_variables compute_h2_vars, pval_variables pval_vars, const unsigned int * d_pmatrix,
             bool covariates, bool get_pval, bool * d_boolean_score, size_t n_voxels, size_t n_subjects, size_t n_permutations,
             int current_stream_number, cudaStream_t stream, cublasHandle_t handle){

  gpuErrchk(cudaMemsetAsync(d_F, 0, sizeof(float)*n_voxels*n_subjects, stream));
  compute_F(d_hat, d_sy, d_evectors,d_F,
            compute_F_vars, covariates,
            n_subjects,n_voxels, stream, handle);

  compute_h2(d_F, d_h2, d_indicator, d_boolean_score,
          compute_h2_vars,  aux_vars,
             n_subjects, n_voxels, stream);



  gpuErrchk(cudaMemcpyAsync(&h2[current_stream_number*batch_size], d_h2, sizeof(float)*n_voxels, cudaMemcpyDeviceToHost, stream));


  gpuErrchk(cudaMemcpyAsync(&indicator[current_stream_number*batch_size], d_indicator, sizeof(float)*n_voxels, cudaMemcpyDeviceToHost, stream));

  if(get_pval){
      gpuErrchk(cudaMemsetAsync(d_pvals, 0, sizeof(float)*n_voxels, stream));
      compute_pvals(d_sy, d_F, d_hat, compute_h2_vars.d_Sigma_A, compute_h2_vars.d_Sigma_E,
               d_pvals, (const unsigned int *)d_pmatrix,
                    d_boolean_score,  aux_vars,  pval_vars, covariates,
                     n_subjects,  n_voxels, n_permutations, stream, handle);
      gpuErrchk(cudaMemcpyAsync(&pvals[current_stream_number*batch_size], d_pvals, sizeof(float)*n_voxels, cudaMemcpyDeviceToHost, stream));
  }
}


size_t run_allocation_test(cuda_fphi_variables_per_stream cudafphi_vars){


	gpuErrchk(cudaSetDevice(cudafphi_vars.shared_device_vars.device_id));

	compute_h2_variables compute_h2_vars;
	compute_F_variables compute_F_vars;
	pval_variables pval_vars;
	float * d_sy;
	float * d_F;
	bool * d_boolean_score;
	float * d_indicator;
	float * d_h2;
	float * d_pval;
	size_t n_subjects = cudafphi_vars.n_subjects;
	size_t n_permutations = cudafphi_vars.n_permutations;


	gpuErrchk(cudaMalloc((void**)&compute_F_vars.d_Y, sizeof(float)*n_subjects*batch_size));

	gpuErrchk(cudaMalloc((void**)&compute_F_vars.mean_or_sigma, sizeof(float)*batch_size));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_A, sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_B, sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_C, sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_D, sizeof(float)) );

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_E, sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_Sigma_P, sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_score, sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_theta, sizeof(float)*2));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_weights, sizeof(float)*n_subjects));
	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_Sigma_A, sizeof(float)*batch_size));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_Sigma_E, sizeof(float)*batch_size));

	if(cudafphi_vars.shared_device_vars.get_pval){
		gpuErrchk(cudaMalloc((void**)&pval_vars.syP, sizeof(float)*n_subjects*(n_permutations + 1)));

		gpuErrchk(cudaMalloc((void**)&pval_vars.d_F, sizeof(float)*n_subjects*(n_permutations + 1)));

		gpuErrchk(cudaMalloc((void**) &pval_vars.d_score, sizeof(float) * (n_permutations + 1)));

		gpuErrchk(cudaMalloc((void**) &pval_vars.d_Ts, sizeof(float) * (n_permutations + 1)));

		gpuErrchk(cudaMalloc((void**) &pval_vars.d_a, sizeof(float) * (n_permutations + 1) * 2));

    	 gpuErrchk(cudaMalloc((void**) &pval_vars.d_sigmaP, sizeof(float)*(n_permutations + 1)));

	}


	gpuErrchk(cudaMalloc((void**)&d_sy, sizeof(float)*n_subjects*batch_size));


	gpuErrchk(cudaMalloc((void**)&d_F, sizeof(float)*batch_size*n_subjects));


	gpuErrchk(cudaMalloc((void**)&d_indicator, sizeof(float)*batch_size));



	gpuErrchk(cudaMalloc((void**)&d_h2, sizeof(float)*batch_size));


	if(cudafphi_vars.shared_device_vars.get_pval){
		gpuErrchk(cudaMalloc((void**)&d_pval, sizeof(float)*batch_size));
	}




	gpuErrchk(cudaMalloc((void**)&d_boolean_score, sizeof(bool)*batch_size));
	size_t freeMem;
	size_t totalMem;
	size_t usedMem;
	gpuErrchk(cudaMemGetInfo(&freeMem, &totalMem));
	usedMem = totalMem - freeMem;

	gpuErrchk(cudaFree(compute_F_vars.d_Y));


	gpuErrchk(cudaFree(compute_F_vars.mean_or_sigma));



	gpuErrchk(cudaFree(compute_h2_vars.d_A));

	gpuErrchk(cudaFree(compute_h2_vars.d_B));

	gpuErrchk(cudaFree(compute_h2_vars.d_C));

	gpuErrchk(cudaFree(compute_h2_vars.d_D));

	gpuErrchk(cudaFree(compute_h2_vars.d_E));

	gpuErrchk(cudaFree(compute_h2_vars.d_Sigma_P));

	gpuErrchk(cudaFree(compute_h2_vars.d_score));

	gpuErrchk(cudaFree(compute_h2_vars.d_theta));

	gpuErrchk(cudaFree(compute_h2_vars.d_weights));


	gpuErrchk(cudaFree(compute_h2_vars.d_Sigma_E));

	gpuErrchk(cudaFree(compute_h2_vars.d_Sigma_A));

	if(cudafphi_vars.shared_device_vars.get_pval){
		gpuErrchk(cudaFree(pval_vars.syP));

		gpuErrchk(cudaFree(pval_vars.d_F));

		gpuErrchk(cudaFree( pval_vars.d_score));

		gpuErrchk(cudaFree( pval_vars.d_Ts));

		gpuErrchk(cudaFree( pval_vars.d_a));

		gpuErrchk(cudaFree( pval_vars.d_sigmaP));

	}


	gpuErrchk(cudaFree(d_sy));


	gpuErrchk(cudaFree(d_F));


	gpuErrchk(cudaFree(d_indicator));



	gpuErrchk(cudaFree(d_h2));


	gpuErrchk(cudaFree(d_boolean_score));


	if(cudafphi_vars.shared_device_vars.get_pval){
		gpuErrchk(cudaFree(d_pval));
	}

	return  usedMem;

}

void * run_cudafphi_pthread(void * cudafphi_args){

   	cuda_fphi_variables_per_stream * cudafphi_vars;
    cudafphi_vars = (cuda_fphi_variables_per_stream *)cudafphi_args;
	gpuErrchk(cudaSetDevice(cudafphi_vars->shared_device_vars.device_id));

    compute_F_variables compute_F_vars;
    compute_h2_variables compute_h2_vars;
    pval_variables pval_vars;

    float * d_sy;
    float * d_F;
    bool * d_boolean_score;
    float * d_pvals;
    float * d_h2;
    float * d_indicator;


	size_t n_subjects = cudafphi_vars->n_subjects;
	size_t n_permutations = cudafphi_vars->n_permutations;
	size_t n_voxels = cudafphi_vars->n_voxels;



	gpuErrchk(cudaMalloc((void**)&compute_F_vars.d_Y, sizeof(float)*n_subjects*n_voxels));

	gpuErrchk(cudaMalloc((void**)&compute_F_vars.mean_or_sigma, sizeof(float)*n_voxels));
	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_A, sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_B, sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_C, sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_D, sizeof(float)) );

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_E, sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_Sigma_P, sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_score, sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_theta, sizeof(float)*2));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_weights, sizeof(float)*n_subjects));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_Sigma_A, sizeof(float)*n_voxels));

	gpuErrchk(cudaMalloc((void**)&compute_h2_vars.d_Sigma_E, sizeof(float)*n_voxels));

	if(cudafphi_vars->shared_device_vars.get_pval){
		gpuErrchk(cudaMalloc((void**)&pval_vars.syP, sizeof(float)*n_subjects*(n_permutations + 1)));

		gpuErrchk(cudaMalloc((void**)&pval_vars.d_F, sizeof(float)*n_subjects*(n_permutations + 1)));

		gpuErrchk(cudaMalloc((void**) &pval_vars.d_score, sizeof(float) * (n_permutations + 1)));

		gpuErrchk(cudaMalloc((void**) &pval_vars.d_Ts, sizeof(float) * (n_permutations + 1)));

		gpuErrchk(cudaMalloc((void**) &pval_vars.d_a, sizeof(float) * (n_permutations + 1) * 2));

    	 gpuErrchk(cudaMalloc((void**) &pval_vars.d_sigmaP, sizeof(float)*(n_permutations + 1)));

	}


	gpuErrchk(cudaMalloc((void**)&d_sy, sizeof(float)*n_subjects*n_voxels));


	gpuErrchk(cudaMalloc((void**)&d_F, sizeof(float)*n_voxels*n_subjects));


	gpuErrchk(cudaMalloc((void**)&d_indicator, sizeof(float)*n_voxels));



	gpuErrchk(cudaMalloc((void**)&d_h2, sizeof(float)*n_voxels));


	if(cudafphi_vars->shared_device_vars.get_pval){
		gpuErrchk(cudaMalloc((void**)&d_pvals, sizeof(float)*n_voxels));
	}




	gpuErrchk(cudaMalloc((void**)&d_boolean_score, sizeof(bool)*n_voxels));



    cudaStream_t stream;

    cublasHandle_t handle;

    cublasErrchk(cublasCreate_v2(&handle));
    gpuErrchk(cudaStreamCreate(&stream));
    cublasErrchk(cublasSetStream_v2(handle, stream));
    float alpha = 1;
    float beta = 0.f;

    gpuErrchk(cudaMemcpyAsync(d_sy, cudafphi_vars->h_y + batch_size*cudafphi_vars->stream_number*cudafphi_vars->n_subjects, sizeof(float)*cudafphi_vars->n_subjects*n_voxels, cudaMemcpyHostToDevice, stream));


    cudafphi(cudafphi_vars->shared_device_vars.h2, cudafphi_vars->shared_device_vars.indicator, cudafphi_vars->shared_device_vars.pvals, d_sy, d_F,cudafphi_vars->shared_device_vars.hat,
    			cudafphi_vars->shared_device_vars.evectors, d_h2, d_indicator, d_pvals,  compute_F_vars,
    			cudafphi_vars->shared_device_vars.aux_vars, compute_h2_vars, pval_vars, cudafphi_vars->shared_device_vars.pmatrix,
    			cudafphi_vars->shared_device_vars.covariates,cudafphi_vars->shared_device_vars.get_pval, d_boolean_score,
    			n_voxels, cudafphi_vars->n_subjects, cudafphi_vars->n_permutations,
    			cudafphi_vars->stream_number, stream, handle);

    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaStreamDestroy(stream));
    cublasErrchk(cublasDestroy_v2(handle));


	gpuErrchk(cudaFree(compute_F_vars.d_Y));


	gpuErrchk(cudaFree(compute_F_vars.mean_or_sigma));



	gpuErrchk(cudaFree(compute_h2_vars.d_A));

	gpuErrchk(cudaFree(compute_h2_vars.d_B));

	gpuErrchk(cudaFree(compute_h2_vars.d_C));

	gpuErrchk(cudaFree(compute_h2_vars.d_D));

	gpuErrchk(cudaFree(compute_h2_vars.d_E));

	gpuErrchk(cudaFree(compute_h2_vars.d_Sigma_P));

	gpuErrchk(cudaFree(compute_h2_vars.d_score));

	gpuErrchk(cudaFree(compute_h2_vars.d_theta));

	gpuErrchk(cudaFree(compute_h2_vars.d_weights));


	gpuErrchk(cudaFree(compute_h2_vars.d_Sigma_E));

	gpuErrchk(cudaFree(compute_h2_vars.d_Sigma_A));

	if(cudafphi_vars->shared_device_vars.get_pval){
		gpuErrchk(cudaFree(pval_vars.syP));

		gpuErrchk(cudaFree(pval_vars.d_F));

		gpuErrchk(cudaFree( pval_vars.d_score));

		gpuErrchk(cudaFree( pval_vars.d_Ts));

		gpuErrchk(cudaFree( pval_vars.d_a));

		gpuErrchk(cudaFree( pval_vars.d_sigmaP));

	}


	gpuErrchk(cudaFree(d_sy));


	gpuErrchk(cudaFree(d_F));


	gpuErrchk(cudaFree(d_indicator));



	gpuErrchk(cudaFree(d_h2));


	gpuErrchk(cudaFree(d_boolean_score));


	if(cudafphi_vars->shared_device_vars.get_pval){
		gpuErrchk(cudaFree(d_pvals));
	}

    int * success = new int;
    *success = 1;
    return (void*)success;

}



