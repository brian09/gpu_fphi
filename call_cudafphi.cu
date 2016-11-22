#include "cudafphi.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include<stdio.h>
#include<time.h>
#include<pthread.h>
#include<vector>
#include <algorithm>
#include <iterator>
#include <chrono>


size_t iterations;
size_t batch_size;
size_t free_voxels;

size_t total_voxels;


static __global__ void Inv4by4(float * ZTZI) {
	float a = ZTZI[0];
	float b = ZTZI[1];
	float c = ZTZI[2];
	float d = ZTZI[3];

	float det = a * d - b * c;
	ZTZI[0] = d / det;
	ZTZI[1] = -c / det;
	ZTZI[2] = -b / det;
	ZTZI[3] = a / det;

}


 extern "C" int call_cudafphi(float * h2, float * indicator, float * pvals,float * h_y,const float * h_Z, const float * h_hat,
             const float * h_evectors,const unsigned int * h_pmatrix, bool covariates, bool get_pval,
             size_t n_voxels, size_t n_subjects, size_t n_permutations, std::vector<int> selected_devices){

  int devices = selected_devices.size();
  if(n_voxels <= BLOCK_SIZE_3){
    batch_size = n_voxels;
    iterations = 1;
    free_voxels = batch_size;

  }else{
    batch_size = BLOCK_SIZE_3;
    iterations = ceil(float(n_voxels)/float(batch_size));
    free_voxels = n_voxels % batch_size;
    if(n_voxels % batch_size == 0)
      free_voxels = batch_size;
  }

  float * h_Y[devices];

  int n_streams = iterations;
  cuda_fphi_variables_per_stream stream_vars[n_streams];
  cuda_fphi_variables_per_device device_vars[devices];

  cudaStream_t device_stream[devices];

  cublasHandle_t device_handles[devices];
  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();


  for(int current_device = 0 ; current_device < devices; current_device++){

	  gpuErrchk(cudaSetDevice(selected_devices[current_device]));
	  device_vars[current_device].device_id = selected_devices[current_device];
	  gpuErrchk(cudaSetDeviceFlags((unsigned int)cudaDeviceScheduleAuto));
	  gpuErrchk(cudaStreamCreate(&device_stream[current_device]));
	  cublasErrchk(cublasCreate_v2(&device_handles[current_device]));
	  cublasErrchk(cublasSetStream_v2(device_handles[current_device], device_stream[current_device]));

  }


  float alpha = 1.f;
  float beta = 0.f;

  for(int current_device = 0; current_device < devices; current_device++){

	  gpuErrchk(cudaSetDevice(device_vars[current_device].device_id));



	  if(covariates){
		  gpuErrchk(cudaMalloc((void**)&device_vars[current_device].hat, sizeof(float)*n_subjects*n_subjects));
	  }

	  gpuErrchk(cudaMalloc((void**)&device_vars[current_device].evectors, sizeof(float)*n_subjects*n_subjects));



      if(get_pval){
    	  gpuErrchk(cudaMalloc((void**)&device_vars[current_device].pmatrix, sizeof(unsigned int)*n_subjects*n_permutations));
      }

      gpuErrchk(cudaMalloc((void**)&device_vars[current_device].aux_vars.d_Z, sizeof(float)*n_subjects*2));



      gpuErrchk(cudaMalloc((void**)&device_vars[current_device].aux_vars.d_ZTZI, sizeof(float)*2*2));

      gpuErrchk(cudaMallocHost((void**)&h_Y[current_device], sizeof(float)*n_subjects*n_voxels, (unsigned int)cudaHostAllocWriteCombined));
      gpuErrchk(cudaMemcpyAsync(h_Y[current_device] , h_y, sizeof(float)*n_subjects*n_voxels, cudaMemcpyHostToHost, device_stream[current_device]));

      device_vars[current_device].device_id = selected_devices[current_device];
      device_vars[current_device].get_pval = get_pval;
      device_vars[current_device].covariates = covariates;

      gpuErrchk(cudaMemcpyAsync((float *)device_vars[current_device].aux_vars.d_Z, h_Z,  sizeof(float)*n_subjects*2, cudaMemcpyHostToDevice, device_stream[current_device]));
      if(get_pval)
    	  gpuErrchk(cudaMemcpyAsync((unsigned int *)device_vars[current_device].pmatrix, h_pmatrix, sizeof(unsigned int)*n_subjects*n_permutations, cudaMemcpyHostToDevice, device_stream[current_device]));
      gpuErrchk(cudaMemcpyAsync((float *)device_vars[current_device].evectors, h_evectors, sizeof(float)*n_subjects*n_subjects, cudaMemcpyHostToDevice, device_stream[current_device]));

      if(covariates)
    	  gpuErrchk(cudaMemcpyAsync((float *)device_vars[current_device].hat, h_hat, sizeof(float)*n_subjects*n_subjects, cudaMemcpyHostToDevice, device_stream[current_device]));
	  cublasErrchk(cublasSgemm_v2(device_handles[current_device], CUBLAS_OP_T, CUBLAS_OP_N, 2, 2, n_subjects, &alpha, device_vars[current_device].aux_vars.d_Z, n_subjects, device_vars[current_device].aux_vars.d_Z, n_subjects, &beta, device_vars[current_device].aux_vars.d_ZTZI, 2));

	  Inv4by4<<<1, 1, 0, device_stream[current_device]>>>(device_vars[current_device].aux_vars.d_ZTZI);

  }

  for(int current_device = 0 ; current_device < devices; current_device++){

	  gpuErrchk(cudaSetDevice(selected_devices[current_device]));

	  gpuErrchk(cudaStreamSynchronize(device_stream[current_device]));

  }

  stream_vars[0].n_permutations = n_permutations;
  stream_vars[0].n_subjects = n_subjects;
  stream_vars[0].shared_device_vars = device_vars[0];

  stream_vars[0].stream_number= 0;
  stream_vars[0].n_voxels = batch_size;

  size_t total_bytes;
  size_t free_bytes;
  unsigned int total_number_of_threads_usable_by_device[devices];
  unsigned int threads_in_use_per_device[n_streams];
  size_t total_usable_threads = 0;
  size_t used_bytes = run_allocation_test(stream_vars[0]);
  for(int device = 0; device < devices ; device++){
      gpuErrchk(cudaSetDevice(selected_devices[device]));
      gpuErrchk(cudaMemGetInfo(&free_bytes, &total_bytes));

      total_number_of_threads_usable_by_device[device] = floor(float(total_bytes)/float(used_bytes));
      total_usable_threads += total_number_of_threads_usable_by_device[device];
      threads_in_use_per_device[device] = 0;

  }

  int device = 0;
  std::vector<int> current_running_streams;
  pthread_t pthreads[n_streams];
  void * return_val;
  int stream_number = 0;


for(int current_stream = 0 ; current_stream < n_streams; current_stream++){


	  do{
		  if(threads_in_use_per_device[device] >= total_number_of_threads_usable_by_device[device])
			  device++;
		  else
			  break;
	  }while(device != devices);

	  if(device == devices){
		  int streamIdx;
		  do{

			  for(std::vector<int>::iterator it = current_running_streams.begin(); it != current_running_streams.end() ;it++){
				  int status = pthread_tryjoin_np(pthreads[*it], &return_val);
				  if(return_val != NULL){
					  device = std::distance(selected_devices.begin(), std::find( selected_devices.begin(), selected_devices.end(), stream_vars[*it].shared_device_vars.device_id));
					  streamIdx = *it;
					  threads_in_use_per_device[device] -= 1;
					  break;
				  }
			  }
		  }while(device == devices);

		  current_running_streams.erase( std::find(current_running_streams.begin(), current_running_streams.end(), streamIdx));
		  return_val = NULL;
	  }
	  stream_vars[current_stream].shared_device_vars = device_vars[device];

	  stream_vars[current_stream].shared_device_vars.h2 = h2;
	  stream_vars[current_stream].shared_device_vars.pvals = pvals;
	  stream_vars[current_stream].shared_device_vars.indicator = indicator;
	  stream_vars[current_stream].h_y = h_Y[device];

	  stream_vars[current_stream].n_permutations = n_permutations;
	  stream_vars[current_stream].n_subjects = n_subjects;
	  stream_vars[current_stream].n_voxels = batch_size;
	  stream_vars[current_stream].stream_number = current_stream;

	  if(current_stream == iterations - 1)
		  stream_vars[current_stream].n_voxels = free_voxels;
 	  pthread_create(&pthreads[current_stream], NULL, run_cudafphi_pthread, (void*)&stream_vars[current_stream]);

	  current_running_streams.push_back(current_stream);
	  threads_in_use_per_device[device] += 1;
	  device++;
	  if(device == devices)
		  device = 0;



  }



  for(int it = 0; it < current_running_streams.size() ; it++){
	  pthread_join(pthreads[current_running_streams[it]], NULL);
  }

  for(std::vector<int>::iterator device_it = selected_devices.begin() ; device_it != selected_devices.end(); device_it++){
	  gpuErrchk(cudaSetDevice(*device_it));
	  const size_t  device_Idx = std::distance(selected_devices.begin(), device_it);
	  if(covariates){
		  gpuErrchk(cudaFree((void*)device_vars[device_Idx].hat));
	  }
	  gpuErrchk(cudaFree((void*)device_vars[device_Idx].evectors));
      if(get_pval){
    	  gpuErrchk(cudaFree((void*)device_vars[device_Idx].pmatrix));
      }
      gpuErrchk(cudaFree((void*)device_vars[device_Idx].aux_vars.d_Z));
      gpuErrchk(cudaFree((void*)device_vars[device_Idx].aux_vars.d_ZTZI));
      gpuErrchk(cudaFreeHost((void*)h_Y[device_Idx]));
	  gpuErrchk(cudaStreamDestroy(device_stream[device_Idx]));
	  cublasErrchk(cublasDestroy_v2(device_handles[device_Idx]));

  }
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();

  printf ("It took %i seconds.\n", (((int)(seconds))));

  return 1;

}
