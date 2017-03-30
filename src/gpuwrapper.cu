/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "support.h"
#include "gpukernel.cu"

extern "C" void gpu_kernel_wrapper(float *input_units,  float *hidden_units, float **input_weights, int inp, int hidden);

extern "C"  void gpu_output_error(float *delta, float *target, float *output, int count, float *err, float *hidden_delta, int hid, float **hidden_weights, float **hidden_units, int out);



void gpu_kernel_wrapper(float *input_units,  float *hidden_units_N, float **input_weights, int inp, int hidden) {

  Timer timer; 
  cudaError_t cuda_ret;
 
  float *input_units_d, *hidden_units_d, *input_weights_d; 
  float *input_weights_h; 
 
  input_weights_h = (float*) malloc(sizeof(float) * inp * hidden); 
 /// hidden_units_N = (float*) malloc(sizeof(float) *  hidden); 
 
  for(int i=0; i < inp; i++) {
      for(int j=0; j < hidden; j++) {
          input_weights_h[i*hidden+j] =  input_weights[i][j]; 
          //printf("i=%d, j=%d, %d\n", i, j, (i*hidden+j));
      }
  }

   //input_units[0] = first;

  // Allocate device variables ----------------------------------------------
  cudaMalloc((void**) &input_units_d, sizeof(float) * inp); 
  cudaMalloc((void**) &hidden_units_d, sizeof(float) * hidden); 
  cudaMalloc((void**) &input_weights_d, sizeof(float) * inp * hidden); 

  cudaDeviceSynchronize();

  // Copy host variables to device ------------------------------------------
  //printf("Copying data from host to device..."); 
  cudaMemcpy(input_units_d, input_units, sizeof(float) *inp, cudaMemcpyHostToDevice);
  cudaMemcpy(input_weights_d, input_weights_h, sizeof(float) *inp * hidden, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();  

  // Launch kernel using standard sgemm interface ---------------------------
  //printf("Launching kernel..."); 
  gpu_bpnn_layerforward(input_units_d, hidden_units_d, input_weights_d, inp, hidden); 

  cudaDeviceSynchronize();


   cudaMemcpy(hidden_units_N, hidden_units_d, sizeof(float) * hidden, cudaMemcpyDeviceToHost);
 
  cudaDeviceSynchronize();


   free(input_weights_h); 
 
   cudaFree(input_units_d);
   cudaFree(hidden_units_d);
   cudaFree(input_weights_d);
}


void gpu_output_error (float *output_delta, float *target, float *output, int count, float *err, float *hidden_delta, int hid, float **hidden_weights, float **hidden_units, int out) {

    float *output_delta_d, *target_d, *output_d; 
    float *hidden_weights_d, *hidden_units_d, *hidden_delta_d; 
    float *hidden_weights_h;
  
    hidden_weights_h = (float*) malloc(sizeof(float) * hid); 
    /// hidden_units_N = (float*) malloc(sizeof(float) *  hidden); 
 
    for(int i=0; i < hid; i++) {
          hidden_weights_h[i] =  hidden_weights[i][1]; 
     }
             
    cudaMalloc((void**) &output_delta_d, sizeof(float) * count); 
    cudaMalloc((void**) &target_d, sizeof(float) * count); 
    cudaMalloc((void**) &output_d, sizeof(float) * count); 

    cudaMalloc((void**) &hidden_units_d, sizeof(float) * hid); 
    cudaMalloc((void**) &hidden_weights_d, sizeof(float) * hid); 
    cudaMalloc((void**) &hidden_delta_d, sizeof(float) * hid); 
    cudaMalloc((void**) &hidden_delta_d, sizeof(float) * hid); 
 
   cudaDeviceSynchronize();

    // Copy host variables to device ------------------------------------------
    //printf("gpu_output_error: Copying data from host to device..."); 
    cudaMemcpy(target_d, target, sizeof(float) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(output_d, output, sizeof(float) * count, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    cudaMemcpy(hidden_units_d, hidden_units, sizeof(float) * hid, cudaMemcpyHostToDevice);
    cudaMemcpy(hidden_weights_d, hidden_weights_h, sizeof(float) * hid , cudaMemcpyHostToDevice);
    //cudaMemcpy(prev_d, prev, sizeof(float) * 2 * hid , cudaMemcpyHostToDevice);


    // Launch kernel using standard sgemm interface ---------------------------
    //printf("gpu_output_error: Launching kernel..."); 
    gpu_output_error_kernel(output_delta_d, target_d, output_d, count, err);
 
    cudaDeviceSynchronize();

    cudaMemcpy(output_delta, output_delta_d, sizeof(float) * count, cudaMemcpyDeviceToHost);
 
    cudaDeviceSynchronize(); 

    gpu_hidden_error_kernel(hidden_delta_d, hid, output_delta_d, out, hidden_weights_d, hidden_units_d);  

    cudaDeviceSynchronize(); 

     cudaMemcpy(hidden_delta, hidden_delta_d, sizeof(float) * hid, cudaMemcpyDeviceToHost);
 
    gpu_weight_adjust(output_delta_d, out, hidden_units_d, hid, hidden_weights_d);
     
    cudaDeviceSynchronize(); 
   

     cudaMemcpy(hidden_weights_h, hidden_weights_d, sizeof(float) * hid, cudaMemcpyDeviceToHost);
 
    cudaDeviceSynchronize(); 

     for(int i=0; i < hid; i++) {
          hidden_weights[i][1] =  hidden_weights_h[i];
     }

    cudaFree(output_delta_d);
    cudaFree(target_d);
    cudaFree(output_d);  
    cudaFree(hidden_weights_d);
    cudaFree(hidden_units_d);
    cudaFree(hidden_delta_d);
}

