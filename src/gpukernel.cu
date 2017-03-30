/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
#include <stdio.h>

#define BLOCK_SIZE 512
#define HIDDEN_SIZE 17
#define ETA 0.3
#define MOMENT 0.3 

/*
 __device__ float  squash(float x ) {
  //float m;
  //x = -x;
  //m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
  //return(1.0 / (1.0 + m));
  return (1.0 / (1.0 + exp(-x)));
}
*/
/*
__global__ void kernel_bpnn_layerforward(float *input_units,  float *hidden_units, float *input_weights, int inp, int hidden) {


   int threadIdx_x =  threadIdx.x;  
   int element     = threadIdx.x + blockDim.x * blockIdx.x;  

   if(element < inp) {

       float val = 0.0;
       for(int i=1; i < hidden  ; i++) {  
           val  = input_units[element] * input_weights[(element * hidden) + i] ;    
            
//          printf("%d, %d, %f, %f, %f\n", element, i,  input_weights[(element *   hidden) + i] ,  input_units[element],  val);
           atomicAdd(&hidden_units[i], val);   
       }   
   } 


//   nt i=1; i < hidden  ; i++) {
//(f(element==0)  {  
//	hidden_units[0] = 0.0;
//    }
}

*/


__global__ void kernel_bpnn_layerforward(float *input_units_master,  float *hidden_units_master, float *input_weights_master, int inp, int hidden) {

   int tx =  threadIdx.x;
   int element     = threadIdx.x + blockDim.x * blockIdx.x;

   // Store Input Units, Input Weights in shared memory  
   __shared__ float input_units[BLOCK_SIZE];     
   __shared__ float input_weights[18*BLOCK_SIZE];     
   __shared__ float hidden_units[17];     

  
   if(element < inp) {

   	// Read Data from Global memory to Shared memory  
   	input_units[tx] = input_units_master[element];
//        printf("PROBLEM------ %d, %d, %f\n", element, tx, input_units[tx]);  
 
   	int i;
   	for(i=0; i<hidden; i++) {
       		input_weights[(tx*hidden)+i] = input_weights_master[(element * hidden) + i];    

		hidden_units[i] = 0.0;
		hidden_units_master[i] = 0.0;
  	}

	// Sync All Threads
        __syncthreads();  

	// Calculate Intermediate results in Shared memory  
	for(i=1; i<hidden; i++) {
		float result = input_units[tx] * input_weights[(tx*hidden)+i]; 
		//hidden_units[i] += result;
		atomicAdd(&(hidden_units[i]), result);
		//printf("Intermediate: %d, %d, %f * %f, %f \n", element, i,input_units[tx] , input_weights[(tx*hidden)+i], result);
	}  


        __syncthreads();


	// Store final results in Main memory 	
	if(tx ==0) { 
		for(i=1; i<hidden; i++) {
			atomicAdd(&(hidden_units_master[i]), hidden_units[i]);
			//hidden_units_master[tx] =  hidden_units[tx];
		//	printf("SUM: %d, %d, %f, %f \n", element, i, hidden_units[i], hidden_units_master[i]);
 		if(element == 0) {
			hidden_units_master[0] = 0.0;
		}
		}
	} 
  }
}






__global__ void gpu_output_error_kernel_function(float *delta, float *t, float *o, int count, float *err) {

   int element = threadIdx.x + blockDim.x * blockIdx.x;  

   if(element != 0 && element < count) {
       delta[element] = o[element] * ( 1.0 - o[element]) * (t[element] - o[element]); 
//	printf("Output err: %d, %f, %f = %f \n", element, o[element], t[element], delta[element]); 
   }
}

__global__ void kernel_squash(float *hidden, int count) {

   int element     = threadIdx.x + blockDim.x * blockIdx.x;  

   if(element >0  && element < count) {
       float orig = hidden[element]; 
       hidden[element] = (1.0 / (1.0 + exp(- orig)));
//	printf("Element: %d, Orig: %f, ,squash: %f \n", element, orig,  hidden[element]);
   }
} 



__global__ void gpu_hidden_error_kernel_function (float *hidden_delta_d, int hid, float *output_delta_d, int out, float *hidden_weights_d, float *hidden_units_d) {

   int element     = threadIdx.x + blockDim.x * blockIdx.x;  

   if(element < hid) { 
//   	for(int i =1; i < out; i++) {
	    float h = hidden_units_d[element] ; 	
            float sum = output_delta_d[1] * hidden_weights_d[element]; 
	    //printf("%d, %f * %f = %f \n", element,  output_delta_d[1],  hidden_weights_d[element], sum);		
            hidden_delta_d[element] = h * (1.0 -h) * sum;
            //printf("%d => %f * (1.0 - %f)  * %f = %f\n", element, h,h, sum, hidden_delta_d[element]);  

//   	} 
   }
    
}



__global__ void gpu_weight_adjust_function(float *delta, int out, float *hidden_units, int hid, float *hidden_weights) {


   int element     = threadIdx.x + blockDim.x * blockIdx.x;  

   if(element < hid) {
   	float new_dw = ((ETA * delta[1] * hidden_units[element]) + (MOMENT * 0));
        hidden_weights[element] += new_dw;
        //printf("Element: %d, new val = %f, %f \n", element, new_dw, hidden_weights[element]);  

   }

} 


/// Algorithm using Naive method 
/*
void gpu_bpnn_layerforward(float *input_units,  float *hidden_units, float *input_weights, int inp, int hidden) {
    // Place holder to complete input sanity check 

    //Allocate Blocks 
    dim3 DimGrid((inp-1)/BLOCK_SIZE +1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE+1, 1, 1); 

   
    // Invoke CUDA kernel -----------------------------------------------------
    kernel_bpnn_layerforward<<<DimGrid, DimBlock>>>(input_units, hidden_units, input_weights, inp, hidden);

    cudaDeviceSynchronize();

    dim3 DimGrid2((hidden-1)/BLOCK_SIZE +1, 1, 1);
    dim3 DimBlock2(BLOCK_SIZE+1, 1, 1); 
 
   // Invoke Squashing kernel 
    kernel_squash<<<DimGrid2, DimBlock2>>>(hidden_units, hidden);
} 
*/


// Algorithm using shared memory 
void gpu_bpnn_layerforward(float *input_units,  float *hidden_units, float *input_weights, int inp, int hidden) {


    dim3 DimGrid((inp-1)/BLOCK_SIZE +1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    kernel_bpnn_layerforward<<<DimGrid, DimBlock>>>(input_units, hidden_units, input_weights, inp, hidden);


    cudaDeviceSynchronize();

    dim3 DimGrid2(1, 1, 1);
    dim3 DimBlock2(hidden, 1, 1);

    // Invoke Squashing kernel
    kernel_squash<<<DimGrid2, DimBlock2>>>(hidden_units, hidden);
}


void gpu_output_error_kernel(float *delta, float *target, float *output, int count, float *err) {

    dim3 DimGrid((count-1)/BLOCK_SIZE +1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1); 

    gpu_output_error_kernel_function<<<DimGrid, DimBlock>>>(delta, target, output, count, err);
} 

void gpu_hidden_error_kernel(float *hidden_delta_d , int hid, float *output_delta_d, int out, float *hidden_weights_d, float *hidden_units_d) {

    dim3 DimGrid((hid-1)/BLOCK_SIZE +1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1); 

    gpu_hidden_error_kernel_function<<<DimGrid, DimBlock>>>(hidden_delta_d, hid, output_delta_d, out, hidden_weights_d, hidden_units_d);
}


void gpu_weight_adjust(float *delta, int out, float *hidden_units, int hid, float *hidden_weights) {

    dim3 DimGrid((hid-1)/BLOCK_SIZE +1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    gpu_weight_adjust_function<<<DimGrid, DimBlock>>>(delta, out, hidden_units, hid, hidden_weights);

}  
