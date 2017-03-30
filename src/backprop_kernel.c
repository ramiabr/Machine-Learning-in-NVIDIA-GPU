#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "backprop.h"
#include "support.h"

#define ERR 2.0

////////////////////////////////////////////////////////////////////////////////


/// GPU Functions below 
extern void gpu_kernel_wrapper(float *l1,  float *l2, float **conn, int n1, int n2);

extern void gpu_output_error(float *delta, float *target, float *output, int count, float *err, float *hidden_delta, int hid, float **hidden_weights, float **hidden_units, int out); 

//extern void gpu_hidden_error(float *delta, float *target, float *output, int count, float *err); 

extern void verify_data(float *gpu, float *cpu, int count, char name[]);

extern void verify_data_2d(float **gpu, float **cpu, int , int j, char name[]);


// CPU Functions 
extern void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);


extern int setup(int argc, char** argv);

extern float **alloc_2d_dbl(int m, int n);

extern float squash(float x);

extern void   bpnn_save_dbg_gpu(BPNN *net, float **hidden_weights);

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	setup(argc, argv);
}



void bpnn_save_dbg_gpu(BPNN *net, float **hidden_weights)
{
  int n1, n2, n3, i, j;
  float **w;

  FILE *pFile;
  pFile = fopen( "out_gpu.txt", "w+" );

  n1 = net->input_n;  n2 = net->hidden_n;  n3 = net->output_n;
  fprintf(pFile, "Saving %dx%dx%d network\n", n1, n2, n3);

  w = hidden_weights;
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
          fprintf(pFile, "%d,%d,%f\n", i,j,w[i][j]);
    }
  }

  fclose(pFile);
  return;
}

void bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   

  float *gpu_hidden_units, *gpu_output_units; 
  float *hidden_delta, *output_delta;
  float *hidden_err, *output_err;  
  float *hidden_prev, **gpu_hidden_weights; 

  int hid_d = hid+1;    
  int in_d = in+1;
  int out_d = out + 1;

  gpu_hidden_units = (float*) malloc(sizeof(float) *  hid_d); 
  gpu_output_units = (float*) malloc(sizeof(float) *  out_d); 
  
  hidden_delta     = (float*) malloc(sizeof(float) *  hid_d); 
  output_delta     = (float*) malloc(sizeof(float) *  out_d); 
 

  gpu_hidden_weights = alloc_2d_dbl( (hid_d+1), (out_d+1)); 

  int i,j;

  for(i=0; i < out_d; i++) {
	for(j=0; j <hid_d; j++) {
		gpu_hidden_weights[j][i] =   net->hidden_weights[j][i]; 		
	}
  } 

   Timer timer;
   startTime(&timer);
   printf("\n\nPerforming GPU Computing\n");
 
 // Below is coded as threshold  
  net->input_units[0] = 1.0; 

  gpu_kernel_wrapper(net->input_units, gpu_hidden_units ,net->input_weights, in_d, hid_d);    

  stopTime(&timer); printf("layerforward : %f s\n", elapsedTime(timer));
  startTime(&timer);

  gpu_kernel_wrapper(gpu_hidden_units, gpu_output_units, net->hidden_weights, hid_d, out_d);

  stopTime(&timer); printf("layerforward : %f s\n", elapsedTime(timer));
  startTime(&timer);
 
  // Below is coded as threshold  
  gpu_hidden_units[0] = 1.0;  

  gpu_output_error(output_delta,  net->target, gpu_output_units, out_d, &output_err, hidden_delta, hid_d, gpu_hidden_weights, gpu_hidden_units, out_d); 

  stopTime(&timer); printf("Calculating Error and Adjusting Weights  : %f s\n", elapsedTime(timer));
  startTime(&timer);
 
 printf("\n\nPerforming CPU computation\n");
  
  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);

  stopTime(&timer); printf("bpnn_layerforward : %f s\n", elapsedTime(timer));
  startTime(&timer);

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);

  stopTime(&timer); printf("bpnn_layerforward : %f s\n", elapsedTime(timer));
  startTime(&timer);
  
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);

  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  

  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

 
  stopTime(&timer); printf("Calculating Error and Adjusting Weights  : %f s\n", elapsedTime(timer));

  printf("\n\nComparison between CPU and GPU:\n"); 
  printf("----------------------------------\n");  
  //verify_data(gpu_hidden_units, net->hidden_units, hid, "Hidden Units"); 
  //verify_data(gpu_output_units, net->output_units, out, "Output Units"); 
  //verify_data(output_delta, net->output_delta, out, "Output Delta"); 
  //verify_data(hidden_delta, net->hidden_delta, hid, "Hidden Delta"); 

   verify_data_2d(gpu_hidden_weights, net->hidden_weights, hid_d, out_d, "Hidden Weights"); 
   

  printf("\n\n");
  bpnn_save_dbg_gpu(net, gpu_hidden_weights); 

}


void verify_data_2d(float **gpu, float **cpu, int hid_d , int out_d, char name[]) {

        int i,j;
	for(i=0; i < out_d; i++) {
        	for(j=0; j <hid_d; j++) {



          	float err = ((cpu[j][i] - gpu[j][i])/ cpu[j][i])*100;
          	float diff = cpu[j][i] - gpu[j][i];

          	if(err < -0.000001)
                  err *= -1;

          	if(diff < -0.0000001)
                  diff *= -1;

          	if(err > 2.0 && diff > 0.00001 )  {
//			if(cpu[j][i] != gpu[j][i]) { 
	    			printf("ERROR: Parameter: %s  Failed for %d,%d, (GPU: %f vs CPU: %f )  \n", name, j, i, gpu[j][i], cpu[j][i]); 
			return;
			}  
        	}
  	} 
    printf("%s  PASSED \n", name); 
}


void verify_data (float *gpu, float *cpu, int count,  char name[]) {

    int i;
    for(i=0; i <= count; i++) {

	  float err = ((cpu[i] - gpu[i])/ cpu[i])*100; 
          float diff = cpu[i] - gpu[i]; 

          if(err < -0.000001) 
         	  err *= -1;

	  if(diff < -0.0000001) 
		  diff *= -1;
	
	  if(err > 2.0 && diff > 0.00001 )  {
	    printf("ERROR: Parameter: %s  Failed for %d, (GPU: %f vs CPU: %f )  %f \n", name, i, gpu[i], cpu[i], diff ); 
	    return;
          }
    }
    printf("%s    PASSED \n", name); 
} 



