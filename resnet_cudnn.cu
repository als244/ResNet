#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "resnet_cudnn.h"

#define SM_COUNT 82
#define WARP_PER_SM 4
#define THREAD_PER_WARP 32
#define MAX_THREAD_PER_BLOCK 1024
#define TILE_WIDTH 32
#define BLOCK_ROWS 8
#define CUDA_BATCH_SIZE 32
#define MAX_SHARED_MEMORY 48000
#define MAX_SHARED_MEM_FLOATS 12000
#define MAX_THREAD_PER_BLOCK_INCL_REG 512




// used to hide all print statements for device data
#define TO_PRINT false

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)


/* DECLARING FUNCTIONS HERE */
void testConvolution(int in_spatial_dim, int kern_dim, int in_filters, int out_filters,  int stride, int batch_size, 
																float * input, float * weights, float * output);


/* START OF KERNELS/FUNCTIONS */

__global__ void setVal(int size, float val, float *out){
 	int ind = blockDim.x * blockIdx.x + threadIdx.x;
 	if (ind >= size){
 		return;
 	}
 	out[ind] = val;
}

void init_weights_gaussian_device(curandGenerator_t * gen, int size, float *X, float mean, float var){
 	float stddev = sqrtf(var);
 	curandStatus_t status = curandGenerateNormal(*gen, X, (size_t) size, mean, stddev);
 }


/* NON-OPTIMIZED CUSTOM KERNELS (non-bottleneck) */

// ASSUME 1-D launch
__global__ void addVec(int size, float * A, float * B, float * out){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size){
		return;
	}
	out[i] = A[i] + B[i];
}

// GRID has dim (ROWS / TILE_WIDTH, COLS/TILE_WIDTH)
// each BLOCK has dim (TILE_WIDTH, TILE_WIDTH)
__global__ void matMul(const float *M, const float *N, int m, int k, int n, float *out){

	
	int row_ind = blockIdx.x * TILE_WIDTH + threadIdx.x;
	int col_ind = blockIdx.y * TILE_WIDTH + threadIdx.y;

	if (row_ind >= m || col_ind >= n){
		return;
	}

	float val = 0;
	for (int z = 0; z < k; z++){
		val += M[row_ind * k + z] * N[z * n + col_ind];
	}
	out[row_ind * n + col_ind] = val;
}

// grid has dim (ROWS / TILE_WIDTH, COLS/TILE_WIDTH)
// each BLOCK has dim (TILE_WIDTH , TILE_WIDTH) = # of threads
__global__ void transpose(const float *in, int rows, int cols, float * out) {

  int row_ind = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int col_ind = blockIdx.y * TILE_WIDTH + threadIdx.y;
  
  
  if (col_ind >= cols || row_ind >= rows){
  	return;
  }

  out[col_ind * rows + row_ind] = in[row_ind * cols + col_ind];
}

// FOR NOW KEEP NAIVE (UN-OPTIMIZED)...
// not bothering with shared memory for now...

// Independent over (output_filter_id, output_spatial_row, output_spatial_col, sample)
// Launch with gridDim (out_spatial_dim, out_spatial_dim, max(1, out_filters / (MAX_THREAD_PER_BLOCK)) and blockDim (batch_size, min(MAX_THREAD_PER_BLOCK / batch_size, output_filters))
// Room to optimize a lot...
__global__ void doConvolution(const float * input, const float * weights, int spatial_dim, int kern_dim, int in_filters, int out_filters, int stride, int batch_size, float * out){

	int out_spatial_row = blockIdx.x;
	int out_spatial_col = blockIdx.y;
	int out_filter_id = blockIdx.z * blockDim.y + threadIdx.y;
	int sample_ind = threadIdx.x;
	int out_spatial_dim = spatial_dim / stride;

	// shoudn't need to check based on launch specs but will anyways
	if ((out_filter_id >= out_filters) || (sample_ind >= batch_size) || (out_spatial_row >= out_spatial_dim) || (out_spatial_col >= out_spatial_dim)) {
		return;
	}

	int in_spatial_row_start = stride * out_spatial_row;
	int in_spatial_col_start = stride * out_spatial_col;

	int half_kernel_dim = kern_dim / 2;
	int in_spatial_row, in_spatial_col, kernel_ind;
	
	// (Calling "Kernel" a 3-D obj of weights where there is 2-D conv filter for each input channel)
	int kernel_size = (kern_dim * kern_dim * in_filters);

	float out_val = 0;
	float in_spatial_val;
	for (int row_offset = -half_kernel_dim; row_offset <= half_kernel_dim; row_offset++){
		for (int col_offset = -half_kernel_dim; col_offset <= half_kernel_dim; col_offset++){
			for (int in_channel = 0; in_channel < in_filters; in_channel++){
						
				// compute spatial value
				in_spatial_row = in_spatial_row_start + row_offset;
				in_spatial_col = in_spatial_col_start + col_offset;
				kernel_ind = kern_dim * kern_dim * in_channel  + kern_dim * (row_offset + half_kernel_dim) + (col_offset + half_kernel_dim);
				if ((in_spatial_row < 0) || (in_spatial_row >= spatial_dim) || (in_spatial_col < 0) || (in_spatial_col >= spatial_dim)) {
					in_spatial_val = 0;
				}
				else{
					in_spatial_val = input[spatial_dim * spatial_dim * in_filters * sample_ind + spatial_dim * in_filters * in_spatial_row + in_filters * in_spatial_col + in_channel];
				}

				// multiply with conv weight
				// threadIdx.x specifies the output filter id
				// kernel_ind specifies the (x, y, input_channel)
				out_val += weights[out_filter_id * kernel_size + kernel_ind] * in_spatial_val;
			}
		}
	}
	out[out_spatial_dim * out_spatial_dim * out_filters * sample_ind + out_spatial_dim * out_filters * out_spatial_row + out_filters * out_spatial_col + out_filter_id] = out_val;
}


// FOR NOW KEEP NAIVE (UN-OPTIMIZED)...
// not bothering with shared memory for now...

// Independent over (input filter, input_x, input_y, sample)
// could use shared memory over conv weights...
// Launch with gridDim (spatial_dim, spatial_dim, max(1, input_filters / (MAX_THREAD_PER_BLOCK / batch_size))) and blockDim (batch_size, min(MAX_THREAD_PER_BLOCK / batch_size, input_filters))
// Can parallelize further with reductions, if want to optimize
__global__ void convolutionDerivInput(const float * input, const float * weights, const float * out_deriv, int spatial_dim, int kern_dim, int in_filters, int out_filters, int stride, int batch_size, bool toAdd,
											float * input_deriv){

	int spatial_row = blockIdx.x;
	int spatial_col = blockIdx.y;
	int in_filter_id = blockIdx.z * blockDim.y + threadIdx.y;
	int sample_ind = threadIdx.x;
	// shouldn't need to check based on launch specs, but will anyways...
	if ((spatial_row >= spatial_dim) || (spatial_col >= spatial_dim) || (in_filter_id >= in_filters) || (sample_ind >= batch_size)){
		return;
	}

	int out_spatial_dim = spatial_dim / stride;
	int half_kernel_dim = kern_dim / 2;
	int out_spatial_row_start = spatial_row / stride;
	int out_spatial_col_start = spatial_col / stride;
	int kern_ind, kern_row_ind, kern_col_ind, out_spatial_ind, out_spatial_row, out_spatial_col;
	int kernel_size = (kern_dim * kern_dim * in_filters);
	float out_spatial_val_deriv;
	float total_deriv = 0;
	for (int out_filt_id = 0; out_filt_id < out_filters; out_filt_id++){
		for (int row_offset = -half_kernel_dim; row_offset <= half_kernel_dim; row_offset++){
			for (int col_offset = -half_kernel_dim; col_offset <= half_kernel_dim; col_offset++){
				// compute output spatial value that used the input spatial value
				out_spatial_row = out_spatial_row_start + row_offset;
				out_spatial_col = out_spatial_col_start + col_offset;
				// index of output spatial val (iterate over samples in batch, then rows, then columns, then channels)
				out_spatial_ind = out_spatial_dim * out_spatial_dim * out_filters * sample_ind + out_spatial_dim * out_filters * out_spatial_row + out_filters * out_spatial_col + out_filt_id;

				// get kernel index used to generate out spatial value for corresponding input spatial value
				kern_row_ind = spatial_row - out_spatial_row * stride + half_kernel_dim;
				kern_col_ind = spatial_col - out_spatial_col * stride + half_kernel_dim;
				kern_ind = kern_dim * kern_dim * in_filter_id + kern_dim * kern_row_ind + kern_col_ind;
				if ((kern_row_ind < 0) || (kern_row_ind >= kern_dim) || (kern_col_ind < 0) || (kern_col_ind >= kern_dim) ||
						(out_spatial_row < 0) || (out_spatial_row >= out_spatial_dim) || (out_spatial_col < 0) || (out_spatial_col >= out_spatial_dim)) {
					out_spatial_val_deriv = 0;
				}
				else{
					out_spatial_val_deriv = weights[out_filt_id * kernel_size + kern_ind] * out_deriv[out_spatial_ind];
				}
				total_deriv += out_spatial_val_deriv;
			}
		}
	}
	int input_spatial_ind = spatial_dim * spatial_dim * in_filters * sample_ind + spatial_dim * in_filters * spatial_row + in_filters * spatial_col + in_filter_id;
	// used because normal backprop + residual adds to deriv
	if (toAdd){
		input_deriv[input_spatial_ind] += total_deriv;
	}
	else{
		input_deriv[input_spatial_ind] = total_deriv;
	}
	
}

// FOR NOW KEEP NAIVE (UN-OPTIMIZED)...
// not bothering with shared memory for now...

// Independent over (input filter, output filter, kern_x, kern_x)
// could use shared memory over input values...
// Launch with gridDim (kern_dim, kern_dim, output_filters) and blockDim (input_filters) [if input_filters > MAX_THREAD_PER_BLOCK switch ordering of input_filters and output_filters in launch]
__global__ void convolutionDerivWeights(const float * input, const float * weights, const float * out_deriv, int spatial_dim, int kern_dim, int in_filters, int out_filters, int stride, int batch_size,
											float * weight_deriv, bool is_block_dim_inp){

	int in_filter_id;
	int out_filter_id;
	if (is_block_dim_inp){
		in_filter_id = threadIdx.x;
		out_filter_id = blockIdx.z;
	}
	else{
		in_filter_id = blockIdx.z;
		out_filter_id = threadIdx.x;
	}
	int kern_row = blockIdx.x;
	int kern_col = blockIdx.y;

	// shouldn't need to check based on launch specs, but will anyways...
	if ((in_filter_id >= in_filters) || (out_filter_id >= out_filters) || (kern_row >= kern_dim) || (kern_col >= kern_dim)){
		return;
	}

	int kern_ind = kern_dim * kern_dim * in_filter_id + kern_dim * kern_row + kern_col;

	int kernel_size = (kern_dim * kern_dim * in_filters);
	int half_kernel_dim = kern_dim / 2;
	int out_spatial_dim = spatial_dim / stride;
	int in_spatial_row, in_spatial_col, in_spatial_ind, out_spatial_ind;
	float out_spatial_val_deriv = 0;
	float total_deriv = 0;
	for (int s = 0; s < batch_size; s++){
		for (int out_row = 0; out_row < out_spatial_dim; out_row++){
			for (int out_col = 0; out_col < out_spatial_dim; out_col++){

				// given out_row, out_col, kern_row, kern_col => get the input value used to generate output
				in_spatial_row = stride * out_row + kern_row - half_kernel_dim;
				in_spatial_col = stride * out_col + kern_col - half_kernel_dim;

				// accounting for input filter and sample in batch get index into the input values
				in_spatial_ind = spatial_dim * spatial_dim * in_filters * s + spatial_dim * in_filters * in_spatial_row + in_filters * in_spatial_col + in_filter_id;

				// going from sample, out_row, out_col, out_filter to get index into out_deriv values
				out_spatial_ind = out_spatial_dim * out_spatial_dim * out_filters * s + out_spatial_dim * out_filters * out_row + out_filters * out_col + out_filter_id;

				if ((in_spatial_row < 0) || (in_spatial_row >= spatial_dim) || (in_spatial_col < 0) || (in_spatial_col >= spatial_dim)){
					out_spatial_val_deriv = 0;
				}
				else{
					out_spatial_val_deriv = input[in_spatial_ind] * out_deriv[out_spatial_ind];
				}
				total_deriv += out_spatial_val_deriv;
			}
		}
	}
	weight_deriv[kernel_size * out_filter_id + kern_ind] = total_deriv;
}


// iterating over each filter separately
// launch with (OUTFILTERS) grid dim and thread dim of 1 (could easily parallelize menas + vars, with reduction, but save for later..)
// could also use shared memory here if want to be faster
// input is the output of convolution
// ASSUME reLU activation function
__global__ void doBatchNormAndActivate(const float * input, const float * gamma, const float * beta,
								int spatial_dim, int filters, int batch_size, float eps, float * means, float * vars, float * normalized_temp, float * normalized, float * activated, bool to_activate){

	int filter_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (filter_id >= filters){
		return;
	}

	float mean, var;
	float sum = 0;
	for (int s = 0; s < batch_size; s++){
		for (int i = 0; i < spatial_dim; i++){
			for (int j = 0; j < spatial_dim; j++){
				sum += input[spatial_dim * spatial_dim * filters * s + spatial_dim * filters * i + filters * j + filter_id];
			}
		}
	}

	mean = sum / (batch_size * spatial_dim * spatial_dim);
	means[filter_id] = mean;

	float var_sum = 0;
	int inp_index;
	for (int s = 0; s < batch_size; s++){
		for (int i = 0; i < spatial_dim; i++){
			for (int j = 0; j < spatial_dim; j++){
				inp_index = spatial_dim * spatial_dim * filters * s + spatial_dim * filters * i + filters * j + filter_id;
				var_sum += (input[inp_index] - mean) * (input[inp_index] - mean);
			}
		}
	}

	var = var_sum / (batch_size * spatial_dim * spatial_dim);
	vars[filter_id] = var;

	float normalized_temp_val, normalized_val;
	for (int s = 0; s < batch_size; s++){
		for (int i = 0; i < spatial_dim; i++){
			for (int j = 0; j < spatial_dim; j++){
				inp_index = spatial_dim * spatial_dim * filters * s + spatial_dim * filters * i + filters * j + filter_id;
				normalized_temp_val = (input[inp_index] - mean) / sqrtf(var + eps);
				normalized_temp[inp_index] = normalized_temp_val;
				normalized_val = gamma[filter_id] * normalized_temp_val + beta[filter_id];
				normalized[inp_index] = normalized_val;
				if (to_activate){
					activated[inp_index] = fmaxf(normalized_val, 0); 
				}
				else{
					activated[inp_index] = normalized_val;
				}
			}
		}
	}
}


// iterating over each filter separately
// launch with (OUTFILTERS) grid dim and thread dim of 1 (could easily parallelize menas + vars, with reduction, but save for later..)
// could also use shared memory here if want to be faster
// input is the output of convolution
// ASSUME reLU activation function
__global__ void activationAndBatchNormDeriv(const float * input, const float * gamma, const float * beta, 
									int spatial_dim, int filters, int batch_size, float eps, const float * means, const float * vars, const float * normalized_temp, const float * activated,
									const float * out_layer_deriv, float * normalized_temp_deriv, float * gamma_deriv, float * beta_deriv, float * input_deriv, bool to_activate_deriv){
	
	
	int filter_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (filter_id >= filters){
		return;
	}

	float n_samples = batch_size * spatial_dim * spatial_dim;
	float gamma_val = gamma[filter_id];
	float mean_val = means[filter_id];
	float var_val = vars[filter_id];

	// first compute dL/activated (relu deriv) and then dL/dNormalized_Temp (== x hat)
	// also can compute dL/dGamma and dL/dBeta (parameters of batch norm)
	int index;
	float dGamma = 0;
	float dBeta = 0;
	float activated_val, out_layer_deriv_val, normalized_temp_val;
	for (int s = 0; s < batch_size; s++){
		for (int i = 0; i < spatial_dim; i++){
			for (int j = 0; j < spatial_dim; j++){
				index = spatial_dim * spatial_dim * filters * s + spatial_dim * filters * i + filters * j + filter_id;
				activated_val = activated[index];
				if (to_activate_deriv && (activated_val <= 0)) {
					normalized_temp_deriv[index] = 0;
				}
				else{
					out_layer_deriv_val = out_layer_deriv[index];
					normalized_temp_val = normalized_temp[index];
					normalized_temp_deriv[index] = out_layer_deriv_val * gamma_val;
					dGamma += out_layer_deriv_val * normalized_temp_val;
					dBeta += out_layer_deriv_val;
				}
			}
		}
	}

	// save down dGamma and dBeta so optimzer can update parameters
	gamma_deriv[filter_id] = dGamma;
	beta_deriv[filter_id] = dBeta;

	// compute dL/dVar and most of dL/dMean
	float dVar = 0;
	float dMean = 0;
	float partial_var_deriv = 0; 
	float norm_temp_deriv_val;
	float filt_var_three_halfs_power = -0.5 * powf(var_val + eps, -1.5);
	float neg_filt_var_recip_sqrt = -1.0 / sqrtf(var_val + eps);
	for (int s = 0; s < batch_size; s++){
		for (int i = 0; i < spatial_dim; i++){
			for (int j = 0; j < spatial_dim; j++){
				index = spatial_dim * spatial_dim * filters * s + spatial_dim * filters * i + filters * j + filter_id;
				norm_temp_deriv_val = normalized_temp_deriv[index];
				dVar += norm_temp_deriv_val * (input[index] - mean_val) * filt_var_three_halfs_power;
				dMean += norm_temp_deriv_val * neg_filt_var_recip_sqrt;
				partial_var_deriv += -2 * (input[index] - mean_val);
			}
		}
	}

	// finish off dL/dMean
	dMean += dVar * partial_var_deriv / n_samples;

	// compute dL/dX (aka w.r.t. to input to batch norm which is typically the output of a conv)
	// saving input_deriv so backprop can continue to previous layer
	for (int s = 0; s < batch_size; s++){
		for (int i = 0; i < spatial_dim; i++){
			for (int j = 0; j < spatial_dim; j++){
				index = spatial_dim * spatial_dim * filters * s + spatial_dim * filters * i + filters * j + filter_id;
				input_deriv[index] = normalized_temp_deriv[index] * (-1 * neg_filt_var_recip_sqrt) + dVar * (2 * (input[index] - mean_val)) / n_samples + dMean / n_samples;
			}
		}
	}
}



// assume grid launch of (SPATIAL_OUT_DIM, SPATIAL_OUT_DIM) and block dim of (FILTERS)
// could parallelize over batches as well, but probably ok. 
// *runs into issues if #filters greater than threads per block
__global__ void doMaxPool(const float * input, int kern_dim, int stride, int batch_size, int * max_inds, float * out){

	int filter_id = threadIdx.x;

	// know this because of launch specification
	int filters = blockDim.x;
	int in_spatial_dim = stride * gridDim.x;
	int out_spatial_dim = gridDim.x;

	int spatial_row_start = stride * blockIdx.x;
	int spatial_col_start = stride * blockIdx.y;

	int half_kernel_dim = kern_dim / 2;

	float max_val, inp_val;
	int spatial_row, spatial_col, max_ind, inp_ind, out_ind;
	for (int s = 0; s < batch_size; s++){
		max_val = -1024;
		max_ind = -1024;
		for (int row_off = -half_kernel_dim; row_off <= half_kernel_dim; row_off++){
			for (int col_off = -half_kernel_dim; col_off <= half_kernel_dim; col_off++){
				spatial_row = spatial_row_start + row_off;
				spatial_col = spatial_col_start + col_off;
				if ((spatial_row < 0) || (spatial_row >= in_spatial_dim) || (spatial_col < 0) || (spatial_col >= in_spatial_dim)){
					continue;
				}
				inp_ind = in_spatial_dim * in_spatial_dim * filters * s + in_spatial_dim * filters * spatial_row + filters * spatial_col + filter_id;
				inp_val = input[inp_ind];
				if (inp_val > max_val){
					max_val = inp_val;
					max_ind = inp_ind;
				}
			}
		}
		out_ind = out_spatial_dim * out_spatial_dim * filters * s + out_spatial_dim * filters * blockIdx.x + filters * blockIdx.y + filter_id;
		max_inds[out_ind] = max_ind;
		out[out_ind] = max_val;
	}
}

// assume grid launch of (SPATIAL_OUT_DIM, SPATIAL_OUT_DIM, OUT_FILTERS) and block dim of (BATCH_SIZE)
// max_inds_populated is mapping from max_pool_out_index -> associated max_index of input (populated from forward pass)
// also assume max_pool_inp_deriv is populated with all 0's to begin with and we overwrite non-zero values
__global__ void maxPoolDeriv(const int *max_inds_populated, const float *out_deriv, int kern_dim, int in_spatial_dim, int stride, int filters, int batch_size, float * max_pool_inp_deriv){

	int out_spatial_dim = in_spatial_dim / stride;

	int out_spatial_row = blockIdx.x;
	int out_spatial_col = blockIdx.y;
	int out_filter_id = blockIdx.z;
	int sample_ind = threadIdx.x;

	// based on launch spec should be ok, but check anyways
	if ((out_spatial_row >= out_spatial_dim) || (out_spatial_col >= out_spatial_dim) || (out_filter_id >= filters) || (sample_ind >= batch_size)){
		return;
	}

	int out_ind = out_spatial_dim * out_spatial_dim * filters * sample_ind + out_spatial_dim * filters * out_spatial_row + filters * out_spatial_col + out_filter_id;
	int max_ind_for_out = max_inds_populated[out_ind];

	max_pool_inp_deriv[max_ind_for_out] = out_deriv[out_ind];
}


// assume grid launch of (# Filters) and block dim of (batch size)
// could parallelize over batches as well, but probably ok. 
// *runs into issues if #filters greater than threads per block
__global__ void doFilterAvgPool(const float * input, int spatial_dim, float * out){

	int filter_id = blockIdx.x;
	int sample_ind = threadIdx.x;

	// know this because of launch specification
	int filters = gridDim.x;

	float sum = 0;
	for (int row = 0; row < spatial_dim; row++){
		for (int col = 0; col < spatial_dim; col++){
			sum += input[spatial_dim * spatial_dim * filters * sample_ind + spatial_dim * filters * row + filters * col + filter_id];
		}
	}

	float avg_val = sum / (spatial_dim * spatial_dim);
	out[filters * sample_ind + filter_id] = avg_val;
}

// assume grid launch of (# Filters) and block dim of (batch size)
// could parallelize over batches as well, but probably ok. 
// *runs into issues if #filters greater than threads per block
__global__ void filterAvgPoolDeriv(const float * pooled_deriv, int filters, int batch_size, int spatial_dim, float * out){

	int filter_id = blockIdx.x;
	int sample_ind = threadIdx.x;

	// unnecessary because of launch conditions, but putting anyways...
	if ((filter_id >= filters) || (sample_ind >= batch_size)){
		return;
	}

	// indexing into (N, 2048) = (batch_size, filters) matrix 
	float pooled_filt_deriv = pooled_deriv[sample_ind * filters + filter_id];
	float avg_pooled_filt_deriv = pooled_filt_deriv / (spatial_dim * spatial_dim);

	// populating the pre-pooled conv block output
	for (int row = 0; row < spatial_dim; row++){
		for (int col = 0; col < spatial_dim; col++){
			out[spatial_dim * spatial_dim * filters * sample_ind + spatial_dim * filters * row + filters * col + filter_id] = avg_pooled_filt_deriv;
		}
	}
}

__global__ void doActivation(int size, float * input, float * output){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= size){
		return;
	}
	output[i] = fmaxf(0, input[i]);
}


__global__ void doActivationDeriv(int size, float *input, float * upstream_deriv, float * output){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= size){
		return;
	}
	if (input[i] > 0){
		output[i] = upstream_deriv[i];
	}
	else{
		output[i] = 0;
	}
}

// assume pass in 1-D block with batch size blocks and 1 thread per block
// could exploit more parallelism here but shouldnt be bottleneck for now...
// assume X is a matrix where # rows = batch size and # columns = output dim
__global__ void softMax(const float * X, int batch_size, int output_len, float * out){
  int i = threadIdx.x;
  if (i < batch_size){
  	float max = X[i * output_len];
  	for (int j = 0; j < output_len; j++){
  		if (X[i * output_len + j] > max){
  			max = X[i * output_len + j];
  		}
  	}
    float sum = 0;
    for (int j = 0; j < output_len; j++){
      sum += expf(X[i * output_len + j] - max);
    }
    for (int j = 0; j < output_len; j++){
      out[i * output_len + j] = expf(X[i * output_len + j] - max) / sum;
    }
  }
}

// launch with gridDim (output_dim) and threadDim (batch_size)
__global__ void averageDerivOverBatchSize(float * output_deriv, int output_dim, int batch_size){

	int output_class = blockIdx.x;
	int sample_ind = threadIdx.x;

	// shouldn't happen because of launch spec but check anyways...
	if ((output_class >= output_dim) || (sample_ind >= batch_size)){
		return;
	}
	output_deriv[sample_ind * output_dim + output_class] /= batch_size;
}


// launch with gridDim = (batch_size), blockDim = (1)
__global__ void crossEntropyDeriv(float * output_deriv, const int * correct_classes, int output_dim, int batch_size){
	int i = threadIdx.x;
	if (i < batch_size){
		output_deriv[i * output_dim + correct_classes[i]] -= 1;
	}
}

// assume large 1-D launch
__global__ void updateMeans(int size, const float * gradients, const float * model_params, float base_mean_decay, float weight_decay, float * prev_means, int loc_ind){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= size){
		return;
	}
	if (isnan(gradients[i])){
		printf("ERROR in Update Means for Parameter at location: %d\nGradient is NAN at index: %d...keeping same running mean\n\n", loc_ind, i);
		return;
	}
	if (isinf(gradients[i])){
		printf("ERROR in Update Means for Parameter at location: %d\nGradient is INF at index: %d...keeping same running mean\n\n", loc_ind, i);
		return;
	}
	float grad_with_decay = gradients[i] + weight_decay * model_params[i];
	prev_means[i] = base_mean_decay * prev_means[i] + (1 - base_mean_decay) * grad_with_decay;
	
}

// assume large 1-D launch
__global__ void updateVars(int size, const float * gradients, const float * model_params, float base_var_decay, float weight_decay, float * prev_vars, int loc_ind){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= size){
		return;
	}
	if (isnan(gradients[i])){
		printf("ERROR in Update Vars for Parameter at location: %d\nGradient is NAN at index: %d...keeping same running var\n", loc_ind, i);
		return;
	}
	if (isinf(gradients[i])){
		printf("ERROR in Update Vars for Parameter at location: %d\nGradient is INF at index: %d...keeping same running var\n", loc_ind, i);
		return;
	}
	float grad_with_decay = gradients[i] + weight_decay * model_params[i];
	prev_vars[i] = base_var_decay * prev_vars[i] + (1 - base_var_decay) * grad_with_decay * grad_with_decay;
}

// assume large 1-D launch
__global__ void updateParams(int size, float * model_params, const float * means, const float * vars, float learning_rate, float weight_decay, float cur_mean_decay, float cur_var_decay, float eps, int loc_ind){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= size){
		return;
	}
	float mean_adj = means[i] / (1 - cur_mean_decay);
	float var_adj = vars[i] / (1 - cur_var_decay);
	float old_model_param = model_params[i];
	model_params[i] = model_params[i] - (learning_rate * (mean_adj / (sqrtf(var_adj) + eps)) + weight_decay * old_model_param);
	if (isnan(model_params[i])){
		printf("ERROR: for Parameter at location: %d\nto NAN at index: %d...resetting to prev value of %f\n", loc_ind, i, old_model_param);
		model_params[i] = old_model_param;
		printf("Var: %f, Var Decay: %f, Var Adj: %f, Sqrt of Var Adj: %f\n\n", vars[i], cur_var_decay, var_adj, sqrtf(var_adj));
		return;
	}
	if (isinf(model_params[i])){
		printf("ERROR: for Parameter at location: %d\nto INF at index: %d...resetting to prev value of %f\n", loc_ind, i, old_model_param);
		model_params[i] = old_model_param;
		return;
	}
}

/* INITIALIZE CORE MODEL STRUCTURES */

Dims * init_dimensions(int input, int init_kernel_dim, int init_conv_filters, int init_conv_stride, int init_maxpool_dim, int init_maxpool_stride, 
							int n_conv_blocks, int * is_block_spatial_reduction, int final_depth, int output){
	
	Dims * dims = (Dims *) malloc(sizeof(Dims));
	dims -> input = input;
	dims -> init_kernel_dim = init_kernel_dim;
	dims -> init_conv_filters = init_conv_filters;
	dims -> init_conv_stride = init_conv_stride;
	dims -> init_maxpool_dim = init_maxpool_dim;
	dims -> init_maxpool_stride = init_maxpool_stride;
	dims -> n_conv_blocks = n_conv_blocks;
	dims -> is_block_spatial_reduction = is_block_spatial_reduction;
	dims -> final_depth = final_depth;
	dims -> output = output;

	return dims;
}

BatchNorm * init_batch_norm(int spatial_dim, int depth, float gamma_val, bool is_zero){
	
	BatchNorm * batch_norm = (BatchNorm *) malloc(sizeof(BatchNorm));

	batch_norm -> spatial_dim = spatial_dim;
	batch_norm -> depth = depth;

	float * gamma, * beta;

	cudaMalloc(&gamma, depth * sizeof(float));
	cudaMemset(gamma, 0, depth * sizeof(float));
	// ZERO-GAMMA INITIALIZE TO SOLVE PROBLEM OF EXPLODING GRADIENTS (Goyal et al. 2017)
	if (!is_zero){
		setVal <<< ceil((float) depth / MAX_THREAD_PER_BLOCK), MAX_THREAD_PER_BLOCK >>> (depth, gamma_val, gamma);
	}

	cudaMalloc(&beta, depth * sizeof(float));
	cudaMemset(beta, 0, depth * sizeof(float));

	batch_norm -> gamma = gamma;
	batch_norm -> beta = beta;

	return batch_norm;

}

ConvBlock * init_conv_block(int incoming_filters, int incoming_spatial_dim, int reduced_depth, int expanded_depth, int stride, curandGenerator_t * gen, bool is_zero){
	
	ConvBlock * conv_block = (ConvBlock *) malloc(sizeof(ConvBlock));
	conv_block -> incoming_filters = incoming_filters;
	conv_block -> incoming_spatial_dim = incoming_spatial_dim;
	conv_block -> reduced_depth = reduced_depth;
	conv_block -> expanded_depth = expanded_depth;
	conv_block -> stride = stride;

	float * depth_reduction, *spatial, *depth_expansion;
	int depth_reduction_size, spatial_size, depth_expansion_size;
	float depth_reduction_fan_in_plus_fan_out, spatial_fan_in_plus_fan_out, depth_expansion_fan_in_plus_fan_out;

	BatchNorm *norm_depth_reduction, *norm_spatial, *norm_expansion, *norm_projection;

	depth_reduction_size = incoming_filters * reduced_depth;
	depth_reduction_fan_in_plus_fan_out = incoming_filters + reduced_depth;
	cudaMalloc(&depth_reduction, depth_reduction_size * sizeof(float));
	cudaMemset(depth_reduction, 0, depth_reduction_size * sizeof(float));
	if (!is_zero){
		init_weights_gaussian_device(gen, depth_reduction_size, depth_reduction, 0, 2.0 / depth_reduction_fan_in_plus_fan_out);
	}

	norm_depth_reduction = init_batch_norm(incoming_spatial_dim, reduced_depth, 1.0, is_zero);


	spatial_size = reduced_depth * reduced_depth * 3 * 3;
	spatial_fan_in_plus_fan_out = (3 * 3) * (reduced_depth + reduced_depth);
	cudaMalloc(&spatial, spatial_size * sizeof(float));
	cudaMemset(spatial, 0, spatial_size * sizeof(float));
	if (!is_zero){
		init_weights_gaussian_device(gen, spatial_size, spatial, 0, 2.0 / spatial_fan_in_plus_fan_out);
	}
	// the spatial decrease happens at middle 3x3 layer, to the last layer of stride block will receive lower spatial dim input
	if (stride == 2){
		incoming_spatial_dim /= 2;
	}
	norm_spatial = init_batch_norm(incoming_spatial_dim, reduced_depth, 1.0, is_zero);

	depth_expansion_size = expanded_depth * reduced_depth;
	depth_expansion_fan_in_plus_fan_out = reduced_depth + expanded_depth;
	cudaMalloc(&depth_expansion, depth_expansion_size * sizeof(float));
	cudaMemset(depth_expansion, 0, depth_expansion_size * sizeof(float));
	if (!is_zero){
		init_weights_gaussian_device(gen, depth_expansion_size, depth_expansion, 0, 2.0 / depth_expansion_fan_in_plus_fan_out);
	}
	conv_block -> depth_reduction = depth_reduction;
	conv_block -> norm_depth_reduction = norm_depth_reduction;

	conv_block -> spatial = spatial;
	conv_block -> norm_spatial = norm_spatial;


	conv_block -> depth_expansion = depth_expansion;

	norm_expansion = init_batch_norm(incoming_spatial_dim, expanded_depth, 1.0, is_zero);
	conv_block -> norm_expansion = norm_expansion;

	float * projection;
	int projection_size;
	if (stride == 2){
		projection_size = 3 * 3 * incoming_filters * expanded_depth;
	}
	else{
		projection_size = incoming_filters * expanded_depth;
	}

	// assuming only project when depths are different (all projections in resnet-50 this way)
	// could later change to adapt to just spatial transform...
	int projection_fan_in_plus_fan_out;
	if (incoming_filters != expanded_depth){
		cudaMalloc(&projection, projection_size * sizeof(float));
		cudaMemset(projection, 0, projection_size * sizeof(float));
		if (stride == 2){
			projection_fan_in_plus_fan_out = 3 * 3 * (incoming_filters + expanded_depth);
		}
		else{
			projection_fan_in_plus_fan_out = incoming_filters + expanded_depth;
		}
		if (!is_zero){
			init_weights_gaussian_device(gen, projection_size, projection, 0, 2.0 / (projection_fan_in_plus_fan_out));
		}
		norm_projection = init_batch_norm(incoming_spatial_dim, expanded_depth, 1.0, is_zero);
	}
	else{
		projection = NULL;
		norm_projection = NULL;
	}

	conv_block -> projection = projection;
	conv_block -> norm_projection = norm_projection;

	return conv_block;
}

Params * init_model_parameters(Dims * model_dims, curandGenerator_t * gen, bool is_zero){

	Params * params = (Params *) malloc(sizeof(Params));

	// dimensions unpacked
	int input_dim = model_dims -> input;
	int n_conv_blocks = model_dims -> n_conv_blocks;
	int init_kernel_dim = model_dims -> init_kernel_dim;
	int init_conv_filters = model_dims -> init_conv_filters;
	int * is_block_spatial_reduction = model_dims -> is_block_spatial_reduction;
	int output_dim = model_dims -> output;

	// init array to hold pointers to weights
	// 3 * 3 weight arrays per conv block (weights, gamma, beta per layer in block) + 3 * inital + fully connected + 4 projections * 3
	int n_locations = 16 + 9 * n_conv_blocks;
	params -> n_locations = n_locations;

	float ** locations = (float **) malloc(n_locations * sizeof(float *));
	int * sizes = (int *) malloc(n_locations * sizeof(int));
	// tracking location ind as we start allocating...
	


	// init first 7 * 7 conv_layer
	float * init_conv_layer;
	int init_conv_size = init_kernel_dim * init_kernel_dim * init_conv_filters * 3;
	float init_conv_fan_in_plus_fan_out = 7 * 7 * (3 + init_conv_filters);
	cudaError_t malloc_err = cudaMalloc(&init_conv_layer,  init_conv_size * sizeof(float));
	cudaError_t memset_err = cudaMemset(init_conv_layer, 0, init_conv_size * sizeof(float));
	if (!is_zero){
		init_weights_gaussian_device(gen, init_conv_size, init_conv_layer, 0, 2.0 / init_conv_fan_in_plus_fan_out);
	}
	params -> init_conv_layer = init_conv_layer;
	int loc_ind = 0;
	locations[loc_ind] = init_conv_layer;
	sizes[loc_ind] = init_kernel_dim * init_kernel_dim * init_conv_filters * 3;
	loc_ind++;

	BatchNorm * norm_init_conv = init_batch_norm(input_dim / model_dims -> init_conv_stride, init_conv_filters, 1.0, is_zero);
	params -> norm_init_conv = norm_init_conv;

	locations[loc_ind] = norm_init_conv -> gamma;
	sizes[loc_ind] = init_conv_filters;
	loc_ind++;

	locations[loc_ind] = norm_init_conv -> beta;
	sizes[loc_ind] = init_conv_filters;
	loc_ind++;
	

	// init conv blocks
	ConvBlock ** conv_blocks = (ConvBlock **) malloc(n_conv_blocks * sizeof(ConvBlock *));
	int incoming_filters = init_conv_filters;
	// assume stride 2 initial conv layer then stride 2 pool before entering conv_blocks
	int incoming_spatial_dim = input_dim / 4;
	int stride = 1;
	int reduced_depth = init_conv_filters;
	int expanded_depth = 4 * init_conv_filters;
	for (int i = 0; i < n_conv_blocks; i++){
		if (is_block_spatial_reduction[i] == 1){
			stride = 2;
			reduced_depth *= 2;
			expanded_depth *= 2;
		}
		else{
			stride = 1;
		}
		conv_blocks[i] = init_conv_block(incoming_filters, incoming_spatial_dim, reduced_depth, expanded_depth, stride, gen, is_zero);
		locations[loc_ind] = conv_blocks[i] -> depth_reduction;
		sizes[loc_ind] = incoming_filters * reduced_depth;
		loc_ind++;
		locations[loc_ind] = conv_blocks[i] -> norm_depth_reduction -> gamma;
		sizes[loc_ind] = reduced_depth;
		loc_ind++;
		locations[loc_ind] = conv_blocks[i] -> norm_depth_reduction -> beta;
		sizes[loc_ind] = reduced_depth;
		loc_ind++;

		locations[loc_ind] = conv_blocks[i] -> spatial;
		sizes[loc_ind] = reduced_depth * reduced_depth * 3 * 3;
		loc_ind++;
		locations[loc_ind] = conv_blocks[i] -> norm_spatial -> gamma;
		sizes[loc_ind] = reduced_depth;
		loc_ind++;
		locations[loc_ind] = conv_blocks[i] -> norm_spatial -> beta;
		sizes[loc_ind] = reduced_depth;
		loc_ind++;

		locations[loc_ind] = conv_blocks[i] -> depth_expansion;
		sizes[loc_ind] = expanded_depth * reduced_depth;
		loc_ind++;

		locations[loc_ind] = conv_blocks[i] -> norm_expansion -> gamma;
		sizes[loc_ind] = expanded_depth;
		loc_ind++;
		locations[loc_ind] = conv_blocks[i] -> norm_expansion -> beta;
		sizes[loc_ind] = expanded_depth;
		loc_ind++;
		
		// if the block needed a projection to make input dim = output dim
		if (conv_blocks[i] -> projection){
			locations[loc_ind] = conv_blocks[i] -> projection;
			if (stride == 2){
				sizes[loc_ind] = 3 * 3 * incoming_filters * expanded_depth;
			}
			else{
				sizes[loc_ind] = incoming_filters * expanded_depth;
			}
			loc_ind++;
			locations[loc_ind] = conv_blocks[i] -> norm_projection -> gamma;
			sizes[loc_ind] = expanded_depth;
			loc_ind++;
			locations[loc_ind] = conv_blocks[i] -> norm_projection -> beta;
			sizes[loc_ind] = expanded_depth;
			loc_ind++;
		}

		// after stride 2 block then reduce spatial dim for next block
		if (is_block_spatial_reduction[i] == 1){
			incoming_spatial_dim /= 2;
		}
		incoming_filters = expanded_depth;
	}
	params -> conv_blocks = conv_blocks;

	float * fully_connected;
	// here expanded depth is the last layer's filters which will go through average pool before FC layer
	// expanded depth should equal dims -> final_depth
	int fully_connected_size = expanded_depth * output_dim;
	float fully_connected_fan_in = expanded_depth;
	cudaMalloc(&fully_connected, fully_connected_size * sizeof(float));
	cudaMemset(fully_connected, 0, fully_connected_size * sizeof(float));
	if (!is_zero){
		init_weights_gaussian_device(gen, fully_connected_size, fully_connected, 0, 0.0001);
	}

	params -> fully_connected = fully_connected;
	locations[loc_ind] = fully_connected;
	sizes[loc_ind] = expanded_depth * output_dim;

	params -> locations = locations;
	params -> sizes = sizes;

	return params;
}

ResNet * init_resnet(Dims * dims, curandGenerator_t * gen){
	ResNet * model = (ResNet *) malloc(sizeof(ResNet));
	model -> dims = dims;
	Params * params = init_model_parameters(dims, gen, false);
	model -> params = params;
	return model;
}


/* INITIALIZE TRAINING STRUCTURES */

Cache_BatchNorm * init_cache_batchnorm(int input_size, int feature_size){
	Cache_BatchNorm * cache_batchnorm = (Cache_BatchNorm *) malloc(sizeof(Cache_BatchNorm));

	cache_batchnorm -> input_size = input_size;
	cache_batchnorm -> feature_size = feature_size;

	float * means, *inv_vars;
	cudaMalloc(&means, feature_size * sizeof(float));
	cudaMemset(means, 0, feature_size * sizeof(float));
	cudaMalloc(&inv_vars, feature_size * sizeof(float));
	cudaMemset(inv_vars, 0, feature_size * sizeof(float));

	cache_batchnorm -> means = means;
	cache_batchnorm -> inv_vars = inv_vars;

	return cache_batchnorm;
}

Activation_ConvBlock * init_activation_convblock(ConvBlock * conv_block, int batch_size){
	Activation_ConvBlock * activation_conv_block = (Activation_ConvBlock *) malloc(sizeof(Activation_ConvBlock));

	int incoming_filters = conv_block -> incoming_filters;
	int incoming_spatial_dim = conv_block -> incoming_spatial_dim;
	int stride = conv_block -> stride;
	int reduced_depth = conv_block -> reduced_depth;
	int expanded_depth = conv_block -> expanded_depth;

	activation_conv_block -> incoming_filters = incoming_filters;
	activation_conv_block -> incoming_spatial_dim = incoming_spatial_dim;
	activation_conv_block -> reduced_depth = reduced_depth;
	activation_conv_block -> expanded_depth = expanded_depth;
	activation_conv_block -> stride = stride;

	float * post_reduced, *post_spatial, *post_expanded, *post_expanded_norm_vals, *transformed_residual, *post_projection_norm_vals, *output, *output_activated;
	float * post_reduced_activated, *post_spatial_activated;
	int post_reduced_size, post_spatial_size, output_size;
	Cache_BatchNorm * norm_post_reduced, *norm_post_spatial, *norm_post_expanded, *norm_post_projection;
	

	post_reduced_size = reduced_depth * incoming_spatial_dim * incoming_spatial_dim * batch_size;
	cudaMalloc(&post_reduced, post_reduced_size * sizeof(float));
	activation_conv_block -> post_reduced = post_reduced;

	norm_post_reduced = init_cache_batchnorm(post_reduced_size, reduced_depth);
	activation_conv_block -> norm_post_reduced = norm_post_reduced;

	cudaMalloc(&post_reduced_activated, post_reduced_size * sizeof(float));
	activation_conv_block -> post_reduced_activated = post_reduced_activated;

	post_spatial_size = reduced_depth * incoming_spatial_dim * incoming_spatial_dim / (stride * stride) * batch_size;
	cudaMalloc(&post_spatial, post_spatial_size * sizeof(float));
	activation_conv_block -> post_spatial = post_spatial;

	norm_post_spatial = init_cache_batchnorm(post_spatial_size, reduced_depth);
	activation_conv_block -> norm_post_spatial = norm_post_spatial;

	cudaMalloc(&post_spatial_activated, post_spatial_size * sizeof(float));
	activation_conv_block -> post_spatial_activated = post_spatial_activated;

	output_size = expanded_depth * incoming_spatial_dim * incoming_spatial_dim / (stride * stride) * batch_size;
	
	cudaMalloc(&post_expanded, output_size * sizeof(float));
	activation_conv_block -> post_expanded = post_expanded;

	norm_post_expanded = init_cache_batchnorm(output_size, expanded_depth);
	activation_conv_block -> norm_post_expanded = norm_post_expanded;

	cudaMalloc(&post_expanded_norm_vals, output_size * sizeof(float));
	activation_conv_block -> post_expanded_norm_vals = post_expanded_norm_vals;

	// only allocate space if transformed, otherwise it will be assumed to be identity of input
	transformed_residual = NULL;
	norm_post_projection = NULL;
	post_projection_norm_vals = NULL;
	if (incoming_filters != expanded_depth){
		cudaMalloc(&transformed_residual, output_size * sizeof(float));
		norm_post_projection = init_cache_batchnorm(output_size, expanded_depth);
		cudaMalloc(&post_projection_norm_vals, output_size * sizeof(float));
	}
	activation_conv_block -> transformed_residual = transformed_residual;
	activation_conv_block -> norm_post_projection = norm_post_projection;
	activation_conv_block -> post_projection_norm_vals = post_projection_norm_vals;

	cudaMalloc(&output, output_size * sizeof(float));
	activation_conv_block -> output = output;

	cudaMalloc(&output_activated, output_size * sizeof(float));
	activation_conv_block -> output_activated = output_activated;

	return activation_conv_block;
}

Activations * init_activations(Dims * dims, ConvBlock ** conv_blocks, int batch_size){
	
	Activations * activations = (Activations *) malloc(sizeof(Activations));

	int input_dim = dims -> input;
	int init_conv_filters = dims -> init_conv_filters;
	int init_conv_stride = dims -> init_conv_stride;
	int maxpool_stride = dims -> init_maxpool_stride;

	float * init_conv_applied;
	int init_conv_applied_size = init_conv_filters * input_dim * input_dim / (init_conv_stride * init_conv_stride) * batch_size; 
	cudaMalloc(&init_conv_applied, init_conv_applied_size * sizeof(float));
	activations -> init_conv_applied = init_conv_applied;

	Cache_BatchNorm * norm_init_conv = init_cache_batchnorm(init_conv_applied_size, init_conv_filters);
	activations -> norm_init_conv = norm_init_conv;

	float * init_conv_activated;
	cudaMalloc(&init_conv_activated, init_conv_applied_size * sizeof(float));
	activations -> init_conv_activated = init_conv_activated;

	int init_convblock_input_size = init_conv_filters * input_dim * input_dim / (init_conv_stride * init_conv_stride) / (maxpool_stride * maxpool_stride) * batch_size;

	int * max_inds;
	cudaMalloc(&max_inds, init_convblock_input_size * sizeof(int));
	activations -> max_inds = max_inds;

	float *init_convblock_input;
	cudaMalloc(&init_convblock_input, init_convblock_input_size * sizeof(float));
	activations -> init_convblock_input = init_convblock_input;

	int n_conv_blocks = dims -> n_conv_blocks;

	Activation_ConvBlock ** activation_conv_blocks = (Activation_ConvBlock **) malloc(n_conv_blocks * sizeof(Activation_ConvBlock *));
	for (int i = 0; i < n_conv_blocks; i++){
		ConvBlock * conv_block = conv_blocks[i];
		activation_conv_blocks[i] = init_activation_convblock(conv_block, batch_size);
	}

	activations -> activation_conv_blocks = activation_conv_blocks;
	activations -> n_conv_blocks = n_conv_blocks;

	int final_depth = dims -> final_depth;
	float * final_conv_output_pooled;
	int final_conv_output_pooled_size = final_depth * batch_size;
	cudaMalloc(&final_conv_output_pooled, final_conv_output_pooled_size * sizeof(float));
	activations -> final_conv_output_pooled = final_conv_output_pooled;

	int output_dim = dims -> output;
	int output_size = output_dim * batch_size;

	float * linear_output;
	cudaMalloc(&linear_output, output_size * sizeof(float));
	activations -> linear_output = linear_output;

	return activations;
}


Forward_Buffer * init_forward_buffer(Dims * dims, ConvBlock ** conv_blocks, int batch_size){

	Forward_Buffer * forward_buffer = (Forward_Buffer *) malloc(sizeof(Forward_Buffer));

	forward_buffer -> activations = init_activations(dims, conv_blocks, batch_size);

	int output_dim = dims -> output;
	int output_size = output_dim * batch_size;

	float * pred;
	cudaMalloc(&pred, output_size * batch_size * sizeof(float));
	forward_buffer -> pred = pred;

	// will be copied to cpu to be able to print values and compute loss on cpu side
	float * pred_cpu = (float *) malloc(output_size * batch_size * sizeof(float));
	forward_buffer -> pred_cpu = pred_cpu;

	return forward_buffer;
}


Backprop_Buffer * init_backprop_buffer(Dims * dims, ConvBlock ** conv_blocks, int batch_size){

	Backprop_Buffer * backprop_buffer = (Backprop_Buffer *) malloc(sizeof(Backprop_Buffer));

	int output_dim = dims -> output;
	int output_size = output_dim * batch_size;

	float * output_layer_deriv;
	cudaMalloc(&output_layer_deriv, output_size * sizeof(float));
	backprop_buffer -> output_layer_deriv = output_layer_deriv;

	backprop_buffer -> param_derivs = init_model_parameters(dims, NULL, true);
	backprop_buffer -> prev_means = init_model_parameters(dims, NULL, true);
	backprop_buffer -> prev_vars = init_model_parameters(dims, NULL, true);
	backprop_buffer -> activation_derivs = init_activations(dims, conv_blocks, batch_size);

	return backprop_buffer;
}


Train_ResNet * init_trainer(ResNet * model, Batch * cur_batch, int batch_size, float learning_rate, float weight_decay, float mean_decay, float var_decay, float eps, int n_epochs, cudnnHandle_t * handle, const char * dump_dir){
	Train_ResNet * trainer = (Train_ResNet *) malloc(sizeof(Train_ResNet));

	trainer -> model = model;

	trainer -> cur_batch = cur_batch;
	trainer -> batch_size = batch_size;

	Dims * dims = model -> dims;
	ConvBlock ** conv_blocks = model -> params -> conv_blocks;
	trainer -> forward_buffer = init_forward_buffer(dims, conv_blocks, batch_size);
	trainer -> backprop_buffer = init_backprop_buffer(dims, conv_blocks, batch_size);

	trainer -> learning_rate = learning_rate;
	trainer -> weight_decay = weight_decay;
	trainer -> base_mean_decay = mean_decay;
	trainer -> base_var_decay = var_decay;
	
	trainer -> cur_mean_decay = 1;
	trainer -> cur_var_decay = 1;
	
	trainer -> eps = eps;

	trainer -> n_epochs = n_epochs;

	trainer -> cur_dump_id = -1;

	trainer -> cur_epoch = 0;

	trainer -> loss_per_epoch = (float *) calloc(n_epochs, sizeof(float));
	trainer -> accuracy_per_epoch = (float *) calloc(n_epochs, sizeof(float));

	trainer -> init_loaded = 0;

	trainer -> cudnnHandle = *handle;

	trainer -> dump_dir = dump_dir;

	return trainer;
}

Batch * init_general_batch(int n_images, int image_size, int image_dim, int shard_n_images){
	Batch * batch = (Batch *) malloc(sizeof(Batch));

	batch -> n_images = n_images;
	// in resnet-50 will be 224 * 224 * 3
	batch -> image_size = image_size;
	batch -> image_dim = image_dim;
	float * images_float_cpu;
	// load batch by first brining into cpu, pinned memory
	cudaError_t status_images_pinned = cudaMallocHost((float **)&images_float_cpu, (size_t) n_images * (size_t) image_size * sizeof(float));
	batch -> images_float_cpu = images_float_cpu;
	
	// allocate memory on gpu so that after loaded on cpu can bring in
	// will be converting from uint8 on CPU to float on GPU
	float * images;
	cudaMalloc(&images, (size_t) n_images * (size_t) image_size * sizeof(float));
	batch -> images = images;

	// pinned memory for correct_classes_cpu
	int * correct_classes_cpu;
	cudaError_t status_classes_pinned = cudaMallocHost((int **)&correct_classes_cpu, n_images * sizeof(int));
	batch -> correct_classes_cpu = correct_classes_cpu;

	int * correct_classes;
	cudaMalloc(&correct_classes, n_images * sizeof(int));
	batch -> correct_classes = correct_classes;

	batch -> cur_shard_id = -1;
	batch -> cur_batch_in_shard = -1;
	
	batch -> shard_n_images = shard_n_images;
	batch -> full_shard_images = (float *) malloc((size_t) shard_n_images * (size_t) image_size * sizeof(float));
	batch -> full_shard_correct_classes = (int *) malloc(shard_n_images * sizeof(int));

	return batch;
}

// (if this takes too long, can do it in parallel with separate process on cpu)
// ASSUMING shard_n_images % batch_size = 0
void load_new_batch(Train_ResNet * trainer, Class_Metadata * class_metadata, Batch * batch_buffer){
	
	int batch_size = batch_buffer -> n_images;
	int image_size = batch_buffer -> image_size;
	size_t total_pixels = (size_t) batch_size * (size_t) image_size;
	
	float * full_shard_images = batch_buffer -> full_shard_images;
	int * full_shard_correct_classes = batch_buffer -> full_shard_correct_classes;	

	float * images_float_cpu = batch_buffer -> images_float_cpu;
	float * images = batch_buffer -> images;

	int * correct_classes_cpu = batch_buffer -> correct_classes_cpu;
	int * correct_classes = batch_buffer -> correct_classes;

	int cur_shard_id = batch_buffer -> cur_shard_id;
	int cur_batch_in_shard = batch_buffer -> cur_batch_in_shard;
	int shard_n_images = batch_buffer -> shard_n_images;

	int cur_dump_id = trainer -> cur_dump_id;

	int init_loaded = trainer -> init_loaded;



	int start_img_num = cur_batch_in_shard * batch_size;
	int n_read;
	int print_ret;

	char * shard_images_filepath, * shard_labels_filepath;
	// cur_shard_id = -1 implies first iteration
	if ((init_loaded) || (cur_shard_id == -1) || (start_img_num >= shard_n_images)) {

		// update new shard id if first iter or passed the bounds
		if (! init_loaded){
			cur_shard_id += 1;
			batch_buffer -> cur_shard_id = cur_shard_id;
		}

		// load new shard into RAM
		print_ret = asprintf(&shard_images_filepath, "/mnt/storage/data/vision/imagenet/2012/train_data_shards/%03d.images", cur_shard_id);
		FILE * shard_images_file = fopen(shard_images_filepath, "rb");
		n_read = fread(full_shard_images, sizeof(float), ((size_t) shard_n_images) * ((size_t) image_size), shard_images_file);
		fclose(shard_images_file);
		free(shard_images_filepath);

		print_ret = asprintf(&shard_labels_filepath, "/mnt/storage/data/vision/imagenet/2012/train_data_shards/%03d.labels", cur_shard_id);
		FILE * shard_labels_file = fopen(shard_labels_filepath, "rb");
		n_read = fread(full_shard_correct_classes, sizeof(int), shard_n_images, shard_labels_file);
		fclose(shard_labels_file);
		free(shard_labels_filepath);

		// reset cur batch in shard to 0 if first iter or passed the bounds
		if (! init_loaded) {
			cur_batch_in_shard = 0;
			batch_buffer -> cur_batch_in_shard = cur_batch_in_shard;
		}

		// don't have to load special first batch from checkpoint anymore
		trainer -> init_loaded = 0;
	}

	// load current batch
	memcpy(images_float_cpu, full_shard_images + cur_batch_in_shard * total_pixels, total_pixels * sizeof(float));
	memcpy(correct_classes_cpu, full_shard_correct_classes + cur_batch_in_shard * batch_size, batch_size * sizeof(int));
	
	/* SAVING BATCH TO FILES FOR INSPECTION... */
	// if (cur_batch_in_shard == 0){
	// 	FILE * test_images_file = fopen("images.buffer", "wb");
	// 	fwrite(images_float_cpu, sizeof(float), total_pixels, test_images_file);
	// 	fclose(test_images_file);

	// 	FILE * test_labels_file = fopen("labels.buffer", "wb");
	// 	fwrite(correct_classes_cpu, sizeof(int), (size_t) batch_size, test_labels_file);
	// 	fclose(test_labels_file);
	// 	exit(0);
	// }

	// copy current batch to GPU

	cudaMemcpy(images, images_float_cpu, total_pixels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(correct_classes, correct_classes_cpu, batch_size * sizeof(int), cudaMemcpyHostToDevice);

	// update cur batch for next iteration of loading
	cur_batch_in_shard++;
	batch_buffer -> cur_batch_in_shard = cur_batch_in_shard;

	cur_dump_id++;
	trainer -> cur_dump_id = cur_dump_id;

}


// READ CLASSES AND LABELS!
// reading a text file line by line into a buffer
// pre-allocate buffer and specify type
void text_file_to_buffer(void * buffer, char * filename, const char * type){

	char ** my_text_buffer = (char **) buffer;
	int * my_int_buffer = (int *) buffer;
	
	FILE * fp;
    char * line = NULL;
    size_t len = 0;

    fp = fopen(filename, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    int cnt = 0;
    while (getline(&line, &len, fp) != -1) {
    	if (strcmp(type, "TEXT") == 0){
        	my_text_buffer[cnt] = strdup(line);
        }
        else if (strcmp(type, "INT") == 0){
        	my_int_buffer[cnt] = atoi(line);
        }
        else{
        	// pass
        }
        cnt++;
    }

    fclose(fp);
    if (line){
    	free(line);
    }
}

Class_Metadata * populate_class_info(char * label_filename, char * synset_filename, char * class_size_filename, int n_classes){
	
	Class_Metadata * classes = (Class_Metadata *) malloc(sizeof(Class_Metadata));

	char ** labels = (char **) malloc(n_classes * sizeof(char *));
	char ** synsets = (char **) malloc(n_classes * sizeof(char *));
	int * counts = (int *) malloc(n_classes * sizeof(int));

	text_file_to_buffer(labels, label_filename, "TEXT");
	text_file_to_buffer(synsets, synset_filename, "TEXT");
	text_file_to_buffer(counts, class_size_filename, "INT");

	classes -> labels = labels;
	classes -> synsets = synsets;
	classes -> counts = counts;
	classes -> n_classes = n_classes;

	return classes;
}

/* PREP AND LAUNCHING CUDA KERNELS! */

void prepareAndDoConvolutionScratch(int in_spatial_dim, int kern_dim, int in_filters, int out_filters,  int stride, int batch_size, 
																float * input, float * weights, float * output){
	int out_spatial_dim = in_spatial_dim / stride;
	int out_filters_block = min(MAX_THREAD_PER_BLOCK / batch_size, out_filters);
	int out_filters_grid = max(1, (int) ceil((float) out_filters / (float) out_filters_block));

	dim3 gridDimConv(out_spatial_dim, out_spatial_dim, out_filters_grid);
	dim3 blockDimConv(batch_size, out_filters_block);

	printf("Grid: (%d, %d, %d)\nBlock: (%d, %d)\n", out_spatial_dim, out_spatial_dim, out_filters_grid, batch_size, out_filters_block);
	doConvolution <<< gridDimConv, blockDimConv>>> (input, weights, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, output);
}


void prepareAndDoConvolution(Train_ResNet * trainer, int in_spatial_dim, int kern_dim, int in_filters, int out_filters,  int stride, int batch_size, 
																float * input, float * weights, float * output){
	cudnnStatus_t status;

	cudnnTensorDescriptor_t input_descriptor;
	status = cudnnCreateTensorDescriptor(&input_descriptor);
	status = cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, batch_size, in_filters, in_spatial_dim, in_spatial_dim);

	cudnnFilterDescriptor_t kernel_descriptor;
	status = cudnnCreateFilterDescriptor(&kernel_descriptor);
	status = cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_filters, in_filters, kern_dim, kern_dim);

	cudnnConvolutionDescriptor_t convolution_descriptor;
	status = cudnnCreateConvolutionDescriptor(&convolution_descriptor);
	status = cudnnSetConvolution2dDescriptor(convolution_descriptor, kern_dim / 2, kern_dim / 2, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

	int out_spatial_dim = in_spatial_dim / stride;

	cudnnTensorDescriptor_t output_descriptor;
	status = cudnnCreateTensorDescriptor(&output_descriptor);
	status = cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, batch_size, out_filters, out_spatial_dim, out_spatial_dim);

	//deprecated as of cuDNN 8
	// cudnnGetConvolutionForwardAlgorithm(trainer -> cudnnHandle, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convolution_algorithm);

	int returned_cnt;
	//cudnnConvolutionFwdAlgoPerf_t top_algo[1];
	//status = cudnnGetConvolutionForwardAlgorithm_v7(trainer -> cudnnHandle, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, 1, &returned_cnt, top_algo);
	cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;

	// const algo_t algos[] = {
    //       CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    //       CUDNN_CONVOLUTION_FWD_ALGO_FFT,
    //       CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
    //       CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    //       CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
    //       CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
    //       CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
    //       CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    //  };

	size_t workspace_bytes = 0;
	status = cudnnGetConvolutionForwardWorkspaceSize(trainer -> cudnnHandle, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, convolution_algorithm, &workspace_bytes);

	void * workspace;
	cudaMalloc(&workspace, workspace_bytes);

	const float alpha = 1, beta = 0;
	status = cudnnConvolutionForward(trainer -> cudnnHandle, &alpha, input_descriptor, input, kernel_descriptor, weights, convolution_descriptor, convolution_algorithm, workspace, workspace_bytes, &beta, output_descriptor, output);

	cudaFree(workspace);
	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);
}

void prepreAndDoConvolutionDeriv(Train_ResNet * trainer, int in_spatial_dim, int kern_dim, int in_filters, int out_filters, int stride, int batch_size, bool toAdd,
												float * input, float * weights, float * out_deriv,
												float * input_deriv, float * weight_deriv, bool toComputeInputDeriv){

	int out_spatial_dim = in_spatial_dim / stride;

	cudnnStatus_t status;

	cudnnTensorDescriptor_t input_descriptor;
	cudnnCreateTensorDescriptor(&input_descriptor);
	cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, batch_size, in_filters, in_spatial_dim, in_spatial_dim);

	cudnnTensorDescriptor_t output_descriptor;
	cudnnCreateTensorDescriptor(&output_descriptor);
	cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, batch_size, out_filters, out_spatial_dim, out_spatial_dim);

	cudnnFilterDescriptor_t kernel_descriptor_nchw;
	cudnnCreateFilterDescriptor(&kernel_descriptor_nchw);
	cudnnSetFilter4dDescriptor(kernel_descriptor_nchw, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_filters, in_filters, kern_dim, kern_dim);

	cudnnFilterDescriptor_t kernel_descriptor_nhwc;
	cudnnCreateFilterDescriptor(&kernel_descriptor_nhwc);
	cudnnSetFilter4dDescriptor(kernel_descriptor_nhwc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, out_filters, in_filters, kern_dim, kern_dim);

	// used to convert kernel weights to same format as tensors (needed for cudnn conv functions)
	cudnnTensorTransformDescriptor_t transform_descriptor;
	cudnnCreateTensorTransformDescriptor(&transform_descriptor);
	cudnnSetTensorTransformDescriptor(transform_descriptor, 4, CUDNN_TENSOR_NHWC, NULL, NULL, NULL, CUDNN_TRANSFORM_UNFOLD);

	cudnnConvolutionDescriptor_t convolution_descriptor;
	cudnnCreateConvolutionDescriptor(&convolution_descriptor);
	cudnnSetConvolution2dDescriptor(convolution_descriptor, kern_dim / 2, kern_dim / 2, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

	const float a_dummy = 1, b_dummy = 0;

	float alpha = 1, beta = 0;

	int returned_cnt;

	size_t workspace_bytes = 0;

	// Compute deriv w.r.t input data
	if (toComputeInputDeriv){

		 // static const algo_t algos[] = {
         // CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
         // CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
         // CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
         // CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
         // CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
         // CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
     	 // };

		//cudnnConvolutionBwdDataAlgoPerf_t top_data_algo[1];
		//cudnnGetConvolutionBackwardDataAlgorithm_v7(trainer -> cudnnHandle, kernel_descriptor, output_descriptor, convolution_descriptor, input_descriptor, 1, &returned_cnt, top_data_algo);
		cudnnConvolutionBwdDataAlgo_t convolution_data_algorithm = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
		
		status = cudnnGetConvolutionBackwardDataWorkspaceSize(trainer -> cudnnHandle, kernel_descriptor_nhwc, output_descriptor, convolution_descriptor, input_descriptor, convolution_data_algorithm, &workspace_bytes);
		// printf("Back Data Workspace Bytes Status: %s\n", cudnnGetErrorString(status));

		void * workspace_data;
		cudaMalloc(&workspace_data, workspace_bytes);


		float * weights_trans;
		cudaMalloc(&weights_trans, in_filters * out_filters * kern_dim * kern_dim * sizeof(float));

		cudnnTransformFilter(trainer -> cudnnHandle, transform_descriptor, &a_dummy, kernel_descriptor_nchw, weights, &b_dummy, kernel_descriptor_nhwc, weights_trans);


		if (toAdd){
			beta = 1;
		}

		status = cudnnConvolutionBackwardData(trainer -> cudnnHandle, &alpha, kernel_descriptor_nhwc, weights_trans, output_descriptor, out_deriv, convolution_descriptor, convolution_data_algorithm, 
										workspace_data, workspace_bytes, &beta, input_descriptor, input_deriv);
		// printf("Back Data Algo Status: %s\n", cudnnGetErrorString(status));



		// float * inp_deriv_cpu = (float *) malloc(in_filters * batch_size * in_spatial_dim * in_spatial_dim * sizeof(float));
		// cudaMemcpy(inp_deriv_cpu, input_deriv, in_filters * batch_size * in_spatial_dim * in_spatial_dim * sizeof(float), cudaMemcpyDeviceToHost);

		// int all_zero = 1;
		// for (size_t i = 0; i < in_filters * out_filters * kern_dim * kern_dim; i++){
		// 	if (inp_deriv_cpu[i] != 0){
		// 		all_zero = 0;
		// 		break;
		// 	}
		// }

		// printf("All Zero Inp Deriv?: %d\n", all_zero);
		
		cudaFree(workspace_data);
		cudaFree(weights_trans);
		workspace_bytes = 0;
	}

	// static const algo_t algos[] = {
    //      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
    //      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
    //      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
    //      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
    //      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
    //      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
    //  };


	// Compute deriv w.r.t filter weights
	//cudnnConvolutionBwdFilterAlgoPerf_t top_filter_algo[1];
	//cudnnGetConvolutionBackwardFilterAlgorithm_v7(trainer -> cudnnHandle, input_descriptor, output_descriptor, convolution_descriptor, kernel_descriptor, 1, &returned_cnt, top_filter_algo);
	cudnnConvolutionBwdFilterAlgo_t convolution_filter_algorithm = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

	cudnnStatus_t status_bytes = cudnnGetConvolutionBackwardFilterWorkspaceSize(trainer -> cudnnHandle, input_descriptor, output_descriptor, convolution_descriptor, kernel_descriptor_nhwc, convolution_filter_algorithm, &workspace_bytes);


	void * workspace_filter;
	cudaMalloc(&workspace_filter, workspace_bytes);

	// printf("Filter Workspace Bytes: %zu\n", workspace_bytes);
	// printf("Workspace Bytes Status: %s\n", cudnnGetErrorString(status_bytes));

	beta = 0;

	float * weight_deriv_temp;
	cudaMalloc(&weight_deriv_temp, out_filters * in_filters * kern_dim * kern_dim * sizeof(float));

	status = cudnnConvolutionBackwardFilter(trainer -> cudnnHandle, &alpha, input_descriptor, input, output_descriptor, out_deriv, convolution_descriptor, convolution_filter_algorithm, 
									workspace_filter, workspace_bytes, &beta, kernel_descriptor_nhwc, weight_deriv_temp);
	// printf("Back Filt Algo Status: %s\n", cudnnGetErrorString(status));
	

	cudnnTransformFilter(trainer -> cudnnHandle, transform_descriptor, &a_dummy, kernel_descriptor_nhwc, weight_deriv_temp, &b_dummy, kernel_descriptor_nchw, weight_deriv);
	
	cudaFree(weight_deriv_temp);
	
	// float * w_deriv_cpu = (float *) malloc(in_filters * out_filters * kern_dim * kern_dim * sizeof(float));
	// cudaMemcpy(w_deriv_cpu, weight_deriv, in_filters * out_filters * kern_dim * kern_dim * sizeof(float), cudaMemcpyDeviceToHost);

	// int all_zero_w = 1;
	// for (size_t i = 0; i < in_filters * out_filters * kern_dim * kern_dim; i++){
	// 	if (w_deriv_cpu[i] != 0){
	// 		all_zero_w = 0;
	// 		break;
	// 	}
	// }

	// printf("All Zero Weight Deriv?: %d\n", all_zero_w);

	// status = cudaGetLastError();
	// printf("Status after backward conv weight deriv: %s\n\n", cudaGetErrorString(status));

	cudaFree(workspace_filter);
	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyFilterDescriptor(kernel_descriptor_nchw);
	cudnnDestroyFilterDescriptor(kernel_descriptor_nhwc);
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);
	cudnnDestroyTensorTransformDescriptor(transform_descriptor);
}


void prepreAndDoConvolutionDerivScratch(int in_spatial_dim, int kern_dim, int in_filters, int out_filters, int stride, int batch_size, bool toAdd,
												float * input, float * weights, float * out_deriv,
												float * input_deriv, float * weight_deriv, bool toComputeInputDeriv){
	
	// first layer conv doesn't take deriv w.r.t input
	int in_filters_block = min(MAX_THREAD_PER_BLOCK / batch_size, in_filters);
	int in_filters_grid = max(1, (int) ceil((float) in_filters / (float) in_filters_block));

	dim3 gridDimDerivInput(in_spatial_dim, in_spatial_dim, in_filters_grid);
	dim3 blockDimDerivInput(batch_size, in_filters_block);
	if (toComputeInputDeriv){
		convolutionDerivInput <<< gridDimDerivInput, blockDimDerivInput >>> (input, weights, out_deriv, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, toAdd, input_deriv);
	}

	int block_dim, grid_dim;
	bool is_block_dim_inp;
	if (in_filters > MAX_THREAD_PER_BLOCK){
		block_dim = out_filters;
		grid_dim = in_filters;
		is_block_dim_inp = false;
	}
	else{
		block_dim = in_filters;
		grid_dim = out_filters;
		is_block_dim_inp = true;
	}
	
	dim3 gridDimDerivWeights(kern_dim, kern_dim, grid_dim);
	dim3 blockDimDerivWeights(block_dim);
	convolutionDerivWeights <<< gridDimDerivWeights, blockDimDerivWeights >>> (input, weights, out_deriv, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, weight_deriv, is_block_dim_inp);
}

void prepareAndDoBatchNormAndActivate(Train_ResNet * trainer, BatchNorm * batch_norm_params, Cache_BatchNorm * batch_norm_cache, int batch_size, float eps, float * input, float * output, bool to_activate){
	// reading values from batch norm params
	int filters = batch_norm_params -> depth;
	int spatial_dim = batch_norm_params -> spatial_dim;
	float * gamma = batch_norm_params -> gamma;
	float * beta = batch_norm_params -> beta;

	// read the output device pointers from batch_norm_cache
	float * means_out = batch_norm_cache -> means;
	float * inv_vars_out = batch_norm_cache -> inv_vars;

	const float alpha_dummy = 1, beta_dummy = 0;

	cudnnTensorDescriptor_t input_descriptor;
	cudnnCreateTensorDescriptor(&input_descriptor);
	cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, batch_size, filters, spatial_dim, spatial_dim);

	cudnnTensorDescriptor_t bn_descriptor;
	cudnnCreateTensorDescriptor(&bn_descriptor);

	cudnnBatchNormMode_t bn_mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

	cudnnDeriveBNTensorDescriptor(bn_descriptor, input_descriptor, bn_mode);

	cudnnBatchNormalizationForwardTraining(trainer -> cudnnHandle, bn_mode, &alpha_dummy, &beta_dummy, input_descriptor, input, input_descriptor, output, bn_descriptor, gamma, beta, 1, NULL, NULL, trainer -> eps, means_out, inv_vars_out);

	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(bn_descriptor);

	if (to_activate){

		size_t bn_output_size = batch_size * filters * spatial_dim * spatial_dim;

		dim3 gridDimBN(ceil((float) (bn_output_size) / MAX_THREAD_PER_BLOCK));
		dim3 blockDimBN(MAX_THREAD_PER_BLOCK);

		doActivation <<< gridDimBN, blockDimBN >>> (bn_output_size, output, output);

	}

}

void prepareAndDoActivationAndBatchNormDeriv(Train_ResNet * trainer, BatchNorm * batch_norm_params, Cache_BatchNorm * batch_norm_cache, BatchNorm * batch_norm_param_derivs, Cache_BatchNorm * batch_norm_cache_derivs, 
																								int batch_size, float eps, float * input, float * activated, float * out_layer_deriv, float * input_deriv, bool to_activate_deriv){
	int filters = batch_norm_params -> depth;
	int spatial_dim = batch_norm_params -> spatial_dim;
	float * gamma = batch_norm_params -> gamma;
	float * beta = batch_norm_params -> beta;
	float * means = batch_norm_cache -> means;
	float * inv_vars = batch_norm_cache -> inv_vars;

	float * gamma_deriv = batch_norm_param_derivs -> gamma;
	float * beta_deriv = batch_norm_param_derivs -> beta;

	const float alpha_data = 1, beta_data = 0, alpha_param = 1, beta_param = 0;


	if (to_activate_deriv){
		size_t bn_output_size = batch_size * filters * spatial_dim * spatial_dim;

		dim3 gridDimBN(ceil((float) (bn_output_size) / MAX_THREAD_PER_BLOCK));
		dim3 blockDimBN(MAX_THREAD_PER_BLOCK);

		doActivationDeriv <<< gridDimBN, blockDimBN >>> (bn_output_size, activated, out_layer_deriv, out_layer_deriv);
	}

	cudnnTensorDescriptor_t layer_descriptor;
	cudnnCreateTensorDescriptor(&layer_descriptor);
	cudnnSetTensor4dDescriptor(layer_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, batch_size, filters, spatial_dim, spatial_dim);

	cudnnTensorDescriptor_t bn_descriptor;
	cudnnCreateTensorDescriptor(&bn_descriptor);

	cudnnBatchNormMode_t bn_mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

	cudnnDeriveBNTensorDescriptor(bn_descriptor, layer_descriptor, bn_mode);


	cudnnBatchNormalizationBackward(trainer -> cudnnHandle, bn_mode, &alpha_data, &beta_data, &alpha_param, &beta_param, 
											layer_descriptor, input, layer_descriptor, out_layer_deriv, layer_descriptor, input_deriv,
											bn_descriptor, gamma, gamma_deriv, beta_deriv, eps, means, inv_vars);

	cudnnDestroyTensorDescriptor(layer_descriptor);
	cudnnDestroyTensorDescriptor(bn_descriptor);
}

// void prepareAndDoActivationAndBatchNormDerivScratch(BatchNorm * batch_norm_params, Cache_BatchNorm * batch_norm_cache, BatchNorm * batch_norm_param_derivs, Cache_BatchNorm * batch_norm_cache_derivs, 
// 																								int batch_size, float eps, float * input, float * activated, float * out_layer_deriv, float * input_deriv, bool to_activate_deriv){
// 	int filters = batch_norm_params -> depth;
// 	int spatial_dim = batch_norm_params -> spatial_dim;
// 	float * gamma = batch_norm_params -> gamma;
// 	float * beta = batch_norm_params -> beta;
// 	float * means = batch_norm_cache -> means;
// 	float * vars = batch_norm_cache -> vars;
// 	float * normalized_temp = batch_norm_cache -> normalized_temp;

// 	float * normalized_temp_deriv = batch_norm_cache_derivs -> normalized_temp;
// 	float * gamma_deriv = batch_norm_param_derivs -> gamma;
// 	float * beta_deriv = batch_norm_param_derivs -> beta;

// 	int num_threads = min(MAX_THREAD_PER_BLOCK_INCL_REG, filters);
// 	int num_blocks = 1;
// 	if (filters > num_threads){
// 		num_blocks = ceil((float) filters / (float) MAX_THREAD_PER_BLOCK_INCL_REG);
// 	}

// 	dim3 gridDimBatchNormDeriv(num_blocks);
// 	dim3 blockDimBatchNormDeriv(num_threads);
// 	activationAndBatchNormDeriv <<< gridDimBatchNormDeriv, blockDimBatchNormDeriv >>> (input, gamma, beta, spatial_dim, filters, batch_size, eps, means, vars, normalized_temp, activated, out_layer_deriv, normalized_temp_deriv, gamma_deriv, beta_deriv, input_deriv, to_activate_deriv);
// }

void prepareAndDoMatMulLeftTranspose(const float * left_orig, const float * right, int left_orig_rows, int left_orig_cols, int right_rows, int right_cols, float * out){
	float * temp_left;
	cudaMalloc(&temp_left, left_orig_rows * left_orig_cols * sizeof(float));

	dim3 gridDimTranspose(ceil((float) left_orig_rows / TILE_WIDTH), ceil((float)left_orig_cols / TILE_WIDTH));
	dim3 blockDimTranspose(TILE_WIDTH, TILE_WIDTH);
	transpose <<< gridDimTranspose, blockDimTranspose >>> (left_orig, left_orig_rows, left_orig_cols, temp_left);

	dim3 gridDimMatMul(ceil((float) left_orig_cols / TILE_WIDTH), ceil((float) right_cols / TILE_WIDTH));
	dim3 blockDimMatMul(TILE_WIDTH, TILE_WIDTH);
	matMul <<< gridDimMatMul, blockDimMatMul >>> (temp_left, right, left_orig_cols, right_rows, right_cols, out);
	cudaFree(temp_left);
}

void prepareAndDoMatMulRightTranspose(const float * left, const float * right_orig, int left_rows, int left_cols, int right_orig_rows, int right_orig_cols, float * out){
	float * temp_right;
	cudaMalloc(&temp_right, right_orig_rows * right_orig_cols * sizeof(float));
	
	dim3 gridDimTranspose(ceil((float) right_orig_rows / TILE_WIDTH), ceil((float)right_orig_cols / TILE_WIDTH));
	dim3 blockDimTranspose(TILE_WIDTH, TILE_WIDTH);

	transpose <<< gridDimTranspose, blockDimTranspose >>> (right_orig, right_orig_rows, right_orig_cols, temp_right);
	
	dim3 gridDimMatMul(ceil((float) left_rows / TILE_WIDTH), ceil((float) right_orig_rows / TILE_WIDTH));
	dim3 blockDimMatMul(TILE_WIDTH, TILE_WIDTH);
	matMul <<< gridDimMatMul, blockDimMatMul >>> (left, temp_right, left_rows, left_cols, right_orig_rows, out);
	cudaFree(temp_right);
}

void printDeviceData(const char * name_of_variable, float * device_variable, int size){
	bool print = TO_PRINT;
	if (print){
		float * cpu_data = (float *) malloc(size * sizeof(float));
		cudaMemcpy(cpu_data, device_variable, size * sizeof(float), cudaMemcpyDeviceToHost);
		printf("VARIABLE NAME: %s\n\n", name_of_variable);
		printf("DATA:\n");
		for (int i = 0; i < size; i++){
			printf("%d: %f\n", i, cpu_data[i]);
		}
		printf("\n\n\n");
		free(cpu_data);
	}
}

void forward_pass(Train_ResNet * trainer){

	Dims * dims = trainer -> model -> dims;

	float eps = trainer -> eps;
	int batch_size = trainer -> batch_size;

	float * input = trainer -> cur_batch -> images;
	float * first_conv = trainer -> model -> params -> init_conv_layer;
	float * first_conv_output = trainer -> forward_buffer -> activations -> init_conv_applied;
	// first apply the convolutions
	// launch grid dimensions as (OUT_SPATIAL_DIM, OUT_SPATIAL_DIM, OUT_FILTER_CHUNK) blocks, and launch with block dim as (out_filt_rows_shared, sub_batch) threads
	
	// 3 colors
	int init_in_filters = 3;
	int init_spatial_dim = dims -> input;
	int init_kernel_dim = dims -> init_kernel_dim;
	int init_out_filters = dims -> init_conv_filters;
	int init_stride = dims -> init_conv_stride;
	int init_out_spatial_dim = init_spatial_dim / init_stride;

	prepareAndDoConvolution(trainer, init_spatial_dim, init_kernel_dim, init_in_filters, init_out_filters, init_stride, batch_size, input, first_conv, first_conv_output);

	int print_size = 10;
	printDeviceData("INIT CONV APPLIED", first_conv_output, print_size);

	BatchNorm * norm_init_conv_params = trainer -> model -> params -> norm_init_conv;
	Cache_BatchNorm * norm_init_conv_cache = trainer -> forward_buffer -> activations -> norm_init_conv;
	float * init_activated = trainer -> forward_buffer -> activations -> init_conv_activated;

	prepareAndDoBatchNormAndActivate(trainer, norm_init_conv_params, norm_init_conv_cache, batch_size, eps, first_conv_output, init_activated, true);

	printDeviceData("INIT CONV ACTIVATED", init_activated, print_size);

	int init_maxpool_dim = dims -> init_maxpool_dim;
	int init_maxpool_stride = dims -> init_maxpool_stride;
	int init_maxpool_out_dim = init_out_spatial_dim / init_maxpool_stride;
	float * init_convblock_input = trainer -> forward_buffer -> activations -> init_convblock_input;
	int * max_ind_buff = trainer -> forward_buffer -> activations -> max_inds;

	dim3 gridDimMaxPool(init_maxpool_out_dim, init_maxpool_out_dim);
	dim3 blockDimMaxPool(init_out_filters);
	doMaxPool <<< gridDimMaxPool , blockDimMaxPool >>> (init_activated, init_maxpool_dim, init_maxpool_stride, batch_size, max_ind_buff, init_convblock_input);

	printDeviceData("MAX POOL OUTPUT", init_convblock_input, print_size);

	/* NOW CAN MOVE ONTO TO CONV_BLOCK LAYERS! */

	int n_conv_blocks = dims -> n_conv_blocks;

	
	ConvBlock ** params_conv_blocks = trainer -> model -> params -> conv_blocks;
	Activation_ConvBlock ** activation_conv_blocks = trainer -> forward_buffer -> activations -> activation_conv_blocks;
	ConvBlock * cur_conv_block_params;
	Activation_ConvBlock * cur_conv_block_activation;
	int in_spatial_dim, kern_dim, in_filters, out_filters, stride, out_spatial_dim, total_size_conv_block_output;

	float * conv_block_input = init_convblock_input;
	float *conv_input, * conv_weights, * conv_output, *norm_input, * norm_output, * conv_block_output, * conv_block_output_activated;
	float *projection_weights, *transformed_residual, *post_projection_norm_vals;
	BatchNorm * cur_batch_norm_params;
	Cache_BatchNorm * cur_batch_norm_cache;
	for (int i = 0; i < n_conv_blocks; i++){
		cur_conv_block_params = params_conv_blocks[i];
		cur_conv_block_activation = activation_conv_blocks[i];

		// do first 1x1 depth_reduce convolution
		in_spatial_dim = cur_conv_block_params -> incoming_spatial_dim;
		in_filters = cur_conv_block_params -> incoming_filters;
		out_filters = cur_conv_block_params -> reduced_depth;
		kern_dim = 1;
		stride = 1;
		// either intialized first time above loop from the maxpool
		// every other block will be the normalized, activated output of previous conv block (previous iteration output) 
		conv_input = conv_block_input;
		conv_weights = cur_conv_block_params -> depth_reduction;
		conv_output = cur_conv_block_activation -> post_reduced;

		prepareAndDoConvolution(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, conv_input, conv_weights, conv_output);

		printDeviceData("REDUCED CONV APPLIED", conv_output, print_size);

		norm_input = conv_output;
		cur_batch_norm_params = cur_conv_block_params -> norm_depth_reduction;
		cur_batch_norm_cache = cur_conv_block_activation -> norm_post_reduced;
		norm_output = cur_conv_block_activation -> post_reduced_activated;

		prepareAndDoBatchNormAndActivate(trainer, cur_batch_norm_params, cur_batch_norm_cache, batch_size, eps, norm_input, norm_output, true);

		printDeviceData("REDUCED CONV NORM & ACTIVATED", norm_output, print_size);

		// do 3x3 spatial convolution

		// same as in first conv
		in_spatial_dim = cur_conv_block_params -> incoming_spatial_dim;
		// now is output filters of 1st conv, which is reduced depth filters
		in_filters = cur_conv_block_params -> reduced_depth;
		// keeps depth the same, just spatial conv
		out_filters = cur_conv_block_params -> reduced_depth;
		kern_dim = 3;
		// if stride is occurring in conv block happens at this kernel
		stride = cur_conv_block_params -> stride;
		conv_input = norm_output;
		conv_weights = cur_conv_block_params -> spatial;;
		conv_output = cur_conv_block_activation -> post_spatial;

		prepareAndDoConvolution(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, conv_input, conv_weights, conv_output);

		printDeviceData("SPATIAL CONV APPLIED", conv_output, print_size);

		norm_input = conv_output;
		cur_batch_norm_params = cur_conv_block_params -> norm_spatial;
		cur_batch_norm_cache = cur_conv_block_activation -> norm_post_spatial;
		norm_output = cur_conv_block_activation -> post_spatial_activated;

		prepareAndDoBatchNormAndActivate(trainer, cur_batch_norm_params, cur_batch_norm_cache, batch_size, eps, norm_input, norm_output, true);

		printDeviceData("SPATIAL CONV NORM & ACTIVATED", norm_output, print_size);

		// do 1x1 depth expansion convolution

		// if stride happened now would need to take that into account
		in_spatial_dim = (cur_conv_block_params -> incoming_spatial_dim) / (cur_conv_block_params -> stride);
		// prev 3x3 conv kept out filters as reduced depth
		in_filters = cur_conv_block_params -> reduced_depth;
		// now creating expanded depth out filters
		out_filters = cur_conv_block_params -> expanded_depth;
		kern_dim = 1;
		stride = 1;
		conv_input = norm_output;
		conv_weights = cur_conv_block_params -> depth_expansion;
		conv_output = cur_conv_block_activation -> post_expanded;

		prepareAndDoConvolution(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, conv_input, conv_weights, conv_output);

		printDeviceData("EXPANDED CONV APPLIED", conv_output, print_size);

		norm_input = conv_output;
		cur_batch_norm_params = cur_conv_block_params -> norm_expansion;
		cur_batch_norm_cache = cur_conv_block_activation -> norm_post_expanded;
		norm_output = cur_conv_block_activation -> post_expanded_norm_vals;

		// do not activate because first need to add to (projection) residual
		prepareAndDoBatchNormAndActivate(trainer, cur_batch_norm_params, cur_batch_norm_cache, batch_size, eps, norm_input, norm_output, false);

		printDeviceData("EXPANDED NORM & ACTIVATED", norm_output, print_size);

		// now need to add identity of conv_block_input (if same dimensions), or project=convolve (different dimensions) and add to conv_output
		// projection is a incoming block filters X expanded depth matrix
		// if stride of 2 in additon to depth change, then 3x3 kernel with stride 2 applied to block input
		// works as a depth-wise 1x1 convolution where in_filters = incoming_filters and out_filters = expanded_depth

		// already updated
		in_spatial_dim = (cur_conv_block_params -> incoming_spatial_dim);
		out_spatial_dim = (cur_conv_block_params -> incoming_spatial_dim) / (cur_conv_block_params -> stride);
		in_filters = cur_conv_block_params -> incoming_filters;
		out_filters = cur_conv_block_params -> expanded_depth;
		stride = cur_conv_block_params -> stride;
		if (stride == 2){
			kern_dim = 3;
		}
		else{
			kern_dim = 1;
		}
		projection_weights = cur_conv_block_params -> projection;

		total_size_conv_block_output = out_spatial_dim * out_spatial_dim * out_filters * batch_size;
		
				
		// the conv_block initializer already handled if we need projection, and if so it allocated weights
		// if there is a projection needed we will do convolution with the above parameters
		if (projection_weights){
			// allocated device memory to store output
			transformed_residual = cur_conv_block_activation -> transformed_residual;
			prepareAndDoConvolution(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, conv_block_input, projection_weights, transformed_residual);
			post_projection_norm_vals = cur_conv_block_activation -> post_projection_norm_vals;
			prepareAndDoBatchNormAndActivate(trainer, cur_conv_block_params -> norm_projection, cur_conv_block_activation -> norm_post_projection, batch_size, eps, transformed_residual, post_projection_norm_vals, false);
		}
		else{
			// would've been null, so renaming for semantic clarity
			post_projection_norm_vals = conv_block_input;
		}

		printDeviceData("(TRANSFORMED) RESIDUAL", transformed_residual, print_size);

		dim3 gridDimConvOutput(ceil((float) total_size_conv_block_output / MAX_THREAD_PER_BLOCK));
		dim3 blockDimConvOutput(MAX_THREAD_PER_BLOCK);

		conv_block_output = cur_conv_block_activation -> output;
		// add identity residual connection (or projected residual connection) to the prior batch norm output
		addVec <<< gridDimConvOutput, blockDimConvOutput >>> (total_size_conv_block_output, norm_output, post_projection_norm_vals, conv_block_output);

		printDeviceData("CONV OUTPUT + (TRANSFORMED) RESIDUAL", conv_block_output, print_size);

		conv_block_output_activated = cur_conv_block_activation -> output_activated;

		doActivation <<< gridDimConvOutput, blockDimConvOutput >>> (total_size_conv_block_output, conv_block_output, conv_block_output_activated);

		printDeviceData("CONV OUTPUT ACTIVATED", conv_block_output, print_size);
		
		// prepare for next block...
		conv_block_input = conv_block_output_activated;
	}

	int final_filters = dims -> final_depth;
	int final_spatial_dim = params_conv_blocks[n_conv_blocks - 1] -> incoming_spatial_dim;
	float * final_conv_block_output = activation_conv_blocks[n_conv_blocks - 1] -> output_activated;
	float * final_avg_pool_values = trainer -> forward_buffer -> activations -> final_conv_output_pooled;

	// NEED TO DO AVERAGE POOL OF LAST LAYER to go from (batch_size, 7, 7, 2048) to (batch size, 1, 1, 2048)

	// format of output is each row is a sample and has a row size of 2048
	dim3 gridDimAvgPool(final_filters);
	dim3 blockDimAvgPool(batch_size);
	doFilterAvgPool <<< gridDimAvgPool, blockDimAvgPool >>> (final_conv_block_output, final_spatial_dim, final_avg_pool_values);

	printDeviceData("FINAL AVG POOL VALUES", final_avg_pool_values, print_size);


	// APPLY FULLY CONNECTED LAYER BETWEEN (2048, 1000)
	float * fc_weights = trainer -> model -> params -> fully_connected;
	float * fc_output = trainer -> forward_buffer -> activations -> linear_output;
	int output_dim = dims -> output;

	// matrix multiply between (N, 2048) and fc weights of (2048, 1000), yields output of (N, 1000)
	// output is each row is a unique sample

	// GRID has dim (OUT_ROWS / TILE_WIDTH, OUT_COLS/TILE_WIDTH)
	// each BLOCK has dim (TILE_WIDTH, TILE_WIDTH)
	dim3 gridDimFCOutput(ceil((float) batch_size / TILE_WIDTH), ceil((float) output_dim / TILE_WIDTH));
	dim3 blockDimFCOutput(TILE_WIDTH, TILE_WIDTH);

	matMul <<< (gridDimFCOutput), (blockDimFCOutput) >>> (final_avg_pool_values, fc_weights, batch_size, final_filters, output_dim, fc_output);

	printDeviceData("FULLY CONNECTED WEIGHTS", fc_weights, print_size);
	printDeviceData("FULLY CONNECTED OUTPUT", fc_output, print_size);

	// DO SOFTMAX
	float * pred = trainer -> forward_buffer -> pred;
	dim3 gridDimSoftMax(1);
	dim3 blockDimSoftMax(batch_size);
	softMax <<< gridDimSoftMax, blockDimSoftMax >>> (fc_output, batch_size, output_dim, pred);

	printDeviceData("SOFTMAX PREDICTIONS", pred, print_size);

	// FINISH UP BY POPULATING PREDICTIONS ONTO CPU
	float * pred_cpu = trainer -> forward_buffer -> pred_cpu;
	cudaMemcpy(pred_cpu, pred, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);
}

void backwards_pass(Train_ResNet * trainer){
	
	Dims * dims = trainer -> model -> dims;
	int batch_size = trainer -> batch_size;
	int output_dim = dims -> output;
	float eps = trainer -> eps;
	Activations * activations = trainer -> forward_buffer -> activations;
	Params * model_params = trainer -> model -> params;
	Backprop_Buffer * backprop_buffer = trainer -> backprop_buffer;
	Params * param_derivs = backprop_buffer -> param_derivs;
	Activations * activation_derivs = backprop_buffer -> activation_derivs;

	int print_size = 10;

	/* STEP 1: LAST LAYER DERIVATIVE */

	// layer has output_dim * batch_size values
	// End of network was: fully connected layer -> softmax
	// Derivative of cross entropy loss w.r.t to fully connected values is: s - y where s is softmax value
	// thus copy softmax values and subtract 1 from the correct index (we know labels y are 0 except correct label of 1)
	int * correct_classes = trainer -> cur_batch -> correct_classes;
	float * pred = trainer -> forward_buffer -> pred;
	float * output_layer_deriv = backprop_buffer -> output_layer_deriv;
	cudaMemcpy(output_layer_deriv, pred, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToDevice);

	dim3 gridDimCrossDeriv(1);
	dim3 blockDimCrossDeriv(batch_size);
	crossEntropyDeriv <<< gridDimCrossDeriv, blockDimCrossDeriv >>> (output_layer_deriv, correct_classes, output_dim, batch_size);

	// divide by the batch size because loss is sum across all batches...
	// NOT SURE IF WE WANT TO DO AVERAGE HERE OR NOT...?
	
	// dim3 gridDimTakeAvgDeriv(output_dim);
	// dim3 blockDimTakeAvgDeriv(batch_size);
	// averageDerivOverBatchSize <<< gridDimTakeAvgDeriv, blockDimTakeAvgDeriv >>> (output_layer_deriv, output_dim, batch_size);

	printDeviceData("CROSS ENTROPY DERIV", output_layer_deriv, print_size);

	/* STEP 2: FC WEIGHT DERIV AND FINAL AVG POOL (SECOND LAST ACTIVTION LAYER) DERIVATIVE */

	// TODO: MAKE SURE THE DIMENSIONS ARE CORRECT ORDER...

	// FC WEIGHTS (2048, 1000) DERIV = matMul(transpose(final_conv_output_pooled), output_layer_deriv)
	int final_depth = dims -> final_depth;
	float * fc_deriv = param_derivs -> fully_connected;
	float * final_conv_output_pooled = activations -> final_conv_output_pooled;
	prepareAndDoMatMulLeftTranspose(final_conv_output_pooled, output_layer_deriv, batch_size, final_depth, batch_size, output_dim, fc_deriv);

	printDeviceData("FC WEIGHT DERIV", fc_deriv, print_size);

	// FINAL AVG POOL (N, 2048) DERIV = matMul(output_layer_deriv, transpose(FC Weight))
	float * fc_weights = model_params -> fully_connected;
	float * final_avg_pool_deriv = activation_derivs -> final_conv_output_pooled;
	prepareAndDoMatMulRightTranspose(output_layer_deriv, fc_weights, batch_size, output_dim, final_depth, output_dim, final_avg_pool_deriv);

	printDeviceData("FINAL AVG POOL ACTIVATION DERIV", final_avg_pool_deriv, print_size);


	/* CONV BLOCK DATA FROM FORWARD PASS */
	int n_conv_blocks = dims -> n_conv_blocks;
	Activation_ConvBlock ** activation_conv_blocks = activations -> activation_conv_blocks;
	ConvBlock ** conv_block_params = model_params -> conv_blocks;

	/* CONV BLOCK DERIV BUFFERS */
	Activation_ConvBlock ** activation_conv_blocks_derivs = activation_derivs -> activation_conv_blocks;
	ConvBlock ** conv_block_param_derivs = param_derivs -> conv_blocks;


	int final_spatial_dim = conv_block_params[n_conv_blocks - 1] -> incoming_spatial_dim;
	
	/* STEP 3: AVG POOL DERIV */

	// get the location for the deriv of final conv block output
	float * final_conv_block_output_deriv = activation_conv_blocks_derivs[n_conv_blocks - 1] -> output_activated;
	// using final_avg_pool_deriv (batch_size, 2048) to populate final_conv_block_output_deriv (batch_size, 7, 7, 2048)
	// each expanded (prior to pooling) spatial index takes on value of given filter's avg_pool_deriv / (spatial_dim^2)
	dim3 gridDimAvgPoolDeriv(final_depth);
	dim3 blockDimAvgPoolDeriv(batch_size);
	filterAvgPoolDeriv <<< gridDimAvgPoolDeriv, blockDimAvgPoolDeriv >>> (final_avg_pool_deriv, final_depth, batch_size, final_spatial_dim, final_conv_block_output_deriv);

	printDeviceData("FINAL CONV BLOCK OUTPUT ACTIVATION DERIV", final_conv_block_output_deriv, print_size);

	
	/* STEP 4: CONV BLOCK & BATCH NORM DERIVS  */
	

	// we are starting with deriv of last conv block output...

	// To go backwards for each block we:
		// 1.) Get deriv of output activated (ReLU so just 0 or 1)
		// 2.) Get deriv projection filter & transformed (if there is a projection of residual, otherwise both derivs are 1)
		// 3.) Multiply the deriv of output activation * deriv of transformed residual and add to the deriv of first layer of conv block (= output activated of prior block)
		// 4.) Multiply the deriv of output activation * deriv of batch norm for expanded conv output (with respect to both its own parameters and also the input to batch norm = expanded conv output)
		// 5.) Get deriv of expanded convolution & deriv of input to expanded convolution (= batch norm output of spatial conv)
		// 6.) Get deriv of batch norm for spatial conv output (with respect to both its own parameters and also the input to batch norm = spatial conv output)
		// 7.) Get deriv of sptial convolution & deriv of input to spatial convolution (= batch norm output of reduced conv)
		// 8.) Get deriv of batch norm for reduced conv output (with respect to both its own parameters and also the input to batch norm = reduced conv output)
		// 9.) Get deriv of reduced convolution & deriv of input to reduced convolution, which is the first layer of conv block (= batch norm output of prior conv block)
		// Items 3.) and 9.) provide the derivative used to repeat process for prior block

	

	// will update these variables throughout loop to pass to batch norm deriv
	float *bn_input, *bn_activated, *bn_out_layer_deriv, *bn_input_deriv;
	BatchNorm *cur_batch_norm_params, *cur_batch_norm_param_derivs;
	Cache_BatchNorm *cur_batch_norm_cache, *cur_batch_norm_cache_derivs;

	// will update these every iteration through conv_blocks
	ConvBlock * cur_conv_block_params, *cur_conv_block_param_derivs;
	Activation_ConvBlock * cur_conv_block_activation, *cur_conv_block_activation_derivs;

	// will update these within every iteration through conv_blocks
	// because multiple convolutions per block, but keep params same for easy calls to functions
	int in_spatial_dim, kern_dim, in_filters, out_filters, stride;
	float *conv_input, *conv_weight, *conv_out_deriv;
	float *conv_input_deriv, *conv_weight_deriv;


	// STARTING POINT FROM BACKPROP COMING FROM UPSTREAM LAYERS IS AT LAST CONV BLOCK ACTIVATION -> OUTPUT_ACTIVATED
	float *conv_block_input, *conv_block_input_deriv, * upstream_deriv, *block_activation_deriv, *final_output_pre_activ;

	// extra temp variables
	int total_size, output_size;

	for (int i = n_conv_blocks - 1; i >= 0; i--){

		// residual deriv and normal backprop deriv added to this
		if (i == 0){
			conv_block_input = activations -> init_convblock_input;
			conv_block_input_deriv = activation_derivs -> init_convblock_input;
		}
		else{
			conv_block_input = activation_conv_blocks[i - 1] -> output_activated;
			conv_block_input_deriv = activation_conv_blocks_derivs[i - 1] -> output_activated;
		}

		// getting current conv block parameters and buffers to hold derivs
		cur_conv_block_params = conv_block_params[i];
		cur_conv_block_param_derivs = conv_block_param_derivs[i];

		// getting current conv block activation values and buffers to hold derivs
		cur_conv_block_activation = activation_conv_blocks[i];
		cur_conv_block_activation_derivs = activation_conv_blocks_derivs[i];

		/* 1: Conv Block Output Activation */
		
		// GIVEN
		upstream_deriv = cur_conv_block_activation_derivs -> output_activated;
		final_output_pre_activ = cur_conv_block_activation -> output;

		// to fill in the ReLU deriv location
		block_activation_deriv = cur_conv_block_activation_derivs -> output;

		output_size = batch_size * cur_conv_block_params -> expanded_depth * cur_conv_block_params -> incoming_spatial_dim * cur_conv_block_params -> incoming_spatial_dim / ((cur_conv_block_params -> stride) * (cur_conv_block_params -> stride));

		dim3 gridDimOutput(ceil((float) output_size / MAX_THREAD_PER_BLOCK));
		dim3 blockDimOutput(MAX_THREAD_PER_BLOCK);
		doActivationDeriv <<< gridDimOutput, blockDimOutput >>> (output_size, final_output_pre_activ, upstream_deriv, block_activation_deriv);


		/* 2: (Transformed) Residual Derivs & Chained/Added to Conv Block Input Deriv (= prior_block_output_deriv) */

		// check if there is a projection (aka convolution over depth/kern_dim=1 or possibly stride=2/kern_dim=3), otherwise the projection deriv is 1
		// If there is a projection need to compute derivative of the projection convolution kernel weights and deriv w.r.t. projection convolution input=conv_block_input=prior_block_output_activated
		if (cur_conv_block_params -> projection){


			// DEAL WITH BATCH NORM
			// update the current batch norm layer pointers
			cur_batch_norm_params = cur_conv_block_params -> norm_projection;
			cur_batch_norm_param_derivs = cur_conv_block_param_derivs -> norm_projection;

			cur_batch_norm_cache = cur_conv_block_activation -> norm_post_projection;
			cur_batch_norm_cache_derivs = cur_conv_block_activation_derivs -> norm_post_projection;

			// fill in details about backprop I/O
			// dL/dBN_Output (given)
			bn_out_layer_deriv = cur_conv_block_activation_derivs -> output;
			// dL/dBN_Input (to fill in)
			bn_input_deriv = cur_conv_block_activation_derivs -> transformed_residual;
			// input to batch norm layer from forward pass
			bn_input = cur_conv_block_activation -> transformed_residual;
			// activated output of batch norm layer from forward pass
			bn_activated = cur_conv_block_activation -> post_projection_norm_vals;
		
			prepareAndDoActivationAndBatchNormDeriv(trainer, cur_batch_norm_params, cur_batch_norm_cache, cur_batch_norm_param_derivs, cur_batch_norm_cache_derivs,
																						batch_size, eps, bn_input, bn_activated, bn_out_layer_deriv, bn_input_deriv, false);


			// CONVOLUTION DIMENSIONS
			in_spatial_dim = (cur_conv_block_params -> incoming_spatial_dim);
			in_filters = cur_conv_block_params -> incoming_filters;
			out_filters = cur_conv_block_params -> expanded_depth;
			stride = cur_conv_block_params -> stride;
			if (stride == 2){
				kern_dim = 3;
			}
			else{
				kern_dim = 1;
			}


			// CONVOLUTION FORWARD DATA
			// transformed residual convolution input is the value at first step of conv block => activated output from previous block
			conv_input = conv_block_input;
			conv_weight = cur_conv_block_params -> projection;
			// from backprop
			conv_out_deriv = cur_conv_block_activation_derivs -> transformed_residual;

			// CONVOLUTION BACKWARDS DERIV DATA BUFFERS
			// because residual
			conv_input_deriv = conv_block_input_deriv;
			conv_weight_deriv = cur_conv_block_param_derivs -> projection;

			prepreAndDoConvolutionDeriv(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, false,
													conv_input, conv_weight, conv_out_deriv,
													conv_input_deriv, conv_weight_deriv, true);

			printDeviceData("PROJECTED CONV INPUT DERIV", conv_input_deriv, print_size);
			printDeviceData("PROJECTED CONV WEIGHT DERIV", conv_weight_deriv, print_size);
		}
		else{
			total_size = batch_size * (cur_conv_block_params -> incoming_spatial_dim) * (cur_conv_block_params -> incoming_spatial_dim) * (cur_conv_block_params -> incoming_filters);

			dim3 gridDimResidual(ceil((float) total_size / MAX_THREAD_PER_BLOCK));
			dim3 blockDimResidual(MAX_THREAD_PER_BLOCK);
			setVal <<< gridDimResidual, blockDimResidual >>> (total_size, 0, conv_block_input_deriv);
			addVec <<< gridDimResidual, blockDimResidual >>> (total_size, conv_block_input_deriv, cur_conv_block_activation_derivs -> output, conv_block_input_deriv);
		}
		

		/* 3: Expanded Convolution And Batch Norm Derivs */

		// update the current batch norm layer pointers
		cur_batch_norm_params = cur_conv_block_params -> norm_expansion;
		
		
		cur_batch_norm_param_derivs = cur_conv_block_param_derivs -> norm_expansion;

		cur_batch_norm_cache = cur_conv_block_activation -> norm_post_expanded;
		cur_batch_norm_cache_derivs = cur_conv_block_activation_derivs -> norm_post_expanded;

		size_t cur_bn_inp_size = cur_batch_norm_cache -> input_size;
		size_t cur_bn_feature_size = cur_batch_norm_cache -> feature_size;

		// fill in details about backprop I/O
		// dL/dBN_Output (given)
		bn_out_layer_deriv = cur_conv_block_activation_derivs -> output;
		// dL/dBN_Input (to fill in)
		bn_input_deriv = cur_conv_block_activation_derivs -> post_expanded;
		// input to batch norm layer from forward pass
		bn_input = cur_conv_block_activation -> post_expanded;
		// activated output of batch norm layer from forward pass
		bn_activated = cur_conv_block_activation -> post_expanded_norm_vals;
		
		prepareAndDoActivationAndBatchNormDeriv(trainer, cur_batch_norm_params, cur_batch_norm_cache, cur_batch_norm_param_derivs, cur_batch_norm_cache_derivs,
																						batch_size, eps, bn_input, bn_activated, bn_out_layer_deriv, bn_input_deriv, false);

		printDeviceData("CONV BLOCK OUTPUT ACTIVATION & NORM DERIV", bn_input_deriv, print_size);

		// CONVOLUTION DIMENSIONS
		in_spatial_dim = (cur_conv_block_params -> incoming_spatial_dim) / (cur_conv_block_params -> stride);
		in_filters = cur_conv_block_params -> reduced_depth;
		out_filters = cur_conv_block_params -> expanded_depth;
		stride = 1;
		kern_dim = 1;

		// CONVOLUTION FORWARD DATA
		conv_input = cur_conv_block_activation -> post_spatial_activated;
		conv_weight = cur_conv_block_params -> depth_expansion;
		// from backprop
		conv_out_deriv = cur_conv_block_activation_derivs -> post_expanded;

		// CONVOLUTION BACKWARDS DERIV DATA BUFFERS
		// because residual
		conv_input_deriv = cur_conv_block_activation_derivs -> post_spatial_activated;
		conv_weight_deriv = cur_conv_block_param_derivs -> depth_expansion;

		prepreAndDoConvolutionDeriv(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, false,
													conv_input, conv_weight, conv_out_deriv,
													conv_input_deriv, conv_weight_deriv, true);
		
		printDeviceData("EXPANDED CONV INPUT DERIV", conv_input_deriv, print_size);
		printDeviceData("EXPANDED CONV WEIGHT DERIV", conv_weight_deriv, print_size);


		/* 4: Spatial Convolution Activation and Batch Norm Derivs */

		// update the current batch norm layer pointers
		cur_batch_norm_params = cur_conv_block_params -> norm_spatial;
		cur_batch_norm_param_derivs = cur_conv_block_param_derivs -> norm_spatial;

		cur_batch_norm_cache = cur_conv_block_activation -> norm_post_spatial;
		cur_batch_norm_cache_derivs = cur_conv_block_activation_derivs -> norm_post_spatial;

		// fill in details about backprop I/O
		// dL/dBN_Output (given)
		bn_out_layer_deriv = cur_conv_block_activation_derivs -> post_spatial_activated;
		// dL/dBN_Input (to fill in)
		bn_input_deriv = cur_conv_block_activation_derivs -> post_spatial;
		// input to batch norm layer from forward pass
		bn_input = cur_conv_block_activation -> post_spatial;
		// activated output of batch norm layer from forward pass
		bn_activated = cur_conv_block_activation -> post_spatial_activated;
		
		prepareAndDoActivationAndBatchNormDeriv(trainer, cur_batch_norm_params, cur_batch_norm_cache, cur_batch_norm_param_derivs, cur_batch_norm_cache_derivs,
																						batch_size, eps, bn_input, bn_activated, bn_out_layer_deriv, bn_input_deriv, true);

		printDeviceData("SPATIAL ACTIVATION & BATCH NORM DERIV", bn_input_deriv, print_size);

		/* 5: Spatial Convolution Derivs */

		// CONVOLUTION DIMENSIONS
		in_spatial_dim = cur_conv_block_params -> incoming_spatial_dim;
		in_filters = cur_conv_block_params -> reduced_depth;
		out_filters = cur_conv_block_params -> reduced_depth;
		stride = cur_conv_block_params -> stride;
		kern_dim = 3;

		// CONVOLUTION FORWARD DATA
		conv_input = cur_conv_block_activation -> post_reduced_activated;
		conv_weight = cur_conv_block_params -> spatial;
		// from backprop
		conv_out_deriv = cur_conv_block_activation_derivs -> post_spatial;

		// CONVOLUTION BACKWARDS DERIV DATA BUFFERS
		// because residual
		conv_input_deriv = cur_conv_block_activation_derivs -> post_reduced_activated;
		conv_weight_deriv = cur_conv_block_param_derivs -> spatial;

		

		prepreAndDoConvolutionDeriv(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, false,
													conv_input, conv_weight, conv_out_deriv,
													conv_input_deriv, conv_weight_deriv, true);

		printDeviceData("SPATIAL CONV INPUT DERIV", conv_input_deriv, print_size);
		printDeviceData("SPATIAL CONV WEIGHT DERIV", conv_weight_deriv, print_size);

		/* 6: Reduced Convolution Activation and Batch Norm Derivs */

		// update the current batch norm layer pointers
		cur_batch_norm_params = cur_conv_block_params -> norm_depth_reduction;
		cur_batch_norm_param_derivs = cur_conv_block_param_derivs -> norm_depth_reduction;

		cur_batch_norm_cache = cur_conv_block_activation -> norm_post_reduced;
		cur_batch_norm_cache_derivs = cur_conv_block_activation_derivs -> norm_post_reduced;

		// fill in details about backprop I/O
		// dL/dBN_Output (given)
		bn_out_layer_deriv = cur_conv_block_activation_derivs -> post_reduced_activated;
		// dL/dBN_Input (to fill in)
		bn_input_deriv = cur_conv_block_activation_derivs -> post_reduced;
		// input to batch norm layer from forward pass
		bn_input = cur_conv_block_activation -> post_reduced;
		// activated output of batch norm layer from forward pass
		bn_activated = cur_conv_block_activation -> post_reduced_activated;
		
		prepareAndDoActivationAndBatchNormDeriv(trainer, cur_batch_norm_params, cur_batch_norm_cache, cur_batch_norm_param_derivs, cur_batch_norm_cache_derivs,
																						batch_size, eps, bn_input, bn_activated, bn_out_layer_deriv, bn_input_deriv, true);

		printDeviceData("REDUCED ACTIVATION & BATCH NORM DERIV", bn_input_deriv, print_size);

		/* 7: Reduced Convolution Derivs */


		// CONVOLUTION DIMENSIONS
		in_spatial_dim = cur_conv_block_params -> incoming_spatial_dim;
		in_filters = cur_conv_block_params -> incoming_filters;
		out_filters = cur_conv_block_params -> reduced_depth;
		stride = 1;
		kern_dim = 1;

		// CONVOLUTION FORWARD DATA
		conv_input = conv_block_input;
		conv_weight = cur_conv_block_params -> depth_reduction;
		// from backprop
		conv_out_deriv = cur_conv_block_activation_derivs -> post_reduced;

		// CONVOLUTION BACKWARDS DERIV DATA BUFFERS
		// because residual
		conv_input_deriv = conv_block_input_deriv;
		conv_weight_deriv = cur_conv_block_param_derivs -> depth_reduction;

		prepreAndDoConvolutionDeriv(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, true,
													conv_input, conv_weight, conv_out_deriv,
													conv_input_deriv, conv_weight_deriv,  true);

		printDeviceData("REDUCED CONV INPUT DERIV", conv_input_deriv, print_size);
		printDeviceData("REDUCED CONV WEIGHT DERIV", conv_weight_deriv, print_size);

	}


	/* STEP 5: MAX POOL DERIV */

	// maxpool dimensions (used in forward pass)
	int maxpool_kern_dim = dims -> init_maxpool_dim;
	int maxpool_stride = dims -> init_maxpool_stride;
	int maxpool_in_spatial_dim = dims -> input / dims -> init_conv_stride;
	int maxpool_out_spatial_dim = maxpool_in_spatial_dim / maxpool_stride;
	int maxpool_filters = dims -> init_conv_filters;

	// backprop up through the init convblock input has been done. the gradient is at:
	float * maxpool_out_deriv = activation_derivs -> init_convblock_input;

	// getting the max inds cached from forward pass to easily do backprop
	int * max_inds = activations -> max_inds;

	// populating the gradient of input to max_pool located at:
	float * maxpool_inp_deriv = activation_derivs -> init_conv_activated;
	// ensure that gradient has 0's, so that maxPoolDeriv kernel can overwrite only at max ind locations
	int maxpool_inp_size = maxpool_in_spatial_dim * maxpool_in_spatial_dim * maxpool_filters * batch_size;
	cudaMemset(maxpool_inp_deriv, 0, maxpool_inp_size * sizeof(float));

	dim3 gridDimMaxPoolDeriv(maxpool_out_spatial_dim, maxpool_out_spatial_dim, maxpool_filters);
	dim3 blockDimMaxPoolDeriv(batch_size);

	// compute max pool deriv (i.e. populate maxpool_inp_deriv)
	maxPoolDeriv <<< gridDimMaxPoolDeriv, blockDimMaxPoolDeriv >>> (max_inds, maxpool_out_deriv, maxpool_kern_dim, maxpool_in_spatial_dim, maxpool_stride, maxpool_filters, batch_size, maxpool_inp_deriv);

	printDeviceData("MAX POOL INPUT ACTIVATION DERIV", maxpool_inp_deriv, print_size);

	/* STEP 6: INIT BATCH NORM & CONV DERIV */

	// BACKPROP OVER THE BATCH NORM OF FIRST CONV LAYER

	// update the current batch norm layer pointers
	cur_batch_norm_params = model_params -> norm_init_conv;
	cur_batch_norm_param_derivs = param_derivs -> norm_init_conv;

	cur_batch_norm_cache = activations -> norm_init_conv;
	cur_batch_norm_cache_derivs = activation_derivs -> norm_init_conv;

	// fill in details about backprop I/O
	// dL/dBN_Output (given)
	bn_out_layer_deriv = activation_derivs -> init_conv_activated;
	// dL/dBN_Input (to fill in)
	bn_input_deriv = activation_derivs -> init_conv_applied;
	// input to batch norm layer from forward pass
	bn_input = activations -> init_conv_applied;
	// activated output of batch norm layer from forward pass
	bn_activated = activations -> init_conv_activated;
		
	prepareAndDoActivationAndBatchNormDeriv(trainer, cur_batch_norm_params, cur_batch_norm_cache, cur_batch_norm_param_derivs, cur_batch_norm_cache_derivs,
																						batch_size, eps, bn_input, bn_activated, bn_out_layer_deriv, bn_input_deriv, true);

	printDeviceData("INIT CONV ACTIVATION & BATCH NORM DERIV", bn_input_deriv, print_size);

	// BACKPROP OVER FIRST CONV LAYER

	// CONVOLUTION DIMENSIONS
	// hardcoded to 3 for the colors
	in_filters = 3;
	out_filters = dims -> init_conv_filters;
	in_spatial_dim = dims -> input;
	stride = dims -> init_conv_stride;
	kern_dim = dims -> init_kernel_dim;

	// CONVOLUTION FORWARD DATA
	conv_input = trainer -> cur_batch -> images;
	conv_weight = model_params -> init_conv_layer;
	// from backprop
	conv_out_deriv = activation_derivs -> init_conv_applied;

	// CONVOLUTION BACKWARDS DERIV DATA BUFFERS
	// because residual
	conv_input_deriv = NULL;
	conv_weight_deriv = param_derivs -> init_conv_layer;

	prepreAndDoConvolutionDeriv(trainer, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, false,
													conv_input, conv_weight, conv_out_deriv,
													conv_input_deriv, conv_weight_deriv, false);

	printDeviceData("INIT CONV WEIGHT DERIV", conv_weight_deriv, print_size);
}

void dump_parameters(int dump_id, Train_ResNet * trainer, const char * special_dir){

	Params * model_params = trainer -> model -> params;
	float ** model_params_locations = model_params -> locations;
	int * param_sizes = model_params -> sizes;
	int n_locations = model_params -> n_locations;

	// values calculated from backprop, will reset these before returning
	Params * current_gradients = trainer -> backprop_buffer -> param_derivs;
	float ** current_gradient_locations = current_gradients -> locations;
	
	// running history values that the optimizer needs, will update these before returning
	Params * prev_grad_means = trainer -> backprop_buffer -> prev_means;
	float ** prev_grad_means_locations = prev_grad_means -> locations;
	Params * prev_grad_vars = trainer -> backprop_buffer -> prev_vars;
	float ** prev_grad_vars_locations = prev_grad_vars -> locations;

	int param_size;
	float *model_location, *grad_location, * mean_location, * var_location;

	float * cpu_param_buff;
	FILE * fp;

	char * model_params_filepath;
	char * gradients_filepath;
	char * means_filepath;
	char * vars_filepath;

	int n_read, print_ret;
	for (int i = n_locations - 1; i >= 0; i--){
		param_size = param_sizes[i];
		cpu_param_buff = (float *) malloc(param_size * sizeof(float));

		model_location = model_params_locations[i];
		cudaMemcpy(cpu_param_buff, model_location, param_size * sizeof(float), cudaMemcpyDeviceToHost);
		print_ret = asprintf(&model_params_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/model_params/%03d.buffer", special_dir, dump_id, i);
		fp = fopen(model_params_filepath, "wb");
		n_read = fwrite(cpu_param_buff, sizeof(float), (size_t) param_size, fp);
		fclose(fp);
		free(model_params_filepath);


		grad_location = current_gradient_locations[i];
		cudaMemcpy(cpu_param_buff, grad_location, param_size * sizeof(float), cudaMemcpyDeviceToHost);
		print_ret = asprintf(&gradients_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/gradients/%03d.buffer", special_dir, dump_id, i);
		fp = fopen(gradients_filepath, "wb");
		n_read = fwrite(cpu_param_buff, sizeof(float), (size_t) param_size, fp);
		fclose(fp);
		free(gradients_filepath);

		mean_location = prev_grad_means_locations[i];
		cudaMemcpy(cpu_param_buff, mean_location, param_size * sizeof(float), cudaMemcpyDeviceToHost);
		print_ret = asprintf(&means_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/means/%03d.buffer", special_dir, dump_id, i);
		fp = fopen(means_filepath, "wb");
		n_read = fwrite(cpu_param_buff, sizeof(float), (size_t) param_size, fp);
		fclose(fp);
		free(means_filepath);

		var_location = prev_grad_vars_locations[i];
		cudaMemcpy(cpu_param_buff, var_location, param_size * sizeof(float), cudaMemcpyDeviceToHost);
		print_ret = asprintf(&vars_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/vars/%03d.buffer", special_dir, dump_id, i);
		fp = fopen(vars_filepath, "wb");
		n_read = fwrite(cpu_param_buff, sizeof(float), (size_t) param_size, fp);
		fclose(fp);
		free(vars_filepath);

		free(cpu_param_buff);
	}
}


void dump_batch_norm_cache(Train_ResNet * trainer, char * filepath, Cache_BatchNorm * cache_batchnorm){

	FILE * fp;
	int n_wrote, print_ret;

	int input_size = cache_batchnorm -> input_size;
	int filters = cache_batchnorm -> feature_size;

	char * filepath_new = NULL;

	print_ret = asprintf(&filepath_new, "%smeans.buffer", filepath);
	float * cpu_means = (float *) malloc(filters * sizeof(float));
	cudaMemcpy(cpu_means, cache_batchnorm -> means, filters * sizeof(float), cudaMemcpyDeviceToHost);
	fp = fopen(filepath_new, "wb");
	n_wrote = fwrite(cpu_means, sizeof(float), filters, fp);
	fclose(fp);
	free(cpu_means);
	free(filepath_new);

	print_ret = asprintf(&filepath_new, "%sinv_vars.buffer", filepath);
	float * cpu_vars = (float *) malloc(filters * sizeof(float));
	cudaMemcpy(cpu_vars, cache_batchnorm -> inv_vars, filters * sizeof(float), cudaMemcpyDeviceToHost);
	fp = fopen(filepath_new, "wb");
	n_wrote = fwrite(cpu_vars, sizeof(float), filters, fp);
	fclose(fp);
	free(cpu_vars);
	free(filepath_new);
}

void dump_conv_block_activation(int dump_id, Train_ResNet * trainer, Activation_ConvBlock * activation_conv_block, int conv_block_ind, bool is_deriv, const char * special_dir){
	FILE * fp;
	int n_wrote, print_ret;

	char * filepath = NULL;

	if (is_deriv){
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/conv_blocks/%02d/", special_dir, dump_id, conv_block_ind);
	}
	else{
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/conv_blocks/%02d/", special_dir, dump_id, conv_block_ind);
	}

	char * filepath_dup = NULL;
	
	char * batchnorm_filepath = NULL;
	if (is_deriv){
		print_ret = asprintf(&batchnorm_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/batch_norms/%02d/", special_dir, dump_id, conv_block_ind);
	}
	else{
		print_ret = asprintf(&batchnorm_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/batch_norms/%02d/", special_dir, dump_id, conv_block_ind);
	}

	char * batchnorm_filepath_dup = NULL; 

	int batch_size = trainer -> batch_size;
	int incoming_spatial_dim = activation_conv_block -> incoming_spatial_dim;
	int reduced_depth = activation_conv_block -> reduced_depth;
	int expanded_depth = activation_conv_block -> expanded_depth;
	int stride = activation_conv_block -> stride;


	/* REDUCTION CONV APPLIED */
	int reduction_size = incoming_spatial_dim * incoming_spatial_dim * reduced_depth * batch_size;
	float * cpu_reduction_applied = (float *) malloc(reduction_size * sizeof(float));
	cudaMemcpy(cpu_reduction_applied, activation_conv_block -> post_reduced, reduction_size * sizeof(float), cudaMemcpyDeviceToHost);
	print_ret = asprintf(&filepath_dup, "%sreduction_applied.buffer", filepath);
	fp = fopen(filepath_dup, "wb");
	n_wrote = fwrite(cpu_reduction_applied, sizeof(float), reduction_size, fp);
	fclose(fp);
	free(filepath_dup);
	free(cpu_reduction_applied);



	/* REDUCTION BATCH NORM */
	print_ret = asprintf(&batchnorm_filepath_dup, "%sreduced/", batchnorm_filepath);
	dump_batch_norm_cache(trainer, batchnorm_filepath_dup, activation_conv_block -> norm_post_reduced);
	free(batchnorm_filepath_dup);


	/* REDUCTION ACTIVATED */
	float * cpu_reduction_activated = (float *) malloc(reduction_size * sizeof(float));
	cudaMemcpy(cpu_reduction_activated, activation_conv_block -> post_reduced_activated, reduction_size * sizeof(float), cudaMemcpyDeviceToHost);
	print_ret = asprintf(&filepath_dup, "%sreduction_activated.buffer", filepath);
	fp = fopen(filepath_dup, "wb");
	n_wrote = fwrite(cpu_reduction_activated, sizeof(float), reduction_size, fp);
	fclose(fp);
	free(filepath_dup);
	free(cpu_reduction_activated);


	/* SPATIAL CONV APPLIED */
	int spatial_size = incoming_spatial_dim * incoming_spatial_dim * reduced_depth * batch_size / (stride * stride);
	float * cpu_spatial_applied = (float *) malloc(spatial_size * sizeof(float));
	cudaMemcpy(cpu_spatial_applied, activation_conv_block -> post_spatial, spatial_size * sizeof(float), cudaMemcpyDeviceToHost);
	print_ret = asprintf(&filepath_dup, "%sspatial_applied.buffer", filepath);
	fp = fopen(filepath_dup, "wb");
	n_wrote = fwrite(cpu_spatial_applied, sizeof(float), spatial_size, fp);
	fclose(fp);
	free(filepath_dup);
	free(cpu_spatial_applied);


	/* SPATIAL BATCH NORM */
	print_ret = asprintf(&batchnorm_filepath_dup, "%sspatial/", batchnorm_filepath);
	dump_batch_norm_cache(trainer, batchnorm_filepath_dup, activation_conv_block -> norm_post_spatial);
	free(batchnorm_filepath_dup);


	/* SPATIAL ACTIVATED */
	float * cpu_spatial_activated = (float *) malloc(spatial_size * sizeof(float));
	cudaMemcpy(cpu_spatial_activated, activation_conv_block -> post_spatial_activated, spatial_size * sizeof(float), cudaMemcpyDeviceToHost);
	print_ret = asprintf(&filepath_dup, "%sspatial_activated.buffer", filepath);
	fp = fopen(filepath_dup, "wb");
	n_wrote = fwrite(cpu_spatial_activated, sizeof(float), spatial_size, fp);
	fclose(fp);
	free(filepath_dup);
	free(cpu_spatial_activated);


	/* EXPANDED CONV APPLIED */
	int expanded_size = incoming_spatial_dim * incoming_spatial_dim * expanded_depth * batch_size / (stride * stride);
	float * cpu_expanded_applied = (float *) malloc(expanded_size * sizeof(float));
	cudaMemcpy(cpu_expanded_applied, activation_conv_block -> post_expanded, expanded_size * sizeof(float), cudaMemcpyDeviceToHost);
	print_ret = asprintf(&filepath_dup, "%sexpanded_applied.buffer", filepath);
	fp = fopen(filepath_dup, "wb");
	n_wrote = fwrite(cpu_expanded_applied, sizeof(float), expanded_size, fp);
	fclose(fp);
	free(filepath_dup);
	free(cpu_expanded_applied);

	/* POST EXPANDED NORM */
	print_ret = asprintf(&batchnorm_filepath_dup, "%sexpanded/", batchnorm_filepath);
	dump_batch_norm_cache(trainer, batchnorm_filepath_dup, activation_conv_block -> norm_post_expanded);
	free(batchnorm_filepath_dup);

	/* EXPANDED NORM VALUES */
	float * cpu_expanded_norm = (float *) malloc(expanded_size * sizeof(float));
	cudaMemcpy(cpu_expanded_norm, activation_conv_block -> post_expanded_norm_vals, expanded_size * sizeof(float), cudaMemcpyDeviceToHost);
	print_ret = asprintf(&filepath_dup, "%sexpanded_post_norm.buffer", filepath);
	fp = fopen(filepath_dup, "wb");
	n_wrote = fwrite(cpu_expanded_norm, sizeof(float), expanded_size, fp);
	fclose(fp);
	free(filepath_dup);
	free(cpu_expanded_norm);


	/* (TRANSFORMED) RESIDUAL */

	// only blocks with projection weights haved a transformed residual. otherwise identity to input
	if (activation_conv_block -> transformed_residual) {
		float * cpu_residual = (float *) malloc(expanded_size * sizeof(float));
		cudaMemcpy(cpu_residual, activation_conv_block -> transformed_residual, expanded_size * sizeof(float), cudaMemcpyDeviceToHost);
		print_ret = asprintf(&filepath_dup, "%stransformed_residual.buffer", filepath);
		fp = fopen(filepath_dup, "wb");
		n_wrote = fwrite(cpu_residual, sizeof(float), expanded_size, fp);
		fclose(fp);
		free(filepath_dup);
		free(cpu_residual);

		print_ret = asprintf(&batchnorm_filepath_dup, "%sprojected/", batchnorm_filepath);
		dump_batch_norm_cache(trainer, batchnorm_filepath_dup, activation_conv_block -> norm_post_projection);
		free(batchnorm_filepath_dup);

	}

	/* EXPANDED + RESIDUAL */
	float * cpu_combined_output = (float *) malloc(expanded_size * sizeof(float));
	cudaMemcpy(cpu_combined_output, activation_conv_block -> output, expanded_size * sizeof(float), cudaMemcpyDeviceToHost);
	print_ret = asprintf(&filepath_dup, "%scombined_output.buffer", filepath);
	fp = fopen(filepath_dup, "wb");
	n_wrote = fwrite(cpu_combined_output, sizeof(float), expanded_size, fp);
	fclose(fp);
	free(filepath_dup);
	free(cpu_combined_output);

	
	/* POST RESIDUAL ACTIVATED */
	float * cpu_combined_output_activated = (float *) malloc(expanded_size * sizeof(float));
	cudaMemcpy(cpu_combined_output_activated, activation_conv_block -> output_activated, expanded_size * sizeof(float), cudaMemcpyDeviceToHost);
	print_ret = asprintf(&filepath_dup, "%soutput_activated.buffer", filepath);
	fp = fopen(filepath_dup, "wb");
	n_wrote = fwrite(cpu_combined_output_activated, sizeof(float), expanded_size, fp);
	fclose(fp);
	free(filepath_dup);
	free(cpu_combined_output_activated);

	free(filepath);
	free(batchnorm_filepath);

	

}

void dump_activations(int dump_id, Train_ResNet * trainer, Activations * activations, bool is_deriv, const char * special_dir){

	size_t batch_size = trainer -> batch_size;
	Dims * dims = trainer -> model -> dims;

	char * filepath = NULL;
	FILE * fp;
	int n_wrote, print_ret;

	// input
	size_t input_size = trainer -> cur_batch -> image_size * batch_size;
	if (!is_deriv){
		float * cpu_images = (float *) malloc(input_size * sizeof(float));
		cudaMemcpy(cpu_images, trainer -> cur_batch -> images, input_size * sizeof(float), cudaMemcpyDeviceToHost);
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/input.buffer", special_dir, dump_id);
		fp = fopen(filepath, "wb");
		n_wrote = fwrite(cpu_images, sizeof(float), input_size, fp);
		fclose(fp);
		free(cpu_images);
		free(filepath);
	}


	/* 1. INIT CONV */

	size_t init_conv_applied_size = batch_size * dims -> init_conv_filters * (dims -> input / dims -> init_conv_stride) * (dims -> input / dims -> init_conv_stride);
	float * cpu_init_conv_applied = (float *) malloc(init_conv_applied_size * sizeof(float));
	cudaMemcpy(cpu_init_conv_applied, activations -> init_conv_applied, init_conv_applied_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (is_deriv){
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/init_conv_applied.buffer", special_dir, dump_id);
	}
	else{
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/init_conv_applied.buffer", special_dir, dump_id);
	}
	fp = fopen(filepath, "wb");
	n_wrote = fwrite(cpu_init_conv_applied, sizeof(float), init_conv_applied_size, fp);
	fclose(fp);
	free(filepath);
	free(cpu_init_conv_applied);


	/* 2. INIT BATCH NORM */
	if (is_deriv){
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/batch_norms/init/", special_dir, dump_id);
	}
	else{
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/batch_norms/init/", special_dir, dump_id);
	}

	dump_batch_norm_cache(trainer, filepath, activations -> norm_init_conv);
	free(filepath);

	/* 3. ACTIVATED BATCH NORM */
	float * cpu_init_conv_activated = (float *) malloc(init_conv_applied_size * sizeof(float));
	cudaMemcpy(cpu_init_conv_activated, activations -> init_conv_activated, init_conv_applied_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (is_deriv){
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/init_conv_activated.buffer", special_dir, dump_id);
	}
	else{
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/init_conv_activated.buffer", special_dir, dump_id);
	}
	fp = fopen(filepath, "wb");
	n_wrote = fwrite(cpu_init_conv_activated, sizeof(float), init_conv_applied_size, fp);
	fclose(fp);
	free(filepath);
	free(cpu_init_conv_activated);

	/* 4. MAX POOL */
	size_t maxpool_size = init_conv_applied_size / (dims -> init_maxpool_stride * dims -> init_maxpool_stride);
	// max inds only populated on forward pass
	if (!is_deriv){
		int * cpu_max_inds = (int *) malloc(maxpool_size * sizeof(int));
		cudaMemcpy(cpu_max_inds, activations -> max_inds, maxpool_size * sizeof(int), cudaMemcpyDeviceToHost);
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/max_inds.buffer", special_dir, dump_id);
		fp = fopen(filepath, "wb");
		n_wrote = fwrite(cpu_max_inds, sizeof(int), maxpool_size, fp);
		fclose(fp);
		free(filepath);
		free(cpu_max_inds);
	}

	float * cpu_init_convblock_input = (float *) malloc(maxpool_size * sizeof(float));
	cudaMemcpy(cpu_init_convblock_input, activations -> init_convblock_input, maxpool_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (is_deriv){
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/init_convblock_input.buffer", special_dir, dump_id);
	}
	else{
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/init_convblock_input.buffer", special_dir, dump_id);
	}
	fp = fopen(filepath, "wb");
	n_wrote = fwrite(cpu_init_convblock_input, sizeof(float), maxpool_size, fp);
	fclose(fp);
	free(filepath);
	free(cpu_init_convblock_input);


	/* 5. CONV BLOCKS */
	int n_conv_blocks = activations -> n_conv_blocks;
	Activation_ConvBlock ** conv_blocks = activations -> activation_conv_blocks;
	Activation_ConvBlock * cur_conv_block;
	for (int i = 0; i < n_conv_blocks; i++){
		cur_conv_block = conv_blocks[i];
		dump_conv_block_activation(dump_id, trainer, cur_conv_block, i, is_deriv, special_dir);
	}


	/* 6. FINAL AVG POOL */
	int final_avg_pool_size = dims -> final_depth * batch_size;
	float * cpu_final_avg_pool = (float *) malloc(final_avg_pool_size * sizeof(float));
	cudaMemcpy(cpu_final_avg_pool, activations -> final_conv_output_pooled, final_avg_pool_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (is_deriv){
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/final_avg_pool.buffer", special_dir, dump_id);
	}
	else{
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/final_avg_pool.buffer", special_dir, dump_id);
	}
	fp = fopen(filepath, "wb");
	n_wrote = fwrite(cpu_final_avg_pool, sizeof(float), final_avg_pool_size, fp);
	fclose(fp);
	free(filepath);
	free(cpu_final_avg_pool);

	/* 7. Fully Connected Output */
	int output_size = dims -> output * batch_size;
	float * cpu_linear_output = (float *) malloc(output_size * sizeof(float));
	cudaMemcpy(cpu_linear_output, activations -> linear_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
	if (is_deriv){
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/fc_output.buffer", special_dir, dump_id);
	}
	else{
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/fc_output.buffer", special_dir, dump_id);
	}
	fp = fopen(filepath, "wb");
	n_wrote = fwrite(cpu_linear_output, sizeof(float), output_size, fp);
	fclose(fp);
	free(filepath);
	free(cpu_linear_output);


	/* 8. Softmax Prediction */
	float * cpu_softmax = (float *) malloc(output_size * sizeof(float));
	if (is_deriv){
		cudaMemcpy(cpu_softmax, trainer -> backprop_buffer -> output_layer_deriv, output_size * sizeof(float), cudaMemcpyDeviceToHost);
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activation_derivs/softmax.buffer", special_dir, dump_id);
	}
	else{
		cudaMemcpy(cpu_softmax, trainer -> forward_buffer -> pred, output_size * sizeof(float), cudaMemcpyDeviceToHost);
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/softmax.buffer", special_dir, dump_id);
	}
	fp = fopen(filepath, "wb");
	n_wrote = fwrite(cpu_softmax, sizeof(float), output_size, fp);
	fclose(fp);
	free(filepath);
	free(cpu_softmax);


	/* 9. Correct Classes */
	if (!is_deriv){
		int * correct_classes_cpu = trainer -> cur_batch -> correct_classes_cpu;
		print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/activations/correct_classes.buffer", special_dir, dump_id);
		fp = fopen(filepath, "wb");
		n_wrote = fwrite(correct_classes_cpu, sizeof(int), batch_size, fp);
		free(filepath);
		fclose(fp);
	}
}

void dump_trainer_meta(int dump_id, Train_ResNet * trainer, const char * special_dir){

	char * filepath = NULL;
	FILE * fp;
	int print_ret;

	print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/trainer_metadata.txt", special_dir, dump_id);
	fp = fopen(filepath, "w");

	// DUMP THE BATCH INFO
	fprintf(fp, "%d\n", trainer -> batch_size);
	fprintf(fp, "%d\n", trainer -> cur_batch -> image_size);
	fprintf(fp, "%d\n", trainer -> cur_batch -> image_dim);
	fprintf(fp, "%d\n", trainer -> cur_batch -> shard_n_images);


	// NOW DO TRAINER METADATA
	fprintf(fp, "%f\n", trainer -> learning_rate);
	fprintf(fp, "%f\n", trainer -> weight_decay);
	fprintf(fp, "%f\n", trainer -> base_mean_decay);
	fprintf(fp, "%f\n", trainer -> base_var_decay);
	fprintf(fp, "%f\n", trainer -> cur_mean_decay);
	fprintf(fp, "%f\n", trainer -> cur_var_decay);
	fprintf(fp, "%f\n", trainer -> eps);
	fprintf(fp, "%d\n", trainer -> n_epochs);
	fprintf(fp, "%d\n", trainer -> cur_dump_id);
	fprintf(fp, "%d\n", trainer -> cur_epoch);

	for (int i = 0; i < trainer -> cur_epoch; i++){
		if (i == 0){
			fprintf(fp, "%f", (trainer -> loss_per_epoch)[i]);
		}
		else{
			fprintf(fp, ",%f", (trainer -> loss_per_epoch)[i]);
		}
	}
	fprintf(fp, "\n");

	for (int i = 0; i < trainer -> cur_epoch; i++){
		if (i == 0){
			fprintf(fp, "%f", (trainer -> accuracy_per_epoch)[i]);
		}
		else{
			fprintf(fp, ",%f", (trainer -> accuracy_per_epoch)[i]);
		}
	}
	fprintf(fp, "\n");

	fclose(fp);
}

void dump_trainer_checkpoint(int dump_id, Train_ResNet * trainer, const char * special_dir){

	char * filepath = NULL;
	FILE * fp;
	int print_ret;

	print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/trainer_checkpoint.txt", special_dir, dump_id);
	fp = fopen(filepath, "w");

	// DUMP THE BATCH INFO
	fprintf(fp, "%d\n", trainer -> cur_batch -> cur_shard_id);
	fprintf(fp, "%d\n", trainer -> cur_batch -> cur_batch_in_shard);

	// NOW DO TRAINER METADATA
	fprintf(fp, "%f\n", trainer -> cur_mean_decay);
	fprintf(fp, "%f\n", trainer -> cur_var_decay);
	fprintf(fp, "%d\n", trainer -> cur_dump_id);
	fprintf(fp, "%d\n", trainer -> cur_epoch);

	fclose(fp);
}

void dump_trainer(int dump_id, Train_ResNet * trainer, const char * special_dir){

	/* DUMP PARAMETERS */
	dump_parameters(dump_id, trainer, special_dir);
	
	/* DUMP FORWARD ACTIVATIONS */
	dump_activations(dump_id, trainer, trainer -> forward_buffer -> activations, false, special_dir);

	/* DUMP BACKPROP ACTIVATION DERIVS */
	dump_activations(dump_id, trainer, trainer -> backprop_buffer -> activation_derivs, true, special_dir);

	/* DUMP TRAINER METADATA */
	dump_trainer_meta(dump_id, trainer, special_dir);

	/* DUMP TRAINER CHECKPOINT */
	dump_trainer_checkpoint(dump_id, trainer, special_dir);

}

/* LOADING MODEL / ALLOCATING MEMORY FOR TRAINER */

// loading from a checkpoint that was dumped
// ASSUME EVERYTHING IS THE SAME AS IN THIS FILE EXCEPT: cur_shard_id, cur_batch_in_shard, cur_mean_decay, cur_var_decay, cur_dump_id, cur_epoch
void overwrite_trainer_hyperparams(Train_ResNet * trainer, int dump_id, const char * special_dir){

	// open the metadata file with hyper params and location of training sequence
	char * filepath = NULL;
	FILE * fp;
	int print_ret;

	print_ret = asprintf(&filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/trainer_checkpoint.txt", special_dir, dump_id);
	fp = fopen(filepath, "r");

	// read the metadata file
	char * line;
	size_t len = 0;
	ssize_t n_read;

	// ASSUME THAT THE ORDERING OF LINES IS FIXED ACCORING TO "dump_trainer_checkpoint"

	// LOAD THE BATCH INFO
	n_read = getline(&line, &len, fp);
	trainer -> cur_batch -> cur_shard_id = atoi(line);
	n_read = getline(&line, &len, fp);
	trainer -> cur_batch -> cur_batch_in_shard = atoi(line);

	// LOAD OPTIMIZATION INFO
	n_read = getline(&line, &len, fp);
	trainer -> cur_mean_decay = atof(line);
	n_read = getline(&line, &len, fp);
	trainer -> cur_var_decay = atof(line);
	
	// LOAD SEQUENCE INFO
	n_read = getline(&line, &len, fp);
	trainer -> cur_dump_id = atoi(line);
	n_read = getline(&line, &len, fp);
	trainer -> cur_epoch = atoi(line);

	trainer -> init_loaded = 1;

	free(line);
	fclose(fp);
}


// LOADING THE MODEL PARAMS AND OPTIMIZATION STATES FROM CHECKPOINT
void overwrite_model_params(Train_ResNet * trainer, int dump_id, const char * special_dir){

	Params * model_params = trainer -> model -> params;
	float ** model_params_locations = model_params -> locations;
	int * param_sizes = model_params -> sizes;
	int n_locations = model_params -> n_locations;
	
	// locations of optimization states
	Params * prev_grad_means = trainer -> backprop_buffer -> prev_means;
	float ** prev_grad_means_locations = prev_grad_means -> locations;
	Params * prev_grad_vars = trainer -> backprop_buffer -> prev_vars;
	float ** prev_grad_vars_locations = prev_grad_vars -> locations;

	size_t param_size;
	float *model_location, * mean_location, * var_location;

	float * cpu_param_buff;
	FILE * fp;

	char * model_params_filepath;
	char * means_filepath;
	char * vars_filepath;

	int n_read, print_ret;
	for (int i = n_locations - 1; i >= 0; i--){
		param_size = param_sizes[i];
		cpu_param_buff = (float *) malloc(param_size * sizeof(float));

		print_ret = asprintf(&model_params_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/model_params/%03d.buffer", special_dir, dump_id, i);
		fp = fopen(model_params_filepath, "rb");
		n_read = fread(cpu_param_buff, sizeof(float), (size_t) param_size, fp);
		model_location = model_params_locations[i];
		cudaMemcpy(model_location, cpu_param_buff, param_size * sizeof(float), cudaMemcpyHostToDevice);
		fclose(fp);
		free(model_params_filepath);

		print_ret = asprintf(&means_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/means/%03d.buffer", special_dir, dump_id, i);
		fp = fopen(means_filepath, "rb");
		n_read = fread(cpu_param_buff, sizeof(float), (size_t) param_size, fp);
		mean_location = prev_grad_means_locations[i];
		cudaMemcpy(mean_location, cpu_param_buff, param_size * sizeof(float), cudaMemcpyHostToDevice);
		fclose(fp);
		free(means_filepath);

		print_ret = asprintf(&vars_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/%08d/vars/%03d.buffer", special_dir, dump_id, i);
		fp = fopen(vars_filepath, "rb");
		n_read = fread(cpu_param_buff, sizeof(float), (size_t) param_size, fp);
		var_location = prev_grad_vars_locations[i];
		cudaMemcpy(var_location, cpu_param_buff, param_size * sizeof(float), cudaMemcpyHostToDevice);
		fclose(fp);
		free(vars_filepath);

		free(cpu_param_buff);
	}
}


// takes in pointers to GPU memory
void check_errors(Train_ResNet * trainer, int param_size, float * model_location, float * grad_location, float * mean_location, float * var_location, int location_ind){

	float * cpu_param_model = (float *) malloc(param_size * sizeof(float));
	cudaMemcpy(cpu_param_model, model_location, param_size * sizeof(float), cudaMemcpyDeviceToHost);

	float * cpu_param_grad = (float *) malloc(param_size * sizeof(float));
	cudaMemcpy(cpu_param_grad, grad_location, param_size * sizeof(float), cudaMemcpyDeviceToHost);

	float * cpu_param_mean = (float *) malloc(param_size * sizeof(float));
	cudaMemcpy(cpu_param_mean, mean_location, param_size * sizeof(float), cudaMemcpyDeviceToHost);

	float * cpu_param_var = (float *) malloc(param_size * sizeof(float));
	cudaMemcpy(cpu_param_var, var_location, param_size * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < param_size; i++){
		if ((isnan(cpu_param_model[i])) || (isnan(cpu_param_grad[i])) || (isnan(cpu_param_mean[i])) || (isnan(cpu_param_var[i]))
				|| (isinf(cpu_param_model[i])) || (isinf(cpu_param_grad[i])) || (isinf(cpu_param_mean[i])) || (isinf(cpu_param_var[i]))){
			printf("ERROR: nan or inf found at location: %d\n", location_ind);
			printf("Dumping data to id=99999999 and exiting...\n");
			dump_trainer(99999999, trainer, trainer -> dump_dir);
			exit(1);
		}
	}

	free(cpu_param_model);
	free(cpu_param_grad);
	free(cpu_param_mean);
	free(cpu_param_var);
}

// doing ADAM optimizer
void update_parameters(Train_ResNet * trainer){
	
	size_t batch_size = (size_t) trainer -> batch_size;
	size_t image_size = (size_t) trainer -> cur_batch -> image_size;

	float learning_rate = trainer -> learning_rate;
	float weight_decay = trainer -> weight_decay;
	float base_mean_decay = trainer -> base_mean_decay;
	float base_var_decay = trainer -> base_var_decay;
	// update the running decays here...
	float cur_mean_decay = trainer -> cur_mean_decay * base_mean_decay;
	float cur_var_decay = trainer -> cur_var_decay * base_var_decay;
	float eps = trainer -> eps;

	Params * model_params = trainer -> model -> params;
	float ** model_params_locations = model_params -> locations;
	int * param_sizes = model_params -> sizes;
	int n_locations = model_params -> n_locations;

	// values calculated from backprop, will reset these before returning
	Params * current_gradients = trainer -> backprop_buffer -> param_derivs;
	float ** current_gradient_locations = current_gradients -> locations;
	
	// running history values that the optimizer needs, will update these before returning
	Params * prev_grad_means = trainer -> backprop_buffer -> prev_means;
	float ** prev_grad_means_locations = prev_grad_means -> locations;
	Params * prev_grad_vars = trainer -> backprop_buffer -> prev_vars;
	float ** prev_grad_vars_locations = prev_grad_vars -> locations;

	int param_size;
	float *model_location, *grad_location, * mean_location, * var_location;

	/* DUMP THE STATE OF TRAINING PROCESS! */
	// dumping every 10 batches, before update
	// also dump when nan or inf occurs (data dumped to id=99999999)
	int cur_dump_id = trainer -> cur_dump_id;

	if (cur_dump_id % 1000 == 0){
		printf("DUMPING TRAINER @ ID: %d!\n\n", cur_dump_id);
		dump_trainer(cur_dump_id, trainer, trainer -> dump_dir);
	}
	
	for (int i = n_locations - 1; i >= 0; i--){
		param_size = param_sizes[i];
		model_location = model_params_locations[i];
		grad_location = current_gradient_locations[i];
		mean_location = prev_grad_means_locations[i];
		var_location = prev_grad_vars_locations[i];

		check_errors(trainer, param_size, model_location, grad_location, mean_location, var_location, i);

		dim3 gridDimUpdate(ceil((float) param_size / MAX_THREAD_PER_BLOCK));
		dim3 blockDimUpdate(MAX_THREAD_PER_BLOCK);
		updateMeans <<< gridDimUpdate, blockDimUpdate >>> (param_size, grad_location, model_location, base_mean_decay, weight_decay, mean_location, i);
		updateVars <<< gridDimUpdate, blockDimUpdate >>> (param_size, grad_location, model_location, base_var_decay, weight_decay, var_location, i);
		updateParams <<< gridDimUpdate, blockDimUpdate >>> (param_size, model_location, mean_location, var_location, learning_rate, weight_decay, cur_mean_decay, cur_var_decay, eps, i);
	}


	

	/* RESET ALL VALUES TO 0 FOR NEXT PASS THROUGH BACKPROP */
	for (int i = 0; i < n_locations; i++){
		param_size = param_sizes[i];
		grad_location = current_gradient_locations[i];
		cudaMemset(grad_location, 0, param_size * sizeof(float));
		// reset_forward_buffer(trainer);
		// reset_backward_buffer(trainer);
	}

	// reset images and classes before next cudaMemcpy
	cudaMemset(trainer -> cur_batch -> images, 0, batch_size * image_size * sizeof(float));
	cudaMemset(trainer -> cur_batch -> correct_classes, 0, batch_size * sizeof(int));

	// change the current mean and var decay...
	trainer -> cur_mean_decay = cur_mean_decay;
	trainer -> cur_var_decay = cur_var_decay;
}


void testTranspose(){

	int orig_rows = 2048;
	int orig_cols = 1000;

	float * origMat_host = (float *) malloc(orig_rows * orig_cols * sizeof(float));
	for (int i = 0; i < orig_rows; i++){
		for (int j = 0; j < orig_cols; j++){
			origMat_host[i * orig_cols + j] = ((float)(rand())/(float)(RAND_MAX));
		}
	}

	float * devOrigMat;
	cudaMalloc(&devOrigMat, orig_rows * orig_cols * sizeof(float));
	cudaMemcpy(devOrigMat, origMat_host, orig_rows * orig_cols * sizeof(float), cudaMemcpyHostToDevice);

	float * devTrans;
	cudaMalloc(&devTrans, orig_cols * orig_rows * sizeof(float));

	dim3 gridDimTranspose(ceil((float) orig_rows / TILE_WIDTH), ceil((float) orig_cols / TILE_WIDTH));
	dim3 blockDimTranspose(TILE_WIDTH, TILE_WIDTH);
	transpose <<< gridDimTranspose, blockDimTranspose >>> (devOrigMat, orig_rows, orig_cols, devTrans);

	float *matTrans_host = (float *) malloc(orig_cols * orig_rows * sizeof(float));

	cudaMemcpy(matTrans_host, devTrans, orig_cols * orig_rows * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(devOrigMat);
	cudaFree(devTrans);

	for (int i = 0; i < orig_cols; i++){
		for (int j = 0; j < orig_rows; j++){
			if (origMat_host[j * orig_cols + i] != matTrans_host[i * orig_rows + j]){
				printf("TRANSPOSE ERROR: @ original row: %d, original col: %d\n", j, i);
			}
		}
	}

	free(origMat_host);
	free(matTrans_host);
}


void testMatMul(){

	int m = 32;
	int k = 2048;
	int n = 1000;

	float * A_host = (float *) malloc(m * k * sizeof(float));
	float * B_host = (float *) malloc(k * n * sizeof(float));
	float * C_host = (float *) calloc(m * n, sizeof(float));

	for (int i = 0; i < m; i++){
		for (int j = 0; j < k; j++){
			A_host[i * k + j] = ((float)(rand())/(float)(RAND_MAX)) * (((int)(rand()) % 2) * 2 - 1);
		}
	}

	for (int i = 0; i < k; i++){
		for (int j = 0; j < n; j++){
			B_host[i * n + j] = ((float)(rand())/(float)(RAND_MAX))  * (((int)(rand()) % 2) * 2 - 1);
		}
	}

	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			for (int c = 0; c < k; c++){
				C_host[i * n + j] += A_host[i * k + c] * B_host[c * n + j];
			}
		}
	}

	float * A_dev, *B_dev, *C_dev;
	cudaMalloc(&A_dev, m * k * sizeof(float));
	cudaMemcpy(A_dev, A_host, m * k * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&B_dev, k * n * sizeof(float));
	cudaMemcpy(B_dev, B_host, k * n * sizeof(float), cudaMemcpyHostToDevice);


	cudaMalloc(&C_dev, m * n * sizeof(float));


	dim3 gridDimMatMul(ceil((float) m / TILE_WIDTH), ceil((float) n / TILE_WIDTH));
	dim3 blockDimMatMul(TILE_WIDTH, TILE_WIDTH);

	matMul <<< gridDimMatMul, blockDimMatMul >>> (A_dev, B_dev, m, k, n, C_dev);

	float * C_kern_result = (float *) malloc(m * n * sizeof(float));

	float eps = 0.00001;

	cudaMemcpy(C_kern_result, C_dev, m * n * sizeof(float), cudaMemcpyDeviceToHost);

	float cpu_val, gpu_val;
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			gpu_val = C_kern_result[i * n + j];
			cpu_val = C_host[i * n + j];
			if ( (gpu_val < (cpu_val - eps)) || (gpu_val > (cpu_val + eps)) ){
				printf("MatMul ERROR: @ row: %d, col: %d\n", j, i);
				printf("CPU Result: %f\n", cpu_val);
				printf("GPU Result: %f\n\n", gpu_val);
			}
		}
	}

	cudaFree(A_dev);
	cudaFree(B_dev);
	cudaFree(C_dev);

	free(A_host);
	free(B_host);
	free(C_host);
	free(C_kern_result);

}

void testConvolution(int in_spatial_dim, int kern_dim, int in_filters, int out_filters,  int stride, int batch_size, 
																float * input, float * weights, float * output){

	printf("\n\n* TESTING THE CONVOLUTION KERNEL *\n\n");
	/* FIRST DO COMPUTATION ON GPU */

	int out_spatial_dim = in_spatial_dim / stride;
	int out_filters_block = min(MAX_THREAD_PER_BLOCK / batch_size, out_filters);
	int out_filters_grid = max(1, (int) ceil((float) out_filters / (float) out_filters_block));

	printf("Conv Params -- Batch Size: %d, In Spatial: %d, Stride: %d, Kern Dim: %d, In Filters: %d, Out Filters %d\n", batch_size, in_spatial_dim, stride, kern_dim, in_filters, out_filters);
	printf("Launch Params -- Out Filters Block: %d, Out Filters Grid: %d\n", out_filters_block, out_filters_grid);
	dim3 gridDimConv(out_spatial_dim, out_spatial_dim, out_filters_grid);
	dim3 blockDimConv(batch_size, out_filters_block);

	printf("Computing Convolution on GPU...\n");

	doConvolution <<< gridDimConv, blockDimConv>>> (input, weights, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, output);

	cudaDeviceSynchronize();

	float * gpu_output_on_cpu = (float *) malloc(batch_size * out_spatial_dim * out_spatial_dim * out_filters * sizeof(float));

	cudaMemcpy(gpu_output_on_cpu, output, batch_size * out_spatial_dim * out_spatial_dim * out_filters * sizeof(float), cudaMemcpyDeviceToHost);

	/* DO COMPUTATION ON CPU */
	
	// COPYING VALUES FROM GPU TO THE CPU...
	float * input_cpu = (float *) malloc(batch_size * in_spatial_dim * in_spatial_dim * in_filters * sizeof(float));
	cudaMemcpy(input_cpu, input, batch_size * in_spatial_dim * in_spatial_dim * in_filters * sizeof(float), cudaMemcpyDeviceToHost);

	float * weights_cpu = (float *) malloc(kern_dim * kern_dim * in_filters * out_filters * sizeof(float));
	cudaMemcpy(weights_cpu, weights, kern_dim * kern_dim * in_filters * out_filters * sizeof(float), cudaMemcpyDeviceToHost);

	float * cpu_output = (float *) malloc(batch_size * out_spatial_dim * out_spatial_dim * out_filters * sizeof(float));

	printf("Computing Convolution on CPU...\n");

	int output_ind, in_spatial_row_start, in_spatial_col_start, in_spatial_row, in_spatial_col, input_ind, kernel_ind;
	int half_kernel_dim = kern_dim / 2;
	int kernel_size = in_filters * kern_dim * kern_dim;
	float in_spatial_val;
	for (int samp = 0; samp < batch_size; samp++){
		for (int out_filt = 0; out_filt < out_filters; out_filt++){
			for (int out_i = 0; out_i < out_spatial_dim; out_i++){
				for (int out_j = 0; out_j < out_spatial_dim; out_j++){
					output_ind = out_spatial_dim * out_spatial_dim * out_filters * samp + out_spatial_dim * out_filters * out_i + out_filters * out_j + out_filt;
					cpu_output[output_ind] = 0;
					in_spatial_row_start = out_i * stride;
					in_spatial_col_start = out_j * stride;
					for (int in_filt = 0; in_filt < in_filters; in_filt++){
						for (int row_offset = -half_kernel_dim; row_offset <= half_kernel_dim; row_offset++){
							for (int col_offset = -half_kernel_dim; col_offset <= half_kernel_dim; col_offset++){
								// compute spatial value
								in_spatial_row = in_spatial_row_start + row_offset;
								in_spatial_col = in_spatial_col_start + col_offset;
								input_ind = in_spatial_dim * in_spatial_dim * in_filters * samp + in_spatial_dim * in_filters * in_spatial_row + in_filters * in_spatial_col + in_filt;
								kernel_ind = kern_dim * kern_dim * in_filt + kern_dim * (row_offset + half_kernel_dim) + (col_offset + half_kernel_dim);
								if ((in_spatial_row < 0) || (in_spatial_row >= in_spatial_dim) || (in_spatial_col < 0) || (in_spatial_col >= in_spatial_dim)) {
									in_spatial_val = 0;
								}
								else{
									in_spatial_val = input_cpu[input_ind];
								}
								// multiply with conv weight
								// threadIdx.x specifies the output filter id
								// kernel_ind specifies the (x, y, input_channel)
								cpu_output[output_ind] += weights_cpu[out_filt * kernel_size + kernel_ind] * in_spatial_val;
							}
						}
					}
				}
			}
		}
	}

	/* COMPARE RESULTS */
	float gpu_val;
	float cpu_val;
	float eps = 0.0001;
	int err_cnt = 0;
	for (int samp = 0; samp < batch_size; samp++){
		for (int filt = 0; filt < out_filters; filt++){
			for (int i = 0; i < out_spatial_dim; i++){
				for (int j = 0; j < out_spatial_dim; j++){
					output_ind = out_spatial_dim * out_spatial_dim * out_filters * samp + out_spatial_dim * out_filters * i + out_filters * j + filt;
					gpu_val = gpu_output_on_cpu[output_ind];
					cpu_val = cpu_output[output_ind];
					if ( (gpu_val < (cpu_val - eps)) || (gpu_val > (cpu_val + eps)) ){
						printf("ERROR: GPU VALUE DIFFERS FROM CPU\n");
						printf("Occurs at:\nSamp: %d\nFilter: %d\nRow: %d\nCol: %d\n", samp, filt, i, j);
						printf("GPU Value:%f vs. CPU Value:%f\n\n", gpu_val, cpu_val);
						err_cnt++;
					}
					if (err_cnt == 10){
						exit(1);
					}
				}
			}
		}
	}

	/* FREE UP STUFF */

	free(gpu_output_on_cpu);
	free(input_cpu);
	free(weights_cpu);
	free(cpu_output);	

}



int main(int argc, char *argv[]) {

	bool debug = false;

	if (debug){
		testMatMul();
		testTranspose();
		return 0;
	}

	int N_CLASSES = 1000;
	
	// GETTING CLASS METADETA
	char * LABEL_FILENAME = (char *) "/mnt/storage/data/vision/imagenet/2012/id_to_label_mapping.txt";
	char * SYNSET_FILENAME = (char *) "/mnt/storage/data/vision/imagenet/2012/id_to_synset_mapping.txt";
	char * COUNTS_FILENAME = (char *) "/mnt/storage/data/vision/imagenet/2012/id_to_img_count_mapping.txt";
	Class_Metadata * class_metadata = populate_class_info(LABEL_FILENAME, SYNSET_FILENAME, COUNTS_FILENAME, N_CLASSES);
	int total_images = 0;
	for (int i = 0; i < N_CLASSES; i++){
		total_images += (class_metadata -> counts)[i];
	}

	// DEFINING MODEL DIMENSIONS
	int INPUT_DIM = 224;
	int INIT_KERNEL_DIM = 7;
	int INIT_CONV_FILTERS = 64;
	int INIT_CONV_STRIDE = 2;
	int INIT_MAXPOOL_DIM = 3;
	int INIT_MAXPOOL_STRIDE = 2;
	int N_CONV_BLOCKS = 16;
	int * IS_BLOCK_SPATIAL_REDUCTION = (int *) calloc(N_CONV_BLOCKS, sizeof(int));
	// transitions between spatial 56 -> 28 -> 14 -> 7
	// transitions between output depth of 256 -> 512 -> 1024 -> 2048
	int FINAL_DEPTH = 2048;
	IS_BLOCK_SPATIAL_REDUCTION[3] = 1;
	IS_BLOCK_SPATIAL_REDUCTION[7] = 1;
	IS_BLOCK_SPATIAL_REDUCTION[13] = 1;
	Dims * dims = init_dimensions(INPUT_DIM, INIT_KERNEL_DIM, INIT_CONV_FILTERS, INIT_CONV_STRIDE, INIT_MAXPOOL_DIM, INIT_MAXPOOL_STRIDE,
									N_CONV_BLOCKS, IS_BLOCK_SPATIAL_REDUCTION, FINAL_DEPTH, N_CLASSES);


	// declaring curandGenerator
	curandGenerator_t gen;
	// INITIALIZING RANDOM NUMBER GENERATOR USED TO INIT WEIGHTS
	curandStatus_t status_create = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandStatus_t status_set_seed = curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

	// INITIALIZING MODEL
	ResNet * model = init_resnet(dims, &gen);

	// INITIALIZING TRAINING

	// Batch Structure (will be modified every iteration of every epoch)
	
	// given when we generated shards...
	int SHARD_N_IMAGES = 32768;

	int BATCH_SIZE = 64;
	// dimensions of INPUT_DIM X INPUT_DIM x 3 color channels
	int IMAGE_SIZE = INPUT_DIM * INPUT_DIM * 3;
	Batch * batch = init_general_batch(BATCH_SIZE, IMAGE_SIZE, INPUT_DIM, SHARD_N_IMAGES);


	// General Training Structure (holds hyperparameters and pointers to structs which have network values)
	float LEARNING_RATE = 0.001;
	float WEIGHT_DECAY = 0;
	float MEAN_DECAY = 0.9;
	float VAR_DECAY = 0.999;
	float EPS = 0.0000001;
	float N_EPOCHS = 40;

	// INIT Cudnn
	cudnnHandle_t cudnn;
	cudnnStatus_t cudnn_status = cudnnCreate(&cudnn);
	//printf("Create Status: %s\n\n", cudnnGetErrorString(cudnn_status));


	const char * MY_DUMP_DIR = "cudnn_test";

	Train_ResNet * trainer = init_trainer(model, batch, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, MEAN_DECAY, VAR_DECAY, EPS, N_EPOCHS, &cudnn, MY_DUMP_DIR);

	// OVERRIDE IF LOADING WEIGHTS
	int LOAD_FROM_DUMP_ID = 88000;

	
	
	if (LOAD_FROM_DUMP_ID != -1){
		overwrite_trainer_hyperparams(trainer, LOAD_FROM_DUMP_ID, MY_DUMP_DIR);
		overwrite_model_params(trainer, LOAD_FROM_DUMP_ID, MY_DUMP_DIR);
	}

	/* PERFORM TRAINING */


	int iterations_per_epoch = ceil((float) total_images / BATCH_SIZE);

	float *pred;
	int * correct;
	float epoch_n_wrong, batch_n_wrong;
	float epoch_loss, batch_loss, avg_batch_loss, epoch_accuracy, batch_accuracy, val_pred_correct;
	float total_images_per_epoch = BATCH_SIZE * iterations_per_epoch;

	int PRINT_FREQ = 1;

	cudaError_t status;

	char * loss_filepath = NULL;
	int print_ret;
	print_ret = asprintf(&loss_filepath, "/mnt/storage/data/vision/imagenet/training_dumps/%s/avg_loss_log.txt", MY_DUMP_DIR);
	FILE * loss_file = fopen(loss_filepath, "w");

	// if this was loaded from checkpoint
	int cur_epoch = trainer -> cur_epoch;
	int cur_iter_in_epoch = (trainer -> cur_dump_id + 1) % iterations_per_epoch;

	for (int epoch = cur_epoch; epoch < N_EPOCHS; epoch++){
		epoch_loss = 0;
		epoch_n_wrong = 0;
		for (int iter = cur_iter_in_epoch; iter < iterations_per_epoch; iter++){

			//printf("************\n");

			/* LOAD NEW BATCH */
			//printf("Loading Batch...: %d\n", iter);
			// values go into trainer -> cur_batch -> [images_cpu|images_float_cpu|images|correct_classes_cpu|correct_classes]
			load_new_batch(trainer, class_metadata, trainer -> cur_batch);

			// cudaDeviceSynchronize();
			// status = cudaGetLastError();
			//printf("Status after loading batch: %s\n\n", cudaGetErrorString(status));
			

			/* DO FORWARD PROP */
			// final predictions go into trainer -> forward_buffer -> [pred|pred_cpu|prediction_label]
			//printf("Making Predictions...\n");
			forward_pass(trainer);

			//cudaDeviceSynchronize();
			//status = cudaGetLastError();
			//printf("Status after forward pass: %s\n\n", cudaGetErrorString(status));
			

			/* RECORD LOSS AND ACCURACY */
			if (iter % 1 == 0){
				cudaDeviceSynchronize();

				// dimensions of pred: (BATCH_SIZE, N_CLASSES)
				pred = trainer -> forward_buffer -> pred_cpu;
				correct = trainer -> cur_batch -> correct_classes_cpu;
				
				// loss
				batch_loss = 0;
				for (int s = 0; s < BATCH_SIZE; s++){
					batch_loss += -1 * logf(pred[s * N_CLASSES + correct[s]]);
				}
				avg_batch_loss = batch_loss / BATCH_SIZE;
				epoch_loss += batch_loss;

				// accuracy
				batch_n_wrong = 0;
				for (int s = 0; s < BATCH_SIZE; s++){
					val_pred_correct = pred[s * N_CLASSES + correct[s]];
					for (int c = 0; c < N_CLASSES; c++){
						if ((c != correct[s]) && (pred[s * N_CLASSES + c] >= val_pred_correct)){
							batch_n_wrong++;
							break;
						}
					}
				}
				epoch_n_wrong += batch_n_wrong;
				batch_accuracy = 100 * ((float) BATCH_SIZE - batch_n_wrong) / ((float) BATCH_SIZE);


				if (iter % PRINT_FREQ == 0){
					printf("Epoch: %d, Batch: %d ----- Avg. Loss: %.4f, Accuracy: %.2f%%\n", epoch, iter, avg_batch_loss, batch_accuracy);
				}
				fprintf(loss_file, "%.4f\n", avg_batch_loss);
				fflush(loss_file);
			}


			/* DO BACKPROP */
			//printf("Backprop to Compute Derivs...\n");
			backwards_pass(trainer);

			//cudaDeviceSynchronize();
			//status = cudaGetLastError();
			//printf("Status after backwards pass: %s\n\n", cudaGetErrorString(status));

			/* OPTIMIZE WEIGHTS */
			//printf("Applying Optimizer to Update Params...\n\n");
			update_parameters(trainer);

			/*cudaDeviceSynchronize();
			status = cudaGetLastError();
			if (status != 0){
				printf("Status after iter: %s\n\n", cudaGetErrorString(status));
			}
			*/

		}

		(trainer -> loss_per_epoch)[epoch] = epoch_loss;
		epoch_accuracy = (total_images_per_epoch - epoch_n_wrong) / total_images_per_epoch;
		(trainer -> accuracy_per_epoch)[epoch] = epoch_accuracy;
		printf("\nEpoch %d, Total Loss: %f\n", epoch, epoch_loss);
		printf("Epoch %d, Total Accuracy: %f\n\n", epoch, epoch_accuracy);

		// reset batch to start from beginning of dataset
		trainer -> cur_batch -> cur_shard_id = -1;
		trainer -> cur_batch -> cur_batch_in_shard = -1;

		trainer -> cur_epoch += 1;
		cur_iter_in_epoch = 0;

	}

	// DO A FINAL DUMP AFTER MODEL FINISHES (stored at 77777777)
	int FINAL_DUMP_ID = 77777777;
	dump_trainer(FINAL_DUMP_ID, trainer, trainer -> dump_dir);

	fclose(loss_file);

}
