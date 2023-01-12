#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <stdint.h>

#include "resnet.h"

#define SM_COUNT 82
#define WARP_PER_SM 4
#define THREAD_PER_WARP 32
#define MAX_THREAD_PER_BLOCK 1024
#define TILE_WIDTH 32
#define BLOCK_ROWS 8
#define CUDA_BATCH_SIZE 32
#define MAX_SHARED_MEMORY 48000
#define MAX_SHARED_MEM_FLOATS 12000

__global__ void sample_gaussian(int size, float *X, float mean, float var) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size){
		return;
	}
	if (var == 0){
		return mean;
	}
	float x = (float)rand() / RAND_MAX;
  	float y = (float)rand() / RAND_MAX;
  	float z = sqrtf(-2 * logf(x)) * cosf(2 * M_PI * y);
  	float std = sqrtf(var);
  	float val = std * z + mean;
  	X[i] = val;
}

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
	__shared__ float M_tile[TILE_WIDTH][TILE_WIDTH + 1];
	__shared__ float N_tile[TILE_WIDTH][TILE_WIDTH + 1];

	int block_x = blockIdx.x;
	int block_y = blockIdx.y;

	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;

	int row_ind = block_y * TILE_WIDTH + thread_y;
	int col_ind = block_x * TILE_WIDTH + thread_x;

	if (row_ind >= m || col_ind >= n){
		return;
	}

	float val = 0;
	for (int phase = 0; phase < ceil((float) k / float(TILE_WIDTH)); phase++) {
		if (phase * TILE_WIDTH + thread_x < k){
			M_tile[thread_y][thread_x] = M[row_ind * k + phase * TILE_WIDTH + thread_x];
		}
		else{
			M_tile[thread_y][thread_x] = 0;
		}
		if (phase * TILE_WIDTH + thread_y < k){
			N_tile[thread_y][thread_x] = N[(phase * TILE_WIDTH + thread_y) * k + col_ind];
		}
		else{
			N_tile[thread_y][thread_x] = 0;
		}

		__syncthreads();

		for (int t = 0; t < TILE_WIDTH; t++){
			val += M_tile[thread_y][t] * N_tile[t][thread_x];
		}
		__syncthreads();
	}
	out[row_ind * n + col_ind] = val;
}

// grid has dim (ROWS / TILE_WIDTH, COLS/TILE_WIDTH)
// each BLOCK has dim (TILE_WIDTH x BLOCK_ROWS) = # of threads
__global__ void transpose(const float *in, int rows, int cols, float * out) {
  __shared__ float tile[TILE_WIDTH][TILE_WIDTH + 1];

  int col_ind = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int row_ind = blockIdx.y * TILE_WIDTH + threadIdx.y;
  
  if (col_ind >= cols || row_ind >= rows){
  	return;
  }

  
  // each thread needs to load TILE_WIDTH / BLOCK_ROWS values
  int row_boundary = min(TILE_WIDTH, rows - row_ind);
  for (int j = 0; j < row_boundary; j += BLOCK_ROWS){
     tile[threadIdx.y+j][threadIdx.x] = in[(row_ind+j)*cols + col_ind];
  }

  __syncthreads();

  int col_boundary = min(TILE_WIDTH, cols - col_ind);
  for (int j = 0; j < col_boundary; j += BLOCK_ROWS){
     out[(col_ind+j)*rows + row_ind] = tile[threadIdx.x][threadIdx.y + j];
  }
}


// TRIED TO OPTIMIZE: MAKE SURE THIS WORKS WITH SHARED MEM AND LAUNCH SPECS (FORGOT HOW I DID IT...)
// 48KB is maximum value for shared memory, passed into this kernel as third param <<< gridDim, blockDim, SHARED_MEM_BYTES >>>
// launch grid dimensions as (OUT_SPATIAL_DIM, OUT_SPATIAL_DIM, OUT_FILTER_CHUNK) blocks, and launch with block dim as (out_filt_rows_shared, sub_batch) threads
// thus 12k floats is max for shared memory per block
// first get as many output filter weights in shared memory as possible, but have separate blocks working on different chunks (OUT_FILTER_CHUNK * out_filt_rows_shared = out_filt)
// then stream samples in batch to compute output value for each sample and output filter. Eac sub_batch will have batch_size / dim(sub_batch) samples to go over
__global__ void doConvolution(const float * input, const float * weights, const float * biases, int spatial_dim, int kern_dim, int in_filters, int out_filters, int stride, int batch_size, float * out){

	// will consist of (shared_out_filt_rows X (kern_dim^2 * in_filt) conv_weight matrix
	extern __shared__ float shared_mem[];


	// (Calling "Kernel" a 3-D obj of weights where there is 2-D conv filter for each input channel)
	int kernel_size = (kern_dim * kern_dim * in_filt);

	int spatial_row_start = stride * blockIdx.x;
	int spatial_col_start = stride * blockIdx.y;
	int out_spatial_dim = spatial_dim / stride;

	int output_filter_off = threadIdx.x;
	int half_kernel_dim = kern_dim / 2;
	int out_filter_start, out_filter_id, spatial_row, spatial_col;
	float out_val, spatial_val;
	out_filter_id = blockIdx.z * blockDim.x + threadIdx.x;
	if (out_filer_id >= out_filters){
		return;
	}

	for (int j = 0; j < kernel_size; j++){
		shared_mem[threadIdx.x * kernel_size + j] = weights[out_filter_id * kernel_size + j];
	}

	int samp_per_subbatch = ceil((float) batch_size / blockDim.y);
	int samp_start = samp_per_subbatch * threadIdx.y;
	int samp_end = min(batch_size, samp_start + samp_per_subbatch);
	// probably could be more efficient by reducing number of output filters in shared mem, and adding tiled spatial....
	for (int sample_ind = samp_start; sample_ind < samp_end; sample_ind++){
		out_val = 0;
		for (int row_offset = -half_kernel_dim; row_offset <= half_kernel_dim; row_offset++){
			for (int col_offset = -half_kernel_dim; col_offset <= half_kernel_dim; col_offset++){
				for (int channel = 0; channel < in_filters; channel++){
						
					// compute spatial value
					spatial_row = spatial_row_start + row_offset;
					spatial_col = spatial_col_start + col_offset;
					kernel_ind = kern_dim * in_filters * (row_offset + half_kernel_dim) + in_filters * (col_offset + half_kernel_dim) + channel;
					if ((spatial_row < 0) || (spatial_row >= spatial_dim) || (spatial_col < 0) || (spatial_col >= spatial_dim)) {
						spatial_val = 0;
					}
					else{
						spatial_val = input[spatial_dim * spatial_dim * in_filters * sample_ind + spatial_dim * in_filters * spatial_row + in_filters * spatial_col + channel];
					}

					// multiply with conv weight
					// threadIdx.x specifies the output filter id
					// kernel_ind specifies the (x, y, input_channel)
					out_val += shared_mem[threadIdx.x * kernel_size + kernel_ind] * spatial_val;
				}
			}
		}
		out[out_spatial_dim * out_spatial_dim * out_filters * sample_ind + out_spatial_dim * out_filters * blockIdx.x + out_filters * blockIdx.y + out_filter_id] = out_val + biases[out_filter_id];
	}
}

// FOR NOW KEEP NAIVE (UN-OPTIMIZED)...
// not bothering with shared memory for now...

// Independent over (input filter, input_x, input_y, sample)
// could use shared memory over conv weights...
// Launch with gridDim (spatial_dim, spatial_dim, input_filters) and blockDim (batch_size)
// Can parallelize further with reductions, if want to optimize
__global__ void convolutionDerivInput(const float * input, const float * weights, const float * out_deriv, int spatial_dim, int kern_dim, int in_filters, int out_filters, int stride, int batch_size, bool toAdd,
											float * input_deriv){

	int spatial_row = blockIdx.x;
	int spatial_col = blockIdx.y;
	int in_filter_id = blockIdx.z;
	int sample_ind = threadIdx.x;
	// shouldn't need to check based on launch specs, but will anyways...
	if ((spatial_row >= spatial_dim) || (spatial_col >= spatial_dim) || (in_filter_id >= in_filters) || (sample_ind >= batch_size)){
		return;
	}

	int out_spatial_dim = spatial_dim / stride;
	int half_kernel_dim = kern_dim / 2;
	int out_spatial_row_start = spatial_row / stride;
	int out_spatial_col_start = spatial_col / stride;
	int kern_ind, out_spatial_ind, kern_row_ind, kern_col_ind;
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
				kern_ind = kern_dim * in_filters * kern_row_ind + in_filters * kern_col_ind + in_filter_id;
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
// Launch with gridDim (input_filters, output_filters) and blockDim (kern_dim, kern_dim)
__global__ void convolutionDerivWeights(const float * input, const float * weights, const float * out_deriv, int spatial_dim, int kern_dim, int in_filters, int out_filters, int stride, int batch_size,
											float * weight_deriv){

	int in_filter_id = blockIdx.x;
	int out_filter_id = blockIdx.y;

	int kern_row = threadIdx.x;
	int kern_col = threadIdx.y;

	// shouldn't need to check based on launch specs, but will anyways...
	if ((in_filter_id >= in_filters) || (out_filter_id >= out_filters) || (kern_row >= kern_dim) || (kern_col >= kern_dim)){
		return;
	}

	int kern_ind = kern_dim * in_filters * kern_row + in_filters * kern_col + in_filter_id;

	int kernel_size = (kern_dim * kern_dim * in_filters);
	int half_kernel_dim = kern_dim / 2;
	int out_spatial_dim = spatial_dim / stride;
	int in_spatial_row, in_spatial_col, in_spatial_ind, out_spatial_ind;
	float out_spatial_val_deriv;
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


// FOR NOW KEEP NAIVE (UN-OPTIMIZED)...
// not bothering with shared memory for now...
// Independent over (out_filters)
// Could incorportate into other deriv kernels (weights easily), but separate for semantic clarity
// To optimize can reduce over samples or in_filters...
// Launch with gridDim (output_filter) and blockDim (1)
__global__ void convolutionDerivBiases(const float * input, const float * weights, const float * out_deriv, int spatial_dim, int kern_dim, int in_filters, int out_filters, int stride, int batch_size,
											float * bias_deriv){
	int out_filter_id = blockIdx.x;
	// shouldn't need to check based on launch specs, but will anyways...
	if (out_filer_id >= out_filters){
		return;
	}

	int out_spatial_dim = spatial_dim / stride;
	int out_spatial_ind;
	float total_deriv = 0;
	for (int s = 0; s < batch_size; s++){
		for (int out_row = 0; out_row < out_spatial_dim; out_row++){
			for (int out_col = 0; out_col < out_spatial_dim; out_col++){
				// going from sample, out_row, out_col, out_filter to get index into out_deriv values
				out_spatial_ind = out_spatial_dim * out_spatial_dim * out_filters * s + out_spatial_dim * out_filters * out_row + out_filters * out_col + out_filter_id;
				total_deriv += out_deriv[out_spatial_ind];
			}
		}
	}
	bias_deriv[out_filter_id] = total_deriv;
}


// iterating over each filter separately
// launch with (OUTFILTERS) grid dim and thread dim of 1 (could easily parallelize menas + vars, with reduction, but save for later..)
// could also use shared memory here if want to be faster
// input is the output of convolution
// ASSUME reLU activation function
__global__ void doBatchNormAndActivate(const float * input, const float * gamma, const float * beta,
								int spatial_dim, int filters, int batch_size, float eps, float * means, float * vars, float * normalized_temp, float * normalized, float * activated){

	int filter_id = blockIdx.x;
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
				acitvated[inp_index] = fmaxf(normalized_val, 0); 
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
									const float * out_layer_deriv, float * normalized_temp_deriv, float * gamma_deriv, float * beta_deriv, float * input_deriv){
	
	int filter_id = blockIdx.x;
	// shouldn't happen based on launch spec, but check anyways...
	if (filter_id >= filters){
		return;
	}

	float n_samples = batch_size * spatial_dim * spatial_dim;
	float gamma_val = gamma[filter_id];
	float beta_val = beta[filter_id];
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
				if (activated_val == 0){
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
	float dVar, dMean, partial_var_deriv, norm_temp_deriv_val;
	float filt_var_three_halfs_power = -.5 * powf(var_val + eps, -1.5);
	float filt_var_recip_sqrt = -1 / sqrtf(var_val + eps);
	for (int s = 0; s < batch_size; s++){
		for (int i = 0; i < spatial_dim; i++){
			for (int j = 0; j < spatial_dim; j++){
				index = spatial_dim * spatial_dim * filters * s + spatial_dim * filters * i + filters * j + filter_id;
				norm_temp_deriv_val = normalized_temp_deriv[index];
				dVar += norm_temp_deriv_val * (input[index] - mean_val) * filt_var_three_halfs_power;
				dMean += norm_temp_deriv_val * filt_var_recip_sqrt;
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
				input_deriv[index] = normalized_temp_deriv[index] * filt_var_recip_sqrt + dVar * (2 * (input[index] - mean_val)) / n_samples + dMean / n_samples;
			}
		}
	}
}



// assume grid launch of (SPATIAL_OUT_DIM, SPATIAL_OUT_DIM) and block dim of (FILTERS)
// could parallelize over batches as well, but probably ok. 
// *runs into issues if #filters greater than threads per block
__global__ void doMaxPool(const float * input, int kern_dim, int stride, int batch_size, float * max_inds, float * out){

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
		max_val = -1;
		max_ind = -1;
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
__global__ void maxPoolDeriv(const float *max_inds_populated, const float *out_deriv, int kern_dim, int in_spatial_dim, int stride, int filters, int batch_size, float * max_pool_inp_deriv){

	int out_spatial_dim = spatial_dim / stride;

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
	int filters = blockDim.x;

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



// hardcoded conv kernel for initial 7x7, stride 2, 64 output filter convolutional layer...
// launching (14, 112, BATCH_SIZE) dim blocks where each block has 112/14=8 phases to utilize shared memory. Each block will have dim (64).
// Each block will contribute 16 unique spatial inds * 64 output filters * 32 Batch Size to the output of layer
// each phase loads stride new rows into shared memory, then multiples new spatial shared_mem with conv_weights, accounting for conv weight col permuation 

/* MAY OR MAY NOT WORK... (commented becuase not used...) */

// __global__ void optimized_init_conv(const float * input, const float * weights, float * out){

// 	__shared__ float conv_weights[64][147];
// 	__shared__ float spatial_vals[147];

// 	// index
// 	int output_filter = threadIdx.x;
// 	int sample_ind = blockIdx.z;

// 	// assume weights are in order of outfilter 0: [R_0,0, B_0,0, G_0,0, R_0,1, G_0,1, B_0,1....R_6,6, G_6,6, B_6,6], outfilter 1: [...], ...., outfilter 63: [...]
// 	for (int kernel_ind = 0; kernel_ind < 147; kernel_ind++){
// 		conv_weights[output_filter][kernel_ind] = weights[output_filter * 147 + kernel_ind];
// 	}o

// 	// 2 * vals because stride of 2
// 	int spatial_row_start = (224 / blockDim.x) * blockIdx.x;
// 	int spatial_col_start = 2 * blockIdx.y;
// 	int spatial_row, spatial_col, kernel_ind;
// 	int half_kernel_dim = 3;
// 	for (int row_offset = -half_kernel_dim; row_offset <= half_kernel_dim;  row_offset++){
// 		for (int col_offset = -half_kernel_dim; col_offset <= half_kernel_dim; col_offset++){
// 			for (int channel = 0; channel < 3; channel++){
// 				spatial_row = spatial_row_start + row_offset;
// 				spatial_col = spatial_col_start + col_offset;
// 				kernel_ind = 7 * 3 * (row_offset + half_kernel_dim) + 3 * (col_offset + half_kernel_dim) + channel;
// 				if ((spatial_row < 0) || (spatial_row >= 224) || (spatial_col < 0) || (spatial_col >= 224)) {
// 					spatial_vals[kernel_ind] = 0;
// 				}
// 				else{
// 					spatial_vals[kernel_ind] = input[224 * 224 * 3 * sample_ind + 224 * 3 * spatial_row + 3 * spatial_col + channel];
// 				}
// 			}
// 		}
// 	}

// 	__syncthreads();

// 	float val = 0;
// 	int circular_row = 0;
// 	int out_spatial_row = (112 / blockDim.x) * blockIdx.x;
// 	int out_spatial_col = blockIdx.y;
// 	int new_top_row = 0;
// 	for (int phase = 0; phase < 8; phase++){

// 		// compute matrix mult to get (output_filt x batch_size) result. this is for a single receptive field across depth and batches
// 		// iterative over phases to get multiple receptive fields and exploit spatial locality
// 		val = 0;
// 		for (int kern_row = 0; kern_row < 7; kern_row++){
// 			for (int kern_col = 0; kern_col < 7; kern_col++){
// 				for (int ch = 0; ch < 3; ch++){
// 					circular_row = (kern_row + 2 * phase) % 7;
// 					val += conv_weights[output_filter][7 * 3 * kern_row + 3 * kern_col + ch] * spatial_vals[7 * 3 * circular_row + 3 * kern_col + ch];
// 				}
// 			}
// 		}

// 		out[112 * 112 * 64 * sample_ind + 112 * 64 * out_spatial_row + 64 * out_spatial_col + output_filter] = val;

// 		__syncthreads();

// 		int row_to_replace, replace_ind;
// 		for (int i = 1; i <= 2; i++){
// 			row_to_replace = (2 * phase) + i % 7;
// 			spatial_row = spatial_row_start + half_kernel_dim + 2 * phase + i; 
// 			for (int col_offset = -half_kernel_dim; col_offset <= half_kernel_dim; col_offset++){
// 				for (int channel = 0; channel < 3; channel++){
// 					spatial_col = spatial_col_start + col_offset;
// 					replace_ind = 7 * 3 * row_to_replace + 3 * (col_offset + half_kernel_dim) + channel;
// 					if ((spatial_row < 0) || (spatial_row >= 224) || (spatial_col < 0) || (spatial_col >= 224)) {
// 						spatial_vals[replace_ind][sample_ind] = 0;
// 					}
// 					else{
// 						spatial_vals[replace_ind][sample_ind] = input[224 * 224 * 3 * sample_ind + 224 * 3 * spatial_row + 3 * spatial_col + channel];
// 					}
// 				}
// 			}
// 		}
// 		out_spatial_row++;

// 		__syncthreads();
// 	}
// }



// assume pass in 1-D block with batch size blocks and 1 thread per block
// could exploit more parallelism here but shouldnt be bottleneck for now...
// assume X is a matrix where # rows = batch size and # columns = output dim
__global__ void softMax(const float * X, int batch_size, int output_len, float * out){
  int i = blockIdx.x;
  if (i < batch_size){
    float sum = 0;
    for (int j = 0; j < output_len; j++){
      sum += __expf(X[i * output_len + j]);
    }
    for (int j = 0; j < output_len; j++){
      out[i * output_len + j] = __expf(X[i * output_len + j]) / sum;
    }
  }
}

// launch with gridDim = (batch_size), blockDim = (1)
__global__ void crossEntropyDeriv(float * output_deriv, const float * correct_classes, int output_dim, int batch_size){
	int i = blockIdx.x;
	if (i < batch_size){
		output_deriv[i * output_dim + correct_classes[i]] -= 1;
	}
}

// assume large 1-D launch
__global__ void updateMeans(int size, const float * gradients, float base_mean_decay, float * prev_means){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= size){
		return;
	}
	prev_means[i] = base_mean_decay * prev_means[i] + (1 - base_mean_decay) * gradients[i];
}

// assume large 1-D launch
__global__ void updateVars(int size, const float * gradients, float base_var_decay, float * prev_vars){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= size){
		return;
	}
	float grad = gradients[i];
	prev_vars[i] = base_var_decay * prev_vars[i] + (1 - base_var_decay) * grad * grad;
}

// assume large 1-D launch
__global__ void updateParams(int size, float * model_params, const float * means, const float * vars, float alpha_t, float eps){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= size){
		return;
	}
	model_params[i] = model_params[i] - alpha_t * means[i] / (sqrtf(vars[i]) + eps);
}

/* INITIALIZE CORE MODEL STRUCTURES */

Dims * init_dimensions(int input, int init_kernel_dim, int init_conv_filters, int init_conv_stride, int init_maxpool_dim, int init_maxpool_stride, 
							int n_conv_blocks, int * is_block_spatial_reduction, int final_depth, int output){
	Dims * dims = malloc(sizeof(Dims));
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
}

BatchNorm * init_batch_norm(int spatial_dim, int depth, bool is_zero){
	BatchNorm * batch_norm = malloc(sizeof(BatchNorm));

	batch_norm -> spatial_dim = spatial_dim;
	batch_norm -> depth = depth;
	batch_norm -> alpha = alpha;

	float * gamma, * beta;

	cudaMalloc(&gamma, depth * sizeof(float));
	cudaMemset(gamma, 0, depth * sizeof(float));
	if (!is_zero){
		cudaMemset(gamma, 1, depth * sizeof(float));
	}

	cudaMalloc(&beta, depth * sizeof(float));
	cudaMemset(beta, 0, depth * sizeof(float));

	batch_norm -> gamma = gamma;
	batch_norm -> beta = beta;

	return batch_norm;

}

ConvBlock * init_conv_block(int incoming_filters, int incoming_spatial_dim, int reduced_depth, int expanded_depth, int stride, float ewma_alpha, bool is_zero){
	ConvBlock * conv_block = malloc(sizeof(ConvBlock));
	conv_block -> incoming_filters = incoming_filters;
	conv_block -> incoming_spatial_dim = incoming_spatial_dim;
	conv_block -> reduced_depth = reduced_depth;
	conv_block -> expanded_depth = expanded_depth;
	conv_block -> stride = stride;

	float * depth_reduction, *spatial, *depth_expansion;
	float * bias_depth_reduction, * bias_spatial, * bias_depth_expansion;
	int depth_reduction_size, spatial_size, depth_expansion_size;
	int bias_depth_reduction_size, bias_spatial_size, bias_depth_expansion_size;
	float depth_reduction_fan_in, spatial_fan_in, depth_expansion_fan_in;

	BatchNorm *norm_depth_reduction, *norm_spatial, *norm_residual_added;

	depth_reduction_size = incoming_filters * reduced_depth;
	depth_reduction_fan_in = incoming_spatial_dim * incoming_spatial_dim * incoming_filters;
	cudaMalloc(&depth_reduction, depth_reduction_size * sizeof(float));
	cudaMemset(depth_reduction, 0, depth_reduction_size * sizeof(float));
	if (!is_zero){
		sample_gaussian <<< SM_COUNT, ceil((float) (depth_reduction_size) / SM_COUNT) >>> (depth_reduction_size, depth_reduction, 0, 2.0 / depth_reduction_fan_in);
	}

	bias_depth_reduction_size = reduced_depth;
	cudaMalloc(&bias_depth_reduction, bias_depth_reduction_size * sizeof(float));
	cudaMemset(bias_depth_reduction, 0, bias_depth_reduction_size * sizeof(float));

	norm_depth_reduction = init_batch_norm(incoming_spatial_dim, reduced_depth, is_zero);


	spatial_size = reduced_depth * reduced_depth * 3 * 3;
	spatial_fan_in = incoming_spatial_dim * incoming_spatial_dim * reduced_depth;
	cudaMalloc(&spatial, spatial_size * sizeof(float));
	cudaMemset(spatial, 0, spatial_size * sizeof(float));
	if (!is_zero){
		sample_gaussian <<< SM_COUNT, ceil((float) (spatial_size) / SM_COUNT) >>> (spatial_size, spatial, 0, 2.0 / spatial_fan_in);
	}

	bias_spatial_size = reduced_depth;
	cudaMalloc(&bias_spatial, bias_spatial_size * sizeof(float));
	cudaMemset(bias_spatial, 0, bias_spatial_size * sizeof(float));

	// the spatial decrease happens at middle 3x3 layer, to the last layer of stride block will receive lower spatial dim input
	if (stride == 2){
		incoming_spatial_dim /= 2;
	}
	norm_spatial = init_batch_norm(incoming_spatial_dim, reduced_depth, is_zero);

	depth_expansion_size = expanded_depth * reduced_depth;
	depth_expansion_fan_in = incoming_spatial_dim * incoming_spatial_dim * reduced_depth;
	cudaMalloc(&depth_expansion, depth_expansion_size * sizeof(float));
	cudaMemset(depth_expansion, 0, depth_expansion_size * sizeof(float));
	if (!is_zero){
		sample_gaussian <<< SM_COUNT, ceil((float) (depth_expansion_size) / SM_COUNT) >>> (depth_expansion_size, depth_expansion, 0, 2.0 / depth_expansion_fan_in);
	}

	bias_depth_expansion_size = expanded_depth;
	cudaMalloc(&bias_depth_expansion, bias_depth_expansion_size * sizeof(float));
	cudaMemset(bias_depth_expansion, 0, bias_depth_expansion_size * sizeof(float));

	conv_block -> depth_reduction = depth_reduction;
	conv_block -> bias_depth_reduction = bias_depth_reduction;
	conv_block -> norm_depth_reduction = norm_depth_reduction;

	conv_block -> spatial = spatial;
	conv_block -> bias_spatial = bias_spatial;
	conv_block -> norm_spatial = norm_spatial;


	conv_block -> depth_expansion = depth_expansion;
	conv_block -> bias_depth_expansion = bias_depth_expansion;

	float * projection, *bias_projection;
	int projection_size;
	if (stride == 2){
		projection_size = 3 * 3 * incoming_filters * expanded_depth;
	}
	else{
		projection_size = incoming_filters * expanded_depth;
	}

	// assuming only project when depths are different (all projections in resnet-50 this way)
	// could later change to adapt to just spatial transform...
	if (incoming_filters != expanded_depth){
		cudaMalloc(&projection, projection_size * sizeof(float));
		cudaMemset(projection, 0, projection_size * sizeof(float));
		if (!is_zero){
			sample_gaussian <<< SM_COUNT, ceil((float) (projection_size) / SM_COUNT) >>> (projection_size, projection, 0, 2.0 / incoming_filters);
		}
		cudaMalloc(&bias_projection, expanded_depth * sizeof(float));
		cudaMemset(bias_projection, 0, expanded_depth * sizeof(float));
	}
	else{
		projection = NULL;
		bias_projection = NULL;
	}

	conv_block -> projection = projection;
	conv_block -> bias_projection = bias_projection;


	norm_residual_added = init_batch_norm(incoming_spatial_dim, expanded_depth, is_zero);
	conv_block -> norm_residual_added = norm_residual_added;

	return conv_block;
}

Params * init_model_parameters(Dims * model_dims, bool is_zero){

	Params * params = malloc(sizeof(Params));

	// dimensions unpacked
	int input_dim = model_dims -> input;
	int n_conv_blocks = model_dims -> n_conv_blocks;
	int init_kernel_dim = model_dims -> init_kernel_dim;
	int init_conv_filters = model_dims -> init_conv_filters;
	int * is_block_spatial_reduction = model_dims -> is_block_spatial_reduction;
	int output_dim = model_dims -> output;

	// init array to hold pointers to weights
	// 3 * 4 weight arrays per conv block (weights, biases, gamma, beta per layer in block) + 4 * inital + fully connected + 4 * 2 projections
	// ignoring biases + batch norm weights for now...
	int n_locations = 13 + 12 * n_conv_blocks;
	params -> n_locations = n_locations;

	float ** locations = malloc(n_locations * sizeof(float *));
	int * sizes = malloc(n_locations * sizeof(int));
	// tracking location ind as we start allocating...
	


	// init first 7 * 7 conv_layer
	float * init_conv_layer;
	int init_conv_size = init_kernel_dim * init_kernel_dim * init_conv_filters;
	float init_conv_fan_in = 3 * input_dim * input_dim;
	cudaMalloc(&init_conv_layer,  init_conv_size * sizeof(float));
	cudaMemset(init_conv_layer, 0, init_conv_size * sizeof(float));
	if (!is_zero){
		sample_gaussian <<< SM_COUNT, ceil((float) (init_conv_size) / SM_COUNT) >>> (init_conv_size, init_conv_layer, 0, 2.0 / init_conv_fan_in);
	}
	params -> init_conv_layer = init_conv_layer;

	int loc_ind = 0;
	locations[loc_ind] = init_conv_layer;
	sizes[loc_ind] = init_kernel_dim * init_kernel_dim * init_conv_filters;
	loc_ind++;

	float * bias_init_conv;
	cudaMalloc(&bias_init_conv, init_conv_filters * sizeof(float));
	cudaMemset(bias_init_conv, 0, init_conv_filters * sizeof(float));

	params -> bias_init_conv = bias_init_conv;

	locations[loc_ind] = bias_init_conv;
	sizes[loc_ind] = init_conv_filters;
	loc_ind++;

	BatchNorm * norm_init_conv = init_batch_norm(input_dim, init_conv_filters, is_zero);
	params -> norm_init_conv = norm_init_conv;

	locations[loc_ind] = norm_init_conv -> gamma;
	sizes[loc_ind] = init_conv_filters;
	loc_ind++;

	locations[loc_ind] = norm_init_conv -> beta;
	sizes[loc_ind] = init_conv_filters;
	loc_ind++;
	

	// init conv blocks
	ConvBlock ** conv_blocks = malloc(n_conv_blocks * sizeof(ConvBlock *));
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
		conv_blocks[i] = init_conv_block(incoming_filters, incoming_spatial_dim, reduced_depth, expanded_depth, stride);
		locations[loc_ind] = conv_blocks[i] -> depth_reduction;
		sizes[loc_ind] = incoming_filters * reduced_depth;
		loc_ind++;
		locations[loc_ind] = conv_blocks[i] -> bias_depth_reduction;
		sizes[loc_ind] = reduced_depth;
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
		locations[loc_ind] = conv_blocks[i] -> bias_spatial;
		sizes[loc_ind] = reduced_depth;
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
		locations[loc_ind] = conv_blocks[i] -> bias_depth_expansion;
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
			locations[loc_ind] = conv_blocks[i] -> bias_projection;
			sizes[loc_ind] = expanded_depth;
			loc_ind++;
		}

		locations[loc_ind] = conv_blocks[i] -> norm_residual_added -> gamma;
		sizes[loc_ind] = expanded_depth;
		loc_ind++;
		locations[loc_ind] = conv_blocks[i] -> norm_residual_added -> beta;
		sizes[loc_ind] = expanded_depth;
		loc_ind++;


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
		sample_gaussian <<< SM_COUNT, ceil((float) (fully_connected_size) / SM_COUNT) >>> (fully_connected_size, fully_connected, 0, 2.0 / fully_connected_fan_in);
	}

	params -> fully_connected = fully_connected;
	locations[loc_ind] = fully_connected;
	sizes[loc_ind] = expanded_depth * output_dim;

	params -> locations = locations;
	params -> sizes = sizes;

	return params;
}

ResNet * init_resnet(Dims * dims){
	ResNet * model = malloc(sizeof(ResNet));
	model -> dims = dims;
	Parms * params = init_model_parameters(dims, false);
	model -> params = params;
	return model;
}


/* INITIALIZE TRAINING STRUCTURES */

Cache_BatchNorm * init_cache_batchnorm(int input_size, int feature_size){
	Cache_BatchNorm * cache_batchnorm = malloc(sizeof(Cache_BatchNorm));

	cache_batchnorm -> input_size = input_size;
	cache_batchnorm -> feature_size = feature_size;

	float * means, *vars, *normalized_temp, *normalized;

	cudaMalloc(&means, feature_size * sizeof(float));
	cudaMalloc(&vars, feature_size * sizeof(float));
	cudaMalloc(&normalized_temp, input_size * sizeof(float));
	cudaMalloc(&normalized, input_size * sizeof(float));

	cache_batchnorm -> means = means;
	cache_batchnorm -> vars = vars;
	cache_batchnorm -> normalized_temp = normalized_temp;
	cache_batchnorm -> normalized = normalized;

	return cache_batchnorm;
}

Activation_ConvBlock * init_activation_convblock(ConvBlock * conv_block, int batch_size){
	Activation_ConvBlock * activation_conv_block = malloc(sizeof(Activation_ConvBlock));

	activation_conv_block -> incoming_filters = conv_block -> incoming_filters;
	activation_conv_block -> incoming_spatial_dim = conv_block -> incoming_spatial_dim;
	activation_conv_block -> reduced_depth = conv_block -> reduced_depth;
	activation_conv_block -> expanded_depth = conv_block -> expanded_depth;
	activation_conv_block -> stride = conv_block -> stride;

	float * post_reduced, *post_spatial, *post_expanded, *transformed_residual, *output, *output_activated;
	float * post_reduced_activated, *post_spatial_activated;
	int post_reduced_size, post_spatial_size, output_size;
	Cache_BatchNorm * norm_post_reduced, *norm_post_spatial, *norm_post_residual_added;
	

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

	// only allocate space if transformed, otherwise it will be assumed to be identity of input
	transformed_residual = NULL;
	if (incoming_filters != expanded_depth){
		cudaMalloc(&transformed_residual, output_size * sizeof(float));
	}
	activation_conv_block -> transformed_residual = transformed_residual;

	cudaMalloc(&output, output_size * sizeof(float));
	activation_conv_block -> output = output;

	norm_post_residual_added = init_cache_batchnorm(output_size, expanded_depth);
	activation_conv_block -> norm_post_residual_added = norm_post_residual_added;

	cudaMalloc(&output_activated, output_size * sizeof(float));
	activation_conv_block -> output_activated = output_activated;

	return activation_conv_block;
}

Activations * init_activations(Dims * dims, ConvBlock ** conv_blocks, int batch_size){
	
	Activations * activations = malloc(sizeof(Activations));

	int input_dim = dims -> input;
	int init_conv_filters = dims -> init_conv_filters;
	int init_conv_stride = dims -> init_conv_stride;
	int maxpool_stride = dims -> init_maxpool_stride;

	float * init_conv_applied;
	int init_conv_applied_size = init_conv_filters * input_dim * input_dim / (init_stride * init_stride) * batch_size; 
	cudaMalloc(&init_conv_applied, init_conv_applied_size * sizeof(float));
	activations -> init_conv_applied = init_conv_applied;

	Cache_BatchNorm * norm_init_conv = init_cache_batchnorm(init_conv_applied_size, init_conv_filters);
	activations -> norm_init_conv = norm_init_conv;

	float * init_conv_activated;
	cudaMalloc(&init_conv_activated, init_conv_applied_size * sizeof(float));
	activation -> init_conv_activated = init_conv_activated;

	int init_convblock_input_size = init_conv_filters * input_dim * input_dim / (init_stride * init_stride) / (maxpool_stride * maxpool_stride) * batch_size;

	int * max_inds;
	cudaMalloc(&max_inds, init_convblock_input_size * sizeof(int));
	activations -> max_inds = max_inds;

	float *init_convblock_input;
	cudaMalloc(&init_convblock_input, init_convblock_input_size * sizeof(float));
	activations -> init_convblock_input = init_convblock_input;

	int n_conv_blocks = dims -> n_conv_blocks;

	Activation_ConvBlock ** activation_conv_blocks = malloc(n_conv_blocks * sizeof(Activation_ConvBlock *));
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
	cudaMalloc(&output, output_size * sizeof(float));
	activations -> linear_output = linear_output;

	return activations;
}


Forward_Buffer * init_forward_buffer(Dims * dims, ConvBlock ** conv_blocks, int batch_size){

	Forward_Buffer * forward_buffer = malloc(sizeof(Forward_Buffer));

	forward_buffer -> activations = init_activations(dims, conv_blocks, batch_size);

	int output_dim = dims -> output;
	int output_size = output_dim * batch_size;

	float * pred;
	cudaMalloc(&pred, output_size * batch_size * sizeof(float));
	forward_buffer -> pred = pred;

	// will be copied to cpu to be able to print values and compute loss on cpu side
	float * pred_cpu = malloc(output_size * batch_size * sizeof(float));
	forward_buffer -> pred_cpu = pred_cpu;

	return forward_buffer;
}


Backprop_Buffer * init_backprop_buffer(Dims * dims, ConvBlock ** conv_blocks, int batch_size){

	Backprop_Buffer * backprop_buffer = malloc(sizeof(Backprop_Buffer));

	int output_dim = dims -> output;
	int output_size = output_dim * batch_size;

	float * output_layer_deriv;
	cudaMalloc(&output_layer_deriv, output_size * sizeof(float));
	backprop_buffer -> output_layer_deriv = output_layer_deriv;

	backprop_buffer -> param_derivs = init_model_parameters(dims, true);
	backprop_buffer -> prev_means = init_model_parameters(dims, true);
	backprop_buffer -> prev_vars = init_model_parameters(dims, true);
	backprop_buffer -> activation_derivs = init_activations(dims, conv_blocks, batch_size);

	return backprop_buffer;
}


Train_ResNet * init_trainer(ResNet * model, Batch * cur_batch, int batch_size, float learning_rate, float mean_decay, float var_decay, float eps, int n_epochs){
	Train_ResNet * trainer = malloc(sizeof(Train_ResNet));

	trainer -> model = model;

	trainer -> cur_batch = cur_batch;
	trainer -> batch_size = batch_size;

	Dims * dims = model -> dims;
	ConvBlock ** conv_blocks = model -> params -> conv_blocks;
	trainer -> forward_buffer = init_forward_buffer(dims, conv_blocks, batch_size);
	trainer -> backprop_buffer = init_backprop_buffer(dims, conv_blocks, batch_size);

	trainer -> learning_rate = learning_rate;
	trainer -> base_mean_decay = mean_decay;
	trainer -> base_var_decay = var_decay;
	trainer -> cur_mean_decay = 1;
	trainer -> cur_var_decay = 1;
	trainer -> eps = eps;

	trainer -> n_epochs = n_epochs;

	trainer -> loss_per_epoch = calloc(n_epochs * sizeof(float));
	trainer -> accuracy_per_epoch = calloc(n_epochs * sizeof(float));

	return trainer;
}

Batch * init_general_batch(int n_images, int image_size, int image_dim){
	Batch * batch = malloc(sizeof(Batch));

	batch -> n_images = n_images;
	// in resnet-50 will be 224 * 224 * 3
	batch -> image_size = image_size;
	batch -> image_dim = image_dim;
	// load batch by first brining into cpu
	batch -> images_cpu = malloc(n_images * image_size * sizeof(uint8_t));
	batch -> images_float_cpu = malloc(n_images * image_size * sizeof(float));

	// allocate memory on gpu so that after loaded on cpu can bring in
	// will be converting from uint8 on CPU to float on GPU
	float * images;
	cudaMalloc(&images, n_images * image_size * sizeof(float));
	batch -> images = images;

	batch -> correct_classes_cpu = malloc(n_images * sizeof(int));

	float * correct_classes;
	cudaMalloc(&correct_classes, n_images * sizeof(int));
	batch -> correct_classes = correct_classes;

	return batch;
}

// (if this takes too long, can do it in parallel with separate process on cpu)
void * load_new_batch(Class_Metadata * class_metadata, Batch * batch_buffer){
	int batch_size = batch_buffer -> n_images;
	int image_size = batch_buffer -> image_size;
	int total_pixels = batch_size * image_size;
	int n_classes = class_metadata -> n_classes;
	int * counts_per_class = class_metadata -> counts;

	uint8_t * images_cpu = batch_buffer -> images_cpu;
	float * images_float_cpu = batch_buffer -> images_float_cpu;
	float * images = batch_buffer -> images;

	int * correct_classes_cpu = batch_buffer -> correct_classes_cpu;
	int * correct_classes = batch_buffer -> correct_classes;

	// randomly select class, then randomly select image within class
	int class_id, image_id;
	FILE * f;
	char file_path[100];
	for (int i = 0; i < batch_size; i++){
		class_id = rand() % n_classes;
		sprintf(file_path, "/mnt/storage/data/vision/imagenet/2012/%08d.buffer", class_id);
		image_id = rand() % counts_per_class[class_id];
		f = fopen(file_path, "rb");
		fseek(f, image_id * image_size, SEEK_SET);
		fread(images_cpu + i * image_size, sizeof(uint8_t), (size_t) image_size, f);
		correct_classes_cpu[i] = class_id;
	}

	// array is linear format where each sequence of image_size [0, image_size) is image 1, then [image_size, 2 * image_size) has image 2
	// each image is also linearized where ording of pixels is - 0, 0: (R, G, B) then 0, 1: (R,G,B), ...

	for (int pixel = 0; pixel < total_pixels; pixel++){
		images_float_cpu[pixel] = ((float) (images_cpu[pixel])) * (2.0 / 255) - 1;
	}

	cudaMemcpy(images, images_float_cpu, total_pixels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(correct_classes, correct_classes_cpu, batch_size * sizeof(int), cudaMemcpyHostToDevice);

}


// READ CLASSES AND LABELS!
Class_Metadata * populate_class_info(char * label_filename, char * synset_filename, char * class_size_filename, int n_classes){
	Class_Metadata classes = malloc(sizeof(Class_Metadata));

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


// reading a text file line by line into a buffer
// pre-allocate buffer and specify type
void text_file_to_buffer(void * buffer, char * filename, const char * type){

	if (strcmp(type, "TEXT") == 0){
        char ** my_buffer = (char **) buffer;
    }
    else if (strcmp(type, "INT") == 0){
        int * my_buffer = (int *) buffer;
    }
    else{
    	// unknown type...
    	void * my_buffer = buffer;
    }


	FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen(filename, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    int cnt = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
    	if (strcmp(type, "TEXT") == 0){
        	my_buffer[cnt] = strdup(line);
        }
        else if (strcmp(type, "INT") == 0){
        	my_buffer[cnt] = atoi(line);
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


void prepareAndDoConvolution(int in_spatial_dim, int kern_dim, int in_filters, int out_filters,  int stride, int batch_size, 
																float * input, float * weights, float * biases, float * output){

	int out_filter_row_size = kern_dim * kern_dim * in_filters;
	int max_out_filter_rows = MAX_SHARED_MEM_FLOATS / outfilter_row_size;
	int out_filter_chunks = ceil((float) out_filters / max_outfilter_rows);
	int shared_mem_size = out_filter_row_size * max_out_filter_rows;
	int out_spatial_dim = in_spatial_dim / stride;
	int max_subatch_size = MAX_THREAD_PER_BLOCK / max_out_filter_rows;

	dim3 gridDimConv(out_spatial_dim, out_spatial_dim, out_filter_chunks);
	dim3 blockDimConv(max_out_filter_rows, max_subatch_size);

	doConvolution <<< gridDimConv, blockDimConv, shared_mem_size >>> (input, weights, biases, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, output);

}


void prepreAndDoConvolutionDeriv(int in_spatial_dim, int kern_dim, int in_filters, int out_filters, int stride, int batch_size, bool toAdd,
												float * input, float * weights, float * out_deriv,
												float * input_deriv, float * weight_deriv, float * bias_deriv, bool toComputeInputDeriv){
	
	// first layer conv doesn't take deriv w.r.t input
	if (toComputeInputDeriv){
		convolutionDerivInput <<< (in_spatial_dim, in_spatial_dim, in_filters), (batch_size) >>> (input, weights, out_deriv, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, toAdd, input_deriv);
	}
	convolutionDerivWeights <<< (in_filters, out_filters), (kern_dim, kern_dim) >>> (input, weights, out_deriv, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, weight_deriv);
	convolutionDerivBiases <<< (out_filters), (1) >>> (input, weights, out_deriv, in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, bias_deriv);
	
}

void prepareAndDoBatchNormAndActivate(BatchNorm * batch_norm_params, Cache_BatchNorm * batch_norm_cache, int batch_size, float eps, float * input, float * activated_out){
	// reading values from batch norm params
	int filters = batch_norm_params -> depth;
	int spatial_dim = batch_norm_params -> spatial_dim;
	float * gamma = batch_norm_params -> gamma;
	float * beta = batch_norm_params -> beta;

	// read the output device pointers from batch_norm_cache
	float * means_out = batch_norm_cache -> means;
	float * vars_out = batch_norm_cache -> vars;
	float * normalized_temp_out = batch_norm_cache -> normalized_temp;
	float * normalized_out = batch_norm_cache -> normalized;

	doBatchNormAndActivate<<< filters, 1 >>> (input, gamma, beta, spatial_dim, filters, batch_size, eps, means_out, vars_out, normalized_temp_out, normalized_out, activated_out);
}

void prepareAndDoActivationAndBatchNormDeriv(BatchNorm * batch_norm_params, Cache_BatchNorm * batch_norm_cache, BatchNorm * batch_norm_param_derivs, Cache_BatchNorm * batch_norm_cache_derivs, 
																								float eps, float * input, float * activated, float * out_layer_deriv, float * input_deriv){
	int filters = batch_norm_params -> depth;
	int spatial_dim = batch_norm_params -> spatial_dim;
	float * gamma = batch_norm_params -> gamma;
	float * beta = batch_norm_params -> beta;
	float * means = batch_norm_params -> means;
	float * vars = batch_norm_params -> vars;
	float * normalized_temp = batch_norm_cache -> normalized_temp;

	float * normalized_temp_deriv = batch_norm_cache_derivs -> normalized_temp;
	float * gamma_deriv = batch_norm_param_derivs -> gamma;
	float * beta_deriv = batch_norm_param_derivs -> beta;

	activationAndBatchNormDeriv <<< (filters), (1) >>> (input, gamma, beta, spatial_dim, filters, eps, means, vars, normalized_temp, activated, out_layer_deriv, normalized_temp_deriv, gamma_deriv, beta_deriv, input_deriv);


}

void prepareAndDoMatMulLeftTranspose(const float * left_orig, const float * right, int left_orig_rows, int left_orig_cols, int right_rows, int right_cols, float * out){
	float * temp_left;
	cudaMalloc(&temp_left, left_orig_rows * left_orig_cols * sizeof(float));
	transpose <<< (left_orig_rows / TILE_WIDTH, left_orig_cols / TILE_WIDTH), (TILE_WIDTH * BLOCK_ROWS) >>> (left_orig, left_orig_rows, left_orig_cols, temp_left);
	matMul <<< (left_orig_cols / TILE_WIDTH, right_cols / TILE_WIDTH), (TILE_WIDTH, TILE_WIDTH) >>> (temp_left, right, left_orig_cols, right_rows, right_cols, out);
	cudaFree(temp_left);
}

void prepareAndDoMatMulRightTranspose(const float * left, const float * right_orig, int left_rows, int left_cols, int right_orig_rows, int right_orig_cols, float * out){
	float * temp_right;
	cudaMalloc(&temp_right, right_orig_rows * right_orig_cols * sizeof(float));
	transpose <<< (right_orig_rows / TILE_WIDTH, right_orig_cols / TILE_WIDTH), (TILE_WIDTH * BLOCK_ROWS) >>> (right_orig, right_orig_rows, right_orig_cols, temp_right);
	matMul <<< (left_rows / TILE_WIDTH, right_orig_rows / TILE_WIDTH), (TILE_WIDTH, TILE_WIDTH) >>> (left, temp_right, left_rows, left_cols, right_orig_rows, out);
	cudaFree(temp_right);
}

void forward_pass(Train_ResNet * trainer){

	Dims * dims = trainer -> model -> dims;

	float eps = trainer -> eps;
	int batch_size = trainer -> batch_size;

	float * input = trainer -> cur_batch -> images;
	float * first_conv = trainer -> model -> params -> init_conv_layer;
	float * first_conv_bias = trainer -> model -> params -> bias_init_conv;
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

	prepareAndDoConvolution(init_spatial_dim, init_kernel_dim, init_in_filters, init_out_filters, init_stride, batch_size, input, first_conv, first_conv_bias, first_conv_output);

	BatchNorm * norm_init_conv_params = trainer -> model -> params -> norm_init_conv;
	Cache_BatchNorm * norm_init_conv_cache = trainer -> forward_buffer -> activations -> norm_init_conv;
	float * init_activated = trainer -> forward_buffer -> activations -> init_conv_activated;

	prepareAndDoBatchNormAndActivate(norm_init_conv_params, norm_init_conv_cache, batch_size, eps, first_conv_output, init_activated);

	int init_maxpool_dim = dims -> init_maxpool_dim;
	int init_maxpool_stride = dims -> init_maxpool_stride;
	int init_maxpool_out_dim = init_out_spatial_dim / init_maxpool_stride;
	float * init_convblock_input = trainer -> forward_buffer -> activations -> init_convblock_input;
	float * max_ind_buff = trainer -> forward_buffer -> activations -> max_inds;
	doMaxPool <<< (init_maxpool_out_dim, init_maxpool_out_dim) , init_out_filters >>> (init_activated, init_maxpool_dim, init_maxpool_stride, batch_size, max_ind_buff, init_convblock_input);


	/* NOW CAN MOVE ONTO TO CONV_BLOCK LAYERS! */

	int n_conv_blocks = dims -> n_conv_blocks;

	
	ConvBlock ** params_conv_blocks = trainer -> model -> params -> conv_blocks;
	Activation_ConvBlock ** activation_conv_blocks = trainer -> forward_buffer -> activations -> activation_conv_blocks;
	ConvBlock * cur_conv_block_params;
	Activation_ConvBlock * cur_conv_block_activation;
	int in_spatial_dim, kern_dim, in_filters, out_filters, stride, out_spatial_dim, total_size_conv_block_output;

	float * conv_block_input = init_convblock_input;
	float *conv_input, * conv_weights, * conv_biases, * conv_output, *norm_input, * norm_output, * conv_block_output;
	float *projection_weights, *projection_biases, *transformed_residual;
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
		conv_biases = cur_conv_block_params -> bias_depth_reduction;
		conv_output = cur_conv_block_activation -> post_reduced;

		prepareAndDoConvolution(in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, conv_input, conv_weights, conv_biases, conv_output);

		norm_input = conv_output;
		cur_batch_norm_params = cur_conv_block_params -> norm_depth_reduction;
		cur_batch_norm_cache = cur_conv_block_params -> norm_post_reduced;
		norm_output = cur_conv_block_activation -> post_reduced_activated;

		prepareAndDoBatchNormAndActivate(cur_batch_norm_params, cur_batch_norm_cache, batch_size, eps, norm_input, norm_output);

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
		conv_weights = cur_conv_block_params -> spatial;
		conv_biases = cur_conv_block_params -> bias_spatial;
		conv_output = cur_conv_block_activation -> post_spatial;

		prepareAndDoConvolution(in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, conv_input, conv_weights, conv_biases, conv_output);

		norm_input = conv_output;
		cur_batch_norm_params = cur_conv_block_params -> norm_spatial;
		cur_batch_norm_cache = cur_conv_block_activation -> norm_post_spatial;
		norm_output = cur_conv_block_activation -> post_spatial_activated;

		prepareAndDoBatchNormAndActivate(cur_batch_norm_params, cur_batch_norm_cache, batch_size, eps, norm_input, norm_output);

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
		conv_biases = cur_conv_block_params -> bias_depth_expansion;
		conv_output = cur_conv_block_activation -> post_expanded;

		prepareAndDoConvolution(in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, conv_input, conv_weights, conv_biases, conv_output);

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
		projection_biases = cur_conv_block_params -> bias_projection;


		total_size_conv_block_output = out_spatial_dim * out_spatial_dim * out_filters * batch_size;
		conv_block_output = cur_conv_block_activation -> output;
				
		// the conv_block initializer already handled if we need projection, and if so it allocated weights
		// if there is a projection needed we will do convolution with the above parameters
		if (projection_weights){
			// allocated device memory to store output
			transformed_residual = cur_conv_block_activation -> transformed_residual;
			prepareAndDoConvolution(in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, conv_block_input, projection_weights, projection_biases, transformed_residual);
		}
		else{
			// would've been null, so renaming for semantic clarity
			transformed_residual = conv_block_input;
		}

		// add identity residual connection (or projected residual connection) to the prior convolutional output
		addVec <<< ceil((float) total_size_conv_block_output / MAX_THREAD_PER_BLOCK), MAX_THREAD_PER_BLOCK >>> (total_size, conv_output, transformed_residual, conv_block_output);

		norm_input = conv_block_output;
		cur_batch_norm_params = cur_conv_block_params -> norm_residual_added;
		cur_batch_norm_cache = cur_conv_block_activation -> norm_post_residual_added;
		norm_output = cur_conv_block_activation -> output_activated;

		prepareAndDoBatchNormAndActivate(cur_batch_norm_params, cur_batch_norm_cache, batch_size, eps, norm_input, norm_output);

		// prepare for next block...
		conv_block_input = norm_output;
	}

	int final_filters = dims -> final_depth;
	int final_spatial_dim = params_conv_blocks[n_conv_blocks - 1] -> incoming_spatial_dim;
	float * final_conv_block_output = activation_conv_blocks[n_conv_blocks - 1] -> output_activated;
	float * final_avg_pool_values = trainer -> forward_buffer -> activations -> final_conv_output_pooled;

	// NEED TO DO AVERAGE POOL OF LAST LAYER to go from (batch_size, 7, 7, 2048) to (batch size, 1, 1, 2048)

	// format of output is each row is a sample and has a row size of 2048
	doFilterAvgPool <<< (final_filters), (batch_size) >>> (final_conv_block_output, final_spatial_dim, final_avg_pool_values);


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

	// DO SOFTMAX
	float * pred = trainer -> forward_buffer -> pred;
	softMax <<< (batch_size), (1) >>> (fc_output, batch_size, output_dim, pred);

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

	/* STEP 1: LAST LAYER DERIVATIVE */

	// layer has output_dim * batch_size values
	// End of network was: fully connected layer -> softmax
	// Derivative of cross entropy loss w.r.t to fully connected values is: s - y where s is softmax value
	// thus copy softmax values and subtract 1 from the correct index (we know labels y are 0 except correct label of 1)
	float * correct_classes = trainer -> cur_batch -> correct_classes;
	float * pred = trainer -> forward_buffer -> pred;
	float * output_layer_deriv = backprop_buffer -> output_layer_deriv;
	cudaMemcpy(output_layer_deriv, pred, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToDevice);
	crossEntropyDeriv <<< (batch_size), (1) >>> (output_layer_deriv, correct_classes, output_dim, batch_size);

	/* STEP 2: FC WEIGHT DERIV AND FINAL AVG POOL (SECOND LAST ACTIVTION LAYER) DERIVATIVE */

	// TODO: MAKE SURE THE DIMENSIONS ARE CORRECT ORDER...

	// FC WEIGHTS (2048, 1000) DERIV = matMul(transpose(final_conv_output_pooled), output_layer_deriv)
	int final_depth = dims -> final_depth;
	float * fc_deriv = param_derivs -> fully_connected;
	float * final_conv_output_pooled = activations -> final_conv_output_pooled;
	prepareAndDoMatMulLeftTranspose(final_conv_output_pooled, output_layer_deriv, batch_size, final_depth, batch_size, output_dim, fc_deriv);

	// FINAL AVG POOL (N, 2048) DERIV = matMul(output_layer_deriv, transpose(FC Weight))
	float * fc_weights = model_params -> fully_connected;
	float * final_avg_pool_deriv = activation_derivs -> final_conv_output_pooled;
	prepareAndDoMatMulRightTranspose(output_layer_deriv, fc_weights, batch_size, output_dim, final_depth, output_dim, final_avg_pool_deriv);


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
	filterAvgPoolDeriv <<< (final_depth), (batch_size) >>> (final_avg_pool_deriv, final_depth, batch_size, final_spatial_dim, final_conv_block_output_deriv);

	
	/* STEP 4: CONV BLOCK & BATCH NORM DERIVS  */
	

	// we are starting with deriv of last conv block output...

	// To go backwards for each block we:
		// 1.) Get deriv of batch norm for residual added to expanded conv output (with respect to both its own parameters and also the input to batch norm = expanded conv output)
		// 2.) Get deriv projection filter & transformed (if there is a projection of residual, otherwise both derivs are 1)
		// 3.) Multiply the deriv of input to batch norm * deriv of transformed residual and add to the deriv of first layer of conv block (= batch norm output of prior block)
		// 4.) Get deriv of expanded convolution & deriv of input to expanded convolution (= batch norm output of spatial conv)
		// 5.) Get deriv of batch norm for spatial conv output (with respect to both its own parameters and also the input to batch norm = spatial conv output)
		// 6.) Get deriv of sptial convolution & deriv of input to spatial convolution (= batch norm output of reduced conv)
		// 7.) Get deriv of batch norm for reduced conv output (with respect to both its own parameters and also the input to batch norm = reduced conv output)
		// 8.) Get deriv of reduced convolution & deriv of input to reduced convolution, which is the first layer of conv block (= batch norm output of prior conv block)
		// Items 3.) and 8.) provide the derivative used to repeat process for prior block

	

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
	float *conv_input_deriv, *conv_weight_deriv, *conv_bias_deriv;


	// STARTING POINT FROM BACKPROP COMING FROM UPSTREAM LAYERS IS AT LAST CONV BLOCK ACTIVATION -> OUTPUT_ACTIVATED
	float *conv_block_input, *conv_block_input_deriv;

	// extra temp variables
	int total_size;

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

		/* 1: Conv Block Output Activation and Batch Norm Derivs */

		// update the current batch norm layer pointers
		cur_batch_norm_params = cur_conv_block_params -> norm_residual_added;
		cur_batch_norm_param_derivs = cur_conv_block_param_derivs -> norm_residual_added;

		cur_batch_norm_cache = cur_conv_block_activation -> norm_post_residual_added;
		cur_batch_norm_cache_derivs = cur_conv_block_activation_derivs -> norm_post_residual_added;

		// fill in details about backprop I/O
		// dL/dBN_Output (given)
		bn_out_layer_deriv = activation_conv_blocks_derivs[i] -> output_activated;
		// dL/dBN_Input (to fill in)
		bn_input_deriv = cur_conv_block_activation_derivs -> output;
		// input to batch norm layer from forward pass
		bn_input = cur_conv_block_activation -> output;
		// activated output of batch norm layer from forward pass
		bn_activated = cur_conv_block_activation -> output_activated;
		
		prepareAndDoActivationAndBatchNormDeriv(cur_batch_norm_params, cur_batch_norm_cache, cur_batch_norm_param_derivs, cur_batch_norm_cache_derivs,
																						eps, bn_input, bn_activated, bn_out_layer_deriv, bn_input_deriv);

		/* 2: (Transformed) Residual Derivs & Chained/Added to Conv Block Input Deriv (= prior_block_output_deriv) */

		// check if there is a projection (aka convolution over depth/kern_dim=1 or possibly stride=2/kern_dim=3), otherwise the projection deriv is 1
		// If there is a projection need to compute derivative of the projection convolution kernel weights and deriv w.r.t. projection convolution input=conv_block_input=prior_block_output_activated
		if (cur_conv_block_params -> projection){


			// CONVOLUTION DIMENSIONS
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


			// CONVOLUTION FORWARD DATA
			// transformed residual convolution input is the value at first step of conv block => activated output from previous block
			conv_input = conv_block_input;
			conv_weight = cur_conv_block_params -> projection;
			// from backprop
			conv_out_deriv = cur_conv_block_activation_derivs -> output;

			// CONVOLUTION BACKWARDS DERIV DATA BUFFERS
			// because residual
			conv_input_deriv = conv_block_input_deriv;
			conv_weight_deriv = cur_conv_block_param_derivs -> projection;
			conv_bias_deriv = cur_conv_block_param_derivs -> bias_projection;

			prepreAndDoConvolutionDeriv(in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, false,
													conv_input, conv_weight, conv_out_deriv,
													conv_input_deriv, conv_weight_deriv, conv_bias_deriv, true);
		}
		else{
			total_size = batch_size * (cur_conv_block_params -> incoming_spatial_dim) * (cur_conv_block_params -> incoming_spatial_dim) * (cur_conv_block_params -> incoming_filters);
			addVec <<< ceil((float) total_size / MAX_THREAD_PER_BLOCK), MAX_THREAD_PER_BLOCK >>> (total_size, conv_block_input_deriv, cur_conv_block_activation_derivs -> output, conv_block_input_deriv);
		}
		

		/* 3: Expanded Convolution Derivs */

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
		conv_out_deriv = cur_conv_block_activation_derivs -> output;

		// CONVOLUTION BACKWARDS DERIV DATA BUFFERS
		// because residual
		conv_input_deriv = cur_conv_block_activation_derivs -> post_spatial_activated;
		conv_weight_deriv = cur_conv_block_param_derivs -> depth_expansion;
		conv_bias_deriv = cur_conv_block_param_derivs -> bias_depth_expansion;

		prepreAndDoConvolutionDeriv(in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, false,
													conv_input, conv_weight, conv_out_deriv,
													conv_input_deriv, conv_weight_deriv, conv_bias_deriv, true);


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
		
		prepareAndDoActivationAndBatchNormDeriv(cur_batch_norm_params, cur_batch_norm_cache, cur_batch_norm_param_derivs, cur_batch_norm_cache_derivs,
																						eps, bn_input, bn_activated, bn_out_layer_deriv, bn_input_deriv);

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
		conv_bias_deriv = cur_conv_block_param_derivs -> bias_spatial;

		prepreAndDoConvolutionDeriv(in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, false,
													conv_input, conv_weight, conv_out_deriv,
													conv_input_deriv, conv_weight_deriv, conv_bias_deriv, true);


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
		
		prepareAndDoActivationAndBatchNormDeriv(cur_batch_norm_params, cur_batch_norm_cache, cur_batch_norm_param_derivs, cur_batch_norm_cache_derivs,
																						eps, bn_input, bn_activated, bn_out_layer_deriv, bn_input_deriv);

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
		conv_bias_deriv = cur_conv_block_param_derivs -> bias_depth_reduction;

		prepreAndDoConvolutionDeriv(in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, true,
													conv_input, conv_weight, conv_out_deriv,
													conv_input_deriv, conv_weight_deriv, conv_bias_deriv, true);

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
	float * max_inds = activations -> max_inds;

	// populating the gradient of input to max_pool located at:
	float * maxpool_inp_deriv = activation_derivs -> init_conv_activated;
	// ensure that gradient has 0's, so that maxPoolDeriv kernel can overwrite only at max ind locations
	int maxpool_inp_size = maxpool_in_spatial_dim * maxpool_in_spatial_dim * maxpool_filters * batch_size;
	cudaMemset(maxpool_inp_deriv, 0, maxpool_inp_size * sizeof(float));


	// compute max pool deriv (i.e. populate maxpool_inp_deriv)
	maxPoolDeriv <<< (maxpool_out_spatial_dim, maxpool_out_spatial_dim, maxpool_filters), (batch_size) >>> (max_inds, maxpool_out_deriv, maxpool_kern_dim, maxpool_in_spatial_dim, maxpool_stride, maxpool_filters, batch_size, maxpool_inp_deriv);


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
		
	prepareAndDoActivationAndBatchNormDeriv(cur_batch_norm_params, cur_batch_norm_cache, cur_batch_norm_param_derivs, cur_batch_norm_cache_derivs,
																						eps, bn_input, bn_activated, bn_out_layer_deriv, bn_input_deriv);

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
	conv_input_deriv = conv_block_input_deriv;
	conv_weight_deriv = param_derivs -> init_conv_layer;
	conv_bias_deriv = param_derivs -> bias_init_conv;

	prepreAndDoConvolutionDeriv(in_spatial_dim, kern_dim, in_filters, out_filters, stride, batch_size, false,
													conv_input, conv_weight, conv_out_deriv,
													conv_input_deriv, conv_weight_deriv, conv_bias_deriv, false);
}	


// doing ADAM optimizer
void update_parameters(Train_ResNet * trainer){
	
	float learning_rate = trainer -> learning_rate;
	float base_mean_decay = trainer -> base_mean_decay;
	float base_var_decay = trainer -> base_var_decay;
	// update the running decays here...
	float cur_mean_decay = trainer -> cur_mean_decay * base_mean_decay;
	float cur_var_decay = trainer -> cur_var_decay * base_mean_decay;
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
	
	// update learning rate
	float alpha_t = learning_rate * sqrtf(1 - cur_var_decay) / (1 - cur_mean_decay);

	for (int i = 0; i < n_locations; i++){
		param_size = param_sizes[i];
		model_location = model_params_locations[i];
		grad_location = current_gradient_locations[i];
		mean_location = prev_grad_means_locations[i];
		var_location = prev_grad_vars_locations[i];

		updateMeans <<< ceil((float) param_size / MAX_THREAD_PER_BLOCK), MAX_THREAD_PER_BLOCK >>> (param_size, grad_location, base_mean_decay, mean_location);
		updateVars <<< ceil((float) param_size / MAX_THREAD_PER_BLOCK), MAX_THREAD_PER_BLOCK >>> (param_size, grad_location, base_var_decay, var_location);
		updateParams <<< ceil((float) param_size / MAX_THREAD_PER_BLOCK), MAX_THREAD_PER_BLOCK >>> (param_size, model_location, mean_location, var_location, alpha_t, eps);

		cudaMemset(grad_location, 0, param_size * sizeof(float));
	}
}



int main(int argc, char *argv[]) {

	char * N_CLASSES = 1000;
	
	// GETTING CLASS METADETA
	char * LABEL_FILENAME = "/mnt/storage/data/vision/imagenet/2012/id_to_label_mapping.txt";
	char * SYNSET_FILENAME = "/mnt/storage/data/vision/imagenet/2012/id_to_synset_mapping.txt";
	char * COUNTS_FILENAME = "/mnt/storage/data/vision/imagenet/2012/id_to_img_count_mapping.txt";
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
	int * IS_BLOCK_SPATIAL_REDUCTION = calloc(N_CONV_BLOCKS * sizeof(int));
	// transitions between spatial 56 -> 28 -> 14 -> 7
	// transitions between output depth of 256 -> 512 -> 1024 -> 2048
	int FINAL_DEPTH = 2048;
	IS_BLOCK_SPATIAL_REDUCTION[3] = 1;
	IS_BLOCK_SPATIAL_REDUCTION[7] = 1;
	IS_BLOCK_SPATIAL_REDUCTION[13] = 1;
	Dims * dims = init_dimensions(INPUT_DIM, INIT_KERNEL_DIM, INIT_CONV_FILTERS, INIT_CONV_STRIDE, INIT_MAXPOOL_DIM, INIT_MAXPOOL_STRIDE,
									N_CONV_BLOCKS, IS_BLOCK_SPATIAL_REDUCTION, FINAL_DEPTH, N_CLASSES);

	// INITIALIZING MODEL
	ResNet * model = init_resnet(dims);


	// INITIALIZING TRAINING

	// Batch Structure (will be modified every iteration of every epoch)
	int BATCH_SIZE = 1;
	// dimensions of INPUT_DIM X INPUT_DIM x 3 color channels
	int IMAGE_SIZE = INPUT_DIM * INPUT_DIM * 3;
	Batch * batch = init_general_batch(BATCH_SIZE, IMAGE_SIZE);


	// General Training Structure (holds hyperparameters and pointers to structs which have network values)
	float LEARNING_RATE = 0.001;
	float MEAN_DECAY = 0.9;
	float VAR_DECAY = 0.999;
	float EPS = 0.00000001;
	float N_EPOCHS = 1;

	Train_ResNet * trainer = init_trainer(model, batch, BATCH_SIZE, LEARNING_RATE, MEAN_DECAY, VAR_DECAY, EPS, N_EPOCHS);
	

	/* PERFORM TRAINING */


	int iterations_per_epoch = ceil((float) total_images / BATCH_SIZE);

	float *pred, *correct;
	float epoch_n_wrong, batch_n_wrong;
	float epoch_loss, batch_loss, avg_batch_loss, epoch_accuracy, batch_accuracy, val_pred_correct;
	float total_images_per_epoch = BATCH_SIZE * iterations_per_epoch;

	int PRINT_FREQ = 100;

	for (int epoch = 0; epoch < N_EPOCHS; epoch++){
		epoch_loss = 0;
		epoch_n_wrong = 0;
		for (int iter = 0; iter < iterations_per_epoch; iter++){

			/* LOAD NEW BATCH */
			// values go into trainer -> cur_batch -> [images_cpu|images_float_cpu|images|correct_classes_cpu|correct_classes]
			load_new_batch(class_metadata, trainer -> cur_batch);

			

			/* DO FORWARD PROP */
			// final predictions go into trainer -> forward_buffer -> [pred|pred_cpu|prediction_label]
			forward_pass(trainer);

			

			/* RECORD LOSS AND ACCURACY */

			// dimensions of pred: (N_CLASSES, BATCH_SIZE)
			pred = trainer -> forward_buffer -> pred_cpu;
			correct = trainer -> cur_batch -> correct_classes_cpu;
			
			// loss
			batch_loss = 0;
			for (int s = 0; s < BATCH_SIZE; s++){
				batch_loss += -1 * logf(pred[correct[s] * BATCH_SIZE + s]);
			}
			avg_batch_loss = batch_loss / BATCH_SIZE;
			epoch_loss += batch_loss;

			// accuracy
			batch_n_wrong = 0;
			for (int s = 0; s < BATCH_SIZE; s++){
				val_pred_correct = pred[correct[s] * BATCH_SIZE + s];
				for (int c = 0; c < N_CLASSES; c++){
					if (pred[c * BATCH_SIZE + s] > val_pred_correct){
						batch_n_wrong++;
						break;
					}
				}
			}
			epoch_n_wrong += batch_n_wrong;
			batch_accuracy = ((float) BATCH_SIZE - batch_n_wrong) / ((float) BATCH_SIZE);

			if (iter % PRINT_FREQ == 0){
				printf("Epoch: %d, Batch: %d ----- Avg. Loss: %d, Accuracy: %d\n", epoch, iter, avg_batch_loss, batch_accuracy);
			}



			/* DO BACKPROP */
			backwards_pass(trainer);

			

			/* OPTIMIZE WEIGHTS */
			update_parameters(trainer);
		}

		(trainer -> loss_per_epoch)[epoch] = epoch_loss;
		epoch_accuracy = (total_images_per_epoch - epoch_n_wrong) / total_images_per_epoch;
		(trainer -> accuracy_per_epoch)[epoch] = epoch_accuracy;

	}

}