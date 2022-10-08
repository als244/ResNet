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

// 48KB is maximum value for shared memory, passed into this kernel as third param <<< gridDim, blockDim, SHARED_MEM_BYTES >>>
// launch grid dimensions as (OUT_SPATIAL_DIM, OUT_SPATIAL_DIM) blocks, and launch with block dim as (out_filt_rows_shared) threads
// thus 12k floats is max for shared memory per block
// first get as many output filter weights in shared memory as possible
// then stream samples in batch to compute output value for each sample and output filter
__global__ void convolution(const float * input, const float * weights, int spatial_dim, int kern_dim, int in_filt, int out_filt, int stride, int batch_size, float * out){

	// will consist of (shared_out_filt_rows X (kern_dim^2 * in_filt) conv_weight matrix
	extern __shared__ float shared_mem[];

	int kernel_size = (kern_dim * kern_dim * in_filt);
	int out_filt_phases = ceil((float) out_filt / blockDim.x);

	int spatial_row_start = stride * blockIdx.x;
	int spatial_col_start = stride * blockIdx.y;
	int out_spatial_dim = spatial_dim / (stride * stride);

	int output_filter_off = threadIdx.x;
	int half_kernel_dim = kern_dim / 2;
	int out_filter_start, out_filter_id, spatial_row, spatial_col;
	float out_val, spatial_val;
	for (int out_filt_ph = 0; out_filt_ph < out_filt_phases; out_filt_ph++){

		out_filter_id = out_filt_ph * blockDim.x + threadIdx.x;
		if (out_filer_id >= out_filt){
			return;
		}

		// overwrite scratchpad when moving phases
		for (int j = 0; j < kernel_size; j++){
			shared_mem[threadIdx.x * kernel_size + j] = weights[out_filter_id * kernel_size + j];
		}

		// make sure finish overwriting before advancing
		__syncthreads();


		for (int sample_ind = 0; sample_ind < batch_size; sample_ind++){
			out_val = 0;
			for (int row_offset = -half_kernel_dim; row_offset <= half_kernel_dim; row_offset++){
				for (int col_offset = -half_kernel_dim; col_offset <= half_kernel_dim; col_offset++){
					for (int channel = 0; channel < in_filt; channel++){
						
						// compute spatial value
						spatial_row = spatial_row_start + row_offset;
						spatial_col = spatial_col_start + col_offset;
						kernel_ind = kern_dim * in_filt * (row_offset + half_kernel_dim) + in_filt * (col_offset + half_kernel_dim) + channel;
						if ((spatial_row < 0) || (spatial_row >= spatial_dim) || (spatial_col < 0) || (spatial_col >= spatial_dim)) {
							spatial_val = 0;
						}
						else{
							spatial_val = input[spatial_dim * spatial_dim * in_filt * sample_ind + spatial_dim * in_filt * spatial_row + in_filt * spatial_col + channel];
						}

						// multiply with conv weight
						out_val += shared_mem[threadIdx.x * kernel_size + kernel_ind] * spatial_val;
					}
				}
			}
			out[out_spatial_dim * out_spatial_dim * out_filt * sample_ind + out_spatial_dim * out_filt * blockIdx.x + out_filt * blockIdx.y + out_filter_id] = out_val;
		}

		// make sure finished with scratchpad before moving on
		__syncthreads();
	}
}

// hardcoded conv kernel for initial 7x7, stride 2, 64 output filter convolutional layer...
// launching (14, 112, BATCH_SIZE) dim blocks where each block has 112/14=8 phases to utilize shared memory. Each block will have dim (64).
// Each block will contribute 16 unique spatial inds * 64 output filters * 32 Batch Size to the output of layer
// each phase loads stride new rows into shared memory, then multiples new spatial shared_mem with conv_weights, accounting for conv weight col permuation 
__global__ void optimized_init_conv(const float * input, const float * weights, float * out){

	__shared__ float conv_weights[64][147];
	__shared__ float spatial_vals[147];

	// index
	int output_filter = threadIdx.x;
	int sample_ind = blockIdx.z;

	// assume weights are in order of outfilter 0: [R_0,0, B_0,0, G_0,0, R_0,1, G_0,1, B_0,1....R_6,6, G_6,6, B_6,6], outfilter 1: [...], ...., outfilter 63: [...]
	for (int kernel_ind = 0; kernel_ind < 147; kernel_ind++){
		conv_weights[output_filter][kernel_ind] = weights[output_filter * 147 + kernel_ind];
	}

	// 2 * vals because stride of 2
	int spatial_row_start = (224 / blockDim.x) * blockIdx.x;
	int spatial_col_start = 2 * blockIdx.y;
	int spatial_row, spatial_col, kernel_ind;
	int half_kernel_dim = 3;
	for (int row_offset = -half_kernel_dim; row_offset <= half_kernel_dim;  row_offset++){
		for (int col_offset = -half_kernel_dim; col_offset <= half_kernel_dim; col_offset++){
			for (int channel = 0; channel < 3; channel++){
				spatial_row = spatial_row_start + row_offset;
				spatial_col = spatial_col_start + col_offset;
				kernel_ind = 7 * 3 * (row_offset + half_kernel_dim) + 3 * (col_offset + half_kernel_dim) + channel;
				if ((spatial_row < 0) || (spatial_row >= 224) || (spatial_col < 0) || (spatial_col >= 224)) {
					spatial_vals[kernel_ind] = 0;
				}
				else{
					spatial_vals[kernel_ind] = input[224 * 224 * 3 * sample_ind + 224 * 3 * spatial_row + 3 * spatial_col + channel];
				}
			}
		}
	}

	__syncthreads();

	float val = 0;
	int circular_row = 0;
	int out_spatial_row = (112 / blockDim.x) * blockIdx.x;
	int out_spatial_col = blockIdx.y;
	int new_top_row = 0;
	for (int phase = 0; phase < 8; phase++){

		// compute matrix mult to get (output_filt x batch_size) result. this is for a single receptive field across depth and batches
		// iterative over phases to get multiple receptive fields and exploit spatial locality
		val = 0;
		for (int kern_row = 0; kern_row < 7; kern_row++){
			for (int kern_col = 0; kern_col < 7; kern_col++){
				for (int ch = 0; ch < 3; ch++){
					circular_row = (kern_row + 2 * phase) % 7;
					val += conv_weights[output_filter][7 * 3 * kern_row + 3 * kern_col + ch] * spatial_vals[7 * 3 * circular_row + 3 * kern_col + ch];
				}
			}
		}

		out[112 * 112 * 64 * sample_ind + 112 * 64 * out_spatial_row + 64 * out_spatial_col + output_filter] = val;

		__syncthreads();

		int row_to_replace, replace_ind;
		for (int i = 1; i <= 2; i++){
			row_to_replace = (2 * phase) + i % 7;
			spatial_row = spatial_row_start + half_kernel_dim + 2 * phase + i; 
			for (int col_offset = -half_kernel_dim; col_offset <= half_kernel_dim; col_offset++){
				for (int channel = 0; channel < 3; channel++){
					spatial_col = spatial_col_start + col_offset;
					replace_ind = 7 * 3 * row_to_replace + 3 * (col_offset + half_kernel_dim) + channel;
					if ((spatial_row < 0) || (spatial_row >= 224) || (spatial_col < 0) || (spatial_col >= 224)) {
						spatial_vals[replace_ind][sample_ind] = 0;
					}
					else{
						spatial_vals[replace_ind][sample_ind] = input[224 * 224 * 3 * sample_ind + 224 * 3 * spatial_row + 3 * spatial_col + channel];
					}
				}
			}
		}
		out_spatial_row++;

		__syncthreads();
	}
}



// assume pass in 1-D block
// assume X is a matrix where # rows = output_len and # columns = batch size
__global__ void softMax(int batch_size, int output_len, float*X){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size){
    float sum = 0;
    for (int j = 0; j < output_len; j++){
      sum += __expf(X[i + batch_size * j]);
    }
    for (int j = 0; j < output_len; j++){
      X[i + batch_size * j] = __expf(X[i + batch_size * j]) / sum;
    }
  }
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

ConvBlock * init_conv_block(int incoming_filters, int incoming_spatial_dim, int reduced_depth, int expanded_depth, int stride, bool is_zero){
	ConvBlock * conv_block = malloc(sizeof(ConvBlock));
	conv_block -> incoming_filters = incoming_filters;
	conv_block -> incoming_spatial_dim = incoming_spatial_dim;
	conv_block -> reduced_depth = reduced_depth;
	conv_block -> expanded_depth = expanded_depth;
	conv_block -> stride = stride;

	float * depth_reduction, *spatial, *depth_expansion;
	int depth_reduction_size, spatial_size, depth_expansion_size;
	float depth_reduction_fan_in, spatial_fan_in, depth_expansion_fan_in;

	depth_reduction_size = incoming_filters * reduced_depth;
	depth_reduction_fan_in = incoming_spatial_dim * incoming_spatial_dim * incoming_filters;
	cudaMalloc(&depth_reduction, depth_reduction_size * sizeof(float));
	cudaMemset(depth_reduction, 0, depth_reduction_size * sizeof(float));
	if (!is_zero){
		sample_gaussian <<< SM_COUNT, ceil((float) (depth_reduction_size) / SM_COUNT) >>> (depth_reduction_size, depth_reduction, 0, 2.0 / depth_reduction_fan_in);
	}

	spatial_size = reduced_depth * reduced_depth * 3 * 3;
	spatial_fan_in = incoming_spatial_dim * incoming_spatial_dim * reduced_depth;
	cudaMalloc(&spatial, spatial_size * sizeof(float));
	cudaMemset(spatial, 0, spatial_size * sizeof(float));
	if (!is_zero){
		sample_gaussian <<< SM_COUNT, ceil((float) (spatial_size) / SM_COUNT) >>> (spatial_size, spatial, 0, 2.0 / spatial_fan_in);
	}

	// the spatial decrease happens at middle 3x3 layer, to the last layer of stride block will receive lower spatial dim input
	if (stride == 2){
		incoming_spatial_dim /= 2;
	}

	depth_expansion_size = expanded_depth * reduced_depth;
	depth_expansion_fan_in = incoming_spatial_dim * incoming_spatial_dim * reduced_depth;
	cudaMalloc(&depth_expansion, depth_expansion_size * sizeof(float));
	cudaMemset(depth_expansion, 0, depth_expansion_size * sizeof(float));
	if (!is_zero){
		sample_gaussian <<< SM_COUNT, ceil((float) (depth_expansion_size) / SM_COUNT) >>> (depth_expansion_size, depth_expansion, 0, 2.0 / depth_expansion_fan_in);
	}

	conv_block -> depth_reduction = depth_reduction;
	conv_block -> spatial = spatial;
	conv_block -> depth_expansion = depth_expansion;


	float * projection;
	int projection_size = incoming_filters * expanded_depth;

	if (incoming_filters != expanded_depth){
		cudaMalloc(&projection, projection_size * sizeof(float));
		cudaMemset(project, 0, projection_size * sizeof(float));
		if (!is_zero){
			sample_gaussian <<< SM_COUNT, ceil((float) (projection_size) / SM_COUNT) >>> (projection_size, projection, 0, 2.0 / incoming_filters);
		}
	}
	else{
		projection = NULL;
	}

	conv_block -> projection = projection;

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
	// 3 weight arrays per conv block + inital + fully connected + 4 projections
	// ignoring biases + batch norm weights for now...
	int n_locations = 6 + 3 * n_conv_blocks;
	params -> n_locations = n_locations;

	float ** locations = malloc(n_locations * sizeof(float *));
	int * sizes = malloc(n_locations * sizeof(int));
	// tracking location ind as we start allocating...
	int loc_ind = 0


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
	locations[loc_ind] = init_conv_layer;
	sizes[loc_ind] = init_kernel_dim * init_kernel_dim * init_conv_filters;
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
		locations[loc_ind] = conv_blocks[i] -> spatial;
		sizes[loc_ind] = reduced_depth * reduced_depth * 3 * 3;
		loc_ind++;
		locations[loc_ind] = conv_blocks[i] -> depth_expansion;
		sizes[loc_ind] = expanded_depth * reduced_depth;
		loc_ind++;
		// if the block needed a projection to make input dim = output dim
		if (conv_blocks[i] -> projection){
			locations[loc_ind] = conv_blocks[i] -> projection;
			sizes[loc_ind] = incoming_filters * expanded_depth;
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

Activation_ConvBlock * init_activation_convblock(ConvBlock * conv_block, int batch_size){
	Activation_ConvBlock * activation_conv_block = malloc(sizeof(Activation_ConvBlock));

	activation_conv_block -> incoming_filters = conv_block -> incoming_filters;
	activation_conv_block -> incoming_spatial_dim = conv_block -> incoming_spatial_dim;
	activation_conv_block -> reduced_depth = conv_block -> reduced_depth;
	activation_conv_block -> expanded_depth = conv_block -> expanded_depth;
	activation_conv_block -> stride = conv_block -> stride;

	float * post_reduced, *post_spatial, *post_expanded, *transformed_residual, *output;
	int post_reduced_size, post_spatial_size, output_size;

	post_reduced_size = reduced_depth * incoming_spatial_dim * incoming_spatial_dim * batch_size;
	cudaMalloc(&post_reduced, post_reduced_size * sizeof(float));
	activation_conv_block -> post_reduced_size = post_reduced;

	post_spatial_size = reduced_depth * incoming_spatial_dim * incoming_spatial_dim / (stride * stride) * batch_size;
	cudaMalloc(&post_spatial, post_spatial_size * sizeof(float));
	activation_conv_block -> post_spatial = post_spatial;

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

	return activation_conv_block;
}

Activations * init_activations(Dims * dims, ConvBlock ** conv_blocks, int batch_size){
	
	Activations * activations = malloc(sizeof(Activations));

	int input_dim = dims -> input;
	int init_conv_filters = dims -> init_conv_filters;
	int init_conv_stride = dims -> init_conv_stride;
	int maxpool_stride = dims -> init_maxpool_stride;

	float * init_conv_activation;
	int init_conv_activation_size = init_conv_filters * input_dim * input_dim / (init_stride * init_stride) * batch_size; 
	cudaMalloc(&init_conv_activation, init_conv_activation_size);
	activations -> init_conv_activation = init_conv_activation;

	float *init_convblock_input;
	int init_convblock_input_size = init_conv_filters * input_dim * input_dim / (init_stride * init_stride) / (maxpool_stride * maxpool_stride) * batch_size;
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

	int input_dim = dims -> input;
	int init_conv_filters = dims -> init_conv_filters;
	int init_conv_stride = dims -> init_conv_stride;
	int maxpool_stride = dims -> init_maxpool_stride;

	int init_convblock_input_size = init_conv_filters * input_dim * input_dim / (init_stride * init_stride) / (maxpool_stride * maxpool_stride) * batch_size;

	int * max_inds;
	cudaMalloc(&max_inds, init_convblock_input_size * sizeof(int));
	forward_buffer -> max_inds = max_inds;

	forward_buffer -> activations = init_activations(dims, conv_blocks, batch_size);

	int output_dim = dims -> output;
	int output_size = output_dim * batch_size;

	float * pred;
	cudaMalloc(&pred, output_size * sizeof(float));
	forward_buffer -> pred = pred;

	// will be copied to cpu to be able to print values and compute loss on cpu side
	float * pred_cpu = malloc(output_size * sizeof(float));
	// will be the maximum of prediction of each sample in batch and converted to label string
	char ** predicted_labels = malloc(batch_size * sizeof(char *));

	forward_buffer -> pred_cpu = pred_cpu;
	forward_buffer -> predicted_labels = predicted_labels;

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
		images_float_cpu[pixel] = (float) images_cpu[pixel];
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