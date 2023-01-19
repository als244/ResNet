#define  _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// 1,281,167 images in ImageNet dataset
// setting each shard to have 32,768 images = (256 * 256 * 3 * 4 * 32,768 ~ 25 GB)
// total of 40 shards

// each shard will have n_images_in_shard images from random classes 
// all grouped together in one file for easy read access when loading new batches in training
void build_shard(int shard_id, long image_dim_in, long image_dim_out, long channels, long n_images){
	
	size_t image_size_in = image_dim_in * image_dim_in * channels;
	size_t image_size_out = image_dim_out * image_dim_out * channels;

	size_t total_pixels = image_size_out * n_images;

	uint8_t * image_bytes = (uint8_t *) malloc(total_pixels * sizeof(uint8_t));
	

	int * classes = (int *) malloc(n_images * sizeof(int));
	int * img_nums = (int *) malloc(n_images * sizeof(int));
	int * row_offs = (int *) malloc(n_images * sizeof(int));
	int * col_offs = (int *) malloc(n_images * sizeof(int));

	char shard_partition_filepath[76];
	sprintf(shard_partition_filepath, "/mnt/storage/data/vision/imagenet/2012/train_data_partioning/%03d_images.csv", shard_id);
	shard_partition_filepath[75] = '\0';

	FILE * fp;
    char * line = NULL;
    size_t len = 0;

    fp = fopen(shard_partition_filepath, "r");

    printf("Reading Partioning Info...\n\n");
    long img_cnt = 0;
    while (getline(&line, &len, fp) != -1) {

    	char * img_class_str = strndup(line, 3);
    	char * img_num_str = strndup(line + 4, 4);
    	char * row_off_str = strndup(line + 9, 2);
    	char * col_off_str = strndup(line + 12, 2);
    	
    	int img_class = atoi(img_class_str);
    	int img_num = atoi(img_num_str);
    	int row_off = atoi(row_off_str);
    	int col_off = atoi(col_off_str);
    	
    	classes[img_cnt] = img_class;
    	img_nums[img_cnt] = img_num;
    	row_offs[img_cnt] = row_off;
    	col_offs[img_cnt] = col_off; 
    	
    	free(img_class_str);
    	free(img_num_str);
    	free(row_off_str);
    	free(col_off_str);

    	img_cnt++;
    }

    fclose(fp);

    size_t true_total_pixels = img_cnt * image_size_out;
	// reading (image_class, image_num) from partioning file
	
	FILE * f;
	char file_path[66];
	int class_id, image_id, row_offset, col_offset;
	printf("Loading Image Bytes...\n\n");
	for (int i = 0; i < img_cnt; i++){
		if (i % 1000 == 0){
			printf("On Image #%d\n", i);
		}
		class_id = classes[i];
		sprintf(file_path, "/mnt/storage/data/vision/imagenet/2012/train_data/%08d.buffer", class_id);
		file_path[65] = '\0';
		image_id = img_nums[i];
		f = fopen(file_path, "rb");
		if (f == NULL){
			printf("ERROR OPENING IMAGE CLASS FILE %d\n", class_id);
			return;
		}
		row_offset = row_offs[i];
		col_offset = col_offs[i];
		long total_off;
		size_t bytes_in_row = image_dim_out * channels;
		for (int row_id = 0; row_id < image_dim_out; row_id++){
			total_off = (row_offset + row_id) * image_dim_in * channels + col_offset * channels;
			fseek(f, image_id * image_size_in + total_off, SEEK_SET);
			fread(image_bytes + i * image_size_out + row_id * bytes_in_row, sizeof(uint8_t), bytes_in_row, f);
		}
		fclose(f);
	}

	// array is linear format where each sequence of image_size [0, image_size) is image 1, then [image_size, 2 * image_size) has image 2
	// each image is also linearized where ording of pixels is - 0, 0: (R, G, B) then 0, 1: (R,G,B), ...
	printf("Converting Image Bytes to Floats...\n\n");
	float * image_floats = (float *) malloc(true_total_pixels * sizeof(float));
	if (image_floats == NULL){
		printf("Malloc Failed\n");
		printf("true_total_pixels: %ld\n", true_total_pixels);
	}

	for (int pixel = 0; pixel < true_total_pixels; pixel++){
		image_floats[pixel] = ((float) (image_bytes[pixel])) * (2.0 / 255) - 1;
	}

	// write out images_float_cpu and correct_classes_cpu to new shard files
	printf("Writing Image Floats & Labels to Shard Files...\n\n");
	char shard_images_filepath[68];
	char shard_labels_filepath[68];
	sprintf(shard_images_filepath, "/mnt/storage/data/vision/imagenet/2012/train_data_shards/%03d.images", shard_id);
	shard_images_filepath[67] = '\0';
	
	FILE * shard_images_file = fopen(shard_images_filepath, "wb");
	fwrite(image_floats, sizeof(float), true_total_pixels, shard_images_file);
	fclose(shard_images_file);

	sprintf(shard_labels_filepath, "/mnt/storage/data/vision/imagenet/2012/train_data_shards/%03d.labels", shard_id);
	shard_labels_filepath[67] = '\0';

	FILE * shard_labels_file = fopen(shard_labels_filepath, "wb");
	fwrite(classes, sizeof(int), img_cnt, shard_labels_file);
	fclose(shard_labels_file);

	free(image_bytes);
	free(image_floats);
	free(classes);
	free(img_nums);
	free(row_offs);
	free(col_offs);
}


int main(int argc, char *argv[]) {

	int n_shards = 40;
	long image_dim_in = 256;
	long image_dim_out = 224;
	long channels = 3;
	long n_images = 32768;

	for (int shard_id = 0; shard_id < n_shards; shard_id++){
		printf("Building Shard #%d\n\n", shard_id);
		build_shard(shard_id, image_dim_in, image_dim_out, channels, n_images);
		printf("\n\n\n");
	}
}