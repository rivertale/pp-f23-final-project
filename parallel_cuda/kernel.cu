#include <cuda.h>
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void classifyPointsKernel(Color4 *centroid, int *label, Color4 *pixels, int *migration_count, int cluster_count, int total_pixel){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < total_pixel){
        int index = -1;
        int min_dist = 10000000;
        for (int j = 0; j < cluster_count; j++)
        {
            int dist = (pixels[i].r - centroid[j].r) * (pixels[i].r - centroid[j].r)\
             + (pixels[i].g - centroid[j].g) * (pixels[i].g - centroid[j].g) + \
             (pixels[i].b - centroid[j].b) * (pixels[i].b - centroid[j].b) +\
              (pixels[i].a - centroid[j].a) * (pixels[i].a - centroid[j].a);
            //printf("%d ",pixels[i].r);
            if (dist < min_dist)
            {   
                index = j;
                min_dist = dist;
            }
        }
        // printf("%d\n",index);
        if (index != label[i])
            atomicAdd(migration_count, 1);
        label[i] = index;
    }
}

void host_classify_points(Color4 *centroid, int *label, Color4 *pixels, int *migration_count, int cluster_count, int total_pixel)
{
    Color4 *centroid_d;
    cudaMalloc(&centroid_d, cluster_count*sizeof(Color4));
    cudaMemcpy(centroid_d, centroid, cluster_count*sizeof(Color4), cudaMemcpyHostToDevice);

    Color4 *pixels_d;
    cudaMalloc(&pixels_d, total_pixel*sizeof(Color4));
    cudaMemcpy(pixels_d, pixels, total_pixel*sizeof(Color4), cudaMemcpyHostToDevice);

    /*int *pre_label_d;
    cudaMalloc(&pre_label_d, total_pixel*sizeof(int));
    cudaMemcpy(pre_label_d, pre_label, total_pixel*sizeof(int), cudaMemcpyHostToDevice);*/

    int *label_d;
    cudaMalloc(&label_d, total_pixel*sizeof(int));
    cudaMemcpy(label_d, label, total_pixel*sizeof(int), cudaMemcpyHostToDevice);

    int *migration_count_d;
    cudaMalloc(&migration_count_d, sizeof(int));

    dim3 threadsPerBlock(256);
    dim3 numBlocks((total_pixel + threadsPerBlock.x - 1) / threadsPerBlock.x);
    classifyPointsKernel<<<numBlocks, threadsPerBlock>>>(centroid_d, label_d, pixels_d, migration_count_d, cluster_count, total_pixel);
    cudaDeviceSynchronize();

    cudaMemcpy(label, label_d, total_pixel*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(migration_count, migration_count_d, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(centroid_d);
    cudaFree(pixels_d);
    cudaFree(label_d);
    //cudaFree(pre_label_d);
    cudaFree(migration_count_d);
}

__global__ void updateCentroidKernel(Color4_SUM* labelSum, int* labelCount, Color4* pixels, int* label, int total_pixel, int cluster_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory for label count and sum
    extern __shared__ Color4_SUM sharedLabelSum[];
    //extern __shared__ int sharedLabelCount[];

    if (tid == 0){
        for (int i = 0; i < cluster_count; i++)
        {
            labelSum[i].r = 0;
            labelSum[i].g = 0;
            labelSum[i].b = 0;
            labelSum[i].a = 0;

            labelCount[i] = 0;
        }
    }

    if (threadIdx.x == 0){
        for (int i = 0; i < cluster_count; i++)
        {
            sharedLabelSum[i].r = 0;
            sharedLabelSum[i].g = 0;
            sharedLabelSum[i].b = 0;
            sharedLabelSum[i].a = 0;

            //sharedLabelCount[i] = 0;
        }

    }

    __syncthreads();

    // Update shared memory with partial sums and counts
    if (tid < total_pixel)
    {
        int currentLabel = label[tid];
        //atomicAdd(&sharedLabelCount[currentLabel], 1);
        //printf("BlockIdx: %d, sharedLabelCount[0]: %d, sharedLabelSum[0].r: %d\n", blockIdx.x, sharedLabelCount[0], sharedLabelSum[0].r);
        atomicAdd(reinterpret_cast<unsigned int*>(&sharedLabelSum[currentLabel].r), static_cast<unsigned int>(pixels[tid].r));
        atomicAdd(reinterpret_cast<unsigned int*>(&sharedLabelSum[currentLabel].g), static_cast<unsigned int>(pixels[tid].g));
        atomicAdd(reinterpret_cast<unsigned int*>(&sharedLabelSum[currentLabel].b), static_cast<unsigned int>(pixels[tid].b));
        atomicAdd(reinterpret_cast<unsigned int*>(&sharedLabelSum[currentLabel].a), 1);
    }

    __syncthreads();

    // Update global memory with shared memory values
    if (threadIdx.x == 0)
    {   
        for(int i = 0; i < cluster_count; i++)
        {
            //atomicAdd(&labelCount[i], sharedLabelCount[i]);
            //printf("slc: %d ", sharedLabelCount[i]);
            atomicAdd(reinterpret_cast<unsigned int*>(&labelSum[i].r), static_cast<unsigned int>(sharedLabelSum[i].r));
            atomicAdd(reinterpret_cast<unsigned int*>(&labelSum[i].g), static_cast<unsigned int>(sharedLabelSum[i].g));
            atomicAdd(reinterpret_cast<unsigned int*>(&labelSum[i].b), static_cast<unsigned int>(sharedLabelSum[i].b));
            atomicAdd(reinterpret_cast<unsigned int*>(&labelSum[i].a), static_cast<unsigned int>(sharedLabelSum[i].a));
        }
    }
}

void host_update_centroid(Color4 *centroid, int *label, Color4 *pixels, int cluster_count, int total_pixel)
{
    int *label_d;
    cudaMalloc(&label_d, total_pixel*sizeof(int));
    cudaMemcpy(label_d, label, total_pixel*sizeof(int), cudaMemcpyHostToDevice);

    Color4 *pixels_d;
    cudaMalloc(&pixels_d, total_pixel*sizeof(Color4));
    cudaMemcpy(pixels_d, pixels, total_pixel*sizeof(Color4), cudaMemcpyHostToDevice);

    Color4_SUM *label_sum = (Color4_SUM *)malloc(cluster_count * sizeof(Color4_SUM));
    Color4_SUM *label_sum_d;
    cudaMalloc(&label_sum_d, cluster_count*sizeof(Color4_SUM));

    int *label_count = (int *)malloc(cluster_count * sizeof(int));
    int *label_count_d;
    cudaMalloc(&label_count_d, cluster_count*sizeof(int));

    // Calculate each number of each label
    dim3 threadsPerBlock(256);
    dim3 numBlocks((total_pixel + threadsPerBlock.x - 1) / threadsPerBlock.x);
    int sharedMemorySize = sizeof(Color4_SUM) * cluster_count + sizeof(int) * + cluster_count;
    updateCentroidKernel<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(label_sum_d, label_count_d, pixels_d, label_d, total_pixel, cluster_count);
    cudaDeviceSynchronize();
    
    cudaMemcpy(label_sum, label_sum_d, cluster_count*sizeof(Color4_SUM), cudaMemcpyDeviceToHost);
    //cudaMemcpy(label_count, label_count_d, cluster_count*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0;  i < cluster_count; i++)
        label_count[i] = label_sum[i].a;

    //printf("%d\n", total_pixel);
    for (int i = 0; i < cluster_count; i++)
    { 
        printf("%d ", label_count[i]);
        printf("%ld ", label_sum[i].r);
        printf("%ld ", label_sum[i].g);
        printf("%ld ", label_sum[i].b);
        printf("%ld \n", label_sum[i].a);
    }
    printf("\n");

    for (int i = 0; i < cluster_count; i++)
    {
        // printf("%ld",label_sum[i].r/label_count[i]);
        if (label_sum[i].r != 0 && label_count[i] != 0)
            centroid[i].r = label_sum[i].r / label_count[i];
        if (label_sum[i].g != 0 && label_count[i] != 0)
            centroid[i].g = label_sum[i].g / label_count[i];
        if (label_sum[i].b != 0 && label_count[i] != 0)
            centroid[i].b = label_sum[i].b / label_count[i];
        if (label_sum[i].a != 0 && label_count[i] != 0)
            centroid[i].a = 0;
    }
    for(int i = 0; i < cluster_count; i++){
        printf("%d ", centroid[i].r);
        printf("%d ",centroid[i].g);
        printf("%d ",centroid[i].b);
        printf("\n");
    }

    free(label_sum);
    free(label_count);
    cudaFree(label_sum_d);
    cudaFree(label_count_d);
    cudaFree(label_d);
    cudaFree(pixels_d);
}