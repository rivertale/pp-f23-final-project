#ifndef KERNEL_H_
#define KERNEL_H_
#include <stdio.h>
#include <stdlib.h>


typedef struct Color4
{
    unsigned char r, g, b, a;
} Color4;

typedef struct Color4_SUM
{
    long r, g, b, a;
} Color4_SUM;

typedef struct Image
{
    int width, height;
    FILE *handle;
} Image;

//extern "C"
void host_classify_points(Color4 *centroid, int *label, Color4 *pixels, int *migration_count, 
                        int cluster_count, int total_pixel);

void host_update_centroid(Color4 *centroid, int *label, Color4 *pixels, int cluster_count, int total_pixel);

#endif /* KERNEL_H_ */