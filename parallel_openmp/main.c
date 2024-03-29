#include <stdio.h>
#include <malloc.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include "common.h"
#include "profile.h"

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

static int 
load_image_info(Image *image, char *path)
{
    int result = 0;
    clear_memory(image, sizeof(*image));
    FILE *file_handle = fopen(path, "rb");
    if(file_handle)
    {
        int width, height, channel_count;
        if(stbi_info_from_file(file_handle, &width, &height, &channel_count))
        {
            result = 1;
            image->width = width;
            image->height = height;
            image->handle = file_handle;
        }
        else
        {
            fclose(file_handle);
        }
    }
    return result;
}

static void
free_image_info(Image *image)
{
    if(image)
    {
        fclose(image->handle);
        clear_memory(image, sizeof(*image));
    }
}

static void 
load_image_data(Color4 *output, Image *image)
{
    int image_width, image_height, channel_count;
    unsigned char *input_pixels = stbi_load_from_file(image->handle, &image_width, &image_height, &channel_count, 3);
    if(input_pixels)
    {
        Color4 *output_pixel = output;
        for(int y = 0; y < image->height; ++y)
        {
            for(int x = 0; x < image->width; ++x)
            {
                unsigned char *channels = input_pixels + (image_height - 1 - y)*image_width*3 + x*3;
                output_pixel->r = channels[0];
                output_pixel->g = channels[1];
                output_pixel->b = channels[2];
                output_pixel->a = 0;
                ++output_pixel;
            }
        }
    }
    else
    {
        clear_memory(output, image->width * image->height * sizeof(*output));
    }
}

static int 
write_image(char *path, Color4 *bitmap, int width, int height)
{
    int result = 0;
    int *data = (int *)malloc(height * width * sizeof(int));
    if(data)
    {
        for(int y = 0; y < height; ++y)
        {
            for(int x = 0; x < width; ++x)
            {
                Color4 *channel = bitmap + y * width + x;
                int r = channel->r;
                int g = channel->g;
                int b = channel->b;
                int a = 255;
                data[(height - 1 - y) * width + x] = (a << 24) | (b << 16) | (g << 8) | (r << 0);
            }
        }
        result = stbi_write_png(path, width, height, 4, data, width * sizeof(int));
        free(data);
    }
    return result;
}


void AllocateRandomClusters(Color4 * centroid, Color4 * pixels, short clustercount)
{
  printf("Initial randomized cluster centres: \n\n");
  for(int i = 100; i < clustercount; i++)
  {
    centroid[i].r = pixels[i].r;
    centroid[i].g = pixels[i].g;
    centroid[i].b = pixels[i].b;
    centroid[i].a = pixels[i].a;
  }
}

void classify_points(Color4 *centroid, int *pre_label, int *label, Color4 *pixels, int *migration_count, int cluster_count, int total_pixel)
{

    for (int i = 0; i < total_pixel; i++)
    {
        int index = -1;
        int min_dist = 1000000;
        for (int j = 0; j < cluster_count; j++)
        {
            int dist = 0;
            dist = dist + pow(pixels[i].r - centroid[j].r, 2);
            dist = dist + pow(pixels[i].g - centroid[j].g, 2);
            dist = dist + pow(pixels[i].b - centroid[j].b, 2);
            dist = dist + pow(pixels[i].a - centroid[j].a, 2);
            dist = sqrt(dist);
            // printf("%d ",dist);
            if (dist < min_dist)
            {
                index = j;
                min_dist = dist;
            }
        }
        // printf("%d\n",index);
        if (index != pre_label[i])
            (*migration_count)++;
        label[i] = index;
    }
}

void update_centroid(Color4 * centroid, int * label, Color4 * pixels, int  cluster_count, int total_pixel){
    
    Color4_SUM * label_sum = (Color4_SUM*) calloc(cluster_count , sizeof(Color4_SUM));
    int* label_count = (int*) calloc(cluster_count , sizeof(int));

    //Calculate each number of each label
    for(int i = 0; i < total_pixel; i++)
    {   
        label_count[label[i]] +=1;
    }

    for(int i = 0; i < total_pixel; i++)
    {   
        label_sum[label[i]].r += pixels[i].r;
        label_sum[label[i]].g += pixels[i].g;
        label_sum[label[i]].b += pixels[i].b;
        label_sum[label[i]].a += pixels[i].a;
    }

    for(int i = 0; i < cluster_count; i++)
    {   
        if(label_sum[i].r != 0 && label_count[i] != 0)
            centroid[i].r = label_sum[i].r/label_count[i];
        if(label_sum[i].g != 0 && label_count[i] != 0)
            centroid[i].g = label_sum[i].g/label_count[i];
        if(label_sum[i].b != 0 && label_count[i] != 0)
            centroid[i].b = label_sum[i].b/label_count[i];
        if(label_sum[i].a != 0 && label_count[i] != 0)
        centroid[i].a = label_sum[i].a/label_count[i];
    }

    free(label_sum);
    free(label_count);
}

void output_result(int * label, Color4 * output, Color4 * centroid, int cluster_count, int total_pixel){

    for(int i = 0; i < total_pixel; i++)
    {   
        output[i].r = centroid[label[i]].r;
        output[i].g = centroid[label[i]].g;
        output[i].b = centroid[label[i]].b;
        output[i].a = centroid[label[i]].a;
    }
}

static void 
Kmean(Color4 *output, Color4 *pixels, int width, int height, 
                         int cluster_count, int max_iteration, float migration_threshold,
                         int *out_iteration)
{   
    
    int pixel_count = width * height;
    Color4 *centroid = (Color4 *) malloc(cluster_count * sizeof(Color4));
    int* label = (int*) malloc(pixel_count * sizeof(int));
    int *previous_label = (int *)calloc(pixel_count, sizeof(int));

    AllocateRandomClusters(centroid, pixels, cluster_count);

    int migration_count;
    for(int i = 0; i < max_iteration; i++)
    {   
        
        migration_count = 0;
        classify_points(centroid, previous_label, label, pixels, &migration_count, cluster_count, pixel_count);
        update_centroid(centroid, label, pixels, cluster_count, pixel_count);
        //Check threshold
        if (migration_count / (float)pixel_count < migration_threshold)
        {
            *out_iteration = i;
            break;
        }
        memcpy(previous_label, label, pixel_count * sizeof(int));
    }
    output_result(label, output, centroid, cluster_count, pixel_count);
    //*out_iteration = max_iteration;
    
    free(label);
    free(previous_label);
    free(centroid);
}

int main(int arg_count, char **args)
{
    int parsing_arg_index = 1;
    int show_usage = 0;
    int verbose = 1;
    int cluster_count = 4;
    int max_iteration = 200;
    float migration_threshold = 0.01f;

    for(; parsing_arg_index < arg_count; ++parsing_arg_index)
    {
        char *option = args[parsing_arg_index];
        if(option[0] != '-') break;
        
        if(option[1] == 'n' && option[2] == '=')
        {
            cluster_count = atoi(option + 3);
        }
        else if(option[1] == 'm' && option[2] == '=')
        {
            max_iteration = atoi(option + 3);
        }
        else if(option[1] == 'r' && option[2] == '=')
        {
            migration_threshold = atof(option + 3);
        }
        else if(option[1] == 'q' && option[2] == 0)
        {
            verbose = 0;
        }
        else if(option[1] == 'h' && option[2] == 0)
        {
            show_usage = 1;
        }
        else
        {
            printf("unknown option '%s'\n", option);
        }
    }
    
    if(show_usage || (parsing_arg_index + 2 != arg_count))
    {
        char *usage = "usage: kmean [option] ... input_path output_path\n"
                      "options:\n"
                      "    -n={cluster_count}  number of clusters (default is 4)\n"
                      "    -m={max_iteration}  max iteration of kmean clustering (default is 200)\n"
                      "    -r={threshold}      exit when the data point migration ratio between clusters exceeds this value (default is 0.01)\n"
                      "    -q                  quiet mode (no output)\n"
                      "    -h                  print this help information\n";
        //NOTE: pass the string via '%s' to shut up the compiler warning
        if(verbose) printf("%s", usage);
        return 0;
    }
    
    char *input_path = args[parsing_arg_index + 0];
    char *output_path = args[parsing_arg_index + 1];
    size_t output_path_len = string_len(output_path);
    printf("Input_path: %s\nOutput_path:%s\n",input_path,output_path);
    
    if(output_path_len >= 4 && 
       (output_path[output_path_len-4] == '.') && 
       (output_path[output_path_len-3] == 'p' || output_path[output_path_len-3] == 'P') && 
       (output_path[output_path_len-2] == 'n' || output_path[output_path_len-2] == 'N') && 
       (output_path[output_path_len-1] == 'g' || output_path[output_path_len-1] == 'G'))
    {
        Image image;
        if(load_image_info(&image, input_path))
        {
            Color4 *input = (Color4 *)malloc(image.width * image.height * sizeof(Color4));
            Color4 *output = (Color4 *)malloc(image.width * image.height * sizeof(Color4));
            if(input && output)
            {
                load_image_data(input, &image);
                int used_iteration;
                unsigned long long start_time = get_microsecond_from_epoch();
                Kmean(output, input, image.width, image.height, 
                                         cluster_count, max_iteration, migration_threshold,  
                                         &used_iteration);
                unsigned long long end_time = get_microsecond_from_epoch();
                if(verbose)
                {
                    printf("[summary]\n");
                    printf("    used iteration = %d\n", used_iteration);
                    printf("    time = %fs\n", (end_time - start_time) / 1000000.0f);
                }
                
                if(write_image(output_path, output, image.width, image.height))
                {
                    // NOTE: success
                }
                else
                {
                    if(verbose) printf("ERROR: write '%s' failed\n", output_path);
                }
            }
            else
            {
                if(verbose) printf("ERROR: out of memory\n");
            }
            
            if(input) free(input);
            if(output) free(output);
            free_image_info(&image);
        }
        else
        {
            if(verbose) printf("ERROR: read '%s' failed\n", input_path);
        }
    }
    else
    {
       if(verbose) printf("ERROR: output should end with '.png' extension\n");
    }
    return 0;
}