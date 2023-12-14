
#include "common.h"
#include "thread.h"
#include "profile.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include <malloc.h>

typedef struct Color4
{
    unsigned char r, g, b, a;
} Color4;

typedef struct Image
{
    int width, height;
    FILE *handle;
} Image;

typedef struct KmeansFilterWork
{
    int pixel_count;
    int cluster_count;
    Color4 *pixels;
    Color4 *cluster_colors;
    
    int *cluster_indices;
    int out_migration_count;
    
    unsigned long long *out_cluster_sums_r;
    unsigned long long *out_cluster_sums_g;
    unsigned long long *out_cluster_sums_b;
    int *out_cluster_pixel_counts;
} KmeansFilterWork;

typedef struct FillImageWork
{
    int pixel_count;
    int *cluster_indices;
    Color4 *cluster_colors;
    
    Color4 *out_pixels;
} FillImageWork;

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

static void 
do_kmeans_filter_work(void *param)
{
    KmeansFilterWork *work = (KmeansFilterWork *)param;
    work->out_migration_count = 0;
    clear_memory(work->out_cluster_sums_r, work->cluster_count * sizeof(unsigned long long));
    clear_memory(work->out_cluster_sums_g, work->cluster_count * sizeof(unsigned long long));
    clear_memory(work->out_cluster_sums_b, work->cluster_count * sizeof(unsigned long long));
    clear_memory(work->out_cluster_pixel_counts, work->cluster_count * sizeof(int));
    
    for(int pixel_index = 0; pixel_index < work->pixel_count; ++pixel_index)
    {
        int min_test_index = 0;
        float first_r_diff = work->pixels[pixel_index].r - work->cluster_colors[0].r;
        float first_g_diff = work->pixels[pixel_index].g - work->cluster_colors[0].g;
        float first_b_diff = work->pixels[pixel_index].b - work->cluster_colors[0].b;
        float min_diff = first_r_diff*first_r_diff + first_g_diff*first_g_diff + first_b_diff*first_b_diff;
        
        for(int test_index = 1; test_index < work->cluster_count; ++test_index)
        {
            float r_diff = work->pixels[pixel_index].r - work->cluster_colors[test_index].r;
            float g_diff = work->pixels[pixel_index].g - work->cluster_colors[test_index].g;
            float b_diff = work->pixels[pixel_index].b - work->cluster_colors[test_index].b;
            float diff = r_diff*r_diff + g_diff*g_diff + b_diff*b_diff;
            if(diff < min_diff)
            {
                min_test_index = test_index;
                min_diff = diff;
            }
        }
        
        if(work->cluster_indices[pixel_index] != min_test_index)
        {
            ++work->out_migration_count;
            work->cluster_indices[pixel_index] = min_test_index;
        }
        ++work->out_cluster_pixel_counts[min_test_index];
        work->out_cluster_sums_r[min_test_index] += work->pixels[pixel_index].r;
        work->out_cluster_sums_g[min_test_index] += work->pixels[pixel_index].g;
        work->out_cluster_sums_b[min_test_index] += work->pixels[pixel_index].b;
    }
}

static void 
do_fill_image_work(void *param)
{
    FillImageWork *work = (FillImageWork *)param;
    for(int i = 0; i < work->pixel_count; ++i)
    {
        work->out_pixels[i] = work->cluster_colors[work->cluster_indices[i]];
    }
}

static void 
filter_bitmap_with_kmean(Color4 *output, Color4 *pixels, int width, int height, 
                         int cluster_count, int max_iteration, float migration_threshold, 
                         WorkQueue *queue, int thread_count, 
                         int *out_iteration)
{
    int pixel_count = width * height;
    int pixel_per_thread = (pixel_count + thread_count - 1) / thread_count;
    if(cluster_count <= pixel_count)
    {
        size_t working_size_per_thread = align_to(sizeof(KmeansFilterWork) + sizeof(FillImageWork) + 
                                                  cluster_count*sizeof(unsigned long long)*3 + cluster_count*sizeof(int), 
                                                  128);
        char *working_memory = (char *)malloc(thread_count * working_size_per_thread + 128);
        int *cluster_indices = (int *)malloc(pixel_count * sizeof(int));
        Color4 *cluster_colors = (Color4 *)malloc(cluster_count * sizeof(Color4));
        if(working_memory && cluster_indices && cluster_colors)
        {
            clear_memory(working_memory, thread_count * working_size_per_thread + 128);
            clear_memory(cluster_indices, pixel_count * sizeof(int));
            for(int i = 0; i < cluster_count; ++i)
            {
                cluster_colors[i] = pixels[i];
            }
            
            char *initial_ptr_to_allocate = (char *)align_to((size_t)working_memory, 128);
            char *ptr_to_allocate = initial_ptr_to_allocate;
            for(int thread_index = 0; thread_index < thread_count; ++thread_index)
            {
                size_t pixel_remaining = pixel_count - thread_index*pixel_per_thread;
                KmeansFilterWork *kmeans_work = (KmeansFilterWork *)ptr_to_allocate;
                FillImageWork *fill_work = (FillImageWork *)(ptr_to_allocate + sizeof(*kmeans_work));
                kmeans_work->pixel_count = (pixel_remaining < pixel_per_thread) ? pixel_remaining : pixel_per_thread;
                kmeans_work->cluster_count = cluster_count;
                kmeans_work->pixels = pixels + thread_index * pixel_per_thread;
                kmeans_work->cluster_indices = cluster_indices + thread_index * pixel_per_thread;
                kmeans_work->cluster_colors = cluster_colors;
                kmeans_work->out_migration_count = 0;
                kmeans_work->out_cluster_sums_r = (unsigned long long *)(ptr_to_allocate + sizeof(*kmeans_work) + sizeof(*fill_work));
                kmeans_work->out_cluster_sums_g = (unsigned long long *)(ptr_to_allocate + sizeof(*kmeans_work) + sizeof(*fill_work) + 1*cluster_count*sizeof(unsigned long long));
                kmeans_work->out_cluster_sums_b = (unsigned long long *)(ptr_to_allocate + sizeof(*kmeans_work) + sizeof(*fill_work) + 2*cluster_count*sizeof(unsigned long long));
                kmeans_work->out_cluster_pixel_counts = (int *)(ptr_to_allocate + sizeof(*kmeans_work) + sizeof(*fill_work) + 3*cluster_count*sizeof(unsigned long long));
                ptr_to_allocate += working_size_per_thread;
            }
            
            int max_migration = migration_threshold * pixel_count;
            int iteration = 0;
            for(; iteration < max_iteration; ++iteration)
            {
                for(int thread_index = 0; thread_index < thread_count; ++thread_index)
                {
                    KmeansFilterWork *work = (KmeansFilterWork *)(initial_ptr_to_allocate + thread_index*working_size_per_thread);
                    queue_work(queue, do_kmeans_filter_work, work);
                }
                complete_all_works(queue);
                
                for(int cluster_index = 0; cluster_index < cluster_count; ++cluster_index)
                {
                    int count = 0;
                    float r_sum = 0.0f;
                    float g_sum = 0.0f;
                    float b_sum = 0.0f;
                    for(int thread_index = 0; thread_index < thread_count; ++thread_index)
                    {
                        KmeansFilterWork *work = (KmeansFilterWork *)(initial_ptr_to_allocate + thread_index*working_size_per_thread);
                        r_sum += work->out_cluster_sums_r[cluster_index];
                        g_sum += work->out_cluster_sums_g[cluster_index];
                        b_sum += work->out_cluster_sums_b[cluster_index];
                        count += work->out_cluster_pixel_counts[cluster_index];
                    }
                    cluster_colors[cluster_index].r = r_sum / count;
                    cluster_colors[cluster_index].g = g_sum / count;
                    cluster_colors[cluster_index].b = b_sum / count;
                }
                
                int total_migration_count = 0;
                for(int thread_index = 0; thread_index < thread_count; ++thread_index)
                {
                    KmeansFilterWork *work = (KmeansFilterWork *)(initial_ptr_to_allocate + thread_index*working_size_per_thread);
                    total_migration_count += work->out_migration_count;
                }
                if(total_migration_count < max_migration) break;
            }
            
            for(int thread_index = 0; thread_index < thread_count; ++thread_index)
            {
                size_t pixel_remaining = pixel_count - pixel_per_thread * thread_index;
                FillImageWork *work = (FillImageWork *)(initial_ptr_to_allocate + thread_index*working_size_per_thread + sizeof(KmeansFilterWork));
                work->pixel_count = (pixel_remaining < pixel_per_thread) ? pixel_remaining : pixel_per_thread;
                work->cluster_colors = cluster_colors;
                work->cluster_indices = cluster_indices + thread_index * pixel_per_thread;
                work->out_pixels = output + thread_index * pixel_per_thread;
                queue_work(queue, do_fill_image_work, work);
            }
            complete_all_works(queue);
            *out_iteration = iteration;
        }
        
        if(working_memory) free(working_memory);
        if(cluster_indices) free(cluster_indices);
        if(cluster_colors) free(cluster_colors);
    }
    else
    {
        for(int i = 0; i < pixel_count; ++i)
        {
            output[i] = pixels[i];
        }
    }
}

int 
main(int arg_count, char **args)
{
    int thread_count = get_thread_count();
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
        else if(option[1] == 't' && option[2] == '=')
        {
            thread_count = atoi(option + 3);
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
                      "    -t={thread_count}   number of used threads (default is the number of logical core)\n"
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
    if(output_path_len >= 4 && 
       (output_path[output_path_len-4] == '.') && 
       (output_path[output_path_len-3] == 'p' || output_path[output_path_len-3] == 'P') && 
       (output_path[output_path_len-2] == 'n' || output_path[output_path_len-2] == 'N') && 
       (output_path[output_path_len-1] == 'g' || output_path[output_path_len-1] == 'G'))
    {
        Image image;
        WorkQueue work_queue;
        create_work_queue(&work_queue, thread_count - 1);
        if(load_image_info(&image, input_path))
        {
            Color4 *input = (Color4 *)malloc(image.width * image.height * sizeof(Color4));
            Color4 *output = (Color4 *)malloc(image.width * image.height * sizeof(Color4));
            if(input && output)
            {
                load_image_data(input, &image);
                int used_iteration;
                unsigned long long start_time = get_microsecond_from_epoch();
                filter_bitmap_with_kmean(output, input, image.width, image.height, 
                                         cluster_count, max_iteration, migration_threshold, 
                                         &work_queue, thread_count, 
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