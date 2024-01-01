#include <stdlib.h>
#include <assert.h>

#define is_pow_of_two(value) (((value) & ((value) - 1)) == 0)

static size_t
string_len(char *string)
{
    size_t result = 0;
    while(*string++) ++result;
    return result;
}

static size_t
align_to(size_t value, size_t alignment)
{
    assert(is_pow_of_two(alignment));
    return (value + alignment - 1) & ~(alignment - 1);
}

static void 
clear_memory(void *ptr, size_t size)
{
    char *byte_ptr = (char *)ptr;
    char *byte_sentinel = (char *)ptr + size;
    while(byte_ptr != byte_sentinel)
    {
        *byte_ptr++ = 0;
    }
}
