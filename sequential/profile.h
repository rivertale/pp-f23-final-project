
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
static unsigned long long 
get_microsecond_from_epoch(void)
{
    unsigned long long result = 0;
    SYSTEMTIME system_time;
    FILETIME nanosecond_since_epoch_div100;
    GetSystemTime(&system_time);
    if(SystemTimeToFileTime(&system_time, &nanosecond_since_epoch_div100))
    {
        result = (((unsigned long long)nanosecond_since_epoch_div100.dwHighDateTime << 32) | 
                  ((unsigned long long)nanosecond_since_epoch_div100.dwLowDateTime << 0)) / 10;
    }
    return result;
}
#elif defined(__unix__)
#include <sys/time.h>
static unsigned long long 
get_microsecond_from_epoch(void)
{
    unsigned long long result = 0;
    struct timeval current_clock;
    if(gettimeofday(&current_clock, 0) == 0)
    {
        result = (unsigned long long)current_clock.tv_sec * 1000000 + current_clock.tv_usec;
    }
    return result;
}
#else
#error unknown platform
#endif
