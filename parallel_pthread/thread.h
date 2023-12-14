#define WORK_QUEUE_SIZE 64
#define WORK_QUEUE_MASK (WORK_QUEUE_SIZE - 1)

#include <assert.h>

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
#elif defined(__unix__)
    #include <unistd.h>
    #include <pthread.h>
    #include <semaphore.h>
#endif

#if defined(_MSC_VER)
    #define MEMORY_BARRIER _ReadWriteBarrier()
    #define atomic_add(ptr, value) InterlockedExchangeAdd((volatile LONG *)(ptr), value)
    #define atomic_compare_exchange(ptr, expected, desired) InterlockedCompareExchange((volatile LONG *)(ptr), (LONG)desired, (LONG)expected)
#elif defined(__GNUC__)
#define MEMORY_BARRIER asm volatile("" ::: "memory")
    #define atomic_add(ptr, value) __sync_fetch_and_add(ptr, value)
    #define atomic_compare_exchange(ptr, expected, desired) __sync_val_compare_and_swap(ptr, expected, desired)
#else
    #error "unknown compiler, use MSVC or GCC"
#endif


#if defined(_WIN32) || defined(_WIN64)
    #define THREAD_PROC(name) DWORD WINAPI name(LPVOID param)
#elif defined(__unix__)
    #define THREAD_PROC(name) void *name(void *param)
#endif

typedef void WorkQueueEntryCallback(void *param);
typedef THREAD_PROC(ThreadProc);

typedef union Semaphore
{
    int data[8];
} Semaphore;

typedef struct TicketMutex
{
    volatile int ticket;
    volatile int serving;
} TicketMutex;

typedef struct WorkQueueEntry
{
    WorkQueueEntryCallback *callback;
    void *data;
} WorkQueueEntry;

typedef struct WorkQueue
{
    volatile int entry_to_read;
    volatile int entry_to_write;
    volatile int completion_goal;
    volatile int completion_count;
    TicketMutex queue_work_mutex;
    Semaphore semaphore;
    WorkQueueEntry entries[WORK_QUEUE_SIZE];
} WorkQueue;

static void 
begin_ticket_mutex(TicketMutex *mutex)
{
    int ticket = atomic_add(&mutex->ticket, 1);
    while(ticket != mutex->serving);
}

static void 
end_ticket_mutex(TicketMutex *mutex)
{
    atomic_add(&mutex->serving, 1);
}

static int 
get_thread_count(void)
{
    int result = 1;
#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    result = system_info.dwNumberOfProcessors;
#elif defined(__unix__)
    result = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    return result;
}

#include <stdio.h>
static void
create_semaphore(Semaphore *semaphore, int max_count)
{
    clear_memory(semaphore, sizeof(*semaphore));
#if defined(_WIN32) || defined(_WIN64)
    static_assert(sizeof(HANDLE) <= sizeof(Semaphore), "sizeof(Semaphore) must be greater than sizeof(HANDLE)");
    *(HANDLE *)semaphore->data = (HANDLE)CreateSemaphoreA(0, 0, max_count, 0);
#elif defined(__unix__)
    static_assert(sizeof(sem_t) <= sizeof(Semaphore), "sizeof(Semaphore) must be greater than sizeof(sem_t)");
    // TODO: implement max count for linux semaphore, currently we let the thread do busy work 
    // in thread_proc to remove these extra signals
    sem_init((sem_t *)semaphore->data, 0, 0);
#endif
}

static void 
increment_semaphore(Semaphore *semaphore)
{
#if defined(_WIN32) || defined(_WIN64)
    HANDLE handle = *(HANDLE *)semaphore->data;
    ReleaseSemaphore(handle, 1, 0);
#elif defined(__unix__)
    sem_t *handle = (sem_t *)semaphore->data;
    sem_post(handle);
#endif
}

static void 
wait_for_semaphore(Semaphore *semaphore)
{
#if defined(_WIN32) || defined(_WIN64)
    HANDLE handle = *(HANDLE *)semaphore->data;
    WaitForSingleObject(handle, INFINITE);
#elif defined(__unix__)
    sem_t *handle = (sem_t *)semaphore->data;
    sem_wait(handle);
#endif
}

static void 
create_thread(ThreadProc *thread_proc, void *param)
{
#if defined(_WIN32) || defined(_WIN64)
    HANDLE thread_handle = CreateThread(0, 0, thread_proc, param, 0, 0);
    CloseHandle(thread_handle);
#elif defined(__unix__)
    pthread_t thread_handle;
    pthread_create(&thread_handle, 0, thread_proc, param);
#endif
}

static void 
queue_work(WorkQueue *queue, WorkQueueEntryCallback *callback, void *data)
{
    begin_ticket_mutex(&queue->queue_work_mutex);
    int index = queue->entry_to_write;
    WorkQueueEntry *entry = queue->entries + index;
    entry->callback = callback;
    entry->data = data;
    ++queue->completion_goal;
    MEMORY_BARRIER;
    queue->entry_to_write = (queue->entry_to_write + 1) & WORK_QUEUE_MASK;
    end_ticket_mutex(&queue->queue_work_mutex);
    increment_semaphore(&queue->semaphore);
}

static int 
do_next_work(WorkQueue *queue)
{
    int result = 1;
    int current_entry_to_read = queue->entry_to_read;
    int next_entry_to_read = (current_entry_to_read + 1) & WORK_QUEUE_MASK;
    if(current_entry_to_read != queue->entry_to_write)
    {
        result = 0;
        WorkQueueEntry entry = queue->entries[current_entry_to_read];
        int index = atomic_compare_exchange(&queue->entry_to_read, current_entry_to_read, next_entry_to_read);
        if(index == current_entry_to_read)
        {
            entry.callback(entry.data);
            ++queue->completion_count;
        }
    }
    return result;
}

static void 
complete_all_works(WorkQueue *queue)
{
    while(queue->completion_count != queue->completion_goal)
    {
        do_next_work(queue);
    }
}

static 
THREAD_PROC(thread_proc)
{
    WorkQueue *queue = (WorkQueue *)param;
    for(;;)
    {
        if(do_next_work(queue))
        {
            wait_for_semaphore(&queue->semaphore);
        }
    }
    return 0;
}
static void 
create_work_queue(WorkQueue *queue, int thread_count)
{
    clear_memory(queue, sizeof(*queue));
    create_semaphore(&queue->semaphore, thread_count);
    for(int thread_index = 0; thread_index < thread_count; ++thread_index)
    {
        create_thread(thread_proc, queue);
    }
}

