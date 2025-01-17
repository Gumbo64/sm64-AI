#ifndef MEMORY_H
#define MEMORY_H

#include <PR/ultratypes.h>

#include "types.h"

#define MEMORY_POOL_LEFT  0
#define MEMORY_POOL_RIGHT 1

#define GFX_POOL_SIZE      0x800000 //  8MB (Vanilla: 512kB)
#define DEFAULT_POOL_SIZE 0x4000000 // 64MB (Vanilla: ~11MB)

struct DynamicPool
{
    u32 usedSpace;
    struct DynamicPoolNode* nextFree;
    struct DynamicPoolNode* tail;
};

struct DynamicPoolNode
{
    void* ptr;
    u32 size;
    struct DynamicPoolNode* prev;
};

struct GrowingPool
{
    u32 usedSpace;
    u32 nodeSize;
    struct GrowingPoolNode* tail;
};

struct GrowingPoolNode
{
    u32 usedSpace;
    void* ptr;
    struct GrowingPoolNode* prev;
};

struct MarioAnimation;
struct Animation;

extern struct DynamicPool *gLevelPool;

uintptr_t set_segment_base_addr(s32 segment, void *addr);
void *segmented_to_virtual(const void *addr);
void *virtual_to_segmented(u32 segment, const void *addr);

#define load_segment(...)
#define load_to_fixed_pool_addr(...)
#define load_segment_decompress(...)
#define load_segment_decompress_heap(...)
#define load_engine_code_segment(...)

struct DynamicPool* dynamic_pool_init(void);
void* dynamic_pool_alloc(struct DynamicPool *pool, u32 size);
void dynamic_pool_free(struct DynamicPool *pool, void* ptr);
void dynamic_pool_free_pool(struct DynamicPool *pool);

struct GrowingPool* growing_pool_init(struct GrowingPool* pool, u32 nodeSize);
void* growing_pool_alloc(struct GrowingPool *pool, u32 size);
void growing_pool_free_pool(struct GrowingPool *pool);

void alloc_display_list_reset(void);
void *alloc_display_list(u32 size);

void alloc_anim_dma_table(struct MarioAnimation* marioAnim, void *b, struct Animation *targetAnim);
s32 load_patchable_table(struct MarioAnimation *a, u32 b);

#endif // MEMORY_H
