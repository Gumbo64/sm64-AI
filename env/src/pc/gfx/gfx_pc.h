#ifndef GFX_PC_H
#define GFX_PC_H

#include "types.h"

struct GfxRenderingAPI;
struct GfxWindowManagerAPI;

struct GfxDimensions {
    uint32_t width, height;
    float aspect_ratio;
};

extern struct GfxDimensions gfx_current_dimensions;

extern Vec3f gLightingDir;
extern Color gLightingColor;

extern int gImgWidth;
extern int gImgHeight;

#ifdef __cplusplus
extern "C" {
#endif

void gfx_init(struct GfxWindowManagerAPI *wapi, struct GfxRenderingAPI *rapi, const char *window_title);
struct GfxRenderingAPI *gfx_get_current_rendering_api(void);
void gfx_start_frame(void);
void gfx_run(Gfx *commands);
void gfx_end_frame(void);
void gfx_precache_textures(void);
void gfx_shutdown(void);
void gfx_pc_precomp_shader(uint32_t rgb1, uint32_t alpha1, uint32_t rgb2, uint32_t alpha2, uint32_t flags);

struct gameStateStruct {
    unsigned char* pixels;

    int pixelsWidth;
    int pixelsHeight;
    int health;
    float posX;
    float posY;
    float posZ;

    float velX;
    float velY;
    float velZ;

    float heightAboveGround;

};

struct gameStateStruct* gfx_get_pixels(void);
void FLOOOOSH(void);

#ifdef __cplusplus
}
#endif

#endif
