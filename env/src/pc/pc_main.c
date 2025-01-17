#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "sm64.h"

#include "pc/lua/smlua.h"

#include "game/memory.h"
#include "audio/external.h"

#include "network/network.h"
#include "lua/smlua.h"

#include "gfx/gfx_pc.h"

#include "gfx/gfx_opengl.h"
#include "gfx/gfx_direct3d11.h"
#include "gfx/gfx_direct3d12.h"

#include "gfx/gfx_dxgi.h"
#include "gfx/gfx_sdl.h"
#include "gfx/gfx_dummy.h"

#include "audio/audio_api.h"
#include "audio/audio_sdl.h"
#include "audio/audio_null.h"

#include "pc_main.h"
#include "cliopts.h"
#include "configfile.h"
#include "controller/controller_api.h"
#include "controller/controller_keyboard.h"
#include "fs/fs.h"

#include "game/display.h" // for gGlobalTimer
#include "game/game_init.h"
#include "game/main.h"
#include "game/rumble_init.h"

#include "include/bass/bass.h"
#include "include/bass/bass_fx.h"
#include "src/bass_audio/bass_audio_helpers.h"
#include "pc/lua/utils/smlua_audio_utils.h"

#include "pc/network/version.h"
#include "pc/network/socket/domain_res.h"
#include "pc/network/network_player.h"
#include "pc/djui/djui.h"
#include "pc/djui/djui_unicode.h"
#include "pc/djui/djui_panel.h"
#include "pc/djui/djui_panel_modlist.h"
#include "pc/debuglog.h"
#include "pc/utils/misc.h"

#include "pc/mods/mods.h"

#include "debug_context.h"
#include "menu/intro_geo.h"

#include <stdbool.h>

#ifdef DISCORD_SDK
#include "pc/discord/discord.h"
#endif


#include "game/object_list_processor.h"
#include "buffers/framebuffers.h"

#include "engine/math_util.h"
#include "engine/surface_collision.h"
#include "game/camera.h"
#include "game/level_update.h"
#include <sys/time.h>

OSMesg D_80339BEC;
OSMesgQueue gSIEventMesgQueue;

s8 gResetTimer;
s8 D_8032C648;
s8 gDebugLevelSelect;
s8 gShowProfiler;
s8 gShowDebugText;

s32 gRumblePakPfs;
u32 gNumVblanks = 0;

u8 gRenderingInterpolated = 0;
f32 gRenderingDelta = 0;

f64 gGameSpeed = 1.0f; // TODO: should probably remove

struct gameStateStruct* gGameStateStructs[MAX_PLAYERS] = { NULL };
bool gRenderingToggle = FALSE;
bool hideAndSeekMode = FALSE;
bool makeOtherPlayersInvisible = FALSE;

// 1 is true, 0 is false
#define MAX_GAME_SPEED 1

#define FRAMERATE 30
static const f64 sFrameTime = (1.0 / ((double)FRAMERATE));
static f64 sFrameTargetTime = 0;
static f64 sFrameTimeStart;
static f64 sLastFrameTimeStart;
static f32 sAvgFrames = 1;
static f32 sAvgFps = 0;

static struct AudioAPI *audio_api;
struct GfxWindowManagerAPI *wm_api;
static struct GfxRenderingAPI *rendering_api;

extern void gfx_run(Gfx *commands);
extern void thread5_game_loop(void *arg);
extern void create_next_audio_buffer(s16 *samples, u32 num_samples);
void game_loop_one_iteration(void);

void dispatch_audio_sptask(UNUSED struct SPTask *spTask) {
}

void set_vblank_handler(UNUSED s32 index, UNUSED struct VblankHandler *handler, UNUSED OSMesgQueue *queue, UNUSED OSMesg *msg) {
}

static bool inited = false;

void send_display_list(struct SPTask *spTask) {
    if (!inited) return;
    gfx_run((Gfx *)spTask->task.t.data_ptr);
}

#ifdef VERSION_EU
#define SAMPLES_HIGH 560 // gAudioBufferParameters.maxAiBufferLength
#define SAMPLES_LOW 528 // gAudioBufferParameters.minAiBufferLength
#else
#define SAMPLES_HIGH 544
#define SAMPLES_LOW 528
#endif

static void patch_interpolations_before(void) {
    extern void patch_mtx_before(void);
    extern void patch_screen_transition_before(void);
    extern void patch_title_screen_before(void);
    extern void patch_dialog_before(void);
    extern void patch_hud_before(void);
    extern void patch_paintings_before(void);
    extern void patch_bubble_particles_before(void);
    extern void patch_snow_particles_before(void);
    extern void patch_djui_before(void);
    extern void patch_djui_hud_before(void);
    patch_mtx_before();
    patch_screen_transition_before();
    patch_title_screen_before();
    patch_dialog_before();
    patch_hud_before();
    patch_paintings_before();
    patch_bubble_particles_before();
    patch_snow_particles_before();
    patch_djui_before();
    patch_djui_hud_before();
}

static inline void patch_interpolations(f32 delta) {
    extern void patch_mtx_interpolated(f32 delta);
    extern void patch_screen_transition_interpolated(f32 delta);
    extern void patch_title_screen_interpolated(f32 delta);
    extern void patch_dialog_interpolated(f32 delta);
    extern void patch_hud_interpolated(f32 delta);
    extern void patch_paintings_interpolated(f32 delta);
    extern void patch_bubble_particles_interpolated(f32 delta);
    extern void patch_snow_particles_interpolated(f32 delta);
    extern void patch_djui_interpolated(f32 delta);
    extern void patch_djui_hud(f32 delta);
    patch_mtx_interpolated(delta);
    patch_screen_transition_interpolated(delta);
    patch_title_screen_interpolated(delta);
    patch_dialog_interpolated(delta);
    patch_hud_interpolated(delta);
    patch_paintings_interpolated(delta);
    patch_bubble_particles_interpolated(delta);
    patch_snow_particles_interpolated(delta);
    patch_djui_interpolated(delta);
    patch_djui_hud(delta);
}


void produce_interpolation_frames_and_delay(void) {
    gRenderingInterpolated = true;

    // sanity check target time to deal with hangs and such
    // f64 curTime = clock_elapsed_f64();
    // if (fabs(sFrameTargetTime - curTime) > 1) {
    //     sFrameTargetTime = curTime - 0.01f;
    // }

    // u64 frames = 0;
    // while ((curTime = clock_elapsed_f64()) < sFrameTargetTime || ( frames == 0 && MAX_GAME_SPEED)) {

    // interpolate and render
    gfx_start_frame();
    // f32 delta = MIN((curTime - sFrameTimeStart) / (sFrameTargetTime - sFrameTimeStart), 1);
    f32 delta = 0;
    gRenderingDelta = delta;
    if ( !MAX_GAME_SPEED && !gSkipInterpolationTitleScreen && (configFrameLimit > 30 || configUncappedFramerate)) { patch_interpolations(delta); }
    send_display_list(gGfxSPTask);
    gfx_end_frame();

    // // delay
    // if (!configUncappedFramerate && !MAX_GAME_SPEED) {
    //     f64 targetDelta = 1.0 / (f64)configFrameLimit;
    //     f64 now = clock_elapsed_f64();
    //     f64 actualDelta = now - curTime;
    //     if (actualDelta < targetDelta) {
    //         f64 delay = ((targetDelta - actualDelta) * 1000.0);
    //         wm_api->delay((u32)delay);
    //     }
    // }

    // frames++;

    // f32 fps = frames / (clock_elapsed_f64() - sFrameTimeStart);
    // sAvgFps = sAvgFps * 0.6 + fps * 0.4;
    // sAvgFrames = sAvgFrames * 0.9 + frames * 0.1;
    // sFrameTimeStart = sFrameTargetTime;
    // sFrameTargetTime += sFrameTime * gGameSpeed;
    gRenderingInterpolated = false;

    // printf(">>> fpt: %llu, fps: %f :: %f\n", frames, sAvgFps, fps);
}

void produce_one_frame(void) {
    // CTX_BEGIN(CTX_NETWORK);
    // network_update();
    // CTX_END(CTX_NETWORK);

    // if (gRenderingToggle){
    //     CTX_BEGIN(CTX_INTERP);
    //     patch_interpolations_before();
    //     CTX_END(CTX_INTERP);

    //     const f32 master_mod = (f32)configMasterVolume / 127.0f;
    //     set_sequence_player_volume(SEQ_PLAYER_LEVEL, (f32)configMusicVolume / 127.0f * master_mod);
    //     set_sequence_player_volume(SEQ_PLAYER_SFX, (f32)configSfxVolume / 127.0f * master_mod);
    //     set_sequence_player_volume(SEQ_PLAYER_ENV, (f32)configEnvVolume / 127.0f * master_mod);
    // }

    CTX_BEGIN(CTX_GAME_LOOP);
    game_loop_one_iteration();
    CTX_END(CTX_GAME_LOOP);

    CTX_BEGIN(CTX_SMLUA);
    smlua_update();
    CTX_END(CTX_SMLUA);

    // if (gRenderingToggle){
        
    //     thread6_rumble_loop(NULL);

    //     CTX_BEGIN(CTX_AUDIO);
    //     int samples_left = audio_api->buffered();
    //     u32 num_audio_samples = samples_left < audio_api->get_desired_buffered() ? SAMPLES_HIGH : SAMPLES_LOW;
    //     //printf("Audio samples: %d %u\n", samples_left, num_audio_samples);
    //     s16 audio_buffer[SAMPLES_HIGH * 2 * 2];
    //     for (s32 i = 0; i < 2; i++) {
    //         /*if (audio_cnt-- == 0) {
    //             audio_cnt = 2;
    //         }
    //         u32 num_audio_samples = audio_cnt < 2 ? 528 : 544;*/
    //         create_next_audio_buffer(audio_buffer + i * (num_audio_samples * 2), num_audio_samples);
    //     }
    //     //printf("Audio samples before submitting: %d\n", audio_api->buffered());

    //     audio_api->play((u8 *)audio_buffer, 2 * num_audio_samples * 4);
    //     CTX_END(CTX_AUDIO);

    //     CTX_BEGIN(CTX_RENDER);
    //     produce_interpolation_frames_and_delay();
    //     CTX_END(CTX_RENDER);
    // }
}

void audio_shutdown(void) {
    audio_custom_shutdown();
    if (audio_api) {
        if (audio_api->shutdown) audio_api->shutdown();
        audio_api = NULL;
    }
}

void game_deinit(void) {
    configfile_save(configfile_name());
    controller_shutdown();
    audio_custom_shutdown();
    audio_shutdown();
    gfx_shutdown();
    network_shutdown(true, true, false, false);
    smlua_shutdown();
    mods_shutdown();
    inited = false;
}

void game_exit(void) {
    LOG_INFO("exiting cleanly");
    game_deinit();
    exit(0);
}

void inthand(UNUSED int signum) {
    game_exit();
}

Vec3f partnerVelocities[MAX_PLAYERS] = {0};

void cam_focus_player(int playerIndex){
    gNoCamUpdate = TRUE;
    Vec3f campos;
    gSmluaCameraIndex = playerIndex;

    vec3f_copy(campos, gMarioStates[playerIndex].pos);

    if (gTopDownCamera) {
        campos[1] += 400;
        campos[2] += 1;
        vec3f_copy(gLakituState.pos, campos);
        vec3f_copy(gLakituState.focus, gMarioStates[playerIndex].pos);
        gFOVState.fov = 160;
    } else {
        vec3f_set_dist_and_angle(campos, campos, 500, 0, gMarioStates[playerIndex].faceAngle[1]+ DEGREES(180));
        campos[1] += 300;
        vec3f_copy(gLakituState.pos, campos);
        vec3f_copy(gLakituState.focus, gMarioStates[playerIndex].pos);
        gFOVState.fov = 60;
    }
    gHudDisplay.lives = gGlobalTimer;
}

void set_compass_targets(Vec3f targets[MAX_PLAYERS]){
    for (int i=0; i<MAX_PLAYERS;i++){
        vec3f_copy(gSmluaCompassTargets[i], targets[i]);
    }
}

void force_make_frame(int playerIndex) {   
    cam_focus_player(playerIndex);

    for (int i=0; i<MAX_PLAYERS;i++){
        // finding hide&seek partner
        int otherPlayerIndex = playerIndex - 1;
        if (playerIndex % 2 == 0) {
            otherPlayerIndex = playerIndex + 1;
        }

        // if you are rendering yourself OR you are rendering your chaser/evader
        if (i == playerIndex || (hideAndSeekMode && i == otherPlayerIndex) || !makeOtherPlayersInvisible) {
            gMarioStates[i].marioObj->header.gfx.node.flags &= ~GRAPH_RENDER_INVISIBLE;
        }else{
            gMarioStates[i].marioObj->header.gfx.node.flags |= GRAPH_RENDER_INVISIBLE;
        }
    }
    

    // before level script
    config_gfx_pool();
        
    // level script
    init_render_image();
    render_game();
    end_master_display_list();
    alloc_display_list(0);
    // after level script
    display_and_vsync();

    // out of game loop
    CTX_BEGIN(CTX_INTERP);
    patch_interpolations_before();
    CTX_END(CTX_INTERP);
    CTX_BEGIN(CTX_RENDER);
    produce_interpolation_frames_and_delay();
    CTX_END(CTX_RENDER);
    CTX_BEGIN(CTX_INTERP);
    patch_interpolations_before();
    CTX_END(CTX_INTERP);
    CTX_BEGIN(CTX_RENDER);
    produce_interpolation_frames_and_delay();
    CTX_END(CTX_RENDER);

}

void force_make_frame_support() {
    // cam_focus_player(playerIndex);
    gHudDisplay.lives = 420;
    // CTX_BEGIN(CTX_INTERP);
    // patch_interpolations_before();
    // CTX_END(CTX_INTERP);

    // before level script
    config_gfx_pool();

    // level script
    init_render_image();
    render_game();
    end_master_display_list();
    alloc_display_list(0);

    // after level script
    display_and_vsync();

    // out of game loop
    // CTX_BEGIN(CTX_RENDER);
    // produce_interpolation_frames_and_delay();
    // CTX_END(CTX_RENDER);

}


struct inputStruct {
    s16 stickX;
    s16 stickY;
    bool buttonInput[3];
};
void adjust_analog_stick(struct Controller* controller);
void update_controllers(struct inputStruct* inputs){
    for (s32 i = 0; i < MAX_PLAYERS; i++) {

        struct Controller *controller = &gControllers[i];
        struct inputStruct input = inputs[i];

        controller->controllerData = gControllers[0].controllerData;


        controller->rawStickX = input.stickX;
        controller->rawStickY = input.stickY;
        controller->controllerData->button |= input.buttonInput[0] * A_BUTTON;
        controller->controllerData->button |= input.buttonInput[1] * B_BUTTON;
        controller->controllerData->button |= input.buttonInput[2] * Z_TRIG;

        if ( controller->rawStickX != 0 && controller->rawStickY != 0){
            controller->controllerData->button |= INPUT_NONZERO_ANALOG;
        }
        
        controller->extStickX = gControllers[0].controllerData->ext_stick_x;
        controller->extStickY = gControllers[0].controllerData->ext_stick_y;

        controller->buttonPressed = controller->controllerData->button
                        & (controller->controllerData->button ^ controller->buttonDown);

        controller->buttonDown = controller->controllerData->button;

        adjust_analog_stick(controller);
    }

}

void reset_script(void);
void reset(void){
    // dynos_warp_to_level(LEVEL_BOB, 1, 0);
    dynos_warp_restart_level();
    for (int i=0; i<MAX_PLAYERS;i++){
        gMarioStates[i].health = 0x880;
        gMarioStates[i].numLives = 4;
    }
}

struct gameStateStruct** step_pixels(struct inputStruct* inputs, int n_steps){
    // PHYSICS LOOP (for frame skipping)
    for(int i = 0; i<n_steps; i++){
        update_controllers(inputs);
        produce_one_frame();
        // player 0' image gets overwritten without this (yes, twice) and also it updates the animations
        force_make_frame_support();
        force_make_frame_support();
    }

    // RENDERING LOOP (for each player)
    for(int i = 0; i < MAX_PLAYERS; i++){    
        // hide and seek mode calculating extra inputs
        if (hideAndSeekMode){
            int otherPlayerIndex = i - 1;
            // if the player is a seeker
            if (i % 2 == 0) {
                otherPlayerIndex = i + 1;
            }
            vec3f_copy(gSmluaCompassTargets[i], gMarioStates[otherPlayerIndex].pos);
            vec3f_copy(partnerVelocities[i], gMarioStates[otherPlayerIndex].vel);
        }else{
            vec3f_set(gSmluaCompassTargets[i],0,0,0);
        }


        force_make_frame(i);
        if (gGameStateStructs[i]){
            if (gGameStateStructs[i]->pixels) free(gGameStateStructs[i]->pixels);
            free(gGameStateStructs[i]);
        }
        gGameStateStructs[i] = gfx_get_pixels();
        // to get the true health, you must do the >>8
        gGameStateStructs[i]->health = (gMarioStates[i].health >> 8);

        gGameStateStructs[i]->posX = gMarioStates[i].pos[0];
        gGameStateStructs[i]->posY = gMarioStates[i].pos[1];
        gGameStateStructs[i]->posZ = gMarioStates[i].pos[2];

        gGameStateStructs[i]->velX = gMarioStates[i].vel[0];
        gGameStateStructs[i]->velY = gMarioStates[i].vel[1];
        gGameStateStructs[i]->velZ = gMarioStates[i].vel[2];
        // + 100 because otherwise it will clip through the floor when the floor is too close (don't worry, 50 is less that mario's height)
        gGameStateStructs[i]->heightAboveGround =  gMarioStates[i].pos[1] - find_floor_height(gMarioStates[i].pos[0], gMarioStates[i].pos[1] + 50, gMarioStates[i].pos[2]);
        

        gGameStateStructs[i]->partner_x = gSmluaCompassTargets[i][0];
        gGameStateStructs[i]->partner_y = gSmluaCompassTargets[i][1];
        gGameStateStructs[i]->partner_z = gSmluaCompassTargets[i][2];

        gGameStateStructs[i]->partner_vel_x = partnerVelocities[i][0];
        gGameStateStructs[i]->partner_vel_y = partnerVelocities[i][1];
        gGameStateStructs[i]->partner_vel_z = partnerVelocities[i][2];

        gGameStateStructs[i]->deathNotice = gSmluaDeathNotices[i];
        gSmluaDeathNotices[i] = 0;

        // the opposite of DEGREES()
        
    }

    return gGameStateStructs;
}



void makemariolol(){
    for (u32 i = 1; i < MAX_PLAYERS; i++) {
        struct NetworkPlayer* npi = &gNetworkPlayers[i];
        struct NetworkPlayer* npp = &gNetworkPlayers[0];
        network_player_connected(NPT_LOCAL, i, 0, &DEFAULT_MARIO_PALETTE, "Botfam");
        network_player_update_course_level(npi, npp->currCourseNum, npp->currActNum, npp->currLevelNum, npp->currAreaIndex);
    }
    
}
extern int gImgHeight;
extern int gImgWidth;
void main_func(char *relGameDir, char *relUserPath, bool invisible, int collision_type, bool seekMode,bool compassEnabled, int renderWidth,int renderHeight, bool topDownCamera, char* configLanguage) {
    gImgHeight = renderHeight;
    gImgWidth = renderWidth;
    gTopDownCamera = topDownCamera;
    makeOtherPlayersInvisible = invisible;
    hideAndSeekMode = seekMode;
    gDjuiDisabled = !compassEnabled;

    // Ensure it is a server, avoid CLI options
    gCLIOpts.NetworkPort = 7777;
    gCLIOpts.Network = NT_SERVER;
    gCLIOpts.FullScreen = 1;
    // gCLIOpts
    const char *gamedir = relGameDir;
    const char *userpath = relUserPath;
    fs_init(sys_ropaths, gamedir, userpath);
    // printf("---------%s %s----------\n",gamedir,userpath);
    sync_objects_init_system();
    djui_unicode_init();
    djui_init();

    dynos_packs_init();
    mods_init();
    mods_enable("compass");

    

    // load config
    configfile_load();
    if (!djui_language_init(configLanguage)) {
        // snprintf(configLanguage, MAX_CONFIG_STRING, "%s", "");
    }

    dynos_pack_init();

    // If coop_custom_palette_* values are not found in sm64config.txt, the custom palette config will use the default values (Mario's palette)
    // But if no preset is found, that means the current palette is a custom palette
    for (int i = 0; i <= PALETTE_PRESET_MAX; i++) {
        if (i == PALETTE_PRESET_MAX) {
            configCustomPalette = configPlayerPalette;
            configfile_save(configfile_name());
        } else if (memcmp(&configPlayerPalette, &gPalettePresets[i], sizeof(struct PlayerPalette)) == 0) {
            break;
        }
    }

    if (configPlayerModel >= CT_MAX) { configPlayerModel = 0; }

    if (gCLIOpts.FullScreen == 1)
        configWindow.fullscreen = true;
    else if (gCLIOpts.FullScreen == 2)
        configWindow.fullscreen = false;

    #if defined(WAPI_SDL1) || defined(WAPI_SDL2)
    wm_api = &gfx_sdl;
    #elif defined(WAPI_DXGI)
    wm_api = &gfx_dxgi;
    #elif defined(WAPI_DUMMY)
    wm_api = &gfx_dummy_wm_api;
    #else
    #error No window API!
    #endif

    #if defined(RAPI_D3D11)
    rendering_api = &gfx_direct3d11_api;
    # define RAPI_NAME "DirectX 11"
    #elif defined(RAPI_D3D12)
    rendering_api = &gfx_direct3d12_api;
    # define RAPI_NAME "DirectX 12"
    #elif defined(RAPI_GL) || defined(RAPI_GL_LEGACY)
    rendering_api = &gfx_opengl_api;
    # ifdef USE_GLES
    #  define RAPI_NAME "OpenGL ES"
    # else
    #  define RAPI_NAME "OpenGL"
    # endif
    #elif defined(RAPI_DUMMY)
    rendering_api = &gfx_dummy_renderer_api;
    #else
    #error No rendering API!
    #endif

    char* version = get_version_local();
    char window_title[96] = { 0 };
#ifdef GIT_HASH
    snprintf(window_title, 96, "sm64ex-coop: %s [%s]", version, GIT_HASH);
#else
    snprintf(window_title, 96, "sm64ex-coop: %s", version);
#endif

    gfx_init(wm_api, rendering_api, window_title);
    wm_api->set_keyboard_callbacks(keyboard_on_key_down, keyboard_on_key_up, keyboard_on_all_keys_up, keyboard_on_text_input);
    
    #if defined(AAPI_SDL1) || defined(AAPI_SDL2)
    if (audio_api == NULL && audio_sdl.init())
        audio_api = &audio_sdl;
    #endif

    if (audio_api == NULL) {
        audio_api = &audio_null;
    }

    djui_init_late();

    if (gCLIOpts.Network == NT_CLIENT) {
        network_set_system(NS_SOCKET);
        snprintf(gGetHostName, MAX_CONFIG_STRING, "%s", gCLIOpts.JoinIp);
        snprintf(configJoinIp, MAX_CONFIG_STRING, "%s", gCLIOpts.JoinIp);
        configJoinPort = gCLIOpts.NetworkPort;
        network_init(NT_CLIENT, false);
    } else if (gCLIOpts.Network == NT_SERVER) {
        network_set_system(NS_SOCKET);
        configHostPort = gCLIOpts.NetworkPort;
        network_init(NT_SERVER, false);
        djui_panel_shutdown();
        djui_panel_modlist_create(NULL);
    } else {
        network_init(NT_NONE, false);
    }

    gServerSettings.playerInteractions = collision_type;

    audio_init();
    sound_init();
    bassh_init();
    network_player_init(0);
   
    // network_player_init(1);
    // force_make_network_player(0);

    thread5_game_loop(NULL);

    inited = true;

#ifdef EXTERNAL_DATA
    // precache data if needed
    if (configPrecacheRes) {
        fprintf(stdout, "precaching data\n");
        fflush(stdout);
        gfx_precache_textures();
    }
#endif
}

// int main(int argc, char *argv[]) {
//     // parse_cli_opts(argc, argv);
//     // main_func("res",".");
//     return 0;
// }

// for python
int max_players_reminder(void){
    return MAX_PLAYERS;
}




