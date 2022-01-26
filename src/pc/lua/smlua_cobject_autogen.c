#include "smlua.h"
#include "include/types.h"
#include "src/game/area.h"
#include "src/game/camera.h"
#include "src/game/characters.h"

#define LUA_CONTROLLER_FIELD_COUNT 10
static struct LuaObjectField sControllerFields[LUA_CONTROLLER_FIELD_COUNT] = {
    { "rawStickX",      LVT_S16, offsetof(struct Controller, rawStickX),      false, LOT_NONE },
    { "rawStickY",      LVT_S16, offsetof(struct Controller, rawStickY),      false, LOT_NONE },
    { "stickX",         LVT_F32, offsetof(struct Controller, stickX),         false, LOT_NONE },
    { "stickY",         LVT_F32, offsetof(struct Controller, stickY),         false, LOT_NONE },
    { "stickMag",       LVT_F32, offsetof(struct Controller, stickMag),       false, LOT_NONE },
    { "buttonDown",     LVT_U16, offsetof(struct Controller, buttonDown),     false, LOT_NONE },
    { "buttonPressed",  LVT_U16, offsetof(struct Controller, buttonPressed),  false, LOT_NONE },
//  { "statusData",     LVT_???, offsetof(struct Controller, statusData),     false, LOT_???  }, <--- UNIMPLEMENTED
//  { "controllerData", LVT_???, offsetof(struct Controller, controllerData), false, LOT_???  }, <--- UNIMPLEMENTED
    { "port",           LVT_S32, offsetof(struct Controller, port),           false, LOT_NONE },
    { "extStickX",      LVT_S16, offsetof(struct Controller, extStickX),      false, LOT_NONE },
    { "extStickY",      LVT_S16, offsetof(struct Controller, extStickY),      false, LOT_NONE },
};

#define LUA_ANIMATION_FIELD_COUNT 7
static struct LuaObjectField sAnimationFields[LUA_ANIMATION_FIELD_COUNT] = {
    { "flags",             LVT_S16, offsetof(struct Animation, flags),  false, LOT_NONE },
    { "animYTransDivisor", LVT_S16, offsetof(struct Animation, unk02),  false, LOT_NONE },
    { "startFrame",        LVT_S16, offsetof(struct Animation, unk04),  false, LOT_NONE },
    { "loopStart",         LVT_S16, offsetof(struct Animation, unk06),  false, LOT_NONE },
    { "loopEnd",           LVT_S16, offsetof(struct Animation, unk08),  false, LOT_NONE },
    { "unusedBoneCount",   LVT_S16, offsetof(struct Animation, unk0A),  false, LOT_NONE },
//  { "values",            LVT_???, offsetof(struct Animation, values), false, LOT_???  }, <--- UNIMPLEMENTED
//  { "index",             LVT_???, offsetof(struct Animation, index),  false, LOT_???  }, <--- UNIMPLEMENTED
    { "length",            LVT_U32, offsetof(struct Animation, length), false, LOT_NONE },
};

#define LUA_GRAPH_NODE_FIELD_COUNT 6
static struct LuaObjectField sGraphNodeFields[LUA_GRAPH_NODE_FIELD_COUNT] = {
    { "type",     LVT_S16,       offsetof(struct GraphNode, type),     false, LOT_NONE      },
    { "flags",    LVT_S16,       offsetof(struct GraphNode, flags),    false, LOT_NONE      },
    { "prev",     LVT_COBJECT_P, offsetof(struct GraphNode, prev),     false, LOT_GRAPHNODE },
    { "next",     LVT_COBJECT_P, offsetof(struct GraphNode, next),     false, LOT_GRAPHNODE },
    { "parent",   LVT_COBJECT_P, offsetof(struct GraphNode, parent),   false, LOT_GRAPHNODE },
    { "children", LVT_COBJECT_P, offsetof(struct GraphNode, children), false, LOT_GRAPHNODE },
};

#define LUA_GRAPH_NODE_OBJECT_SUB_FIELD_COUNT 11
static struct LuaObjectField sGraphNodeObject_subFields[LUA_GRAPH_NODE_OBJECT_SUB_FIELD_COUNT] = {
    { "animID",                 LVT_S16,       offsetof(struct GraphNodeObject_sub, animID),                 false, LOT_NONE      },
    { "animYTrans",             LVT_S16,       offsetof(struct GraphNodeObject_sub, animYTrans),             false, LOT_NONE      },
    { "curAnim",                LVT_COBJECT_P, offsetof(struct GraphNodeObject_sub, curAnim),                false, LOT_ANIMATION },
    { "animFrame",              LVT_S16,       offsetof(struct GraphNodeObject_sub, animFrame),              false, LOT_NONE      },
    { "animTimer",              LVT_U16,       offsetof(struct GraphNodeObject_sub, animTimer),              false, LOT_NONE      },
    { "animFrameAccelAssist",   LVT_S32,       offsetof(struct GraphNodeObject_sub, animFrameAccelAssist),   false, LOT_NONE      },
    { "animAccel",              LVT_S32,       offsetof(struct GraphNodeObject_sub, animAccel),              false, LOT_NONE      },
    { "prevAnimFrame",          LVT_S16,       offsetof(struct GraphNodeObject_sub, prevAnimFrame),          false, LOT_NONE      },
    { "prevAnimID",             LVT_S16,       offsetof(struct GraphNodeObject_sub, prevAnimID),             false, LOT_NONE      },
    { "prevAnimFrameTimestamp", LVT_U32,       offsetof(struct GraphNodeObject_sub, prevAnimFrameTimestamp), false, LOT_NONE      },
    { "prevAnimPtr",            LVT_COBJECT_P, offsetof(struct GraphNodeObject_sub, prevAnimPtr),            false, LOT_ANIMATION },
};

#define LUA_GRAPH_NODE_OBJECT_FIELD_COUNT 19
static struct LuaObjectField sGraphNodeObjectFields[LUA_GRAPH_NODE_OBJECT_FIELD_COUNT] = {
    { "node",                       LVT_COBJECT,   offsetof(struct GraphNodeObject, node),                       true,  LOT_GRAPHNODE           },
    { "sharedChild",                LVT_COBJECT_P, offsetof(struct GraphNodeObject, sharedChild),                false, LOT_GRAPHNODE           },
    { "unk18",                      LVT_S8,        offsetof(struct GraphNodeObject, unk18),                      false, LOT_NONE                },
    { "unk19",                      LVT_S8,        offsetof(struct GraphNodeObject, unk19),                      false, LOT_NONE                },
    { "angle",                      LVT_COBJECT,   offsetof(struct GraphNodeObject, angle),                      true,  LOT_VEC3S               },
    { "pos",                        LVT_COBJECT,   offsetof(struct GraphNodeObject, pos),                        true,  LOT_VEC3F               },
    { "prevAngle",                  LVT_COBJECT,   offsetof(struct GraphNodeObject, prevAngle),                  true,  LOT_VEC3S               },
    { "prevPos",                    LVT_COBJECT,   offsetof(struct GraphNodeObject, prevPos),                    true,  LOT_VEC3F               },
    { "prevTimestamp",              LVT_U32,       offsetof(struct GraphNodeObject, prevTimestamp),              false, LOT_NONE                },
    { "prevShadowPos",              LVT_COBJECT,   offsetof(struct GraphNodeObject, prevShadowPos),              true,  LOT_VEC3F               },
    { "prevShadowPosTimestamp",     LVT_U32,       offsetof(struct GraphNodeObject, prevShadowPosTimestamp),     false, LOT_NONE                },
    { "scale",                      LVT_COBJECT,   offsetof(struct GraphNodeObject, scale),                      true,  LOT_VEC3F               },
    { "prevScale",                  LVT_COBJECT,   offsetof(struct GraphNodeObject, prevScale),                  true,  LOT_VEC3F               },
    { "prevScaleTimestamp",         LVT_U32,       offsetof(struct GraphNodeObject, prevScaleTimestamp),         false, LOT_NONE                },
    { "animInfo",                   LVT_COBJECT,   offsetof(struct GraphNodeObject, unk38),                      true,  LOT_GRAPHNODEOBJECT_SUB },
    { "unk4C",                      LVT_COBJECT_P, offsetof(struct GraphNodeObject, unk4C),                      false, LOT_SPAWNINFO           },
//  { "throwMatrix",                LVT_???,       offsetof(struct GraphNodeObject, throwMatrix),                false, LOT_???                 }, <--- UNIMPLEMENTED
//  { "prevThrowMatrix",            LVT_???,       offsetof(struct GraphNodeObject, prevThrowMatrix),            false, LOT_???                 }, <--- UNIMPLEMENTED
    { "prevThrowMatrixTimestamp",   LVT_U32,       offsetof(struct GraphNodeObject, prevThrowMatrixTimestamp),   false, LOT_NONE                },
//  { "throwMatrixInterpolated",    LVT_???,       offsetof(struct GraphNodeObject, throwMatrixInterpolated),    false, LOT_???                 }, <--- UNIMPLEMENTED
    { "cameraToObject",             LVT_COBJECT,   offsetof(struct GraphNodeObject, cameraToObject),             true,  LOT_VEC3F               },
    { "skipInterpolationTimestamp", LVT_U32,       offsetof(struct GraphNodeObject, skipInterpolationTimestamp), false, LOT_NONE                },
};

#define LUA_OBJECT_NODE_FIELD_COUNT 3
static struct LuaObjectField sObjectNodeFields[LUA_OBJECT_NODE_FIELD_COUNT] = {
    { "gfx",  LVT_COBJECT,   offsetof(struct ObjectNode, gfx),  true,  LOT_GRAPHNODEOBJECT },
    { "next", LVT_COBJECT_P, offsetof(struct ObjectNode, next), false, LOT_OBJECTNODE      },
    { "prev", LVT_COBJECT_P, offsetof(struct ObjectNode, prev), false, LOT_OBJECTNODE      },
};

#define LUA_OBJECT_FIELD_COUNT 21
static struct LuaObjectField sObjectFields[LUA_OBJECT_FIELD_COUNT] = {
    { "header",                          LVT_COBJECT,   offsetof(struct Object, header),                          true,  LOT_OBJECTNODE },
    { "parentObj",                       LVT_COBJECT_P, offsetof(struct Object, parentObj),                       false, LOT_OBJECT     },
    { "prevObj",                         LVT_COBJECT_P, offsetof(struct Object, prevObj),                         false, LOT_OBJECT     },
    { "collidedObjInteractTypes",        LVT_U32,       offsetof(struct Object, collidedObjInteractTypes),        false, LOT_NONE       },
    { "activeFlags",                     LVT_S16,       offsetof(struct Object, activeFlags),                     false, LOT_NONE       },
    { "numCollidedObjs",                 LVT_S16,       offsetof(struct Object, numCollidedObjs),                 false, LOT_NONE       },
//  { "collidedObjs",                    LVT_COBJECT_P, offsetof(struct Object, collidedObjs),                    false, LOT_???        }, <--- UNIMPLEMENTED
//  { "rawData",                         LVT_???,       offsetof(struct Object, rawData),                         false, LOT_???        }, <--- UNIMPLEMENTED
//  { "ptrData",                         LVT_???,       offsetof(struct Object, ptrData),                         false, LOT_???        }, <--- UNIMPLEMENTED
    { "unused1",                         LVT_U32,       offsetof(struct Object, unused1),                         false, LOT_NONE       },
//  { "curBhvCommand",                   LVT_???,       offsetof(struct Object, curBhvCommand),                   false, LOT_???        }, <--- UNIMPLEMENTED
    { "bhvStackIndex",                   LVT_U32,       offsetof(struct Object, bhvStackIndex),                   false, LOT_NONE       },
//  { "bhvStack",                        LVT_???,       offsetof(struct Object, bhvStack),                        false, LOT_???        }, <--- UNIMPLEMENTED
    { "bhvDelayTimer",                   LVT_S16,       offsetof(struct Object, bhvDelayTimer),                   false, LOT_NONE       },
    { "respawnInfoType",                 LVT_S16,       offsetof(struct Object, respawnInfoType),                 false, LOT_NONE       },
    { "hitboxRadius",                    LVT_F32,       offsetof(struct Object, hitboxRadius),                    false, LOT_NONE       },
    { "hitboxHeight",                    LVT_F32,       offsetof(struct Object, hitboxHeight),                    false, LOT_NONE       },
    { "hurtboxRadius",                   LVT_F32,       offsetof(struct Object, hurtboxRadius),                   false, LOT_NONE       },
    { "hurtboxHeight",                   LVT_F32,       offsetof(struct Object, hurtboxHeight),                   false, LOT_NONE       },
    { "hitboxDownOffset",                LVT_F32,       offsetof(struct Object, hitboxDownOffset),                false, LOT_NONE       },
//  { "behavior",                        LVT_???,       offsetof(struct Object, behavior),                        false, LOT_???        }, <--- UNIMPLEMENTED
    { "heldByPlayerIndex",               LVT_U32,       offsetof(struct Object, heldByPlayerIndex),               false, LOT_NONE       },
    { "platform",                        LVT_COBJECT_P, offsetof(struct Object, platform),                        false, LOT_OBJECT     },
//  { "collisionData",                   LVT_???,       offsetof(struct Object, collisionData),                   false, LOT_???        }, <--- UNIMPLEMENTED
//  { "transform",                       LVT_???,       offsetof(struct Object, transform),                       false, LOT_???        }, <--- UNIMPLEMENTED
//  { "respawnInfo",                     LVT_???,       offsetof(struct Object, respawnInfo),                     false, LOT_???        }, <--- UNIMPLEMENTED
    { "createdThroughNetwork",           LVT_U8,        offsetof(struct Object, createdThroughNetwork),           false, LOT_NONE       },
//  { "areaTimerType",                   LVT_???,       offsetof(struct Object, areaTimerType),                   false, LOT_???        }, <--- UNIMPLEMENTED
    { "areaTimer",                       LVT_U32,       offsetof(struct Object, areaTimer),                       false, LOT_NONE       },
    { "areaTimerDuration",               LVT_U32,       offsetof(struct Object, areaTimerDuration),               false, LOT_NONE       },
//  { "areaTimerRunOnceCallback)(void)", LVT_???,       offsetof(struct Object, areaTimerRunOnceCallback)(void)), false, LOT_???        }, <--- UNIMPLEMENTED
    { "globalPlayerIndex",               LVT_U8,        offsetof(struct Object, globalPlayerIndex),               false, LOT_NONE       },
};

#define LUA_OBJECT_HITBOX_FIELD_COUNT 9
static struct LuaObjectField sObjectHitboxFields[LUA_OBJECT_HITBOX_FIELD_COUNT] = {
    { "interactType",      LVT_U32, offsetof(struct ObjectHitbox, interactType),      false, LOT_NONE },
    { "downOffset",        LVT_U8,  offsetof(struct ObjectHitbox, downOffset),        false, LOT_NONE },
    { "damageOrCoinValue", LVT_S8,  offsetof(struct ObjectHitbox, damageOrCoinValue), false, LOT_NONE },
    { "health",            LVT_S8,  offsetof(struct ObjectHitbox, health),            false, LOT_NONE },
    { "numLootCoins",      LVT_S8,  offsetof(struct ObjectHitbox, numLootCoins),      false, LOT_NONE },
    { "radius",            LVT_S16, offsetof(struct ObjectHitbox, radius),            false, LOT_NONE },
    { "height",            LVT_S16, offsetof(struct ObjectHitbox, height),            false, LOT_NONE },
    { "hurtboxRadius",     LVT_S16, offsetof(struct ObjectHitbox, hurtboxRadius),     false, LOT_NONE },
    { "hurtboxHeight",     LVT_S16, offsetof(struct ObjectHitbox, hurtboxHeight),     false, LOT_NONE },
};

#define LUA_WAYPOINT_FIELD_COUNT 2
static struct LuaObjectField sWaypointFields[LUA_WAYPOINT_FIELD_COUNT] = {
    { "flags", LVT_S16,     offsetof(struct Waypoint, flags), false, LOT_NONE  },
    { "pos",   LVT_COBJECT, offsetof(struct Waypoint, pos),   true,  LOT_VEC3S },
};

#define LUA_SURFACE_FIELD_COUNT 16
static struct LuaObjectField sSurfaceFields[LUA_SURFACE_FIELD_COUNT] = {
    { "type",              LVT_S16,       offsetof(struct Surface, type),              false, LOT_NONE   },
    { "force",             LVT_S16,       offsetof(struct Surface, force),             false, LOT_NONE   },
    { "flags",             LVT_S8,        offsetof(struct Surface, flags),             false, LOT_NONE   },
    { "room",              LVT_S8,        offsetof(struct Surface, room),              false, LOT_NONE   },
    { "lowerY",            LVT_S16,       offsetof(struct Surface, lowerY),            false, LOT_NONE   },
    { "upperY",            LVT_S16,       offsetof(struct Surface, upperY),            false, LOT_NONE   },
    { "vertex1",           LVT_COBJECT,   offsetof(struct Surface, vertex1),           true,  LOT_VEC3S  },
    { "vertex2",           LVT_COBJECT,   offsetof(struct Surface, vertex2),           true,  LOT_VEC3S  },
    { "vertex3",           LVT_COBJECT,   offsetof(struct Surface, vertex3),           true,  LOT_VEC3S  },
    { "normal",            LVT_COBJECT,   offsetof(struct Surface, normal),            true,  LOT_VEC3F  },
    { "originOffset",      LVT_F32,       offsetof(struct Surface, originOffset),      false, LOT_NONE   },
    { "object",            LVT_COBJECT_P, offsetof(struct Surface, object),            false, LOT_OBJECT },
    { "prevVertex1",       LVT_COBJECT,   offsetof(struct Surface, prevVertex1),       true,  LOT_VEC3S  },
    { "prevVertex2",       LVT_COBJECT,   offsetof(struct Surface, prevVertex2),       true,  LOT_VEC3S  },
    { "prevVertex3",       LVT_COBJECT,   offsetof(struct Surface, prevVertex3),       true,  LOT_VEC3S  },
    { "modifiedTimestamp", LVT_U32,       offsetof(struct Surface, modifiedTimestamp), false, LOT_NONE   },
};

#define LUA_MARIO_BODY_STATE_FIELD_COUNT 12
static struct LuaObjectField sMarioBodyStateFields[LUA_MARIO_BODY_STATE_FIELD_COUNT] = {
    { "action",              LVT_U32,     offsetof(struct MarioBodyState, action),              false, LOT_NONE  },
    { "capState",            LVT_S8,      offsetof(struct MarioBodyState, capState),            false, LOT_NONE  },
    { "eyeState",            LVT_S8,      offsetof(struct MarioBodyState, eyeState),            false, LOT_NONE  },
    { "handState",           LVT_S8,      offsetof(struct MarioBodyState, handState),           false, LOT_NONE  },
    { "wingFlutter",         LVT_S8,      offsetof(struct MarioBodyState, wingFlutter),         false, LOT_NONE  },
    { "modelState",          LVT_S16,     offsetof(struct MarioBodyState, modelState),          false, LOT_NONE  },
    { "grabPos",             LVT_S8,      offsetof(struct MarioBodyState, grabPos),             false, LOT_NONE  },
    { "punchState",          LVT_U8,      offsetof(struct MarioBodyState, punchState),          false, LOT_NONE  },
    { "torsoAngle",          LVT_COBJECT, offsetof(struct MarioBodyState, torsoAngle),          true,  LOT_VEC3S },
    { "headAngle",           LVT_COBJECT, offsetof(struct MarioBodyState, headAngle),           true,  LOT_VEC3S },
    { "heldObjLastPosition", LVT_COBJECT, offsetof(struct MarioBodyState, heldObjLastPosition), true,  LOT_VEC3F },
    { "torsoPos",            LVT_COBJECT, offsetof(struct MarioBodyState, torsoPos),            true,  LOT_VEC3F },
//  { "handFootPos",         LVT_???,     offsetof(struct MarioBodyState, handFootPos),         false, LOT_???   }, <--- UNIMPLEMENTED
};

#define LUA_OFFSET_SIZE_PAIR_FIELD_COUNT 2
static struct LuaObjectField sOffsetSizePairFields[LUA_OFFSET_SIZE_PAIR_FIELD_COUNT] = {
    { "offset", LVT_U32, offsetof(struct OffsetSizePair, offset), false, LOT_NONE },
    { "size",   LVT_U32, offsetof(struct OffsetSizePair, size),   false, LOT_NONE },
};

#define LUA_MARIO_ANIMATION_FIELD_COUNT 1
static struct LuaObjectField sMarioAnimationFields[LUA_MARIO_ANIMATION_FIELD_COUNT] = {
//  { "animDmaTable",    LVT_COBJECT_P, offsetof(struct MarioAnimation, animDmaTable),    false, LOT_???       }, <--- UNIMPLEMENTED
//  { "currentAnimAddr", LVT_???,       offsetof(struct MarioAnimation, currentAnimAddr), false, LOT_???       }, <--- UNIMPLEMENTED
    { "targetAnim",      LVT_COBJECT_P, offsetof(struct MarioAnimation, targetAnim),      false, LOT_ANIMATION },
//  { "padding",         LVT_???,       offsetof(struct MarioAnimation, padding),         false, LOT_???       }, <--- UNIMPLEMENTED
};

#define LUA_MARIO_STATE_FIELD_COUNT 72
static struct LuaObjectField sMarioStateFields[LUA_MARIO_STATE_FIELD_COUNT] = {
    { "playerIndex",              LVT_U16,       offsetof(struct MarioState, playerIndex),              false, LOT_NONE              },
    { "input",                    LVT_U16,       offsetof(struct MarioState, input),                    false, LOT_NONE              },
    { "flags",                    LVT_U32,       offsetof(struct MarioState, flags),                    false, LOT_NONE              },
    { "particleFlags",            LVT_U32,       offsetof(struct MarioState, particleFlags),            false, LOT_NONE              },
    { "action",                   LVT_U32,       offsetof(struct MarioState, action),                   false, LOT_NONE              },
    { "prevAction",               LVT_U32,       offsetof(struct MarioState, prevAction),               false, LOT_NONE              },
    { "terrainSoundAddend",       LVT_U32,       offsetof(struct MarioState, terrainSoundAddend),       false, LOT_NONE              },
    { "actionState",              LVT_U16,       offsetof(struct MarioState, actionState),              false, LOT_NONE              },
    { "actionTimer",              LVT_U16,       offsetof(struct MarioState, actionTimer),              false, LOT_NONE              },
    { "actionArg",                LVT_U32,       offsetof(struct MarioState, actionArg),                false, LOT_NONE              },
    { "intendedMag",              LVT_F32,       offsetof(struct MarioState, intendedMag),              false, LOT_NONE              },
    { "intendedYaw",              LVT_S16,       offsetof(struct MarioState, intendedYaw),              false, LOT_NONE              },
    { "invincTimer",              LVT_S16,       offsetof(struct MarioState, invincTimer),              false, LOT_NONE              },
    { "framesSinceA",             LVT_U8,        offsetof(struct MarioState, framesSinceA),             false, LOT_NONE              },
    { "framesSinceB",             LVT_U8,        offsetof(struct MarioState, framesSinceB),             false, LOT_NONE              },
    { "wallKickTimer",            LVT_U8,        offsetof(struct MarioState, wallKickTimer),            false, LOT_NONE              },
    { "doubleJumpTimer",          LVT_U8,        offsetof(struct MarioState, doubleJumpTimer),          false, LOT_NONE              },
    { "faceAngle",                LVT_COBJECT,   offsetof(struct MarioState, faceAngle),                true,  LOT_VEC3S             },
    { "angleVel",                 LVT_COBJECT,   offsetof(struct MarioState, angleVel),                 true,  LOT_VEC3S             },
    { "slideYaw",                 LVT_S16,       offsetof(struct MarioState, slideYaw),                 false, LOT_NONE              },
    { "twirlYaw",                 LVT_S16,       offsetof(struct MarioState, twirlYaw),                 false, LOT_NONE              },
    { "pos",                      LVT_COBJECT,   offsetof(struct MarioState, pos),                      true,  LOT_VEC3F             },
    { "vel",                      LVT_COBJECT,   offsetof(struct MarioState, vel),                      true,  LOT_VEC3F             },
    { "forwardVel",               LVT_F32,       offsetof(struct MarioState, forwardVel),               false, LOT_NONE              },
    { "slideVelX",                LVT_F32,       offsetof(struct MarioState, slideVelX),                false, LOT_NONE              },
    { "slideVelZ",                LVT_F32,       offsetof(struct MarioState, slideVelZ),                false, LOT_NONE              },
    { "wall",                     LVT_COBJECT_P, offsetof(struct MarioState, wall),                     false, LOT_SURFACE           },
    { "ceil",                     LVT_COBJECT_P, offsetof(struct MarioState, ceil),                     false, LOT_SURFACE           },
    { "floor",                    LVT_COBJECT_P, offsetof(struct MarioState, floor),                    false, LOT_SURFACE           },
    { "ceilHeight",               LVT_F32,       offsetof(struct MarioState, ceilHeight),               false, LOT_NONE              },
    { "floorHeight",              LVT_F32,       offsetof(struct MarioState, floorHeight),              false, LOT_NONE              },
    { "floorAngle",               LVT_S16,       offsetof(struct MarioState, floorAngle),               false, LOT_NONE              },
    { "waterLevel",               LVT_S16,       offsetof(struct MarioState, waterLevel),               false, LOT_NONE              },
    { "interactObj",              LVT_COBJECT_P, offsetof(struct MarioState, interactObj),              false, LOT_OBJECT            },
    { "heldObj",                  LVT_COBJECT_P, offsetof(struct MarioState, heldObj),                  false, LOT_OBJECT            },
    { "usedObj",                  LVT_COBJECT_P, offsetof(struct MarioState, usedObj),                  false, LOT_OBJECT            },
    { "riddenObj",                LVT_COBJECT_P, offsetof(struct MarioState, riddenObj),                false, LOT_OBJECT            },
    { "marioObj",                 LVT_COBJECT_P, offsetof(struct MarioState, marioObj),                 false, LOT_OBJECT            },
    { "spawnInfo",                LVT_COBJECT_P, offsetof(struct MarioState, spawnInfo),                false, LOT_SPAWNINFO         },
    { "area",                     LVT_COBJECT_P, offsetof(struct MarioState, area),                     false, LOT_AREA              },
    { "statusForCamera",          LVT_COBJECT_P, offsetof(struct MarioState, statusForCamera),          false, LOT_PLAYERCAMERASTATE },
    { "marioBodyState",           LVT_COBJECT_P, offsetof(struct MarioState, marioBodyState),           false, LOT_MARIOBODYSTATE    },
    { "controller",               LVT_COBJECT_P, offsetof(struct MarioState, controller),               false, LOT_CONTROLLER        },
    { "animation",                LVT_COBJECT_P, offsetof(struct MarioState, animation),                false, LOT_MARIOANIMATION    },
    { "collidedObjInteractTypes", LVT_U32,       offsetof(struct MarioState, collidedObjInteractTypes), false, LOT_NONE              },
    { "numCoins",                 LVT_S16,       offsetof(struct MarioState, numCoins),                 false, LOT_NONE              },
    { "numStars",                 LVT_S16,       offsetof(struct MarioState, numStars),                 false, LOT_NONE              },
    { "numKeys",                  LVT_S8,        offsetof(struct MarioState, numKeys),                  false, LOT_NONE              },
    { "numLives",                 LVT_S8,        offsetof(struct MarioState, numLives),                 false, LOT_NONE              },
    { "health",                   LVT_S16,       offsetof(struct MarioState, health),                   false, LOT_NONE              },
    { "unkB0",                    LVT_S16,       offsetof(struct MarioState, unkB0),                    false, LOT_NONE              },
    { "hurtCounter",              LVT_U8,        offsetof(struct MarioState, hurtCounter),              false, LOT_NONE              },
    { "healCounter",              LVT_U8,        offsetof(struct MarioState, healCounter),              false, LOT_NONE              },
    { "squishTimer",              LVT_U8,        offsetof(struct MarioState, squishTimer),              false, LOT_NONE              },
    { "fadeWarpOpacity",          LVT_U8,        offsetof(struct MarioState, fadeWarpOpacity),          false, LOT_NONE              },
    { "capTimer",                 LVT_U16,       offsetof(struct MarioState, capTimer),                 false, LOT_NONE              },
    { "prevNumStarsForDialog",    LVT_S16,       offsetof(struct MarioState, prevNumStarsForDialog),    false, LOT_NONE              },
    { "peakHeight",               LVT_F32,       offsetof(struct MarioState, peakHeight),               false, LOT_NONE              },
    { "quicksandDepth",           LVT_F32,       offsetof(struct MarioState, quicksandDepth),           false, LOT_NONE              },
    { "unkC4",                    LVT_F32,       offsetof(struct MarioState, unkC4),                    false, LOT_NONE              },
    { "currentRoom",              LVT_S16,       offsetof(struct MarioState, currentRoom),              false, LOT_NONE              },
    { "heldByObj",                LVT_COBJECT_P, offsetof(struct MarioState, heldByObj),                false, LOT_OBJECT            },
    { "isSnoring",                LVT_U8,        offsetof(struct MarioState, isSnoring),                false, LOT_NONE              },
    { "bubbleObj",                LVT_COBJECT_P, offsetof(struct MarioState, bubbleObj),                false, LOT_OBJECT            },
    { "freeze",                   LVT_U8,        offsetof(struct MarioState, freeze),                   false, LOT_NONE              },
//  { "splineKeyframe",           LVT_???,       offsetof(struct MarioState, splineKeyframe),           false, LOT_???               }, <--- UNIMPLEMENTED
    { "splineKeyframeFraction",   LVT_F32,       offsetof(struct MarioState, splineKeyframeFraction),   false, LOT_NONE              },
    { "splineState",              LVT_S32,       offsetof(struct MarioState, splineState),              false, LOT_NONE              },
    { "nonInstantWarpPos",        LVT_COBJECT,   offsetof(struct MarioState, nonInstantWarpPos),        true,  LOT_VEC3F             },
    { "character",                LVT_COBJECT_P, offsetof(struct MarioState, character),                false, LOT_CHARACTER         },
    { "wasNetworkVisible",        LVT_U8,        offsetof(struct MarioState, wasNetworkVisible),        false, LOT_NONE              },
    { "minimumBoneY",             LVT_F32,       offsetof(struct MarioState, minimumBoneY),             false, LOT_NONE              },
    { "curAnimOffset",            LVT_F32,       offsetof(struct MarioState, curAnimOffset),            false, LOT_NONE              },
};

#define LUA_WARP_NODE_FIELD_COUNT 4
static struct LuaObjectField sWarpNodeFields[LUA_WARP_NODE_FIELD_COUNT] = {
    { "id",        LVT_U8, offsetof(struct WarpNode, id),        false, LOT_NONE },
    { "destLevel", LVT_U8, offsetof(struct WarpNode, destLevel), false, LOT_NONE },
    { "destArea",  LVT_U8, offsetof(struct WarpNode, destArea),  false, LOT_NONE },
    { "destNode",  LVT_U8, offsetof(struct WarpNode, destNode),  false, LOT_NONE },
};

#define LUA_OBJECT_WARP_NODE_FIELD_COUNT 3
static struct LuaObjectField sObjectWarpNodeFields[LUA_OBJECT_WARP_NODE_FIELD_COUNT] = {
    { "node",   LVT_COBJECT,   offsetof(struct ObjectWarpNode, node),   true,  LOT_WARPNODE       },
    { "object", LVT_COBJECT_P, offsetof(struct ObjectWarpNode, object), false, LOT_OBJECT         },
    { "next",   LVT_COBJECT_P, offsetof(struct ObjectWarpNode, next),   false, LOT_OBJECTWARPNODE },
};

#define LUA_INSTANT_WARP_FIELD_COUNT 3
static struct LuaObjectField sInstantWarpFields[LUA_INSTANT_WARP_FIELD_COUNT] = {
    { "id",           LVT_U8,      offsetof(struct InstantWarp, id),           false, LOT_NONE  },
    { "area",         LVT_U8,      offsetof(struct InstantWarp, area),         false, LOT_NONE  },
    { "displacement", LVT_COBJECT, offsetof(struct InstantWarp, displacement), true,  LOT_VEC3S },
};

#define LUA_SPAWN_INFO_FIELD_COUNT 7
static struct LuaObjectField sSpawnInfoFields[LUA_SPAWN_INFO_FIELD_COUNT] = {
    { "startPos",        LVT_COBJECT,   offsetof(struct SpawnInfo, startPos),        true,  LOT_VEC3S     },
    { "startAngle",      LVT_COBJECT,   offsetof(struct SpawnInfo, startAngle),      true,  LOT_VEC3S     },
    { "areaIndex",       LVT_S8,        offsetof(struct SpawnInfo, areaIndex),       false, LOT_NONE      },
    { "activeAreaIndex", LVT_S8,        offsetof(struct SpawnInfo, activeAreaIndex), false, LOT_NONE      },
    { "behaviorArg",     LVT_U32,       offsetof(struct SpawnInfo, behaviorArg),     false, LOT_NONE      },
//  { "behaviorScript",  LVT_???,       offsetof(struct SpawnInfo, behaviorScript),  false, LOT_???       }, <--- UNIMPLEMENTED
    { "unk18",           LVT_COBJECT_P, offsetof(struct SpawnInfo, unk18),           false, LOT_GRAPHNODE },
    { "next",            LVT_COBJECT_P, offsetof(struct SpawnInfo, next),            false, LOT_SPAWNINFO },
};

#define LUA_WHIRLPOOL_FIELD_COUNT 2
static struct LuaObjectField sWhirlpoolFields[LUA_WHIRLPOOL_FIELD_COUNT] = {
    { "pos",      LVT_COBJECT, offsetof(struct Whirlpool, pos),      true,  LOT_VEC3S },
    { "strength", LVT_S16,     offsetof(struct Whirlpool, strength), false, LOT_NONE  },
};

#define LUA_AREA_FIELD_COUNT 10
static struct LuaObjectField sAreaFields[LUA_AREA_FIELD_COUNT] = {
    { "index",             LVT_S8,        offsetof(struct Area, index),             false, LOT_NONE           },
    { "flags",             LVT_S8,        offsetof(struct Area, flags),             false, LOT_NONE           },
    { "terrainType",       LVT_U16,       offsetof(struct Area, terrainType),       false, LOT_NONE           },
//  { "unk04",             LVT_COBJECT_P, offsetof(struct Area, unk04),             false, LOT_???            }, <--- UNIMPLEMENTED
//  { "terrainData",       LVT_???,       offsetof(struct Area, terrainData),       false, LOT_???            }, <--- UNIMPLEMENTED
//  { "surfaceRooms",      LVT_???,       offsetof(struct Area, surfaceRooms),      false, LOT_???            }, <--- UNIMPLEMENTED
//  { "macroObjects",      LVT_???,       offsetof(struct Area, macroObjects),      false, LOT_???            }, <--- UNIMPLEMENTED
    { "warpNodes",         LVT_COBJECT_P, offsetof(struct Area, warpNodes),         false, LOT_OBJECTWARPNODE },
    { "paintingWarpNodes", LVT_COBJECT_P, offsetof(struct Area, paintingWarpNodes), false, LOT_WARPNODE       },
    { "instantWarps",      LVT_COBJECT_P, offsetof(struct Area, instantWarps),      false, LOT_INSTANTWARP    },
    { "objectSpawnInfos",  LVT_COBJECT_P, offsetof(struct Area, objectSpawnInfos),  false, LOT_SPAWNINFO      },
    { "camera",            LVT_COBJECT_P, offsetof(struct Area, camera),            false, LOT_CAMERA         },
//  { "unused28",          LVT_COBJECT_P, offsetof(struct Area, unused28),          false, LOT_???            }, <--- UNIMPLEMENTED
//  { "whirlpools",        LVT_COBJECT_P, offsetof(struct Area, whirlpools),        false, LOT_???            }, <--- UNIMPLEMENTED
//  { "dialog",            LVT_???,       offsetof(struct Area, dialog),            false, LOT_???            }, <--- UNIMPLEMENTED
    { "musicParam",        LVT_U16,       offsetof(struct Area, musicParam),        false, LOT_NONE           },
    { "musicParam2",       LVT_U16,       offsetof(struct Area, musicParam2),       false, LOT_NONE           },
//  { "cachedBehaviors",   LVT_???,       offsetof(struct Area, cachedBehaviors),   false, LOT_???            }, <--- UNIMPLEMENTED
//  { "cachedPositions",   LVT_???,       offsetof(struct Area, cachedPositions),   false, LOT_???            }, <--- UNIMPLEMENTED
};

#define LUA_WARP_TRANSITION_DATA_FIELD_COUNT 10
static struct LuaObjectField sWarpTransitionDataFields[LUA_WARP_TRANSITION_DATA_FIELD_COUNT] = {
    { "red",            LVT_U8,  offsetof(struct WarpTransitionData, red),            false, LOT_NONE },
    { "green",          LVT_U8,  offsetof(struct WarpTransitionData, green),          false, LOT_NONE },
    { "blue",           LVT_U8,  offsetof(struct WarpTransitionData, blue),           false, LOT_NONE },
    { "startTexRadius", LVT_S16, offsetof(struct WarpTransitionData, startTexRadius), false, LOT_NONE },
    { "endTexRadius",   LVT_S16, offsetof(struct WarpTransitionData, endTexRadius),   false, LOT_NONE },
    { "startTexX",      LVT_S16, offsetof(struct WarpTransitionData, startTexX),      false, LOT_NONE },
    { "startTexY",      LVT_S16, offsetof(struct WarpTransitionData, startTexY),      false, LOT_NONE },
    { "endTexX",        LVT_S16, offsetof(struct WarpTransitionData, endTexX),        false, LOT_NONE },
    { "endTexY",        LVT_S16, offsetof(struct WarpTransitionData, endTexY),        false, LOT_NONE },
    { "texTimer",       LVT_S16, offsetof(struct WarpTransitionData, texTimer),       false, LOT_NONE },
};

#define LUA_WARP_TRANSITION_FIELD_COUNT 5
static struct LuaObjectField sWarpTransitionFields[LUA_WARP_TRANSITION_FIELD_COUNT] = {
    { "isActive",       LVT_U8,      offsetof(struct WarpTransition, isActive),       false, LOT_NONE               },
    { "type",           LVT_U8,      offsetof(struct WarpTransition, type),           false, LOT_NONE               },
    { "time",           LVT_U8,      offsetof(struct WarpTransition, time),           false, LOT_NONE               },
    { "pauseRendering", LVT_U8,      offsetof(struct WarpTransition, pauseRendering), false, LOT_NONE               },
    { "data",           LVT_COBJECT, offsetof(struct WarpTransition, data),           true,  LOT_WARPTRANSITIONDATA },
};

#define LUA_PLAYER_CAMERA_STATE_FIELD_COUNT 7
static struct LuaObjectField sPlayerCameraStateFields[LUA_PLAYER_CAMERA_STATE_FIELD_COUNT] = {
    { "action",       LVT_U32,       offsetof(struct PlayerCameraState, action),       false, LOT_NONE   },
    { "pos",          LVT_COBJECT,   offsetof(struct PlayerCameraState, pos),          true,  LOT_VEC3F  },
    { "faceAngle",    LVT_COBJECT,   offsetof(struct PlayerCameraState, faceAngle),    true,  LOT_VEC3S  },
    { "headRotation", LVT_COBJECT,   offsetof(struct PlayerCameraState, headRotation), true,  LOT_VEC3S  },
    { "unused",       LVT_S16,       offsetof(struct PlayerCameraState, unused),       false, LOT_NONE   },
    { "cameraEvent",  LVT_S16,       offsetof(struct PlayerCameraState, cameraEvent),  false, LOT_NONE   },
    { "usedObj",      LVT_COBJECT_P, offsetof(struct PlayerCameraState, usedObj),      false, LOT_OBJECT },
};

#define LUA_TRANSITION_INFO_FIELD_COUNT 9
static struct LuaObjectField sTransitionInfoFields[LUA_TRANSITION_INFO_FIELD_COUNT] = {
    { "posPitch",   LVT_S16,     offsetof(struct TransitionInfo, posPitch),   false, LOT_NONE  },
    { "posYaw",     LVT_S16,     offsetof(struct TransitionInfo, posYaw),     false, LOT_NONE  },
    { "posDist",    LVT_F32,     offsetof(struct TransitionInfo, posDist),    false, LOT_NONE  },
    { "focPitch",   LVT_S16,     offsetof(struct TransitionInfo, focPitch),   false, LOT_NONE  },
    { "focYaw",     LVT_S16,     offsetof(struct TransitionInfo, focYaw),     false, LOT_NONE  },
    { "focDist",    LVT_F32,     offsetof(struct TransitionInfo, focDist),    false, LOT_NONE  },
    { "framesLeft", LVT_S32,     offsetof(struct TransitionInfo, framesLeft), false, LOT_NONE  },
    { "marioPos",   LVT_COBJECT, offsetof(struct TransitionInfo, marioPos),   true,  LOT_VEC3F },
    { "pad",        LVT_U8,      offsetof(struct TransitionInfo, pad),        false, LOT_NONE  },
};

#define LUA_HANDHELD_SHAKE_POINT_FIELD_COUNT 3
static struct LuaObjectField sHandheldShakePointFields[LUA_HANDHELD_SHAKE_POINT_FIELD_COUNT] = {
    { "index", LVT_S8,      offsetof(struct HandheldShakePoint, index), false, LOT_NONE  },
    { "pad",   LVT_U32,     offsetof(struct HandheldShakePoint, pad),   false, LOT_NONE  },
    { "point", LVT_COBJECT, offsetof(struct HandheldShakePoint, point), true,  LOT_VEC3S },
};

#define LUA_CAMERA_TRIGGER_FIELD_COUNT 8
static struct LuaObjectField sCameraTriggerFields[LUA_CAMERA_TRIGGER_FIELD_COUNT] = {
    { "area",      LVT_S8,  offsetof(struct CameraTrigger, area),      false, LOT_NONE },
//  { "event",     LVT_???, offsetof(struct CameraTrigger, event),     false, LOT_???  }, <--- UNIMPLEMENTED
    { "centerX",   LVT_S16, offsetof(struct CameraTrigger, centerX),   false, LOT_NONE },
    { "centerY",   LVT_S16, offsetof(struct CameraTrigger, centerY),   false, LOT_NONE },
    { "centerZ",   LVT_S16, offsetof(struct CameraTrigger, centerZ),   false, LOT_NONE },
    { "boundsX",   LVT_S16, offsetof(struct CameraTrigger, boundsX),   false, LOT_NONE },
    { "boundsY",   LVT_S16, offsetof(struct CameraTrigger, boundsY),   false, LOT_NONE },
    { "boundsZ",   LVT_S16, offsetof(struct CameraTrigger, boundsZ),   false, LOT_NONE },
    { "boundsYaw", LVT_S16, offsetof(struct CameraTrigger, boundsYaw), false, LOT_NONE },
};

#define LUA_CUTSCENE_FIELD_COUNT 1
static struct LuaObjectField sCutsceneFields[LUA_CUTSCENE_FIELD_COUNT] = {
//  { "shot",     LVT_???, offsetof(struct Cutscene, shot),     false, LOT_???  }, <--- UNIMPLEMENTED
    { "duration", LVT_S16, offsetof(struct Cutscene, duration), false, LOT_NONE },
};

#define LUA_CAMERA_FOVSTATUS_FIELD_COUNT 8
static struct LuaObjectField sCameraFOVStatusFields[LUA_CAMERA_FOVSTATUS_FIELD_COUNT] = {
    { "fovFunc",          LVT_U8,  offsetof(struct CameraFOVStatus, fovFunc),          false, LOT_NONE },
    { "fov",              LVT_F32, offsetof(struct CameraFOVStatus, fov),              false, LOT_NONE },
    { "fovOffset",        LVT_F32, offsetof(struct CameraFOVStatus, fovOffset),        false, LOT_NONE },
    { "unusedIsSleeping", LVT_U32, offsetof(struct CameraFOVStatus, unusedIsSleeping), false, LOT_NONE },
    { "shakeAmplitude",   LVT_F32, offsetof(struct CameraFOVStatus, shakeAmplitude),   false, LOT_NONE },
    { "shakePhase",       LVT_S16, offsetof(struct CameraFOVStatus, shakePhase),       false, LOT_NONE },
    { "shakeSpeed",       LVT_S16, offsetof(struct CameraFOVStatus, shakeSpeed),       false, LOT_NONE },
    { "decay",            LVT_S16, offsetof(struct CameraFOVStatus, decay),            false, LOT_NONE },
};

#define LUA_CUTSCENE_SPLINE_POINT_FIELD_COUNT 3
static struct LuaObjectField sCutsceneSplinePointFields[LUA_CUTSCENE_SPLINE_POINT_FIELD_COUNT] = {
    { "index", LVT_S8,      offsetof(struct CutsceneSplinePoint, index), false, LOT_NONE  },
    { "speed", LVT_U8,      offsetof(struct CutsceneSplinePoint, speed), false, LOT_NONE  },
    { "point", LVT_COBJECT, offsetof(struct CutsceneSplinePoint, point), true,  LOT_VEC3S },
};

#define LUA_PLAYER_GEOMETRY_FIELD_COUNT 13
static struct LuaObjectField sPlayerGeometryFields[LUA_PLAYER_GEOMETRY_FIELD_COUNT] = {
    { "currFloor",       LVT_COBJECT_P, offsetof(struct PlayerGeometry, currFloor),       false, LOT_SURFACE },
    { "currFloorHeight", LVT_F32,       offsetof(struct PlayerGeometry, currFloorHeight), false, LOT_NONE    },
    { "currFloorType",   LVT_S16,       offsetof(struct PlayerGeometry, currFloorType),   false, LOT_NONE    },
    { "currCeil",        LVT_COBJECT_P, offsetof(struct PlayerGeometry, currCeil),        false, LOT_SURFACE },
    { "currCeilType",    LVT_S16,       offsetof(struct PlayerGeometry, currCeilType),    false, LOT_NONE    },
    { "currCeilHeight",  LVT_F32,       offsetof(struct PlayerGeometry, currCeilHeight),  false, LOT_NONE    },
    { "prevFloor",       LVT_COBJECT_P, offsetof(struct PlayerGeometry, prevFloor),       false, LOT_SURFACE },
    { "prevFloorHeight", LVT_F32,       offsetof(struct PlayerGeometry, prevFloorHeight), false, LOT_NONE    },
    { "prevFloorType",   LVT_S16,       offsetof(struct PlayerGeometry, prevFloorType),   false, LOT_NONE    },
    { "prevCeil",        LVT_COBJECT_P, offsetof(struct PlayerGeometry, prevCeil),        false, LOT_SURFACE },
    { "prevCeilHeight",  LVT_F32,       offsetof(struct PlayerGeometry, prevCeilHeight),  false, LOT_NONE    },
    { "prevCeilType",    LVT_S16,       offsetof(struct PlayerGeometry, prevCeilType),    false, LOT_NONE    },
    { "waterHeight",     LVT_F32,       offsetof(struct PlayerGeometry, waterHeight),     false, LOT_NONE    },
};

#define LUA_LINEAR_TRANSITION_POINT_FIELD_COUNT 5
static struct LuaObjectField sLinearTransitionPointFields[LUA_LINEAR_TRANSITION_POINT_FIELD_COUNT] = {
    { "focus", LVT_COBJECT, offsetof(struct LinearTransitionPoint, focus), true,  LOT_VEC3F },
    { "pos",   LVT_COBJECT, offsetof(struct LinearTransitionPoint, pos),   true,  LOT_VEC3F },
    { "dist",  LVT_F32,     offsetof(struct LinearTransitionPoint, dist),  false, LOT_NONE  },
    { "pitch", LVT_S16,     offsetof(struct LinearTransitionPoint, pitch), false, LOT_NONE  },
    { "yaw",   LVT_S16,     offsetof(struct LinearTransitionPoint, yaw),   false, LOT_NONE  },
};

#define LUA_MODE_TRANSITION_INFO_FIELD_COUNT 6
static struct LuaObjectField sModeTransitionInfoFields[LUA_MODE_TRANSITION_INFO_FIELD_COUNT] = {
    { "newMode",         LVT_S16,     offsetof(struct ModeTransitionInfo, newMode),         false, LOT_NONE                  },
    { "lastMode",        LVT_S16,     offsetof(struct ModeTransitionInfo, lastMode),        false, LOT_NONE                  },
    { "max",             LVT_S16,     offsetof(struct ModeTransitionInfo, max),             false, LOT_NONE                  },
    { "frame",           LVT_S16,     offsetof(struct ModeTransitionInfo, frame),           false, LOT_NONE                  },
    { "transitionStart", LVT_COBJECT, offsetof(struct ModeTransitionInfo, transitionStart), true,  LOT_LINEARTRANSITIONPOINT },
    { "transitionEnd",   LVT_COBJECT, offsetof(struct ModeTransitionInfo, transitionEnd),   true,  LOT_LINEARTRANSITIONPOINT },
};

#define LUA_PARALLEL_TRACKING_POINT_FIELD_COUNT 4
static struct LuaObjectField sParallelTrackingPointFields[LUA_PARALLEL_TRACKING_POINT_FIELD_COUNT] = {
    { "startOfPath", LVT_S16,     offsetof(struct ParallelTrackingPoint, startOfPath), false, LOT_NONE  },
    { "pos",         LVT_COBJECT, offsetof(struct ParallelTrackingPoint, pos),         true,  LOT_VEC3F },
    { "distThresh",  LVT_F32,     offsetof(struct ParallelTrackingPoint, distThresh),  false, LOT_NONE  },
    { "zoom",        LVT_F32,     offsetof(struct ParallelTrackingPoint, zoom),        false, LOT_NONE  },
};

#define LUA_CAMERA_STORED_INFO_FIELD_COUNT 4
static struct LuaObjectField sCameraStoredInfoFields[LUA_CAMERA_STORED_INFO_FIELD_COUNT] = {
    { "pos",           LVT_COBJECT, offsetof(struct CameraStoredInfo, pos),           true,  LOT_VEC3F },
    { "focus",         LVT_COBJECT, offsetof(struct CameraStoredInfo, focus),         true,  LOT_VEC3F },
    { "panDist",       LVT_F32,     offsetof(struct CameraStoredInfo, panDist),       false, LOT_NONE  },
    { "cannonYOffset", LVT_F32,     offsetof(struct CameraStoredInfo, cannonYOffset), false, LOT_NONE  },
};

#define LUA_CUTSCENE_VARIABLE_FIELD_COUNT 5
static struct LuaObjectField sCutsceneVariableFields[LUA_CUTSCENE_VARIABLE_FIELD_COUNT] = {
    { "unused1",     LVT_S32,     offsetof(struct CutsceneVariable, unused1),     false, LOT_NONE  },
    { "point",       LVT_COBJECT, offsetof(struct CutsceneVariable, point),       true,  LOT_VEC3F },
    { "unusedPoint", LVT_COBJECT, offsetof(struct CutsceneVariable, unusedPoint), true,  LOT_VEC3F },
    { "angle",       LVT_COBJECT, offsetof(struct CutsceneVariable, angle),       true,  LOT_VEC3S },
    { "unused2",     LVT_S16,     offsetof(struct CutsceneVariable, unused2),     false, LOT_NONE  },
};

#define LUA_CAMERA_FIELD_COUNT 12
static struct LuaObjectField sCameraFields[LUA_CAMERA_FIELD_COUNT] = {
    { "mode",       LVT_U8,      offsetof(struct Camera, mode),       false, LOT_NONE  },
    { "defMode",    LVT_U8,      offsetof(struct Camera, defMode),    false, LOT_NONE  },
    { "yaw",        LVT_S16,     offsetof(struct Camera, yaw),        false, LOT_NONE  },
    { "focus",      LVT_COBJECT, offsetof(struct Camera, focus),      true,  LOT_VEC3F },
    { "pos",        LVT_COBJECT, offsetof(struct Camera, pos),        true,  LOT_VEC3F },
    { "unusedVec1", LVT_COBJECT, offsetof(struct Camera, unusedVec1), true,  LOT_VEC3F },
    { "areaCenX",   LVT_F32,     offsetof(struct Camera, areaCenX),   false, LOT_NONE  },
    { "areaCenZ",   LVT_F32,     offsetof(struct Camera, areaCenZ),   false, LOT_NONE  },
    { "cutscene",   LVT_U8,      offsetof(struct Camera, cutscene),   false, LOT_NONE  },
//  { "filler31",   LVT_???,     offsetof(struct Camera, filler31),   false, LOT_???   }, <--- UNIMPLEMENTED
    { "nextYaw",    LVT_S16,     offsetof(struct Camera, nextYaw),    false, LOT_NONE  },
//  { "filler3C",   LVT_???,     offsetof(struct Camera, filler3C),   false, LOT_???   }, <--- UNIMPLEMENTED
    { "doorStatus", LVT_U8,      offsetof(struct Camera, doorStatus), false, LOT_NONE  },
    { "areaCenY",   LVT_F32,     offsetof(struct Camera, areaCenY),   false, LOT_NONE  },
};

#define LUA_LAKITU_STATE_FIELD_COUNT 35
static struct LuaObjectField sLakituStateFields[LUA_LAKITU_STATE_FIELD_COUNT] = {
    { "curFocus",                         LVT_COBJECT, offsetof(struct LakituState, curFocus),                         true,  LOT_VEC3F },
    { "curPos",                           LVT_COBJECT, offsetof(struct LakituState, curPos),                           true,  LOT_VEC3F },
    { "goalFocus",                        LVT_COBJECT, offsetof(struct LakituState, goalFocus),                        true,  LOT_VEC3F },
    { "goalPos",                          LVT_COBJECT, offsetof(struct LakituState, goalPos),                          true,  LOT_VEC3F },
//  { "filler30",                         LVT_???,     offsetof(struct LakituState, filler30),                         false, LOT_???   }, <--- UNIMPLEMENTED
    { "mode",                             LVT_U8,      offsetof(struct LakituState, mode),                             false, LOT_NONE  },
    { "defMode",                          LVT_U8,      offsetof(struct LakituState, defMode),                          false, LOT_NONE  },
//  { "filler3E",                         LVT_???,     offsetof(struct LakituState, filler3E),                         false, LOT_???   }, <--- UNIMPLEMENTED
    { "focusDistance",                    LVT_F32,     offsetof(struct LakituState, focusDistance),                    false, LOT_NONE  },
    { "oldPitch",                         LVT_S16,     offsetof(struct LakituState, oldPitch),                         false, LOT_NONE  },
    { "oldYaw",                           LVT_S16,     offsetof(struct LakituState, oldYaw),                           false, LOT_NONE  },
    { "oldRoll",                          LVT_S16,     offsetof(struct LakituState, oldRoll),                          false, LOT_NONE  },
    { "shakeMagnitude",                   LVT_COBJECT, offsetof(struct LakituState, shakeMagnitude),                   true,  LOT_VEC3S },
    { "shakePitchPhase",                  LVT_S16,     offsetof(struct LakituState, shakePitchPhase),                  false, LOT_NONE  },
    { "shakePitchVel",                    LVT_S16,     offsetof(struct LakituState, shakePitchVel),                    false, LOT_NONE  },
    { "shakePitchDecay",                  LVT_S16,     offsetof(struct LakituState, shakePitchDecay),                  false, LOT_NONE  },
    { "unusedVec1",                       LVT_COBJECT, offsetof(struct LakituState, unusedVec1),                       true,  LOT_VEC3F },
    { "unusedVec2",                       LVT_COBJECT, offsetof(struct LakituState, unusedVec2),                       true,  LOT_VEC3S },
//  { "filler72",                         LVT_???,     offsetof(struct LakituState, filler72),                         false, LOT_???   }, <--- UNIMPLEMENTED
    { "roll",                             LVT_S16,     offsetof(struct LakituState, roll),                             false, LOT_NONE  },
    { "yaw",                              LVT_S16,     offsetof(struct LakituState, yaw),                              false, LOT_NONE  },
    { "nextYaw",                          LVT_S16,     offsetof(struct LakituState, nextYaw),                          false, LOT_NONE  },
    { "focus",                            LVT_COBJECT, offsetof(struct LakituState, focus),                            true,  LOT_VEC3F },
    { "pos",                              LVT_COBJECT, offsetof(struct LakituState, pos),                              true,  LOT_VEC3F },
    { "shakeRollPhase",                   LVT_S16,     offsetof(struct LakituState, shakeRollPhase),                   false, LOT_NONE  },
    { "shakeRollVel",                     LVT_S16,     offsetof(struct LakituState, shakeRollVel),                     false, LOT_NONE  },
    { "shakeRollDecay",                   LVT_S16,     offsetof(struct LakituState, shakeRollDecay),                   false, LOT_NONE  },
    { "shakeYawPhase",                    LVT_S16,     offsetof(struct LakituState, shakeYawPhase),                    false, LOT_NONE  },
    { "shakeYawVel",                      LVT_S16,     offsetof(struct LakituState, shakeYawVel),                      false, LOT_NONE  },
    { "shakeYawDecay",                    LVT_S16,     offsetof(struct LakituState, shakeYawDecay),                    false, LOT_NONE  },
    { "focHSpeed",                        LVT_F32,     offsetof(struct LakituState, focHSpeed),                        false, LOT_NONE  },
    { "focVSpeed",                        LVT_F32,     offsetof(struct LakituState, focVSpeed),                        false, LOT_NONE  },
    { "posHSpeed",                        LVT_F32,     offsetof(struct LakituState, posHSpeed),                        false, LOT_NONE  },
    { "posVSpeed",                        LVT_F32,     offsetof(struct LakituState, posVSpeed),                        false, LOT_NONE  },
    { "keyDanceRoll",                     LVT_S16,     offsetof(struct LakituState, keyDanceRoll),                     false, LOT_NONE  },
    { "lastFrameAction",                  LVT_U32,     offsetof(struct LakituState, lastFrameAction),                  false, LOT_NONE  },
    { "unused",                           LVT_S16,     offsetof(struct LakituState, unused),                           false, LOT_NONE  },
    { "skipCameraInterpolationTimestamp", LVT_U32,     offsetof(struct LakituState, skipCameraInterpolationTimestamp), false, LOT_NONE  },
};

#define LUA_CHARACTER_FIELD_COUNT 54
static struct LuaObjectField sCharacterFields[LUA_CHARACTER_FIELD_COUNT] = {
//  { "name",                  LVT_???, offsetof(struct Character, name),                  false, LOT_???  }, <--- UNIMPLEMENTED
//  { "hudHead",               LVT_???, offsetof(struct Character, hudHead),               false, LOT_???  }, <--- UNIMPLEMENTED
//  { "hudHeadTexture",        LVT_???, offsetof(struct Character, hudHeadTexture),        false, LOT_???  }, <--- UNIMPLEMENTED
    { "cameraHudHead",         LVT_U32, offsetof(struct Character, cameraHudHead),         false, LOT_NONE },
    { "modelId",               LVT_U32, offsetof(struct Character, modelId),               false, LOT_NONE },
    { "capModelId",            LVT_U32, offsetof(struct Character, capModelId),            false, LOT_NONE },
    { "capMetalModelId",       LVT_U32, offsetof(struct Character, capMetalModelId),       false, LOT_NONE },
    { "capWingModelId",        LVT_U32, offsetof(struct Character, capWingModelId),        false, LOT_NONE },
    { "capMetalWingModelId",   LVT_U32, offsetof(struct Character, capMetalWingModelId),   false, LOT_NONE },
    { "capEnemyLayer",         LVT_U8,  offsetof(struct Character, capEnemyLayer),         false, LOT_NONE },
//  { "capEnemyGfx",           LVT_???, offsetof(struct Character, capEnemyGfx),           false, LOT_???  }, <--- UNIMPLEMENTED
//  { "capEnemyDecalGfx",      LVT_???, offsetof(struct Character, capEnemyDecalGfx),      false, LOT_???  }, <--- UNIMPLEMENTED
    { "animOffsetEnabled",     LVT_U8,  offsetof(struct Character, animOffsetEnabled),     false, LOT_NONE },
    { "animOffsetLowYPoint",   LVT_F32, offsetof(struct Character, animOffsetLowYPoint),   false, LOT_NONE },
    { "animOffsetFeet",        LVT_F32, offsetof(struct Character, animOffsetFeet),        false, LOT_NONE },
    { "animOffsetHand",        LVT_F32, offsetof(struct Character, animOffsetHand),        false, LOT_NONE },
    { "soundFreqScale",        LVT_F32, offsetof(struct Character, soundFreqScale),        false, LOT_NONE },
    { "soundYahWahHoo",        LVT_S32, offsetof(struct Character, soundYahWahHoo),        false, LOT_NONE },
    { "soundHoohoo",           LVT_S32, offsetof(struct Character, soundHoohoo),           false, LOT_NONE },
    { "soundYahoo",            LVT_S32, offsetof(struct Character, soundYahoo),            false, LOT_NONE },
    { "soundUh",               LVT_S32, offsetof(struct Character, soundUh),               false, LOT_NONE },
    { "soundHrmm",             LVT_S32, offsetof(struct Character, soundHrmm),             false, LOT_NONE },
    { "soundWah2",             LVT_S32, offsetof(struct Character, soundWah2),             false, LOT_NONE },
    { "soundWhoa",             LVT_S32, offsetof(struct Character, soundWhoa),             false, LOT_NONE },
    { "soundEeuh",             LVT_S32, offsetof(struct Character, soundEeuh),             false, LOT_NONE },
    { "soundAttacked",         LVT_S32, offsetof(struct Character, soundAttacked),         false, LOT_NONE },
    { "soundOoof",             LVT_S32, offsetof(struct Character, soundOoof),             false, LOT_NONE },
    { "soundOoof2",            LVT_S32, offsetof(struct Character, soundOoof2),            false, LOT_NONE },
    { "soundHereWeGo",         LVT_S32, offsetof(struct Character, soundHereWeGo),         false, LOT_NONE },
    { "soundYawning",          LVT_S32, offsetof(struct Character, soundYawning),          false, LOT_NONE },
    { "soundSnoring1",         LVT_S32, offsetof(struct Character, soundSnoring1),         false, LOT_NONE },
    { "soundSnoring2",         LVT_S32, offsetof(struct Character, soundSnoring2),         false, LOT_NONE },
    { "soundWaaaooow",         LVT_S32, offsetof(struct Character, soundWaaaooow),         false, LOT_NONE },
    { "soundHaha",             LVT_S32, offsetof(struct Character, soundHaha),             false, LOT_NONE },
    { "soundHaha_2",           LVT_S32, offsetof(struct Character, soundHaha_2),           false, LOT_NONE },
    { "soundUh2",              LVT_S32, offsetof(struct Character, soundUh2),              false, LOT_NONE },
    { "soundUh2_2",            LVT_S32, offsetof(struct Character, soundUh2_2),            false, LOT_NONE },
    { "soundOnFire",           LVT_S32, offsetof(struct Character, soundOnFire),           false, LOT_NONE },
    { "soundDying",            LVT_S32, offsetof(struct Character, soundDying),            false, LOT_NONE },
    { "soundPantingCold",      LVT_S32, offsetof(struct Character, soundPantingCold),      false, LOT_NONE },
    { "soundPanting",          LVT_S32, offsetof(struct Character, soundPanting),          false, LOT_NONE },
    { "soundCoughing1",        LVT_S32, offsetof(struct Character, soundCoughing1),        false, LOT_NONE },
    { "soundCoughing2",        LVT_S32, offsetof(struct Character, soundCoughing2),        false, LOT_NONE },
    { "soundCoughing3",        LVT_S32, offsetof(struct Character, soundCoughing3),        false, LOT_NONE },
    { "soundPunchYah",         LVT_S32, offsetof(struct Character, soundPunchYah),         false, LOT_NONE },
    { "soundPunchHoo",         LVT_S32, offsetof(struct Character, soundPunchHoo),         false, LOT_NONE },
    { "soundMamaMia",          LVT_S32, offsetof(struct Character, soundMamaMia),          false, LOT_NONE },
    { "soundGroundPoundWah",   LVT_S32, offsetof(struct Character, soundGroundPoundWah),   false, LOT_NONE },
    { "soundDrowning",         LVT_S32, offsetof(struct Character, soundDrowning),         false, LOT_NONE },
    { "soundPunchWah",         LVT_S32, offsetof(struct Character, soundPunchWah),         false, LOT_NONE },
    { "soundYahooWahaYippee",  LVT_S32, offsetof(struct Character, soundYahooWahaYippee),  false, LOT_NONE },
    { "soundDoh",              LVT_S32, offsetof(struct Character, soundDoh),              false, LOT_NONE },
    { "soundGameOver",         LVT_S32, offsetof(struct Character, soundGameOver),         false, LOT_NONE },
    { "soundHello",            LVT_S32, offsetof(struct Character, soundHello),            false, LOT_NONE },
    { "soundPressStartToPlay", LVT_S32, offsetof(struct Character, soundPressStartToPlay), false, LOT_NONE },
    { "soundTwirlBounce",      LVT_S32, offsetof(struct Character, soundTwirlBounce),      false, LOT_NONE },
    { "soundSnoring3",         LVT_S32, offsetof(struct Character, soundSnoring3),         false, LOT_NONE },
    { "soundSoLongaBowser",    LVT_S32, offsetof(struct Character, soundSoLongaBowser),    false, LOT_NONE },
    { "soundImaTired",         LVT_S32, offsetof(struct Character, soundImaTired),         false, LOT_NONE },
};

struct LuaObjectTable sLuaObjectAutogenTable[LOT_AUTOGEN_MAX - LOT_AUTOGEN_MIN] = {
    { LOT_CONTROLLER,            sControllerFields,            LUA_CONTROLLER_FIELD_COUNT              },
    { LOT_ANIMATION,             sAnimationFields,             LUA_ANIMATION_FIELD_COUNT               },
    { LOT_GRAPHNODE,             sGraphNodeFields,             LUA_GRAPH_NODE_FIELD_COUNT              },
    { LOT_GRAPHNODEOBJECT_SUB,   sGraphNodeObject_subFields,   LUA_GRAPH_NODE_OBJECT_SUB_FIELD_COUNT   },
    { LOT_GRAPHNODEOBJECT,       sGraphNodeObjectFields,       LUA_GRAPH_NODE_OBJECT_FIELD_COUNT       },
    { LOT_OBJECTNODE,            sObjectNodeFields,            LUA_OBJECT_NODE_FIELD_COUNT             },
    { LOT_OBJECT,                sObjectFields,                LUA_OBJECT_FIELD_COUNT                  },
    { LOT_OBJECTHITBOX,          sObjectHitboxFields,          LUA_OBJECT_HITBOX_FIELD_COUNT           },
    { LOT_WAYPOINT,              sWaypointFields,              LUA_WAYPOINT_FIELD_COUNT                },
    { LOT_SURFACE,               sSurfaceFields,               LUA_SURFACE_FIELD_COUNT                 },
    { LOT_MARIOBODYSTATE,        sMarioBodyStateFields,        LUA_MARIO_BODY_STATE_FIELD_COUNT        },
    { LOT_OFFSETSIZEPAIR,        sOffsetSizePairFields,        LUA_OFFSET_SIZE_PAIR_FIELD_COUNT        },
    { LOT_MARIOANIMATION,        sMarioAnimationFields,        LUA_MARIO_ANIMATION_FIELD_COUNT         },
    { LOT_MARIOSTATE,            sMarioStateFields,            LUA_MARIO_STATE_FIELD_COUNT             },
    { LOT_WARPNODE,              sWarpNodeFields,              LUA_WARP_NODE_FIELD_COUNT               },
    { LOT_OBJECTWARPNODE,        sObjectWarpNodeFields,        LUA_OBJECT_WARP_NODE_FIELD_COUNT        },
    { LOT_INSTANTWARP,           sInstantWarpFields,           LUA_INSTANT_WARP_FIELD_COUNT            },
    { LOT_SPAWNINFO,             sSpawnInfoFields,             LUA_SPAWN_INFO_FIELD_COUNT              },
    { LOT_WHIRLPOOL,             sWhirlpoolFields,             LUA_WHIRLPOOL_FIELD_COUNT               },
    { LOT_AREA,                  sAreaFields,                  LUA_AREA_FIELD_COUNT                    },
    { LOT_WARPTRANSITIONDATA,    sWarpTransitionDataFields,    LUA_WARP_TRANSITION_DATA_FIELD_COUNT    },
    { LOT_WARPTRANSITION,        sWarpTransitionFields,        LUA_WARP_TRANSITION_FIELD_COUNT         },
    { LOT_PLAYERCAMERASTATE,     sPlayerCameraStateFields,     LUA_PLAYER_CAMERA_STATE_FIELD_COUNT     },
    { LOT_TRANSITIONINFO,        sTransitionInfoFields,        LUA_TRANSITION_INFO_FIELD_COUNT         },
    { LOT_HANDHELDSHAKEPOINT,    sHandheldShakePointFields,    LUA_HANDHELD_SHAKE_POINT_FIELD_COUNT    },
    { LOT_CAMERATRIGGER,         sCameraTriggerFields,         LUA_CAMERA_TRIGGER_FIELD_COUNT          },
    { LOT_CUTSCENE,              sCutsceneFields,              LUA_CUTSCENE_FIELD_COUNT                },
    { LOT_CAMERAFOVSTATUS,       sCameraFOVStatusFields,       LUA_CAMERA_FOVSTATUS_FIELD_COUNT        },
    { LOT_CUTSCENESPLINEPOINT,   sCutsceneSplinePointFields,   LUA_CUTSCENE_SPLINE_POINT_FIELD_COUNT   },
    { LOT_PLAYERGEOMETRY,        sPlayerGeometryFields,        LUA_PLAYER_GEOMETRY_FIELD_COUNT         },
    { LOT_LINEARTRANSITIONPOINT, sLinearTransitionPointFields, LUA_LINEAR_TRANSITION_POINT_FIELD_COUNT },
    { LOT_MODETRANSITIONINFO,    sModeTransitionInfoFields,    LUA_MODE_TRANSITION_INFO_FIELD_COUNT    },
    { LOT_PARALLELTRACKINGPOINT, sParallelTrackingPointFields, LUA_PARALLEL_TRACKING_POINT_FIELD_COUNT },
    { LOT_CAMERASTOREDINFO,      sCameraStoredInfoFields,      LUA_CAMERA_STORED_INFO_FIELD_COUNT      },
    { LOT_CUTSCENEVARIABLE,      sCutsceneVariableFields,      LUA_CUTSCENE_VARIABLE_FIELD_COUNT       },
    { LOT_CAMERA,                sCameraFields,                LUA_CAMERA_FIELD_COUNT                  },
    { LOT_LAKITUSTATE,           sLakituStateFields,           LUA_LAKITU_STATE_FIELD_COUNT            },
    { LOT_CHARACTER,             sCharacterFields,             LUA_CHARACTER_FIELD_COUNT               },
};

struct LuaObjectField* smlua_get_object_field_autogen(u16 lot, const char* key) {
    struct LuaObjectTable* ot = &sLuaObjectAutogenTable[lot - LOT_AUTOGEN_MIN - 1];
    // TODO: change this to binary search or hash table or something
    for (int i = 0; i < ot->fieldCount; i++) {
        if (!strcmp(ot->fields[i].key, key)) {
            return &ot->fields[i];
        }
    }
    return NULL;
}

