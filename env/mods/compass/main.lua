-- MOD BY squishy6094 on sm64ex-coop discord

-- name: Compass
-- description: 
-- incompatible: compass



function s16(num)
    num = math.floor(num) & 0xFFFF
    if num >= 32768 then return num - 65536 end
    return num
end

--- @param m MarioState
--- @param target Object
--- @param iconTexture TextureInfo
function render_hud_radar(target, iconTexture, texW, texH, x, y)
    local m = gMarioStates[0]

    -- direction
    local angle = s16(
        atan2s(
            target.oPosZ - m.pos.z,
            target.oPosX - m.pos.x
        )
    )
    
    local dist = vec3f_dist({ x = target.oPosX, y = target.oPosY, z = target.oPosZ }, m.pos)

    local distToScale = 100 - math.floor(dist*0.01)
    if distToScale < 0 then distToScale = 0 end
    if distToScale > 100 then distToScale = 100 end
    distToScale = distToScale*0.01

    djui_hud_set_rotation(angle, 0.5/distToScale, 2.5/distToScale)
    djui_hud_render_texture(iconTexture, x + 4, y - 12, distToScale/2, distToScale/2)
    djui_hud_set_rotation(0, 0, 0)
end

function compass_render()
    local m = gMarioStates[get_current_camera_index()]
    local m2 = gMarioStates[get_current_compass_target_index()]
    local scale = 4
    local x_offset = 120
    local y_offset = 120

    djui_hud_set_font(FONT_NORMAL)
    djui_hud_set_resolution(RESOLUTION_N64)
    djui_hud_set_color(255, 255, 255, 255)
    djui_hud_render_texture(get_texture_info("back"), djui_hud_get_screen_width() - x_offset, djui_hud_get_screen_height() - y_offset, scale, scale)
    -- if gLakituState.yaw ~= nil then
    --     djui_hud_set_rotation(gLakituState.yaw , 0.5, 0.5)
    --     djui_hud_render_texture(get_texture_info("camera-dial"), djui_hud_get_screen_width() - x_offset, djui_hud_get_screen_height() - y_offset, scale, scale)
    -- end

    
    if m.faceAngle.y ~= nil then
        local angle = s16(
            atan2s(
                m2.pos.z - m.pos.z,
                m2.pos.x - m.pos.x
            )
        )
        angle = angle - m.faceAngle.y
        djui_hud_set_rotation(angle , 0.5, 0.5)
        djui_hud_render_texture(get_texture_info("player-dial"), djui_hud_get_screen_width() - x_offset , djui_hud_get_screen_height() - y_offset, scale, scale)
    end
    djui_hud_set_rotation(0, 0, 0)
    
    -- -- red coin
    -- if obj_get_nearest_object_with_behavior_id(gMarioStates[0].marioObj, id_bhvRedCoin) ~= nil then
    --     local r = obj_get_nearest_object_with_behavior_id(gMarioStates[0].marioObj, id_bhvRedCoin)
    --     djui_hud_set_color(255,0,0,255)
    --     render_hud_radar(r, gTextures.coin, 1, 1, djui_hud_get_screen_width() - 44, djui_hud_get_screen_height() - 62)
    -- end
    -- djui_hud_set_color(255,255,255,255)
    -- -- spawnable star
    -- if obj_get_nearest_object_with_behavior_id(gMarioStates[0].marioObj, 412) ~= nil then
    --     local s = obj_get_nearest_object_with_behavior_id(gMarioStates[0].marioObj, 412)
    --     render_hud_radar(s, gTextures.star, 1, 1, djui_hud_get_screen_width() - 44, djui_hud_get_screen_height() - 62)
    -- -- star from a box
    -- elseif obj_get_nearest_object_with_behavior_id(gMarioStates[0].marioObj, 538) ~= nil then
    --     local s = obj_get_nearest_object_with_behavior_id(gMarioStates[0].marioObj, 538)
    --     render_hud_radar(s, gTextures.star, 1, 1, djui_hud_get_screen_width() - 44, djui_hud_get_screen_height() - 62)
    -- -- regular star
    -- elseif obj_get_nearest_object_with_behavior_id(gMarioStates[0].marioObj, 409) ~= nil then
    --     local s = obj_get_nearest_object_with_behavior_id(gMarioStates[0].marioObj, 409)
    --     render_hud_radar(s, gTextures.star, 1, 1, djui_hud_get_screen_width() - 44, djui_hud_get_screen_height() - 62)
    -- end
    -- -- secret
    -- if obj_get_nearest_object_with_behavior_id(gMarioStates[0].marioObj, id_bhvHiddenStarTrigger) ~= nil then
    --     local sc = obj_get_nearest_object_with_behavior_id(gMarioStates[0].marioObj, id_bhvHiddenStarTrigger)
    --     djui_hud_set_color(255,255,0,255)
    --     render_hud_radar(sc, get_texture_info("secret"), 1, 1, djui_hud_get_screen_width() - 44, djui_hud_get_screen_height() - 62)
    -- end
end

hook_event(HOOK_ON_HUD_RENDER, compass_render)

-- dialog eater
hook_event(HOOK_ON_DIALOG, function () return false end)