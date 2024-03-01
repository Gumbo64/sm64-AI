-- Compass by squishy6094 on sm64ex-coop discord

-- name: Compass
-- description: 
-- incompatible: compass

-- gLevelValues.entryLevel = LEVEL_BOB
gLevelValues.entryLevel = LEVEL_BITDW
local my_start_pos = {x = 0, y = 0, z = 0}

function s16(num)
    num = math.floor(num) & 0xFFFF
    if num >= 32768 then return num - 65536 end
    return num
end

function compass_render()
    local m = gMarioStates[get_current_camera_index()]

    local target = {}
    get_current_compass_target(target)

    local scale = 4
    local x_offset = 120
    local y_offset = 120

    djui_hud_set_font(FONT_NORMAL)
    djui_hud_set_resolution(RESOLUTION_N64)
    djui_hud_set_color(255, 255, 255, 255)
    djui_hud_render_texture(get_texture_info("back"), djui_hud_get_screen_width() - x_offset, djui_hud_get_screen_height() - y_offset, scale, scale)
    
    if m.faceAngle.y ~= nil then
        local angle = s16(
            atan2s(
                target.z - m.pos.z,
                target.x - m.pos.x
            )
        )
        angle = angle - m.faceAngle.y
        djui_hud_set_rotation(angle , 0.5, 0.5)
        djui_hud_render_texture(get_texture_info("player-dial"), djui_hud_get_screen_width() - x_offset , djui_hud_get_screen_height() - y_offset, scale, scale)
    end
    djui_hud_set_rotation(0, 0, 0)
end

hook_event(HOOK_ON_HUD_RENDER, compass_render)

-- dialog eater
hook_event(HOOK_ON_DIALOG, function () return false end)


-- function on_update()
--     local m = gMarioStates[0]
--     print(m.pos.x, m.pos.y, m.pos.z, my_start_pos.x-m.pos.x, my_start_pos.y-m.pos.y, my_start_pos.z-m.pos.z)
-- end

-- hook_event(HOOK_MARIO_UPDATE, on_update)

function on_init()
    vec3f_set(my_start_pos, gMarioStates[0].pos.x, gMarioStates[0].pos.y, gMarioStates[0].pos.z)
    -- print(my_start_pos.x, my_start_pos.y, my_start_pos.z)
end
hook_event(HOOK_ON_LEVEL_INIT, on_init)

-- death function
function on_death(m)

    m.hurtCounter = 0
    m.health = 0x880
    set_mario_action(m, ACT_IDLE, 0)
    vec3f_set(m.pos, my_start_pos.x, my_start_pos.y, my_start_pos.z)
    set_death_notice(m.playerIndex)
    -- soft_reset_camera(m.area.camera)
    -- m.area.camera.cutscene = 0

    return false
end




hook_event(HOOK_ON_DEATH, on_death)