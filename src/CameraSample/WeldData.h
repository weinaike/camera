
#pragma once

#include <QMetaType>
struct WeldResult {
    bool    is_enable;
    int     camera_id;
    int     frame_id;
    int     weld_status;
    float   status_score;
    float   weld_depth;
    float   front_quality;
    float   back_quality; 
};


struct LaserControlData
{
    int     command_id;     // command id    
    float   power;          // Laser power
    float   de_focus;       // De-focus
    float   speed;          // Welding speed
};


Q_DECLARE_METATYPE(WeldResult)