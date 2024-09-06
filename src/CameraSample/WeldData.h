
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

Q_DECLARE_METATYPE(WeldResult)