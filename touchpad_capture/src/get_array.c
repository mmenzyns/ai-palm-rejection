#include "debug_v4l.h"

#include <stdlib.h>
#include <time.h>
#include <assert.h>


struct hm_cfg* init() {
    int devno = 0;
    int input = 0;

    struct hm_cfg *cfg = calloc(1, sizeof(struct hm_cfg));
    *cfg = (struct hm_cfg){
        .width = 0,
        .file = NULL,
        .print = false,
        .single_capture = false,
        .rate = 1,
    };

    const struct timespec ts = {
        .tv_sec = 1 / cfg->rate,
        .tv_nsec = ((cfg->rate > 1)?(1000 * 1000 * 1000 / cfg->rate):0)
    };
    cfg->ts = ts;


    snprintf(cfg->path, sizeof(cfg->path), "/dev/v4l-touch%d", devno);

    if (hm_v4l_init(cfg, input) < 0)
        return NULL;

    return cfg;
}


int *get_image(struct hm_cfg *cfg) {
    size_t len = hm_v4l_get_frame(cfg); // Get a new buffer frame

    nanosleep(&cfg->ts, NULL); // Rate of data capturing

    int *array = malloc(sizeof(int) * cfg->width * cfg->height);

    for (size_t i = 0; i < len; i++) { // Cycle through the frame
        int blob = hm_v4l_get_value(cfg, i);
        assert(blob <= 255 && blob >= -255);
        array[i] = blob;
    }

    return array;
}

int main() {
    struct hm_cfg* cfg = init();
    int *image = get_image(cfg);
    return 0;
}

