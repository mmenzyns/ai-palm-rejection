#include "main.h"

void sigintHandler() {
    interrupt_flag_set = 1;
}

int main(int argc, char *argv[]) {
    int devno = 0;
    int input = 0;
    int opt;

    struct hm_cfg cfg = {
        .rate = 100,
        .width = 0,
        .min = INT_MAX,
        .max = INT_MIN,
        .auto_min = true,
        .auto_max = true,
        .print = false,
        .visual = false,
        .single_capture = false,
        .file = NULL,
    };

    while ((opt = getopt(argc, argv, "pvsf:r:")) != -1) {
        switch (opt) {
            case 'p': cfg.print = true; break; // Print values
            case 'v': cfg.visual = true; break; // Show heatmap
            case 's': cfg.single_capture = true; break; // Save value only when key is pressed
            case 'f':
                cfg.file = fopen(optarg, "w");
                if (!cfg.file) {
                    fprintf(stderr, "Erro with selected path %s\n", optarg);
                    return 1;
                }
                break;
            case 'r': cfg.rate = strtol(optarg, NULL, 10); break;
            default:
                fprintf(stderr, "Unrecognized argument -%c (%d)\n", opt, opt);
                return 1;
        }
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(struct sigaction));
    sa.sa_handler = sigintHandler;
    sa.sa_flags = 0;// not SA_RESTART!;

    sigaction(SIGINT, &sa, NULL);

    snprintf(cfg.path, sizeof(cfg.path), "/dev/v4l-touch%d", devno);

    if (hm_v4l_init(&cfg, input) < 0)
        return 1;

    if (cfg.file)
        fprintf(cfg.file, "width\t%d\theight\t%d\n", cfg.width, cfg.height);

    if (cfg.single_capture)
        return singleCapture(&cfg);
    else
        return continuousCapture(&cfg);

    return 0;
}

int singleCapture(struct hm_cfg *cfg) {
    size_t len;
    bool first_number = true;

    int err = 0;
    while (1) {
        printf("\n\nPress ENTER key to Continue\n");
        getchar();

        if (interrupt_flag_set)
            break;

        len = hm_v4l_get_frame(cfg);

        if (len == 0) {
            if (err > 5)
                return 1;
            err++;
            continue;
        }

        for (size_t i = 0; i < len; i++) {
            int blob = hm_v4l_get_value(cfg, i);
            assert(blob <= 255 && blob >= -255);

            if (cfg->auto_min && blob < cfg->min)
                cfg->min = blob;
            if (cfg->auto_max && blob > cfg->max)
                cfg->max = blob;

            if (cfg->print) {
                if (i % cfg->width == 0)
                    printf("\n");
                printf("%4d ", blob);
            }

            if (cfg->file) {
                if (first_number)
                    first_number = false;
                else
                    fprintf(cfg->file, "\t");
                fprintf(cfg->file, "%d", blob);
            }
        }
    }
    if (cfg->print)
        printf("\n\n");

    if (cfg->file) {
        fprintf(cfg->file, "\nmin\t%d\tmax\t%d\n", cfg->min, cfg->max);
        fclose(cfg->file);
    }
    return 0;
}

int continuousCapture(struct hm_cfg *cfg) {
    size_t len;
    bool first_number = true;

    const struct timespec ts = {
        .tv_sec = 1 / cfg->rate,
        .tv_nsec = ((cfg->rate > 1)?(1000 * 1000 * 1000 / cfg->rate):0)
    };

    int err = 0;
    //for (int cnt = 0; cnt < 50; cnt++) {
    while(1) {
        if (interrupt_flag_set)
            break;

        nanosleep(&ts, NULL); // Rate of data capturing

        len = hm_v4l_get_frame(cfg);

        if (len == 0) {
            if (err > 5)
                return 1;
            err++;
            continue;
        }

        for (size_t i = 0; i < len; i++) {
            int blob = hm_v4l_get_value(cfg, i);
            assert(blob <= 255 && blob >= -255);

            if (cfg->auto_min && blob < cfg->min)
                cfg->min = blob;
            if (cfg->auto_max && blob < cfg->max)
                cfg->max = blob;

            if (cfg->print) {
                if (i % cfg->width == 0)
                    printf("\n");
                printf("%4d ", blob); // Standardize? the data
            }

            if (cfg->file) {
                if (first_number)
                    first_number = false;
                else
                    fprintf(cfg->file, "\t");
                fprintf(cfg->file, "%d", blob);
            }
        }
        if (cfg->print) {
            printf("\n");
        }
    }
    if (cfg->print)
        printf("\n\nlowest: %d, highest %d\n", cfg->min, cfg->max);
    return 0;
}
