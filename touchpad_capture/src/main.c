#include "main.h"

void sigintHandler() {
    interrupt_flag_set = 1;
}

int main(int argc, char *argv[]) {
    int devno = 0;
    int input = 0;
    int opt;

    struct hm_cfg cfg = {
        .width = 0,
        .file = NULL,
        .print = false,
        .single_capture = false,
        .rate = 1,
    };

    while ((opt = getopt(argc, argv, "f:psr:h")) != -1) {
        switch (opt) {
            case 'f':
                cfg.file = fopen(optarg, "w");
                if (!cfg.file) {
                    fprintf(stderr, "Erro with selected path %s\n", optarg);
                    return 1;
                }
                break;
            case 'p': cfg.print = true; break; // Print values into stdout
            case 's': cfg.single_capture = true; break; // Save value only when key is pressed
            case 'r': cfg.rate = strtol(optarg, NULL, 10); break;
            case 'h': {
                printf("%s [-f PATH, -s, -r, -p, -h]\n", argv[0]);
                char *str = "\t -p \t\t Print captured values into stdout \n"
                            "\t -s \t\t Use single capture mode instead of a continuous one. A data array will be collected only on a press of an Enter key \n"
                            "\t -f PATH \t Path where to save captured data \n"
                            "\t -r \t\t Rate (captures per second) in which to collect data in continous mode, defaults to 1 \n"
                            "\t -r \t\t Print this hint \n";
                printf("%s", str);
                return 0;
            }
            default:
                fprintf(stderr, "Unrecognized argument -%c\n", opt);
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

    return capture(&cfg);
}

int capture(struct hm_cfg *cfg) {
    size_t len;
    bool first_number = true;

    const struct timespec ts = {
        .tv_sec = 1 / cfg->rate,
        .tv_nsec = ((cfg->rate > 1)?(1000 * 1000 * 1000 / cfg->rate):0)
    };

    int err = 0;
    while (1) {
        if (cfg->single_capture) {
            printf("\n\nPress ENTER key to capture\n");
            getchar();
        }

        if (interrupt_flag_set)
            break;

        if (!cfg->single_capture)
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
        if(cfg->print)
            printf("\n");
    }

    return 0;
}
