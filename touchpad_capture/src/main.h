#include <stdlib.h>

#include <stdint.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
#include <assert.h>
#include <signal.h>
#include <stdbool.h>
#include <string.h>

#include "debug_v4l.h"

#define SIGINT 2

volatile sig_atomic_t interrupt_flag_set = 0;

void sigintHandler();

int continuousCapture(struct hm_cfg *cfg);
int singleCapture(struct hm_cfg *cfg);
