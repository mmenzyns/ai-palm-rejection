CC=gcc
CFLAGS=-Wall -g
SRC = src/
BUILD = build/

_HFILES = main.h debug_v4l.h
_CFILES = main.c debug_v4l.c
_OBJ = main.o debug_v4l.o

OBJ = $(patsubst %,$(BUILD)%,$(_OBJ))
CFILES = $(patsubst %,$(SRC)%,$(_CFILES))
HFILES = $(patsubst %,$(SRC)%,$(_HFILES))

$(BUILD)%.o: $(SRC)%.c $(HFILES)
	$(CC) -c -o $@ $< $(CFLAGS)

capture: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f $(BUILD)*.o src/*~ capture
