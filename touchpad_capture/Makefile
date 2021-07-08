CC=gcc
CFLAGS=-Wall -g
HFILES = src/debug_v4l.h src/main.h
CFILES = src/main.c src/debug_v4l.c
SRC = src/
BUILD = build/

proj: $(CFILES) $(HFILES)
	$(CC) -o $@ $(CFILES) $(CFLAGS)

clean:
	rm -f $(SRC)*.o $(SRC)*~ proj
