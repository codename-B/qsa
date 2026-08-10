CC ?= gcc

# QSACONV

TARGET_CONV ?= qsaconv
CFLAGS_CONV ?= -std=c99 -O3
LFLAGS_CONV ?= -lm

# For mp3 and flac support, put dr_mp3.h and dr_flac.h in this directory,
# then call `make HAS_DRLIBS=true`. Refer to the README.
ifeq ($(HAS_DRLIBS), true)
	CFLAGS_CONV := $(CFLAGS_CONV) -D QSACONV_HAS_DRMP3 -D QSACONV_HAS_DRFLAC
endif

# QSAPLAY
# Put sokol_audio.h in this directory. Refer to the README.
# FIXME: not tested on Windows or macOS
TARGET_PLAY ?= qsaplay
CFLAGS_PLAY ?= -std=gnu99 -O3

# QSATEST

TARGET_TEST ?= qsatest
CFLAGS_TEST ?= -std=c99 -O3
LFLAGS_TEST ?= -lm

ifeq ($(OS),Windows_NT)
	LFLAGS_PLAY ?= # a #pragma in sokol_audio.h sets these
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Darwin)
		LFLAGS_PLAY ?= -pthread -framework AudioToolbox
	else
		LFLAGS_PLAY ?= -pthread -lasound
	endif
endif

all: $(TARGET_PLAY) $(TARGET_CONV)

play: $(TARGET_PLAY)
$(TARGET_PLAY):$(TARGET_PLAY).c qsa.h
	$(CC) $(CFLAGS_PLAY) $(TARGET_PLAY).c -o $(TARGET_PLAY) $(LFLAGS_PLAY)

conv: $(TARGET_CONV)
$(TARGET_CONV):$(TARGET_CONV).c qsa.h
	$(CC) $(CFLAGS_CONV) $(TARGET_CONV).c -o $(TARGET_CONV) $(LFLAGS_CONV)

$(TARGET_TEST):$(TARGET_TEST).c qsa.h
	$(CC) $(CFLAGS_TEST) $(TARGET_TEST).c -o $(TARGET_TEST) $(LFLAGS_TEST)

.PHONY: test
test: $(TARGET_TEST)
	./$(TARGET_TEST)

.PHONY: clean
clean:
	$(RM) $(TARGET_PLAY) $(TARGET_CONV) $(TARGET_TEST)
