/*

Copyright (c) 2023, Dominic Szablewski - https://phoboslab.org
SPDX-License-Identifier: MIT


Command line tool. It plays QSA files.

Put sokol_audio.h in this directory before you compile. Refer to the README.

To compile:
	gcc qsaplay.c -std=gnu99 -lasound -pthread -O3 -o qsaplay

*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SOKOL_AUDIO_IMPL
#include "sokol_audio.h"

#define QSA_IMPLEMENTATION
#include "qsa.h"


/* -----------------------------------------------------------------------------
	qsaplay */

/* Decodes a QSA file one chunk at a time and keeps the full file in memory.
QSA uses about 2.7 kB for each second, thus one hour is less than 10 MB.

The file holds an LMS state every 16 chunks. Thus a seek resumes from the
nearest stored state. */

typedef struct {
	qsa_desc info;
	const unsigned char *bytes;
	unsigned int size;

	qsa_lms_t lms;
	unsigned int next_chunk;
	unsigned int sample_pos;

	unsigned int sample_data_pos;
	unsigned int sample_data_len;
	short *sample_data;
} qsaplay_desc;

qsaplay_desc *qsaplay_open(const char *path) {
	FILE *file = fopen(path, "rb");
	if (!file) {
		return NULL;
	}

	fseek(file, 0, SEEK_END);
	long size = ftell(file);
	if (size <= 0) {
		fclose(file);
		return NULL;
	}
	fseek(file, 0, SEEK_SET);

	unsigned char *bytes = malloc(size);
	int read = bytes && fread(bytes, size, 1, file);
	fclose(file);
	if (!read) {
		free(bytes);
		return NULL;
	}

	qsa_desc info;
	if (!qsa_decode_header(bytes, size, &info)) {
		free(bytes);
		return NULL;
	}

	qsaplay_desc *qp = malloc(sizeof(qsaplay_desc) + info.chunk_samples * sizeof(short));
	memset(qp, 0, sizeof(qsaplay_desc));

	qp->info = info;
	qp->bytes = bytes;
	qp->size = size;
	qp->sample_data = (short *)(((unsigned char *)qp) + sizeof(qsaplay_desc));
	qsa_lms_init(&qp->lms);
	return qp;
}

void qsaplay_close(qsaplay_desc *qp) {
	free((void *)qp->bytes);
	free(qp);
}

unsigned int qsaplay_decode_chunk(qsaplay_desc *qp) {
	if (qp->next_chunk >= qp->info.chunks) {
		return 0;
	}
	unsigned int start = qsa_read_u32(qp->bytes + QSA_HEADER_SIZE + qp->next_chunk * 4);
	unsigned int end = qsa_read_u32(qp->bytes + QSA_HEADER_SIZE + (qp->next_chunk + 1) * 4);
	if ((start & 3) || start > end || end > qp->size || end - start < 4) {
		return 0;
	}

	unsigned int decoded = qsa_decode_chunk(
		qp->bytes + start, end - start, &qp->info, &qp->lms, qp->sample_data
	);

	/* The pad samples in the last slice are not part of the file. */
	unsigned int remaining = qp->info.samples - qp->next_chunk * qp->info.chunk_samples;
	if (decoded > remaining) {
		decoded = remaining;
	}

	qp->next_chunk++;
	qp->sample_data_pos = 0;
	qp->sample_data_len = decoded;
	return decoded;
}

void qsaplay_rewind(qsaplay_desc *qp) {
	qsa_lms_init(&qp->lms);
	qp->next_chunk = 0;
	qp->sample_pos = 0;
	qp->sample_data_len = 0;
	qp->sample_data_pos = 0;
}

unsigned int qsaplay_decode(qsaplay_desc *qp, float *sample_data, int num_samples) {
	int src_index = qp->sample_data_pos;
	int dst_index = 0;
	for (int i = 0; i < num_samples; i++) {

		/* Decode more samples when the buffer becomes empty. */
		if (qp->sample_data_len - qp->sample_data_pos == 0) {
			if (!qsaplay_decode_chunk(qp)) {
				/* Go back to the start */
				qsaplay_rewind(qp);
				if (!qsaplay_decode_chunk(qp)) {
					sample_data[dst_index++] = 0;
					continue;
				}
			}
			src_index = 0;
		}

		/* Change the sample to a float in the range -1 to 1 */
		sample_data[dst_index++] = qp->sample_data[src_index++] / 32768.0;
		qp->sample_data_pos++;
		qp->sample_pos++;
	}
	return num_samples;
}

double qsaplay_get_duration(qsaplay_desc *qp) {
	return (double)qp->info.samples / (double)qp->info.samplerate;
}

double qsaplay_get_time(qsaplay_desc *qp) {
	return (double)qp->sample_pos / (double)qp->info.samplerate;
}

int qsaplay_get_chunk(qsaplay_desc *qp) {
	return qp->sample_pos / qp->info.chunk_samples;
}

void qsaplay_seek_chunk(qsaplay_desc *qp, int chunk) {
	if (chunk < 0) {
		chunk = 0;
	}
	if ((unsigned int)chunk >= qp->info.chunks) {
		chunk = qp->info.chunks - 1;
	}

	/* Restore the nearest stored LMS state. */
	unsigned int checkpoint = (unsigned int)chunk -
		(unsigned int)chunk % QSA_SEEK_CHUNK_INTERVAL;
	if (!qsa_decode_seek_state(qp->bytes, qp->size, &qp->info,
		checkpoint, &qp->lms)) {
		return;
	}
	qp->next_chunk = checkpoint;
	qp->sample_pos = checkpoint * qp->info.chunk_samples;
	qp->sample_data_len = 0;
	qp->sample_data_pos = 0;
}




/* -----------------------------------------------------------------------------
	The application */

/* getch() for Windows, macOS and Linux */

#if defined(_WIN32)
	#include <conio.h>
#else
	#if defined(__APPLE__)
		#include <unistd.h>
	#endif
	#include <termios.h>
	int getch(void) {
		struct termios oldattr, newattr;
		int ch;
		tcgetattr(STDIN_FILENO, &oldattr);
		newattr = oldattr;
		newattr.c_lflag &= ~(ICANON | ECHO);
		tcsetattr(STDIN_FILENO, TCSANOW, &newattr);
		ch = getchar();
		tcsetattr(STDIN_FILENO, TCSANOW, &oldattr);
		return ch;
	}
#endif


/* The audio callback. The audio backend calls it when it needs more samples.
All decode operations occur here. */

static void sokol_audio_cb(float* sample_data, int num_samples, int num_channels, void *user_data) {
	qsaplay_desc *qsaplay = (qsaplay_desc *)user_data;
	if (num_channels != 1) {
		printf("Audio cb channels %d, but QSA is mono\n", num_channels);
		exit(1);
	}

	qsaplay_decode(qsaplay, sample_data, num_samples);

	printf("\r %6.2f / %.2f sec", qsaplay_get_time(qsaplay), qsaplay_get_duration(qsaplay));
	fflush(stdout);
}


int main(int argc, char **argv) {
	if (argc < 2) {
		printf("Usage: qsaplay <file.qsa>\n");
		exit(1);
	}

	qsaplay_desc *qsaplay = qsaplay_open(argv[1]);

	if (!qsaplay) {
		printf("Failed to load %s\n", argv[1]);
		exit(1);
	}

	printf(
		"%s: mono, samplerate: %d hz, samples: %d, duration: %d sec\n",
		argv[1],
		qsaplay->info.samplerate,
		qsaplay->info.samples,
		qsaplay->info.samples/qsaplay->info.samplerate
	);
	printf("Controls: [x] rewind / [c] skip / [q] quit\n");

	saudio_setup(&(saudio_desc){
		.sample_rate = qsaplay->info.samplerate,
		.num_channels = 1,
		.stream_userdata_cb = sokol_audio_cb,
		.user_data = qsaplay
	});

	int wants_to_quit = 0;
	while (!wants_to_quit) {
		char c = getch();
		switch (c) {
			case 'c': qsaplay_seek_chunk(qsaplay, qsaplay_get_chunk(qsaplay) + 48); break;
			case 'x': qsaplay_seek_chunk(qsaplay, qsaplay_get_chunk(qsaplay) - 48); break;
			case 'q': wants_to_quit = 1; break;
		}
	}

	saudio_shutdown();
	qsaplay_close(qsaplay);

	printf("\n");
	return 0;
}
