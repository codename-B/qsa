/*

Copyright (c) 2026, codename-B
SPDX-License-Identifier: MIT

QSA - "Quantised Sliced Audio"

Based on QOA - the "Quite OK Audio" format by Dominic Szablewski
Copyright (c) 2023, Dominic Szablewski - https://phoboslab.org

-- Data Format

QSA is a mono descendant of QOA for decoders that cannot afford QOA's 3.2
bits per sample. It keeps QOA's sign-sign LMS predictor and QOA's idea of a
4-bit quantiser scale per slice, but fixes the residual at 2 bits and
stretches the slice to 64 samples, landing at roughly 2.1 bits per sample.
The sample rate is carried in the header; the default is 9360 hz, which fits
90 minutes of audio in about 13 MB.

All values are little endian. The file layout is:

struct {
	struct {
		char     magic[4];      // magic bytes "QSA2"
		uint32_t samples;        // total samples in this file
		uint32_t samplerate;     // samples per second
		uint16_t chunk_samples;  // samples per chunk, multiple of 8 and of slice_samples
		uint16_t slice_samples;  // samples per scale, multiple of 4, divides chunk_samples
		uint32_t chunks;         // number of chunks
	} file_header;

	uint32_t index[chunks + 1];  // byte offset of each chunk, plus the file size

	struct {
		uint8_t  bits;           // always 2
		uint8_t  reserved;       // 0
		uint16_t samples;        // samples in this chunk, multiple of slice_samples
		uint8_t  scales[(samples / slice_samples + 1) / 2]; // packed nibbles, low first
		uint8_t  indices[samples / 4];  // one codebook index per 4 samples
		uint8_t  padding[];      // zeroes, to a 4 byte boundary
	} chunks[chunks];
} qsa_file_t;

Every chunk but the last holds exactly chunk_samples samples. The last holds
the remainder rounded up to a whole number of slices, zero padded; those pad
samples are encoded and decoded like any other but fall outside the file's
sample count.

Residuals are vector quantised: each index byte selects a 4-sample vector
from qsa_codebook, a fixed 256-entry table that is part of this format. The
i-th residual of the group is dequantized as

	(qsa_codebook[index][i] * qsa_scale_tab[scale]) >> 8

with an arithmetic shift, then added to the LMS prediction and clamped to 16
bits. The codebook was trained with closed-loop k-means on film audio, seeded
from the product codebook {-4, -1, 1, 4}^4 (in 8.8 fixed point) that an
earlier revision of this format used as independent 2-bit scalar residuals.
Training moved the entries toward correlated shapes, attacks and ringing,
that the scalar grid could not represent; on held-out material this is worth
about 2 dB at the same 2 bits per sample.

Entry 153 is the reserved all-zero vector. A codebook with no zero forces
the encoder to inject energy into digital silence, and the LMS feedback
turns that into a periodic, audible idle tone; with the zero vector, silent
input encodes to zero residuals and decodes to exact digital silence.

The LMS filter is QOA's, with two changes. First, a leak: every fourth sample
each weight loses weight >> 7 before the sign-sign update; without it the low
bit depth lets the weights run away. Second, the prediction dot product is
specified as wrapping 32-bit arithmetic, so a fixed-point ARM decoder and this
reference are bit-identical on any input. The filter state is never stored in
the file, so a stream must be decoded from its first chunk.

The encoder picks each slice's scale by exhaustive search, minimising the
noise-shaped cost e^2 + lambda * (e[n] - e[n-1])^2. Penalising the error slope
tilts the quantisation noise floor down to follow the signal spectrum, where
the ear masks it far better than flat MSE's high-frequency hiss. The shaping
memory is encoder-only state; the bitstream and decoder know nothing of it.

*/


/* -----------------------------------------------------------------------------
	Header - Public functions */

#ifndef QSA_H
#define QSA_H

#ifdef __cplusplus
extern "C" {
#endif

#define QSA_MAGIC 0x32415351 /* 'QSA2' */
#define QSA_HEADER_SIZE 20
#define QSA_MIN_FILESIZE 28
#define QSA_DEFAULT_SAMPLERATE 9360
#define QSA_LMS_LEN 4
#define QSA_LMS_LEAK_SHIFT 7
#define QSA_SCALE_COUNT 16
#define QSA_MIN_CHUNK_SAMPLES 8
#define QSA_MAX_CHUNK_SAMPLES 65528
#define QSA_DEFAULT_CHUNK_SAMPLES 2048
#define QSA_DEFAULT_SLICE_SAMPLES 64
#define QSA_DEFAULT_SHAPE_LAMBDA 0.5

#define QSA_CHUNK_SIZE(samples, slice) \
	((4 + (((samples) / (slice) + 1) >> 1) + ((samples) >> 2) + 3) & ~3)

typedef struct {
	int history[QSA_LMS_LEN];
	int weights[QSA_LMS_LEN];
	unsigned int sample_index;
} qsa_lms_t;

typedef struct {
	unsigned int samplerate;
	unsigned int samples;
	unsigned int chunk_samples;
	unsigned int slice_samples;
	unsigned int chunks;
	double shape_lambda;
	#ifdef QSA_RECORD_TOTAL_ERROR
		double error;
	#endif
} qsa_desc;

/* Fills desc with the defaults qsa_encode() needs beyond samples. */
void qsa_desc_init(qsa_desc *desc);
void qsa_lms_init(qsa_lms_t *lms);

unsigned int qsa_max_encoded_size(const qsa_desc *desc);
void *qsa_encode(const short *sample_data, qsa_desc *desc, unsigned int *out_len);

unsigned int qsa_decode_header(const unsigned char *bytes, unsigned int size, qsa_desc *desc);
/* Decodes one chunk, bytes pointing at its header. Returns the samples written,
   which includes the last chunk's padding. */
unsigned int qsa_decode_chunk(const unsigned char *bytes, unsigned int size, const qsa_desc *desc, qsa_lms_t *lms, short *sample_data);
short *qsa_decode(const unsigned char *bytes, unsigned int size, qsa_desc *desc);

#ifndef QSA_NO_STDIO

int qsa_write(const char *filename, const short *sample_data, qsa_desc *desc);
void *qsa_read(const char *filename, qsa_desc *desc);

#endif /* QSA_NO_STDIO */


#ifdef __cplusplus
}
#endif
#endif /* QSA_H */


/* -----------------------------------------------------------------------------
	Implementation */

#ifdef QSA_IMPLEMENTATION
#include <stdlib.h>

#ifndef QSA_MALLOC
	#define QSA_MALLOC(sz) malloc(sz)
	#define QSA_FREE(p) free(p)
#endif

static const int qsa_scale_tab[QSA_SCALE_COUNT] = {
	16, 24, 32, 48, 64, 96, 128, 192,
	256, 384, 512, 768, 1024, 1536, 2048, 3072
};

/* The residual codebook in 8.8 fixed point: 256 vectors of 4 samples. Part
of the format; see the header comment for how it was trained. */

static const short qsa_codebook[256][4] = {
	{ -1602,  -1770,  -1777,  -1607}, {  -664,  -1518,  -2119,  -1510},
	{   -26,   -907,  -1856,  -1767}, {   190,    -63,  -1634,  -2113},
	{  -905,  -1018,  -1355,  -1601}, {  -363,   -470,  -1155,  -1282},
	{   206,   -318,  -1241,  -1058}, {  1051,    -41,  -1641,  -1580},
	{  -990,   -336,  -1012,  -1640}, {  -109,   -189,   -654,  -1772},
	{   304,    127,   -839,  -1238}, {  1044,    547,  -1209,  -1094},
	{ -1078,    970,   -634,  -1643}, {   -45,    851,  -1085,  -1552},
	{   540,    470,   -602,  -2127}, {  1432,    911,   -817,  -1661},
	{ -1400,  -1283,   -916,  -1187}, {  -670,   -955,   -915,  -1050},
	{   -18,   -948,   -757,  -1145}, {   750,   -739,  -1051,  -1159},
	{  -980,   -538,   -543,   -923}, {  -359,   -414,   -605,   -842},
	{   142,   -265,   -538,   -838}, {   824,    -13,   -608,   -972},
	{  -762,     38,   -403,  -1311}, {  -240,    109,   -384,   -882},
	{   318,    300,   -316,   -838}, {   990,    555,   -411,   -791},
	{ -1841,    237,    -88,  -1503}, {  -111,    518,   -100,  -1437},
	{   558,    850,     25,  -1390}, {  1459,   1178,    -39,  -1154},
	{ -2172,  -1501,   -936,   -865}, {  -701,  -1113,   -157,   -986},
	{   117,  -1162,   -102,   -710}, {  1039,   -662,     52,   -905},
	{ -1780,   -714,   -448,   -965}, {  -432,   -345,    -11,   -758},
	{   203,   -211,     10,   -788}, {   852,     48,     76,   -791},
	{ -1137,    -34,     52,   -851}, {  -290,    249,    148,   -699},
	{   289,    470,    178,   -620}, {   950,    686,    216,   -639},
	{  -974,    781,    298,  -1281}, {  -221,   1202,    193,  -1131},
	{   575,   1502,     59,   -830}, {  1597,   1834,    202,   -618},
	{ -1355,   -815,    494,   -901}, {  -553,   -299,   1361,   -264},
	{  -109,   -871,    857,  -1127}, {   886,  -1110,   1185,   -826},
	{ -2582,   -550,    333,   -714}, {  -408,   -166,    287,  -1432},
	{   218,     23,    629,  -1058}, {   870,     24,    939,   -500},
	{ -1443,     13,    949,  -1044}, {  -517,    358,    883,   -780},
	{   225,    750,    906,   -759}, {   980,   1149,    946,   -836},
	{ -1832,    812,    980,   -400}, {  -670,   1413,   1133,   -663},
	{   220,   1688,    915,   -651}, {  1151,   1903,   1061,   -139},
	{ -1316,  -1546,  -1418,   -685}, {  -604,  -1428,  -1375,   -807},
	{    26,  -1072,  -1568,   -688}, {   533,   -578,  -2175,   -812},
	{ -1105,   -680,  -1415,   -604}, {  -369,   -607,  -1074,   -547},
	{   264,   -466,   -939,   -454}, {   987,   -179,  -1272,   -644},
	{  -974,    -49,   -796,   -530}, {  -220,    -30,   -933,   -456},
	{   367,    124,   -821,   -411}, {  1698,    148,   -760,   -585},
	{  -636,    312,  -1640,   -791}, {  -295,    633,   -817,   -606},
	{   421,    815,   -730,   -712}, {  1653,   1258,   -853,   -449},
	{ -1109,  -1068,   -776,   -557}, {  -566,   -885,   -654,   -467},
	{   -48,   -713,   -515,   -367}, {   717,   -826,   -536,   -326},
	{  -771,   -522,   -408,   -376}, {  -309,   -317,   -389,   -341},
	{   102,   -218,   -396,   -307}, {   635,   -163,   -416,   -306},
	{  -674,     63,   -284,   -403}, {  -108,     50,   -201,   -313},
	{   312,    217,   -209,   -277}, {   770,    313,   -224,   -218},
	{ -1247,    553,   -237,   -327}, {  -197,    646,   -214,   -442},
	{   484,    890,   -108,   -358}, {  1388,    782,   -176,   -322},
	{ -1273,  -1022,   -160,   -257}, {  -494,   -705,    -34,   -279},
	{    66,   -560,     91,   -214}, {   631,   -534,    296,    -34},
	{ -1102,   -417,     89,   -236}, {  -395,   -171,     25,   -181},
	{   126,     -6,     76,   -147}, {   676,     37,    265,   -125},
	{  -818,    121,    290,   -285}, {  -231,    226,    231,   -134},
	{   242,    382,    190,    -83}, {   709,    520,    258,    -80},
	{  -738,    755,    329,   -441}, {   -41,    839,    350,   -291},
	{   578,   1053,    475,   -111}, {  1406,   1218,    495,   -147},
	{ -2331,  -1760,   -358,    134}, {  -814,  -1300,    538,    -53},
	{    75,   -682,    711,   -210}, {   585,   -663,   1182,    137},
	{ -1778,   -377,    529,   -143}, {  -446,   -291,    555,   -313},
	{   156,     -1,    634,   -251}, {  1555,    315,    521,    -31},
	{ -1183,    144,    824,   -106}, {  -342,    424,    767,   -113},
	{   308,    541,    693,    -76}, {   840,    825,   1000,     13},
	{  -815,    709,   1590,   -124}, {   -83,   1082,   1011,     60},
	{   358,   1593,   1409,    223}, {  2019,   1902,   1127,    435},
	{ -1856,  -2355,  -1508,   -492}, {  -496,  -1394,  -1016,   -110},
	{    98,  -1079,  -1025,    -83}, {   754,   -885,  -1495,    -66},
	{  -990,   -815,   -705,     14}, {  -377,   -584,   -722,     41},
	{   260,   -420,   -781,    165}, {  1085,   -263,  -1027,     51},
	{ -1682,   -389,   -541,    -58}, {  -276,     58,   -613,    180},
	{   309,    145,   -539,    230}, {  1160,    328,   -644,     77},
	{ -1032,   1101,   -704,     62}, {  -147,    806,   -693,    247},
	{   600,    866,   -754,     77}, {  2340,   1288,    147,     35},
	{ -1517,  -1396,   -686,    -44}, {  -618,  -1205,   -330,    105},
	{    46,   -861,   -351,    269}, {   717,   -789,   -341,    435},
	{  -753,   -486,   -223,    131}, {  -261,   -379,   -220,     84},
	{   200,   -284,   -212,    165}, {   830,   -167,   -277,    214},
	{  -680,     60,   -135,    242}, {     0,      0,      0,      0},
	{   383,    176,    -29,    190}, {  1138,    332,    -26,    257},
	{  -435,   1533,     78,     -5}, {  -180,    558,    -93,    209},
	{   501,    726,    -35,    288}, {  1131,    871,    206,    210},
	{ -1519,   -935,     29,    405}, {  -450,   -793,    125,    350},
	{   143,   -631,    312,    479}, {  1220,   -463,    303,    521},
	{  -853,   -372,    347,    241}, {  -287,   -193,    175,    246},
	{   123,    -38,    205,    307}, {   645,    -10,    284,    462},
	{  -638,     99,    459,    329}, {   -80,    246,    376,    288},
	{   309,    366,    368,    326}, {   755,    485,    416,    379},
	{  -786,    664,    475,    289}, {     0,    758,    460,    345},
	{   605,    851,    675,    513}, {  1152,   1041,    763,    514},
	{ -2161,  -1306,    518,    311}, {  -533,   -929,    923,    678},
	{   -40,   -161,   1507,    590}, {  1017,  -1326,    449,    262},
	{ -1638,   -410,   1090,    383}, {  -269,   -151,    705,    395},
	{   268,     11,    759,    401}, {   869,     59,   1013,    550},
	{  -941,     94,   1180,    564}, {  -248,    414,    960,    423},
	{   285,    608,    901,    463}, {  1201,    608,   1305,    687},
	{  -811,    914,   1324,    682}, {    -7,    984,   1593,    581},
	{   589,   1136,   1194,    766}, {  1221,   1488,   1307,    694},
	{  -905,  -2311,  -1105,    290}, {   -47,  -1846,  -2048,    -95},
	{   301,  -1488,  -1060,    713}, {  1205,  -1289,  -1265,    698},
	{  -962,  -1232,  -1102,    667}, {  -280,   -714,   -903,    740},
	{   521,   -455,   -785,    921}, {  1765,   -226,   -850,    615},
	{  -871,   -114,  -1107,    292}, {  -228,    -18,   -606,   1125},
	{   622,    264,   -982,    866}, {  2040,    486,   -379,    617},
	{  -444,   1072,  -1155,   1046}, {   264,    103,  -1605,     97},
	{   531,    791,   -177,   1296}, {  1159,   1007,   -448,    684},
	{ -1453,  -1932,   -330,    772}, {  -403,  -1553,   -536,    827},
	{    91,   -930,   -336,   1262}, {  1363,   -835,   -396,   1038},
	{  -997,   -743,   -244,    642}, {  -347,   -440,   -191,    648},
	{   338,   -275,   -129,    734}, {  1035,     19,   -276,    910},
	{ -1509,    -43,    152,    534}, {  -181,     72,     15,    711},
	{   453,    291,    -14,    745}, {  1654,    417,    358,   1024},
	{  -885,    421,   -345,    744}, {  -117,    707,    -52,    926},
	{   486,   1521,    350,    530}, {  1561,   1369,    456,    773},
	{ -1127,  -1472,    117,   1079}, {  -592,  -1009,     -1,   1317},
	{    65,   -606,    286,   1307}, {   851,   -673,    161,   1556},
	{ -1024,   -617,    389,    840}, {  -413,   -347,    312,    775},
	{   204,   -100,    370,    919}, {   796,     43,    366,   1267},
	{  -923,    -61,    465,    965}, {  -136,    228,    552,    790},
	{   353,    374,    563,    829}, {   989,    565,    491,    884},
	{  -696,    547,    594,   1116}, {   -42,    933,    742,   1039},
	{   681,    952,    683,   1206}, {  1723,   1272,   1165,   1365},
	{ -1819,  -1067,    748,   1306}, {  -626,   -674,    655,   1859},
	{   324,   -487,   1056,   1280}, {   158,  -1544,    186,    611},
	{ -1403,   -345,   1241,   1249}, {  -423,   -153,    785,   1193},
	{    47,    174,    830,   1733}, {   917,    325,   1069,   1788},
	{  -836,     24,   1527,   1591}, {  -180,    362,   1133,   1094},
	{   456,    521,   1087,   1044}, {  1001,   1053,   1210,   1358},
	{  -465,    675,   1900,   1306}, {   193,    876,   1582,   1630},
	{   745,   1521,   1866,   1471}, {  1702,   1893,   1826,   1550},
};


static inline int qsa_clamp_s16(int v) {
	if ((unsigned int)(v + 32768) > 65535) {
		if (v < -32768) { return -32768; }
		if (v >  32767) { return  32767; }
	}
	return v;
}

static inline unsigned int qsa_read_u16(const unsigned char *bytes) {
	return bytes[0] | ((unsigned int)bytes[1] << 8);
}

static inline unsigned int qsa_read_u32(const unsigned char *bytes) {
	return bytes[0] | ((unsigned int)bytes[1] << 8) |
		((unsigned int)bytes[2] << 16) | ((unsigned int)bytes[3] << 24);
}

static inline void qsa_write_u16(unsigned int v, unsigned char *bytes) {
	bytes[0] = v & 0xff;
	bytes[1] = (v >> 8) & 0xff;
}

static inline void qsa_write_u32(unsigned int v, unsigned char *bytes) {
	bytes[0] = v & 0xff;
	bytes[1] = (v >> 8) & 0xff;
	bytes[2] = (v >> 16) & 0xff;
	bytes[3] = (v >> 24) & 0xff;
}

/* Wrapping 32-bit dot product, per the spec; matches an ARM mul/mla chain. */
static int qsa_lms_predict(const qsa_lms_t *lms) {
	unsigned int prediction = 0;
	for (int i = 0; i < QSA_LMS_LEN; i++) {
		prediction += (unsigned int)lms->weights[i] * (unsigned int)lms->history[i];
	}
	return (int)prediction >> 13;
}

static void qsa_lms_update(qsa_lms_t *lms, int sample, int residual) {
	int delta = residual >> 4;
	int leak = (lms->sample_index & 3) == 0;

	for (int i = 0; i < QSA_LMS_LEN; i++) {
		if (leak) {
			lms->weights[i] -= lms->weights[i] >> QSA_LMS_LEAK_SHIFT;
		}
		lms->weights[i] += lms->history[i] < 0 ? -delta : delta;
	}

	for (int i = 0; i < QSA_LMS_LEN-1; i++) {
		lms->history[i] = lms->history[i+1];
	}
	lms->history[QSA_LMS_LEN-1] = sample;
	lms->sample_index++;
}

void qsa_lms_init(qsa_lms_t *lms) {
	lms->weights[0] = 0;
	lms->weights[1] = 0;
	lms->weights[2] = -(1<<13);
	lms->weights[3] =  (1<<14);

	for (int i = 0; i < QSA_LMS_LEN; i++) {
		lms->history[i] = 0;
	}
	lms->sample_index = 0;
}

void qsa_desc_init(qsa_desc *desc) {
	desc->samplerate = QSA_DEFAULT_SAMPLERATE;
	desc->samples = 0;
	desc->chunk_samples = QSA_DEFAULT_CHUNK_SAMPLES;
	desc->slice_samples = QSA_DEFAULT_SLICE_SAMPLES;
	desc->chunks = 0;
	desc->shape_lambda = QSA_DEFAULT_SHAPE_LAMBDA;
	#ifdef QSA_RECORD_TOTAL_ERROR
		desc->error = 0;
	#endif
}



/* -----------------------------------------------------------------------------
	Encoder */

/* The LMS state plus the noise shaping memory; err_prev never leaves the
encoder, so a trial slice and its winner carry both together. */

typedef struct {
	qsa_lms_t lms;
	int err_prev;
} qsa_enc_state_t;

/* Quantises one 4-sample group at a fixed scale by exhaustive codebook
search. Returns the noise-shaped cost used for scale selection, stores the
winning index and the winner's plain squared error, and advances the state.
The search costs 256 trial encodes per group; entries are abandoned as soon
as their running cost exceeds the best so far, which cuts most of that. */

static double qsa_encode_group(const short *samples, int scale, double lambda, qsa_enc_state_t *state, unsigned char *index_out, double *plain_energy) {
	int best = 0;
	double best_cost = -1;
	double best_plain = 0;
	qsa_enc_state_t best_state;

	for (int e = 0; e < 256; e++) {
		qsa_enc_state_t trial = *state;
		double cost = 0;
		double plain = 0;

		for (int i = 0; i < 4; i++) {
			int residual = (qsa_codebook[e][i] * qsa_scale_tab[scale]) >> 8;
			int reconstructed = qsa_clamp_s16(qsa_lms_predict(&trial.lms) + residual);
			double error = (double)samples[i] - reconstructed;
			double slope = error - trial.err_prev;
			cost += error * error + lambda * slope * slope;
			if (best_cost >= 0 && cost >= best_cost) { break; }
			plain += error * error;
			trial.err_prev = (int)error;
			qsa_lms_update(&trial.lms, reconstructed, residual);
		}

		if (best_cost < 0 || cost < best_cost) {
			best = e;
			best_cost = cost;
			best_plain = plain;
			best_state = trial;
		}
	}

	*index_out = best;
	*plain_energy = best_plain;
	*state = best_state;
	return best_cost;
}

unsigned int qsa_max_encoded_size(const qsa_desc *desc) {
	unsigned int chunks = (desc->samples + desc->chunk_samples - 1) / desc->chunk_samples;
	return QSA_HEADER_SIZE + (chunks + 1) * 4 +
		chunks * QSA_CHUNK_SIZE(desc->chunk_samples, desc->slice_samples);
}

void *qsa_encode(const short *sample_data, qsa_desc *desc, unsigned int *out_len) {
	if (
		desc->samples == 0 ||
		desc->samplerate == 0 ||
		desc->chunk_samples < QSA_MIN_CHUNK_SAMPLES ||
		desc->chunk_samples > QSA_MAX_CHUNK_SAMPLES ||
		(desc->chunk_samples & 7) ||
		desc->slice_samples < 4 ||
		(desc->slice_samples & 3) ||
		desc->chunk_samples % desc->slice_samples ||
		desc->shape_lambda < 0
	) {
		return NULL;
	}

	desc->chunks = (desc->samples + desc->chunk_samples - 1) / desc->chunk_samples;
	#ifdef QSA_RECORD_TOTAL_ERROR
		desc->error = 0;
	#endif

	unsigned int slice = desc->slice_samples;
	unsigned char *bytes = QSA_MALLOC(qsa_max_encoded_size(desc));
	short *chunk = QSA_MALLOC(desc->chunk_samples * sizeof(short));
	unsigned char *codes = QSA_MALLOC(slice / 4);
	unsigned char *best_codes = QSA_MALLOC(desc->chunk_samples / 4);

	qsa_enc_state_t state;
	qsa_lms_init(&state.lms);
	state.err_prev = 0;

	unsigned int index_end = QSA_HEADER_SIZE + (desc->chunks + 1) * 4;
	unsigned int p = index_end;

	for (unsigned int c = 0; c < desc->chunks; c++) {
		unsigned int start = c * desc->chunk_samples;
		unsigned int valid = desc->samples - start;
		if (valid > desc->chunk_samples) { valid = desc->chunk_samples; }
		unsigned int count = ((valid + slice - 1) / slice) * slice;

		for (unsigned int i = 0; i < valid; i++) {
			chunk[i] = sample_data[start + i];
		}
		for (unsigned int i = valid; i < count; i++) {
			chunk[i] = 0;
		}

		qsa_write_u32(p, bytes + QSA_HEADER_SIZE + c * 4);

		unsigned int slice_count = count / slice;
		unsigned int nibble_bytes = (slice_count + 1) >> 1;
		unsigned int chunk_size = QSA_CHUNK_SIZE(count, slice);
		for (unsigned int i = 0; i < chunk_size; i++) {
			bytes[p + i] = 0;
		}
		bytes[p] = 2;
		qsa_write_u16(count, bytes + p + 2);
		unsigned char *nibbles = bytes + p + 4;
		unsigned char *payload = nibbles + nibble_bytes;

		for (unsigned int s = 0; s < slice_count; s++) {
			double best_cost = -1;
			double best_plain = 0;
			int best_scale = 0;
			qsa_enc_state_t best_state = state;

			unsigned int groups = slice / 4;
			for (int scale = 0; scale < QSA_SCALE_COUNT; scale++) {
				qsa_enc_state_t trial = state;
				double cost = 0;
				double plain = 0;
				for (unsigned int g = 0; g < groups; g++) {
					double group_plain;
					cost += qsa_encode_group(
						chunk + s * slice + g * 4, scale, desc->shape_lambda,
						&trial, codes + g, &group_plain
					);
					if (best_cost >= 0 && cost >= best_cost) { break; }
					plain += group_plain;
				}
				if (best_cost < 0 || cost < best_cost) {
					best_cost = cost;
					best_plain = plain;
					best_scale = scale;
					best_state = trial;
					for (unsigned int g = 0; g < groups; g++) {
						best_codes[(s * slice) / 4 + g] = codes[g];
					}
				}
			}

			state = best_state;
			#ifdef QSA_RECORD_TOTAL_ERROR
				desc->error += best_plain;
			#else
				(void)best_plain;
			#endif

			if (s & 1) { nibbles[s >> 1] |= best_scale << 4; }
			else       { nibbles[s >> 1] |= best_scale; }
		}

		for (unsigned int i = 0; i < count / 4; i++) {
			payload[i] = best_codes[i];
		}
		p += chunk_size;
	}

	qsa_write_u32(QSA_MAGIC, bytes);
	qsa_write_u32(desc->samples, bytes + 4);
	qsa_write_u32(desc->samplerate, bytes + 8);
	qsa_write_u16(desc->chunk_samples, bytes + 12);
	qsa_write_u16(desc->slice_samples, bytes + 14);
	qsa_write_u32(desc->chunks, bytes + 16);
	qsa_write_u32(p, bytes + QSA_HEADER_SIZE + desc->chunks * 4);

	QSA_FREE(best_codes);
	QSA_FREE(codes);
	QSA_FREE(chunk);

	*out_len = p;
	return bytes;
}



/* -----------------------------------------------------------------------------
	Decoder */

unsigned int qsa_decode_header(const unsigned char *bytes, unsigned int size, qsa_desc *desc) {
	if (size < QSA_MIN_FILESIZE || qsa_read_u32(bytes) != QSA_MAGIC) {
		return 0;
	}

	qsa_desc_init(desc);
	desc->samples = qsa_read_u32(bytes + 4);
	desc->samplerate = qsa_read_u32(bytes + 8);
	desc->chunk_samples = qsa_read_u16(bytes + 12);
	desc->slice_samples = qsa_read_u16(bytes + 14);
	desc->chunks = qsa_read_u32(bytes + 16);

	if (
		desc->samples == 0 || desc->chunks == 0 ||
		desc->samplerate == 0 ||
		desc->chunk_samples < QSA_MIN_CHUNK_SAMPLES ||
		(desc->chunk_samples & 7) ||
		desc->slice_samples < 4 ||
		(desc->slice_samples & 3) ||
		desc->chunk_samples % desc->slice_samples ||
		desc->chunks > (size - QSA_HEADER_SIZE) / 4 - 1 ||
		desc->chunks != (desc->samples + desc->chunk_samples - 1) / desc->chunk_samples
	) {
		return 0;
	}

	unsigned int index_end = QSA_HEADER_SIZE + (desc->chunks + 1) * 4;
	if (
		qsa_read_u32(bytes + QSA_HEADER_SIZE) != index_end ||
		qsa_read_u32(bytes + QSA_HEADER_SIZE + desc->chunks * 4) != size
	) {
		return 0;
	}

	return QSA_HEADER_SIZE;
}

unsigned int qsa_decode_chunk(const unsigned char *bytes, unsigned int size, const qsa_desc *desc, qsa_lms_t *lms, short *sample_data) {
	if (size < 4) {
		return 0;
	}

	unsigned int slice = desc->slice_samples;
	unsigned int count = qsa_read_u16(bytes + 2);

	if (
		bytes[0] != 2 || bytes[1] != 0 ||
		count == 0 || slice == 0 || count % slice ||
		count > desc->chunk_samples
	) {
		return 0;
	}

	unsigned int slice_count = count / slice;
	unsigned int nibble_bytes = (slice_count + 1) >> 1;
	if (4 + nibble_bytes + (count >> 2) > size) {
		return 0;
	}

	const unsigned char *nibbles = bytes + 4;
	const unsigned char *residuals = nibbles + nibble_bytes;

	for (unsigned int s = 0; s < slice_count; s++) {
		/* A nibble cannot exceed the table bound, so it needs no check;
		neither can an index byte. */
		int scale = qsa_scale_tab[(s & 1) ? (nibbles[s >> 1] >> 4) : (nibbles[s >> 1] & 15)];

		for (unsigned int g = 0; g < slice / 4; g++) {
			unsigned int n = s * slice + g * 4;
			const short *vector = qsa_codebook[residuals[n >> 2]];
			for (int i = 0; i < 4; i++) {
				int dequantized = (vector[i] * scale) >> 8;
				int reconstructed = qsa_clamp_s16(qsa_lms_predict(lms) + dequantized);
				sample_data[n + i] = reconstructed;
				qsa_lms_update(lms, reconstructed, dequantized);
			}
		}
	}

	return count;
}

short *qsa_decode(const unsigned char *bytes, unsigned int size, qsa_desc *desc) {
	if (!qsa_decode_header(bytes, size, desc)) {
		return NULL;
	}

	unsigned int slice = desc->slice_samples;
	unsigned int padded = ((desc->samples + slice - 1) / slice) * slice;
	short *sample_data = QSA_MALLOC(padded * sizeof(short));
	qsa_lms_t lms;
	qsa_lms_init(&lms);

	unsigned int sample_index = 0;
	for (unsigned int c = 0; c < desc->chunks; c++) {
		unsigned int start = qsa_read_u32(bytes + QSA_HEADER_SIZE + c * 4);
		unsigned int end = qsa_read_u32(bytes + QSA_HEADER_SIZE + (c + 1) * 4);
		unsigned int remaining = desc->samples - sample_index;
		unsigned int expected = c + 1 < desc->chunks
			? desc->chunk_samples
			: ((remaining + slice - 1) / slice) * slice;

		/* The chunk's own sample count has to be checked before decoding it,
		as that is what bounds the write into sample_data. */
		if (
			(start & 3) || start > end || end > size || end - start < 4 ||
			qsa_read_u16(bytes + start + 2) != expected ||
			qsa_decode_chunk(bytes + start, end - start, desc, &lms, sample_data + sample_index) != expected
		) {
			QSA_FREE(sample_data);
			return NULL;
		}

		sample_index += expected < remaining ? expected : remaining;
	}

	return sample_data;
}



/* -----------------------------------------------------------------------------
	File read/write convenience functions */

#ifndef QSA_NO_STDIO
#include <stdio.h>

int qsa_write(const char *filename, const short *sample_data, qsa_desc *desc) {
	FILE *f = fopen(filename, "wb");
	unsigned int size;
	void *encoded;

	if (!f) {
		return 0;
	}

	encoded = qsa_encode(sample_data, desc, &size);
	if (!encoded) {
		fclose(f);
		return 0;
	}

	fwrite(encoded, 1, size, f);
	fclose(f);

	QSA_FREE(encoded);
	return size;
}

void *qsa_read(const char *filename, qsa_desc *desc) {
	FILE *f = fopen(filename, "rb");
	int size, bytes_read;
	void *data;
	short *sample_data;

	if (!f) {
		return NULL;
	}

	fseek(f, 0, SEEK_END);
	size = ftell(f);
	if (size <= 0) {
		fclose(f);
		return NULL;
	}
	fseek(f, 0, SEEK_SET);

	data = QSA_MALLOC(size);
	if (!data) {
		fclose(f);
		return NULL;
	}

	bytes_read = fread(data, 1, size, f);
	fclose(f);

	sample_data = qsa_decode(data, bytes_read, desc);
	QSA_FREE(data);
	return sample_data;
}

#endif /* QSA_NO_STDIO */
#endif /* QSA_IMPLEMENTATION */
