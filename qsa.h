/*

Copyright (c) 2026, codename-B
SPDX-License-Identifier: MIT

QSA - "Quantised Sliced Audio"

Based on QOA - the "Quite OK Audio" format by Dominic Szablewski
Copyright (c) 2023, Dominic Szablewski - https://phoboslab.org

-- Data Format

QSA is a mono descendant of QOA. Use QSA when a decoder cannot use QOA's 3.2
bits per sample. QSA keeps QOA's sign-sign LMS predictor and QOA's 4-bit
quantiser scale for each slice. But QSA holds the residual at 2 bits and
makes the slice 64 samples long. The result is about 2.1 bits per sample. The
header gives the sample rate. The default rate is 9360 hz, which holds 90
minutes of audio in about 13 MB.

All values are little endian. This is the file layout:

struct {
	struct {
		char     magic[4];      // magic bytes "QSA4" or "QSA5"
		uint32_t samples;        // total samples in this file
		uint32_t samplerate;     // QSA4: samples per second
		                         // QSA5: low 24 bits are samples per second;
		                         // high byte is the de-emphasis coefficient
		uint16_t chunk_samples;  // samples per chunk, multiple of 8 and of slice_samples
		uint16_t slice_samples;  // samples per scale, multiple of 4, divides chunk_samples
		uint32_t chunks;         // number of chunks
	} file_header;

	int16_t codebook[256][4];    // the residual codebook, 8.8 fixed point;
	                             // entry 0 must be the all-zero vector

	uint32_t index[chunks + 1];  // byte offset of each chunk, plus the file size

	struct {
		int32_t history[4];      // LMS state entering chunks 16, 32, ...
		int32_t weights[4];
	} seek_states[(chunks - 1) / 16];

	struct {
		uint8_t  bits;           // always 2
		uint8_t  reserved;       // 0
		uint16_t samples;        // samples in this chunk, multiple of slice_samples
		uint8_t  scales[(samples / slice_samples + 1) / 2]; // packed nibbles, low first
		uint8_t  indices[samples / 4];  // one codebook index per 4 samples
		uint8_t  padding[];      // zeroes, to a 4 byte boundary
	} chunks[chunks];
} qsa_file_t;

All chunks but the last one hold chunk_samples samples. The last chunk
increases the remainder to a full slice and adds zeros. The decoder decodes
these pad samples, but they are not part of the file's sample count.

QSA quantises the residuals as vectors. Each index byte selects one 4-sample
vector from the file's codebook. That table has 256 entries and comes
immediately after the fixed header. To dequantise residual i of a group,
calculate

	(codebook[index][i] * qsa_scale_tab[scale]) >> 8

with an arithmetic shift. Then add the LMS prediction and clamp the result to
16 bits.

From QSA3, the codebook travels with the file. Thus a file can carry a table
trained on its own material. qsa_encode() writes the table that
qsa_desc.codebook points at, and qsa_desc_init() points it at the built-in
table.

Closed-loop k-means made the built-in table. The first entries were the
product codebook {-4, -1, 1, 4}^4 in 8.8 fixed point. The training moved the
entries to correlated shapes that a scalar grid cannot make. This gives about
2 dB at the same bit rate.

Entry 0 is the reserved vector of zeros, and qsa_decode_header() refuses a
file that disobeys this rule. Without a zero vector, the encoder must add
energy to digital silence. The LMS feedback then makes an audible idle tone.
With the zero vector, silent input encodes to zero residuals and decodes to
exact digital silence.

The LMS filter is QOA's filter with two changes. First, the weights leak. At
every fourth sample, each weight loses weight >> 7 before the sign-sign
update. Without the leak, the coarse residuals make the weights too large.
Second, the prediction dot product uses 32-bit arithmetic that wraps. QSA4
stores the LMS state entering every 16th chunk, so seeking can resume from the
nearest stored state without decoding from the start.

QSA5 adds pre-emphasis during encoding and de-emphasis during decoding. Its
coefficient is stored in the high byte of the samplerate field. A coefficient
of zero writes a QSA4 file and leaves samples unchanged.

For each slice, the encoder tests all 16 scales. It keeps the scale with the
minimum shaped cost e^2 + lambda * (e[n] - e[n-1])^2. The penalty on the
error slope moves the noise below the signal, where the ear hears it less
than the flat hiss of plain MSE. The shaping memory stays in the encoder. The
bitstream and the decoder do not use it.

*/


/* -----------------------------------------------------------------------------
	Header - Public functions */

#ifndef QSA_H
#define QSA_H

#ifdef __cplusplus
extern "C" {
#endif

#define QSA_MAGIC 0x34415351 /* 'QSA4' */
#define QSA_MAGIC5 0x35415351 /* 'QSA5': de-emphasis byte in the header */
#define QSA_MAX_QSA5_SAMPLERATE 0xffffff
#define QSA_CODEBOOK_SIZE 2048          /* 256 entries x 4 x int16 */
#define QSA_HEADER_SIZE (20 + QSA_CODEBOOK_SIZE)
#define QSA_MIN_FILESIZE (QSA_HEADER_SIZE + 8)
#define QSA_DEFAULT_SAMPLERATE 9360
#define QSA_LMS_LEN 4
#define QSA_LMS_LEAK_SHIFT 7
#define QSA_SCALE_COUNT 16
#define QSA_MIN_CHUNK_SAMPLES 8
#define QSA_MAX_CHUNK_SAMPLES 65528
#define QSA_DEFAULT_CHUNK_SAMPLES 2048
#define QSA_DEFAULT_SLICE_SAMPLES 64
#define QSA_DEFAULT_SHAPE_LAMBDA 0.5
#define QSA_SEEK_CHUNK_INTERVAL 16
#define QSA_SEEK_STATE_SIZE (QSA_LMS_LEN * 2 * 4)
#define QSA_SEEK_STATE_COUNT(chunks) \
	((chunks) > 1 ? ((chunks) - 1) / QSA_SEEK_CHUNK_INTERVAL : 0)

#define QSA_CHUNK_SIZE(samples, slice) \
	((4 + (((samples) / (slice) + 1) >> 1) + ((samples) >> 2) + 3) & ~3)

typedef struct {
	int history[QSA_LMS_LEN];
	int weights[QSA_LMS_LEN];
	unsigned int sample_index;
	int deemph_prev;
} qsa_lms_t;

typedef struct {
	unsigned int samplerate;
	unsigned int samples;
	unsigned int chunk_samples;
	unsigned int slice_samples;
	unsigned int chunks;
	double shape_lambda;
	double energy_mu;
	double pns_threshold;
	unsigned int deemph;
	/* The residual codebook for this stream. qsa_desc_init() points it at
	the built-in table. qsa_decode_header() points it at the table in the
	file, which it reads in place because the format is little endian.
	Entry 0 must be the vector of zeros. */
	const short (*codebook)[4];
	unsigned int search_beam;
	unsigned int threads;
	#ifdef QSA_RECORD_TOTAL_ERROR
		double error;
	#endif
} qsa_desc;

/* Sets desc to the defaults. Set desc->samples before you encode. */
void qsa_desc_init(qsa_desc *desc);
void qsa_lms_init(qsa_lms_t *lms);

unsigned int qsa_max_encoded_size(const qsa_desc *desc);
void *qsa_encode(const short *sample_data, qsa_desc *desc, unsigned int *out_len);

unsigned int qsa_decode_header(const unsigned char *bytes, unsigned int size, qsa_desc *desc);
int qsa_decode_seek_state(const unsigned char *bytes, unsigned int size, const qsa_desc *desc, unsigned int chunk, qsa_lms_t *lms);
/* Decodes one chunk. bytes must point at the chunk header. Returns the number
   of samples written, which includes the last chunk's pad samples. */
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
#include <math.h>
#define qsa_sqrt sqrt

#ifndef QSA_MALLOC
	#define QSA_MALLOC(sz) malloc(sz)
	#define QSA_FREE(p) free(p)
#endif

static const int qsa_scale_tab[QSA_SCALE_COUNT] = {
	16, 24, 32, 48, 64, 96, 128, 192,
	256, 384, 512, 768, 1024, 1536, 2048, 3072
};

/* The built-in residual codebook in 8.8 fixed point: 256 vectors of 4
samples. The encoder uses it by default, and each file holds its own copy.
Entry 0 is the reserved vector of zeros. */

static const short qsa_codebook[256][4] = {
	{     0,      0,      0,      0}, {  -664,  -1518,  -2119,  -1510},
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
	{  -680,     60,   -135,    242}, { -1602,  -1770,  -1777,  -1607},
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

/* The dot product uses 32-bit arithmetic that wraps, as the format shows. */
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
	lms->deemph_prev = 0;
}

void qsa_desc_init(qsa_desc *desc) {
	desc->samplerate = QSA_DEFAULT_SAMPLERATE;
	desc->samples = 0;
	desc->chunk_samples = QSA_DEFAULT_CHUNK_SAMPLES;
	desc->slice_samples = QSA_DEFAULT_SLICE_SAMPLES;
	desc->chunks = 0;
	desc->shape_lambda = QSA_DEFAULT_SHAPE_LAMBDA;
	desc->energy_mu = 0;
	desc->pns_threshold = 0;
	desc->deemph = 0;
	desc->codebook = qsa_codebook;
	desc->search_beam = 1;
	desc->threads = 1;
	#ifdef QSA_RECORD_TOTAL_ERROR
		desc->error = 0;
	#endif
}



/* -----------------------------------------------------------------------------
	Encoder */

/* The LMS state and the noise shaping memory. Only the encoder uses err_prev. */

typedef struct {
	qsa_lms_t lms;
	int err_prev;
} qsa_enc_state_t;

#ifdef QSA_AVX2

#include <immintrin.h>
#include <cpuid.h>

__attribute__((target("avx2")))
static void qsa_avx2_costs256(const short *s4, const int *dq, double lambda,
	double mu, double wf,
	const qsa_enc_state_t *state, double init_cost, double *cost_out)
{
	const __m256i clamp_hi = _mm256_set1_epi32(32767);
	const __m256i clamp_lo = _mm256_set1_epi32(-32768);
	const __m256i zero = _mm256_setzero_si256();
	const __m256d lam = _mm256_set1_pd(lambda);
	const __m256d wfv = _mm256_set1_pd(wf);
	const __m256d muv = _mm256_set1_pd(mu);
	int leak[4];
	for (int i = 0; i < 4; i++) {
		leak[i] = ((state->lms.sample_index + i) & 3) == 0;
	}
	double es = 0;
	for (int i = 0; i < 4; i++) {
		es += (double)s4[i] * s4[i];
	}
	const __m256d ses = _mm256_set1_pd(qsa_sqrt(es));

	for (int blk = 0; blk < 256; blk += 8) {
		__m256i h0 = _mm256_set1_epi32(state->lms.history[0]);
		__m256i h1 = _mm256_set1_epi32(state->lms.history[1]);
		__m256i h2 = _mm256_set1_epi32(state->lms.history[2]);
		__m256i h3 = _mm256_set1_epi32(state->lms.history[3]);
		__m256i w0 = _mm256_set1_epi32(state->lms.weights[0]);
		__m256i w1 = _mm256_set1_epi32(state->lms.weights[1]);
		__m256i w2 = _mm256_set1_epi32(state->lms.weights[2]);
		__m256i w3 = _mm256_set1_epi32(state->lms.weights[3]);
		__m256i ep = _mm256_set1_epi32(state->err_prev);
		__m256d clo = _mm256_set1_pd(init_cost);
		__m256d chi = clo;
		__m256d erlo = _mm256_setzero_pd();
		__m256d erhi = erlo;

		for (int i = 0; i < 4; i++) {
			__m256i acc = _mm256_add_epi32(
				_mm256_add_epi32(_mm256_mullo_epi32(h0, w0),
				                 _mm256_mullo_epi32(h1, w1)),
				_mm256_add_epi32(_mm256_mullo_epi32(h2, w2),
				                 _mm256_mullo_epi32(h3, w3)));
			__m256i pred = _mm256_srai_epi32(acc, 13);
			__m256i r = _mm256_loadu_si256(
				(const __m256i *)(const void *)(dq + i * 256 + blk));
			__m256i recon = _mm256_min_epi32(_mm256_max_epi32(
				_mm256_add_epi32(pred, r), clamp_lo), clamp_hi);
			__m256i err = _mm256_sub_epi32(_mm256_set1_epi32(s4[i]), recon);
			__m256i slope = _mm256_sub_epi32(err, ep);
			ep = err;

			__m256d elo = _mm256_cvtepi32_pd(_mm256_castsi256_si128(err));
			__m256d ehi = _mm256_cvtepi32_pd(_mm256_extracti128_si256(err, 1));
			__m256d slo = _mm256_cvtepi32_pd(_mm256_castsi256_si128(slope));
			__m256d shi = _mm256_cvtepi32_pd(_mm256_extracti128_si256(slope, 1));
			__m256d e2lo = _mm256_mul_pd(elo, elo);
			__m256d e2hi = _mm256_mul_pd(ehi, ehi);
			clo = _mm256_add_pd(clo, _mm256_mul_pd(wfv, _mm256_add_pd(e2lo,
				_mm256_mul_pd(_mm256_mul_pd(lam, slo), slo))));
			chi = _mm256_add_pd(chi, _mm256_mul_pd(wfv, _mm256_add_pd(e2hi,
				_mm256_mul_pd(_mm256_mul_pd(lam, shi), shi))));
			if (mu > 0) {
				__m256d rdlo = _mm256_cvtepi32_pd(_mm256_castsi256_si128(recon));
				__m256d rdhi = _mm256_cvtepi32_pd(_mm256_extracti128_si256(recon, 1));
				erlo = _mm256_add_pd(erlo, _mm256_mul_pd(rdlo, rdlo));
				erhi = _mm256_add_pd(erhi, _mm256_mul_pd(rdhi, rdhi));
			}

			__m256i delta = _mm256_srai_epi32(r, 4);
			if (leak[i]) {
				w0 = _mm256_sub_epi32(w0, _mm256_srai_epi32(w0, QSA_LMS_LEAK_SHIFT));
				w1 = _mm256_sub_epi32(w1, _mm256_srai_epi32(w1, QSA_LMS_LEAK_SHIFT));
				w2 = _mm256_sub_epi32(w2, _mm256_srai_epi32(w2, QSA_LMS_LEAK_SHIFT));
				w3 = _mm256_sub_epi32(w3, _mm256_srai_epi32(w3, QSA_LMS_LEAK_SHIFT));
			}
			__m256i m;
			m = _mm256_cmpgt_epi32(zero, h0);
			w0 = _mm256_add_epi32(w0, _mm256_sub_epi32(_mm256_xor_si256(delta, m), m));
			m = _mm256_cmpgt_epi32(zero, h1);
			w1 = _mm256_add_epi32(w1, _mm256_sub_epi32(_mm256_xor_si256(delta, m), m));
			m = _mm256_cmpgt_epi32(zero, h2);
			w2 = _mm256_add_epi32(w2, _mm256_sub_epi32(_mm256_xor_si256(delta, m), m));
			m = _mm256_cmpgt_epi32(zero, h3);
			w3 = _mm256_add_epi32(w3, _mm256_sub_epi32(_mm256_xor_si256(delta, m), m));
			h0 = h1; h1 = h2; h2 = h3; h3 = recon;
		}

		if (mu > 0) {
			__m256d de = _mm256_sub_pd(ses, _mm256_sqrt_pd(erlo));
			clo = _mm256_add_pd(clo,
				_mm256_mul_pd(_mm256_mul_pd(muv, de), de));
			de = _mm256_sub_pd(ses, _mm256_sqrt_pd(erhi));
			chi = _mm256_add_pd(chi,
				_mm256_mul_pd(_mm256_mul_pd(muv, de), de));
		}

		_mm256_storeu_pd(cost_out + blk, clo);
		_mm256_storeu_pd(cost_out + blk + 4, chi);
	}
}

static double qsa_sim_group(const short *s4, const int *dq, int e,
	qsa_enc_state_t *trial)
{
	double plain = 0;
	for (int i = 0; i < 4; i++) {
		int residual = dq[i * 256 + e];
		int reconstructed = qsa_clamp_s16(qsa_lms_predict(&trial->lms) + residual);
		double error = (double)s4[i] - reconstructed;
		plain += error * error;
		trial->err_prev = (int)error;
		qsa_lms_update(&trial->lms, reconstructed, residual);
	}
	return plain;
}

static int qsa_avx2_ready(void) {
	unsigned int eax, ebx, ecx, edx;
	if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx) ||
		(ecx & bit_AVX) == 0 || (ecx & bit_OSXSAVE) == 0) {
		return 0;
	}
	unsigned int xcr0_lo, xcr0_hi;
	__asm__ volatile ("xgetbv" : "=a" (xcr0_lo), "=d" (xcr0_hi) : "c" (0));
	(void)xcr0_hi;
	if ((xcr0_lo & 6) != 6) { return 0; }
	if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) { return 0; }
	return (ebx & bit_AVX2) != 0;
}

#endif /* QSA_AVX2 */

/* Quantises one group of 4 samples at a given scale, and tests all codebook
entries. Returns the shaped cost, writes the best index and its squared
error, and moves the state forward. Stops a trial when its cost becomes more
than the best cost. A group of zeros takes entry 0 with no search, thus
digital silence encodes very quickly. */

static double qsa_encode_group(const short *samples, const short (*codebook)[4], int scale, double lambda, double mu, double wf, qsa_enc_state_t *state, unsigned char *index_out, double *plain_energy, const int *dq) {
	int best = 0;
	double best_cost = -1;
	double best_plain = 0;
	qsa_enc_state_t best_state;

	int entries = 256;
	if (samples[0] == 0 && samples[1] == 0 && samples[2] == 0 && samples[3] == 0) {
		entries = 1;
	}

	#ifdef QSA_AVX2
	if (dq && entries == 256) {
		double ecost[256];
		qsa_avx2_costs256(samples, dq, lambda, mu, wf, state, 0, ecost);
		for (int e = 1; e < 256; e++) {
			if (ecost[e] < ecost[best]) { best = e; }
		}
		double plain = qsa_sim_group(samples, dq, best, state);
		*index_out = best;
		*plain_energy = plain;
		return ecost[best];
	}
	#else
		(void)dq;
	#endif

	for (int e = 0; e < entries; e++) {
		qsa_enc_state_t trial = *state;
		double cost = 0;
		double plain = 0;
		double es = 0, er = 0;
		int i;

		for (i = 0; i < 4; i++) {
			int residual = (codebook[e][i] * qsa_scale_tab[scale]) >> 8;
			int reconstructed = qsa_clamp_s16(qsa_lms_predict(&trial.lms) + residual);
			double error = (double)samples[i] - reconstructed;
			double slope = error - trial.err_prev;
			cost += wf * (error * error + lambda * slope * slope);
			if (best_cost >= 0 && cost >= best_cost) { break; }
			plain += error * error;
			es += (double)samples[i] * samples[i];
			er += (double)reconstructed * reconstructed;
			trial.err_prev = (int)error;
			qsa_lms_update(&trial.lms, reconstructed, residual);
		}
		if (i == 4 && mu > 0) {
			double de = qsa_sqrt(es) - qsa_sqrt(er);
			cost += mu * de * de;
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

#define QSA_BEAM_MAX 16
#define QSA_BEAM_MAX_GROUPS 64

typedef struct {
	qsa_enc_state_t state;
	double cost;
	double plain;
} qsa_beam_t;

#ifdef QSA_AVX2
typedef struct {
	double cost;
	unsigned char parent;
	unsigned char entry;
} qsa_pick_t;
#endif

static double qsa_encode_slice(const short *samples, unsigned int groups,
	const short (*codebook)[4], int scale, double lambda, double mu, double wf,
	unsigned int beam,
	qsa_enc_state_t *state, unsigned char *codes_out, double *plain_out,
	const int *dq, double abandon)
{
	if (beam < 2 || groups > QSA_BEAM_MAX_GROUPS) {
		double cost = 0;
		double plain = 0;
		for (unsigned int g = 0; g < groups; g++) {
			double group_plain;
			cost += qsa_encode_group(samples + g * 4, codebook, scale,
				lambda, mu, wf, state, codes_out + g, &group_plain, dq);
			if (abandon >= 0 && cost > abandon) { return -1; }
			plain += group_plain;
		}
		*plain_out = plain;
		return cost;
	}
	if (beam > QSA_BEAM_MAX) { beam = QSA_BEAM_MAX; }

	qsa_beam_t gen[2][QSA_BEAM_MAX];
	qsa_beam_t *cur = gen[0], *nxt = gen[1];
	unsigned char back_parent[QSA_BEAM_MAX_GROUPS][QSA_BEAM_MAX];
	unsigned char back_code[QSA_BEAM_MAX_GROUPS][QSA_BEAM_MAX];
	unsigned int n_cur = 1;
	cur[0].state = *state;
	cur[0].cost = 0;
	cur[0].plain = 0;

	for (unsigned int g = 0; g < groups; g++) {
		const short *s4 = samples + g * 4;
		int entries = 256;
		if (s4[0] == 0 && s4[1] == 0 && s4[2] == 0 && s4[3] == 0) {
			entries = 1;
		}
		unsigned int n_nxt = 0;
		unsigned char next_parent[QSA_BEAM_MAX];
		unsigned char next_code[QSA_BEAM_MAX];

		#ifdef QSA_AVX2
		if (dq && entries == 256) {
			qsa_pick_t picks[QSA_BEAM_MAX];
			unsigned int n_picks = 0;
			for (unsigned int b = 0; b < n_cur; b++) {
				double ecost[256];
				qsa_avx2_costs256(s4, dq, lambda, mu, wf, &cur[b].state,
					cur[b].cost, ecost);
				for (int e = 0; e < 256; e++) {
					double cost = ecost[e];
					if (n_picks == beam && cost >= picks[beam - 1].cost) { continue; }
					if (abandon >= 0 && cost > abandon) { continue; }
					unsigned int pos = n_picks;
					while (pos > 0 && cost < picks[pos - 1].cost) { pos--; }
					if (pos == beam) { continue; }
					unsigned int last = (n_picks < beam) ? n_picks : beam - 1;
					for (unsigned int m = last; m > pos; m--) {
						picks[m] = picks[m - 1];
					}
					picks[pos].cost = cost;
					picks[pos].parent = (unsigned char)b;
					picks[pos].entry = (unsigned char)e;
					if (n_picks < beam) { n_picks++; }
				}
			}
			if (n_picks == 0) { return -1; }
			for (unsigned int k = 0; k < n_picks; k++) {
				unsigned int b = picks[k].parent;
				unsigned int e = picks[k].entry;
				qsa_enc_state_t trial = cur[b].state;
				double group_plain = qsa_sim_group(s4, dq, e, &trial);
				nxt[k].state = trial;
				nxt[k].cost = picks[k].cost;
				nxt[k].plain = cur[b].plain + group_plain;
				next_parent[k] = (unsigned char)b;
				next_code[k] = (unsigned char)e;
			}
			n_nxt = n_picks;
			for (unsigned int k = 0; k < n_nxt; k++) {
				back_parent[g][k] = next_parent[k];
				back_code[g][k] = next_code[k];
			}
			qsa_beam_t *swap = cur; cur = nxt; nxt = swap;
			n_cur = n_nxt;
			continue;
		}
		#endif

		for (unsigned int b = 0; b < n_cur; b++) {
			for (int e = 0; e < entries; e++) {
				qsa_enc_state_t trial = cur[b].state;
				double cost = cur[b].cost;
				double plain = cur[b].plain;
				double worst = (n_nxt == beam) ? nxt[beam - 1].cost : -1;

				int alive = 1;
				double es = 0, er = 0;
				for (int i = 0; i < 4; i++) {
					int residual = (codebook[e][i] * qsa_scale_tab[scale]) >> 8;
					int reconstructed = qsa_clamp_s16(qsa_lms_predict(&trial.lms) + residual);
					double error = (double)s4[i] - reconstructed;
					double slope = error - trial.err_prev;
					cost += wf * (error * error + lambda * slope * slope);
					if ((worst >= 0 && cost >= worst) ||
						(abandon >= 0 && cost > abandon)) { alive = 0; break; }
					plain += error * error;
					es += (double)s4[i] * s4[i];
					er += (double)reconstructed * reconstructed;
					trial.err_prev = (int)error;
					qsa_lms_update(&trial.lms, reconstructed, residual);
				}
				if (!alive) { continue; }
				if (mu > 0) {
					double de = qsa_sqrt(es) - qsa_sqrt(er);
					cost += mu * de * de;
					if ((worst >= 0 && cost >= worst) ||
						(abandon >= 0 && cost > abandon)) { continue; }
				}

				unsigned int pos = n_nxt;
				while (pos > 0 && cost < nxt[pos - 1].cost) { pos--; }
				if (pos == beam) { continue; }
				unsigned int last = (n_nxt < beam) ? n_nxt : beam - 1;
				for (unsigned int m = last; m > pos; m--) {
					nxt[m] = nxt[m - 1];
					next_parent[m] = next_parent[m - 1];
					next_code[m] = next_code[m - 1];
				}
				nxt[pos].state = trial;
				nxt[pos].cost = cost;
				nxt[pos].plain = plain;
				next_parent[pos] = (unsigned char)b;
				next_code[pos] = (unsigned char)e;
				if (n_nxt < beam) { n_nxt++; }
			}
		}

		if (n_nxt == 0) { return -1; }
		for (unsigned int k = 0; k < n_nxt; k++) {
			back_parent[g][k] = next_parent[k];
			back_code[g][k] = next_code[k];
		}
		qsa_beam_t *swap = cur; cur = nxt; nxt = swap;
		n_cur = n_nxt;
	}

	*state = cur[0].state;
	*plain_out = cur[0].plain;
	unsigned int winner = 0;
	for (unsigned int m = groups; m-- > 0;) {
		codes_out[m] = back_code[m][winner];
		winner = back_parent[m][winner];
	}
	return cur[0].cost;
}

static void qsa_make_scale_order(unsigned int hint,
	unsigned char order[QSA_SCALE_COUNT])
{
	if (hint >= QSA_SCALE_COUNT) { hint = 0; }
	unsigned int n = 0;
	order[n++] = (unsigned char)hint;
	for (unsigned int d = 1; n < QSA_SCALE_COUNT; d++) {
		if (d <= hint) { order[n++] = (unsigned char)(hint - d); }
		if (hint + d < QSA_SCALE_COUNT) {
			order[n++] = (unsigned char)(hint + d);
		}
	}
}

#ifdef QSA_THREADS

#include <pthread.h>

typedef union { double d; unsigned long long u; } qsa_bound_bits_t;

typedef struct {
	double cost;
	double plain;
	qsa_enc_state_t state;
	unsigned char codes[QSA_BEAM_MAX_GROUPS];
} qsa_scale_trial_t;

typedef struct {
	pthread_mutex_t mu;
	pthread_cond_t cv_start;
	pthread_cond_t cv_done;
	pthread_t th[QSA_SCALE_COUNT];
	unsigned int nworkers;
	int generation;
	int running;
	int shutdown;
	const short *samples;
	unsigned int groups;
	const short (*codebook)[4];
	double lambda;
	double emu;
	double wfw;
	unsigned int beam;
	const qsa_enc_state_t *in_state;
	const int *dq;
	unsigned char scale_order[QSA_SCALE_COUNT];
	int next_scale;
	unsigned long long bound_bits;
	qsa_scale_trial_t trial[QSA_SCALE_COUNT];
} qsa_pool_t;

static void qsa_pool_work(qsa_pool_t *p) {
	for (;;) {
		int job = __atomic_fetch_add(&p->next_scale, 1, __ATOMIC_RELAXED);
		if (job >= QSA_SCALE_COUNT) { return; }
		int scale = p->scale_order[job];
		qsa_scale_trial_t *t = &p->trial[scale];
		qsa_bound_bits_t abandon;
		abandon.u = __atomic_load_n(&p->bound_bits, __ATOMIC_RELAXED);
		t->state = *p->in_state;
		t->plain = 0;
		t->cost = qsa_encode_slice(p->samples, p->groups, p->codebook,
			scale, p->lambda, p->emu, p->wfw, p->beam, &t->state, t->codes,
			&t->plain,
			p->dq ? p->dq + scale * 1024 : (const int *)0, abandon.d);
		if (t->cost < 0) { continue; }
		for (;;) {
			qsa_bound_bits_t cur, next;
			cur.u = __atomic_load_n(&p->bound_bits, __ATOMIC_RELAXED);
			if (cur.d >= 0 && cur.d <= t->cost) { break; }
			next.d = t->cost;
			if (__atomic_compare_exchange_n(&p->bound_bits, &cur.u, next.u,
					0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) { break; }
		}
	}
}

static void *qsa_pool_thread(void *arg) {
	qsa_pool_t *p = arg;
	int gen = 0;
	pthread_mutex_lock(&p->mu);
	for (;;) {
		while (p->generation == gen && !p->shutdown) {
			pthread_cond_wait(&p->cv_start, &p->mu);
		}
		if (p->shutdown) { break; }
		gen = p->generation;
		pthread_mutex_unlock(&p->mu);
		qsa_pool_work(p);
		pthread_mutex_lock(&p->mu);
		if (--p->running == 0) { pthread_cond_signal(&p->cv_done); }
	}
	pthread_mutex_unlock(&p->mu);
	return NULL;
}

static qsa_pool_t *qsa_pool_create(unsigned int threads) {
	if (threads > QSA_SCALE_COUNT) { threads = QSA_SCALE_COUNT; }
	if (threads < 2) { return NULL; }
	qsa_pool_t *p = QSA_MALLOC(sizeof(qsa_pool_t));
	if (!p) { return NULL; }
	p->nworkers = threads - 1;
	p->generation = 0;
	p->running = 0;
	p->shutdown = 0;
	pthread_mutex_init(&p->mu, NULL);
	pthread_cond_init(&p->cv_start, NULL);
	pthread_cond_init(&p->cv_done, NULL);
	for (unsigned int i = 0; i < p->nworkers; i++) {
		if (pthread_create(&p->th[i], NULL, qsa_pool_thread, p) != 0) {
			p->nworkers = i;
			break;
		}
	}
	return p;
}

static void qsa_pool_destroy(qsa_pool_t *p) {
	if (!p) { return; }
	pthread_mutex_lock(&p->mu);
	p->shutdown = 1;
	pthread_cond_broadcast(&p->cv_start);
	pthread_mutex_unlock(&p->mu);
	for (unsigned int i = 0; i < p->nworkers; i++) {
		pthread_join(p->th[i], NULL);
	}
	pthread_cond_destroy(&p->cv_done);
	pthread_cond_destroy(&p->cv_start);
	pthread_mutex_destroy(&p->mu);
	QSA_FREE(p);
}

static int qsa_pool_search(qsa_pool_t *p, const short *samples,
	unsigned int groups, const short (*codebook)[4], double lambda,
	double mu, double wf,
	unsigned int beam, const qsa_enc_state_t *state, const int *dq,
	unsigned int scale_hint)
{
	qsa_bound_bits_t none;
	none.d = -1;
	p->dq = dq;
	p->samples = samples;
	p->groups = groups;
	p->codebook = codebook;
	p->lambda = lambda;
	p->emu = mu;
	p->wfw = wf;
	p->beam = beam;
	p->in_state = state;
	qsa_make_scale_order(scale_hint, p->scale_order);
	__atomic_store_n(&p->next_scale, 0, __ATOMIC_RELAXED);
	__atomic_store_n(&p->bound_bits, none.u, __ATOMIC_RELAXED);

	pthread_mutex_lock(&p->mu);
	p->running = (int)p->nworkers;
	p->generation++;
	pthread_cond_broadcast(&p->cv_start);
	pthread_mutex_unlock(&p->mu);

	qsa_pool_work(p);

	pthread_mutex_lock(&p->mu);
	while (p->running) { pthread_cond_wait(&p->cv_done, &p->mu); }
	pthread_mutex_unlock(&p->mu);

	int best = -1;
	for (int scale = 0; scale < QSA_SCALE_COUNT; scale++) {
		if (p->trial[scale].cost >= 0 &&
			(best < 0 || p->trial[scale].cost < p->trial[best].cost)) {
			best = scale;
		}
	}
	return best;
}

#endif /* QSA_THREADS */

unsigned int qsa_max_encoded_size(const qsa_desc *desc) {
	unsigned int chunks = (desc->samples + desc->chunk_samples - 1) / desc->chunk_samples;
	return QSA_HEADER_SIZE + (chunks + 1) * 4 +
		QSA_SEEK_STATE_COUNT(chunks) * QSA_SEEK_STATE_SIZE +
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
		desc->shape_lambda < 0 ||
		desc->deemph > 255 ||
		(desc->deemph && desc->samplerate > QSA_MAX_QSA5_SAMPLERATE) ||
		desc->codebook == NULL ||
		desc->codebook[0][0] || desc->codebook[0][1] ||
		desc->codebook[0][2] || desc->codebook[0][3]
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

	#ifdef QSA_THREADS
		qsa_pool_t *pool = (desc->threads > 1 &&
			slice / 4 <= QSA_BEAM_MAX_GROUPS)
			? qsa_pool_create(desc->threads) : NULL;
	#endif

	int *dq = (int *)0;
	#ifdef QSA_AVX2
	if (qsa_avx2_ready()) {
		dq = QSA_MALLOC(QSA_SCALE_COUNT * 4 * 256 * sizeof(int));
		if (dq) {
			for (int sc = 0; sc < QSA_SCALE_COUNT; sc++) {
				for (int e = 0; e < 256; e++) {
					for (int i = 0; i < 4; i++) {
						dq[sc * 1024 + i * 256 + e] =
							(desc->codebook[e][i] * qsa_scale_tab[sc]) >> 8;
					}
				}
			}
		}
	}
	#endif

	unsigned int index_end = QSA_HEADER_SIZE + (desc->chunks + 1) * 4;
	unsigned int seek_states = QSA_SEEK_STATE_COUNT(desc->chunks);
	unsigned int p = index_end + seek_states * QSA_SEEK_STATE_SIZE;
	unsigned int scale_hint = 0;
	int dm = (int)desc->deemph;
	int pre_prev = 0;

	for (unsigned int c = 0; c < desc->chunks; c++) {
		if (c && c % QSA_SEEK_CHUNK_INTERVAL == 0) {
			unsigned int state_pos = index_end +
				(c / QSA_SEEK_CHUNK_INTERVAL - 1) * QSA_SEEK_STATE_SIZE;
			for (unsigned int i = 0; i < QSA_LMS_LEN; i++) {
				qsa_write_u32((unsigned int)state.lms.history[i], bytes + state_pos + i * 4);
				qsa_write_u32((unsigned int)state.lms.weights[i],
					bytes + state_pos + (QSA_LMS_LEN + i) * 4);
			}
		}

		unsigned int start = c * desc->chunk_samples;
		unsigned int valid = desc->samples - start;
		if (valid > desc->chunk_samples) { valid = desc->chunk_samples; }
		unsigned int count = ((valid + slice - 1) / slice) * slice;

		for (unsigned int i = 0; i < valid; i++) {
			int x = sample_data[start + i];
			chunk[i] = (short)qsa_clamp_s16(x - ((dm * pre_prev) >> 8));
			pre_prev = x;
		}
		for (unsigned int i = valid; i < count; i++) {
			chunk[i] = (short)qsa_clamp_s16(-((dm * pre_prev) >> 8));
			pre_prev = 0;
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

			double sl_mu = desc->energy_mu;
			double sl_wf = 1.0;
			if (desc->pns_threshold > 0) {
				double s2 = 0, d2 = 0;
				for (unsigned int i = 0; i < slice; i++) {
					double v = chunk[s * slice + i];
					s2 += v * v;
					if (i) {
						double d = (double)chunk[s * slice + i] -
							chunk[s * slice + i - 1];
						d2 += d * d;
					}
				}
				if (s2 > (double)slice * 16 * 16 &&
					d2 / (2 * s2) > desc->pns_threshold) {
					sl_wf = 0.25;
					sl_mu = sl_mu > 1.0 ? sl_mu * 4 : 4.0;
				}
			}

			#ifdef QSA_THREADS
			if (pool) {
				best_scale = qsa_pool_search(pool, chunk + s * slice,
					groups, desc->codebook, desc->shape_lambda,
					sl_mu, sl_wf,
					desc->search_beam, &state, dq, scale_hint);
				qsa_scale_trial_t *win = &pool->trial[best_scale];
				best_plain = win->plain;
				best_state = win->state;
				for (unsigned int g = 0; g < groups; g++) {
					best_codes[(s * slice) / 4 + g] = win->codes[g];
				}
				state = best_state;
				scale_hint = (unsigned int)best_scale;
				#ifdef QSA_RECORD_TOTAL_ERROR
					desc->error += best_plain;
				#endif
				if (s & 1) { nibbles[s >> 1] |= best_scale << 4; }
				else       { nibbles[s >> 1] |= best_scale; }
				continue;
			}
			#endif

			unsigned char scale_order[QSA_SCALE_COUNT];
			qsa_make_scale_order(scale_hint, scale_order);
			for (int job = 0; job < QSA_SCALE_COUNT; job++) {
				int scale = scale_order[job];
				qsa_enc_state_t trial = state;
				double plain = 0;
				double cost = qsa_encode_slice(
					chunk + s * slice, groups, desc->codebook,
					scale, desc->shape_lambda, sl_mu, sl_wf,
					desc->search_beam,
					&trial, codes, &plain,
					dq ? dq + scale * 1024 : (const int *)0, best_cost
				);
				if (cost >= 0 && (best_cost < 0 || cost < best_cost ||
					(cost == best_cost && scale < best_scale))) {
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
			scale_hint = (unsigned int)best_scale;
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

	qsa_write_u32(desc->deemph ? QSA_MAGIC5 : QSA_MAGIC, bytes);
	qsa_write_u32(desc->samples, bytes + 4);
	qsa_write_u32(desc->deemph
		? (desc->samplerate | (desc->deemph << 24))
		: desc->samplerate, bytes + 8);
	qsa_write_u16(desc->chunk_samples, bytes + 12);
	qsa_write_u16(desc->slice_samples, bytes + 14);
	qsa_write_u32(desc->chunks, bytes + 16);
	for (unsigned int i = 0; i < 256 * 4; i++) {
		qsa_write_u16((unsigned short)desc->codebook[i >> 2][i & 3],
			bytes + 20 + i * 2);
	}
	qsa_write_u32(p, bytes + QSA_HEADER_SIZE + desc->chunks * 4);

	#ifdef QSA_THREADS
		qsa_pool_destroy(pool);
	#endif
	if (dq) { QSA_FREE(dq); }
	QSA_FREE(best_codes);
	QSA_FREE(codes);
	QSA_FREE(chunk);

	*out_len = p;
	return bytes;
}



/* -----------------------------------------------------------------------------
	Decoder */

unsigned int qsa_decode_header(const unsigned char *bytes, unsigned int size, qsa_desc *desc) {
	unsigned int magic;
	if (size < QSA_MIN_FILESIZE) {
		return 0;
	}
	magic = qsa_read_u32(bytes);
	if (magic != QSA_MAGIC && magic != QSA_MAGIC5) {
		return 0;
	}

	qsa_desc_init(desc);
	desc->samples = qsa_read_u32(bytes + 4);
	desc->samplerate = qsa_read_u32(bytes + 8);
	if (magic == QSA_MAGIC5) {
		desc->deemph = desc->samplerate >> 24;
		desc->samplerate &= QSA_MAX_QSA5_SAMPLERATE;
	}
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
	unsigned int chunks_start = index_end +
		QSA_SEEK_STATE_COUNT(desc->chunks) * QSA_SEEK_STATE_SIZE;
	if (
		chunks_start < index_end || chunks_start > size ||
		qsa_read_u32(bytes + QSA_HEADER_SIZE) != chunks_start ||
		qsa_read_u32(bytes + QSA_HEADER_SIZE + desc->chunks * 4) != size
	) {
		return 0;
	}

	/* Read the codebook in place. It is little endian, and it is 4-aligned
	after the 20 fixed bytes when the file is aligned. Entry 0 must be the
	reserved vector of zeros. */
	if (
		qsa_read_u16(bytes + 20) || qsa_read_u16(bytes + 22) ||
		qsa_read_u16(bytes + 24) || qsa_read_u16(bytes + 26)
	) {
		return 0;
	}
	desc->codebook = (const short (*)[4])(const void *)(bytes + 20);

	return QSA_HEADER_SIZE;
}

static int qsa_read_s32(const unsigned char *bytes) {
	unsigned int value = qsa_read_u32(bytes);
	return value < 0x80000000u ? (int)value : -(int)(~value) - 1;
}

int qsa_decode_seek_state(const unsigned char *bytes, unsigned int size,
	const qsa_desc *desc, unsigned int chunk, qsa_lms_t *lms)
{
	if (chunk >= desc->chunks || chunk % QSA_SEEK_CHUNK_INTERVAL) {
		return 0;
	}
	if (chunk == 0) {
		qsa_lms_init(lms);
		return 1;
	}

	unsigned int index_end = QSA_HEADER_SIZE + (desc->chunks + 1) * 4;
	unsigned int state_pos = index_end +
		(chunk / QSA_SEEK_CHUNK_INTERVAL - 1) * QSA_SEEK_STATE_SIZE;
	if (state_pos > size || size - state_pos < QSA_SEEK_STATE_SIZE) {
		return 0;
	}

	for (unsigned int i = 0; i < QSA_LMS_LEN; i++) {
		lms->history[i] = qsa_read_s32(bytes + state_pos + i * 4);
		lms->weights[i] = qsa_read_s32(bytes + state_pos + (QSA_LMS_LEN + i) * 4);
	}
	lms->sample_index = chunk * desc->chunk_samples;
	lms->deemph_prev = 0;
	return 1;
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
	const short (*codebook)[4] = desc->codebook;
	int dm = (int)desc->deemph;

	for (unsigned int s = 0; s < slice_count; s++) {
		/* A nibble and an index byte are always in range, thus no test is
		necessary. */
		int scale = qsa_scale_tab[(s & 1) ? (nibbles[s >> 1] >> 4) : (nibbles[s >> 1] & 15)];

		for (unsigned int g = 0; g < slice / 4; g++) {
			unsigned int n = s * slice + g * 4;
			const short *vector = codebook[residuals[n >> 2]];
			for (int i = 0; i < 4; i++) {
				int dequantized = (vector[i] * scale) >> 8;
				int reconstructed = qsa_clamp_s16(qsa_lms_predict(lms) + dequantized);
				int out = qsa_clamp_s16(reconstructed +
					((dm * lms->deemph_prev) >> 8));
				lms->deemph_prev = out;
				sample_data[n + i] = out;
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

		/* Test the chunk's sample count first. It limits the write. */
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
