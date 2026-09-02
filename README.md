# QSA - Quantised Sliced Audio

QSA is a tiny, lossy audio codec for machines that cannot afford much of a
codec at all. It is mono, works at any sample rate, and codes residuals as
one byte per 4 samples, an index into a 256-entry trained codebook carried
in the file (2 kB, once), plus a 4-bit quantiser scale per 64-sample slice.
That comes out at about 2.1 bits per sample, or 19.6 kbit/s at the default
9360 hz, so 90 minutes of audio is about 13 MB. The decoder is integer only:
no floating point, and the per-sample path is one table read, a multiply, a
shift and an add.

Single-file MIT licensed library for C/C++. See [qsa.h](qsa.h) for the
documentation and format specification.

This is a fork of [phoboslab/qoa](https://github.com/phoboslab/qoa), the
"Quite OK Audio Format". QSA is a direct descendant of it. The 4-tap
sign-sign LMS predictor, the scale table and the slice idea are all QOA's.

⚠️ This implementation has not yet been fuzzed. Don't use it with untrusted input.


## How QSA relates to QOA

QOA already quantises in slices: 20 samples share a 4-bit scalefactor, with
3-bit residuals, for ~3.2 bits per sample at any sample rate. QSA is what
happens when you need to halve that: it keeps QOA's per-slice scaling, with a
4-bit scale every 64 samples, and pins the residual at 2 bits per sample.

Rather than four independent 2-bit residuals, QSA spends its 8 bits per
4-sample group on one index into a 256-entry codebook of residual vectors.
Since QSA3 the codebook is stored in the file header (2 kB), so material can
carry a table trained on itself -- `qsaconv --codebook file.bin` -- with a
built-in table, made by closed-loop k-means, as the default. Entry 0 is
always the all-zero vector, so digital silence encodes to exact digital
silence, and the encoder spots all-zero groups without searching. A codebook
of vectors can represent correlated shapes that a scalar grid cannot, and it
makes the decoder simpler: one table read per group instead of bit
unpacking.

QSA changes a few more things on top of QOA's design. The LMS weights leak:
every fourth sample each weight loses `weight >> 7`, because at 2 bits the
residuals are too coarse to keep the filter stable without it. The prediction
dot product is specified as wrapping 32-bit arithmetic, so any conforming
integer decoder reproduces this implementation bit for bit. The encoder picks
codes and scales by minimising `e^2 + lambda * (e[n] - e[n-1])^2` rather than
plain squared error; penalising the error slope shapes the quantisation noise
to follow the signal instead of spreading it flat. This shaping is
encoder-only; the bitstream and decoder know nothing of it. Finally, chunks
(default 2048 samples) are indexed by byte offset, so a player can locate any
chunk directly. The LMS state is encoded in every 16th chunk, so while playing
you can seek to the nearest stored state without decoding from the start.
QSA5 optionally applies pre-emphasis while encoding and de-emphasis while
decoding; its coefficient is stored in the high byte of the sample rate.


## Tools

`qsaconv` converts WAV (and optionally MP3/FLAC) to QSA and back. Sources
must be mono 16-bit; the source's sample rate is carried in the QSA header.
Resample with e.g. `ffmpeg -i in.flac -ac 1 -ar 9360 -c:a pcm_s16le in.wav`
first.

```bash
./qsaconv in.wav out.qsa
./qsaconv out.qsa decoded.wav
```

`--chunk-samples` (default 2048) sets the chunk length, `--slice-samples`
(default 64) the scale period, and `--shape` (default 0.5) the noise shaping
strength lambda. `--shape 0` gives plain MSE. `--beam` (default 1) keeps more
candidate paths during encoding; values from 4 to 8 improve quality but take
longer. `--threads` sets the worker count when compiled with `QSA_THREADS`.
`--energy` favours matching the source energy, while `--pns` does the same for
noise-like slices instead of matching their exact waveform. Both options
affect the encoder only. `--deemph A` writes QSA5 with pre-emphasis and
de-emphasis; `A` must be at least 0 and less than 1, and 0 writes QSA4.

`qsaplay` plays QSA files:

```bash
./qsaplay file.qsa
```


## Compiling

Call `make` to build `qsaconv` and `qsaplay`. By default `qsaconv` is compiled
without MP3 and FLAC support.

To compile `qsaconv` with MP3 and FLAC support, download the
[dr_*.h files](https://github.com/mackron/dr_libs) and pass `HAS_DRLIBS=true`
to make:

```bash
curl https://raw.githubusercontent.com/mackron/dr_libs/refs/heads/master/dr_mp3.h -o dr_mp3.h
curl https://raw.githubusercontent.com/mackron/dr_libs/refs/heads/master/dr_flac.h -o dr_flac.h
make HAS_DRLIBS=true
```

`qsaplay` requires
[sokol_audio.h](https://github.com/floooh/sokol/blob/master/sokol_audio.h).

Call `make test` to build and run `qsatest`, which round trips a fixed,
integer-generated signal against golden hashes and reports the bit rate and
SNR it measured.
