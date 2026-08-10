#include <immintrin.h>
#include <stdbool.h>
#include <stdint.h>

static inline void maskload_accum(float *C, __m256 C_accum[6][2],
                                  __m256i packed_masks[2], int M, int nr) {
  for (int j = 0; j < nr; j++) {
    C_accum[j][0] = _mm256_maskload_ps(&C[j * M], packed_masks[0]);
    C_accum[j][1] = _mm256_maskload_ps(&C[j * M + 8], packed_masks[1]);
  }
}

static inline void load_accum(float *C, __m256 C_accum[6][2], int M, int nr) {
  for (int j = 0; j < nr; j++) {
    C_accum[j][0] = _mm256_loadu_ps(&C[j * M]);
    C_accum[j][1] = _mm256_loadu_ps(&C[j * M + 8]);
  }
}

static inline void maskstore_accum(float *C, __m256 C_accum[6][2],
                                   __m256i packed_masks[2], int M, int nr) {
  for (int j = 0; j < nr; j++) {
    _mm256_maskstore_ps(&C[j * M], packed_masks[0], C_accum[j][0]);
    _mm256_maskstore_ps(&C[j * M + 8], packed_masks[1], C_accum[j][1]);
  }
}

static inline void store_accum(float *C, __m256 C_accum[6][2], int M, int nr) {
  for (int j = 0; j < nr; j++) {
    _mm256_storeu_ps(&C[j * M], C_accum[j][0]);
    _mm256_storeu_ps(&C[j * M + 8], C_accum[j][1]);
  }
}

static inline void fma_loop(float *blockA_packed, float *blockB_packed,
                            __m256 C_accum[6][2], __m256 a0_packFloat8,
                            __m256 a1_packFloat8, __m256 b_packFloat8, int kc) {

  for (int p = 0; p < kc; p++) {
    a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
    a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
    C_accum[0][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[0][0]);
    C_accum[0][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[0][1]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
    C_accum[1][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[1][0]);
    C_accum[1][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[1][1]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
    C_accum[2][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[2][0]);
    C_accum[2][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[2][1]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
    C_accum[3][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[3][0]);
    C_accum[3][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[3][1]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
    C_accum[4][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[4][0]);
    C_accum[4][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[4][1]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 5);
    C_accum[5][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[5][0]);
    C_accum[5][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[5][1]);

    blockA_packed += 16;
    blockB_packed += 6;
  }
}

static inline void build_masks(__m256i packed_masks[2], int mr) {
  const uint32_t bit_mask = 65535;
  packed_masks[0] = _mm256_setr_epi32(
      bit_mask << (mr + 15), bit_mask << (mr + 14), bit_mask << (mr + 13),
      bit_mask << (mr + 12), bit_mask << (mr + 11), bit_mask << (mr + 10),
      bit_mask << (mr + 9), bit_mask << (mr + 8));
  packed_masks[1] = _mm256_setr_epi32(
      bit_mask << (mr + 7), bit_mask << (mr + 6), bit_mask << (mr + 5),
      bit_mask << (mr + 4), bit_mask << (mr + 3), bit_mask << (mr + 2),
      bit_mask << (mr + 1), bit_mask << mr);
}

void kernel_16x6_accum(float *blockA_packed, float *blockB_packed, float *C,
                       int mr, int nr, int kc, int M, bool load) {

  __m256 C_accum[6][2] = {};
  __m256 b_packFloat8 = {};
  __m256 a0_packFloat8 = {};
  __m256 a1_packFloat8 = {};

  if (mr != 16) {
    __m256i packed_masks[2];
    build_masks(packed_masks, mr);
    if (load)
      maskload_accum(C, C_accum, packed_masks, M, nr);
    fma_loop(blockA_packed, blockB_packed, C_accum, a0_packFloat8,
             a1_packFloat8, b_packFloat8, kc);
    maskstore_accum(C, C_accum, packed_masks, M, nr);
  } else {
    if (load)
      load_accum(C, C_accum, M, nr);
    fma_loop(blockA_packed, blockB_packed, C_accum, a0_packFloat8,
             a1_packFloat8, b_packFloat8, kc);
    store_accum(C, C_accum, M, nr);
  }
}
