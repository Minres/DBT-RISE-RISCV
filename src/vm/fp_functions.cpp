////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, MINRES Technologies GmbH
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Contributors:
//       eyck@minres.com - initial API and implementation
////////////////////////////////////////////////////////////////////////////////

#include "fp_functions.h"
#include "softfloat_types.h"
#include <array>
#include <cstdint>

extern "C" {
#include "internals.h"
#include "specialize.h"
#include <softfloat.h>
}

#include <limits>

using this_t = uint8_t*;
template <typename T> T constexpr defaultNaN();
template <> uint16_t constexpr defaultNaN<uint16_t>() { return defaultNaNF16UI; }
template <> uint32_t constexpr defaultNaN<uint32_t>() { return defaultNaNF32UI; }
template <> uint64_t constexpr defaultNaN<uint64_t>() { return defaultNaNF64UI; }
template <typename T> T constexpr posInf();
template <> uint16_t constexpr posInf<uint16_t>() { return 0x7C00; }
template <> uint32_t constexpr posInf<uint32_t>() { return 0x7F800000; }
template <> uint64_t constexpr posInf<uint64_t>() { return 0x7FF0000000000000; }
template <typename T> T constexpr negInf();
template <> uint16_t constexpr negInf<uint16_t>() { return 0xFC00; }
template <> uint32_t constexpr negInf<uint32_t>() { return 0xFF800000; }
template <> uint64_t constexpr negInf<uint64_t>() { return 0xFFF0000000000000; }
template <typename T> T constexpr negZero();
template <> uint16_t constexpr negZero<uint16_t>() { return 0x8000; }
template <> uint32_t constexpr negZero<uint32_t>() { return 0x80000000; }
template <> uint64_t constexpr negZero<uint64_t>() { return 0x8000000000000000; }
// this does not inlcude any reserved rm or the DYN rm, as DYN rm should be taken care of in the vm_impl
template <typename T> bool rsqrt_check(T fclass_val, bool& subnormal, T& ret_val) {
    softfloat_exceptionFlags = 0;
    switch(fclass_val) {
    case 0x0001: {
        softfloat_exceptionFlags |= softfloat_flag_invalid;
        ret_val = defaultNaN<T>();
        return true;
    }
    case 0x0002: {
        softfloat_exceptionFlags |= softfloat_flag_invalid;
        ret_val = defaultNaN<T>();
        return true;
    }
    case 0x0004: {
        softfloat_exceptionFlags |= softfloat_flag_invalid;
        ret_val = defaultNaN<T>();
        return true;
    }
    case 0x0100: {
        softfloat_exceptionFlags |= softfloat_flag_invalid;
        ret_val = defaultNaN<T>();
        return true;
    }
    case 0x0200: {
        ret_val = defaultNaN<T>();
        return true;
    }
    case 0x0008: {
        softfloat_exceptionFlags |= softfloat_flag_infinite;
        ret_val = negInf<T>();
        return true;
    }
    case 0x0010: {
        softfloat_exceptionFlags |= softfloat_flag_infinite;
        ret_val = posInf<T>();
        return true;
    }
    case 0x0080: {
        ret_val = 0;
        return true;
    }
    case 0x0020: {
        subnormal = true;
    }
    default:
        return false;
    }
}
static constexpr std::array<std::array<uint64_t, 64>, 2> rsqrt_table{
    {{
         52, 51, 50, 48, 47, 46, 44, 43, 42, 41, 40, 39, 38, 36, 35, 34, 33, 32, 31, 30, 30, 29, 28, 27, 26, 25, 24, 23, 23, 22, 21, 20,
         19, 19, 18, 17, 16, 16, 15, 14, 14, 13, 12, 12, 11, 10, 10, 9,  9,  8,  7,  7,  6,  6,  5,  4,  4,  3,  3,  2,  2,  1,  1,  0,
     },
     {127, 125, 123, 121, 119, 118, 116, 114, 113, 111, 109, 108, 106, 105, 103, 102, 100, 99, 97, 96, 95, 93,
      92,  91,  90,  88,  87,  86,  85,  84,  83,  82,  80,  79,  78,  77,  76,  75,  74,  73, 72, 71, 70, 70,
      69,  68,  67,  66,  65,  64,  63,  63,  62,  61,  60,  59,  59,  58,  57,  56,  56,  55, 54, 53}}};

uint64_t constexpr frsqrt7_general(const unsigned s, const unsigned e, const uint64_t sign, const int64_t exp, const uint64_t sig,
                                   const bool subnormal) {
    int64_t normalized_exp = exp;
    uint64_t normalized_sig = sig;
    if(subnormal) {
        signed nr_leadingzeros = __builtin_clzll(sig) - (64 - s);
        normalized_exp = -nr_leadingzeros;
        normalized_sig = (sig << (1 + nr_leadingzeros)) & ((1ULL << s) - 1);
    }
    unsigned exp_idx = normalized_exp & 1;
    unsigned sig_idx = (normalized_sig >> (s - 6)) & 0x3f;
    // The output of the table becomes the seven high bits of the result significand (after the leading one); the remainder of the
    // result significand is zero.
    uint64_t out_sig = rsqrt_table[exp_idx][sig_idx] << (s - 7);
    // The output exponent equals floor((3*B - 1 - the normalized input exponent) / 2), where B is the exponent bias.
    unsigned bias = (1UL << (e - 1)) - 1;
    uint64_t out_exp = (3 * bias - 1 - normalized_exp) / 2;
    // The output sign equals the input sign.
    return (sign << (s + e)) | (out_exp << s) | out_sig;
}
template <typename T> bool recip_check(T fclass_val, bool& subnormal, uint64_t& ret_val) {
    softfloat_exceptionFlags = 0;
    switch(fclass_val) {
    case 0x0001: {
        ret_val = negZero<T>();
        return true;
    }
    case 0x0080: {
        ret_val = 0;
        return true;
    }
    case 0x0008: {
        softfloat_exceptionFlags |= softfloat_flag_infinite;
        ret_val = negInf<T>();
        return true;
    }
    case 0x0010: {
        softfloat_exceptionFlags |= softfloat_flag_infinite;
        ret_val = posInf<T>();
        return true;
    }
    case 0x0100: {
        softfloat_exceptionFlags |= softfloat_flag_invalid;
        ret_val = defaultNaN<T>();
        return true;
    }
    case 0x0200: {
        ret_val = defaultNaN<T>();
        return true;
    }
    case 0x0004: {
        subnormal = true;
        return false;
    }
    case 0x0020: {
        subnormal = true;
        return false;
    }
    default: {
        subnormal = false;
        return false;
    }
    }
}
static constexpr std::array<uint64_t, 128> rec_table{
    {127, 125, 123, 121, 119, 117, 116, 114, 112, 110, 109, 107, 105, 104, 102, 100, 99, 97, 96, 94, 93, 91, 90, 88, 87, 85,
     84,  83,  81,  80,  79,  77,  76,  75,  74,  72,  71,  70,  69,  68,  66,  65,  64, 63, 62, 61, 60, 59, 58, 57, 56, 55,
     54,  53,  52,  51,  50,  49,  48,  47,  46,  45,  44,  43,  42,  41,  40,  40,  39, 38, 37, 36, 35, 35, 34, 33, 32, 31,
     31,  30,  29,  28,  28,  27,  26,  25,  25,  24,  23,  23,  22,  21,  21,  20,  19, 19, 18, 17, 17, 16, 15, 15, 14, 14,
     13,  12,  12,  11,  11,  10,  9,   9,   8,   8,   7,   7,   6,   5,   5,   4,   4,  3,  3,  2,  2,  1,  1,  0}};
bool frec_general(uint64_t& res, const unsigned s, const unsigned e, const uint64_t sign, const int64_t exp, const uint64_t sig,
                  const bool subnormal, uint8_t mode) {
    int nr_leadingzeros = __builtin_clzll(sig) - (64 - s);
    int64_t normalized_exp = subnormal ? -nr_leadingzeros : exp;
    uint64_t normalized_sig = subnormal ? ((sig << (1 + nr_leadingzeros)) & ((1ULL << s) - 1)) : sig;
    unsigned idx = (normalized_sig >> (s - 7)) & 0x7f;
    unsigned bias = (1UL << (e - 1)) - 1;
    uint64_t mid_exp = 2 * (bias)-1 - normalized_exp;
    uint64_t mid_sig = rec_table[idx] << (s - 7);

    uint64_t out_exp = mid_exp;
    uint64_t out_sig = mid_sig;
    if(mid_exp == 0) {
        out_exp = mid_exp;
        out_sig = (mid_sig >> 1) | (1ULL << (s - 1));
    } else if(mid_exp == (1ULL << e) - 1) {
        out_exp = 0;
        out_sig = (mid_sig >> 2) | (1ULL << (s - 2));
    }
    if(subnormal && nr_leadingzeros > 1) {
        if((mode == 0b001) || (mode == 0b010 && sign == 0b0) || (mode == 0b011 && sign == 0b1)) {
            res = (sign << (s + e)) | ((1ULL << (e - 1)) - 1) << s | ((1ULL << s) - 1);
            return true;
        } else {
            res = (sign << (s + e)) | ((1ULL << e) - 1) << s;
            return true;
        }
    }
    res = (sign << (s + e)) | (out_exp << s) | out_sig;
    return false;
}

extern "C" {

uint32_t fget_flags() { return softfloat_exceptionFlags & 0x1f; }
uint16_t fadd_h(uint16_t v1, uint16_t v2, uint8_t mode) {
    float16_t v1f{v1}, v2f{v2};
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float16_t r = f16_add(v1f, v2f);
    return r.v;
}

uint16_t fsub_h(uint16_t v1, uint16_t v2, uint8_t mode) {
    float16_t v1f{v1}, v2f{v2};
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float16_t r = f16_sub(v1f, v2f);
    return r.v;
}

uint16_t fmul_h(uint16_t v1, uint16_t v2, uint8_t mode) {
    float16_t v1f{v1}, v2f{v2};
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float16_t r = f16_mul(v1f, v2f);
    return r.v;
}

uint16_t fdiv_h(uint16_t v1, uint16_t v2, uint8_t mode) {
    float16_t v1f{v1}, v2f{v2};
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float16_t r = f16_div(v1f, v2f);
    return r.v;
}

uint16_t fsqrt_h(uint16_t v1, uint8_t mode) {
    float16_t v1f{v1};
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float16_t r = f16_sqrt(v1f);
    return r.v;
}

uint16_t fcmp_h(uint16_t v1, uint16_t v2, uint16_t op) {
    float16_t v1f{v1}, v2f{v2};
    softfloat_exceptionFlags = 0;
    bool nan = v1 == defaultNaNF16UI || v2 & defaultNaNF16UI;
    bool snan = softfloat_isSigNaNF16UI(v1) || softfloat_isSigNaNF16UI(v2);
    switch(op) {
    case 0:
        if(nan | snan) {
            if(snan)
                softfloat_raiseFlags(softfloat_flag_invalid);
            return 0;
        } else
            return f16_eq(v1f, v2f) ? 1 : 0;
    case 1:
        if(nan | snan) {
            softfloat_raiseFlags(softfloat_flag_invalid);
            return 0;
        } else
            return f16_le(v1f, v2f) ? 1 : 0;
    case 2:
        if(nan | snan) {
            softfloat_raiseFlags(softfloat_flag_invalid);
            return 0;
        } else
            return f16_lt(v1f, v2f) ? 1 : 0;
    default:
        break;
    }
    return -1;
}

uint16_t fmadd_h(uint16_t v1, uint16_t v2, uint16_t v3, uint16_t op, uint8_t mode) {
    uint16_t F16_SIGN = 1UL << 15;
    switch(op) {
    case 0: // FMADD_S
        break;
    case 1: // FMSUB_S
        v3 ^= F16_SIGN;
        break;
    case 2: // FNMADD_S
        v1 ^= F16_SIGN;
        v3 ^= F16_SIGN;
        break;
    case 3: // FNMSUB_S
        v1 ^= F16_SIGN;
        break;
    }
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float16_t res = softfloat_mulAddF16(v1, v2, v3, 0);
    return res.v;
}

uint16_t fsel_h(uint16_t v1, uint16_t v2, uint16_t op) {
    softfloat_exceptionFlags = 0;
    bool v1_nan = (v1 & defaultNaNF16UI) == defaultNaNF16UI;
    bool v2_nan = (v2 & defaultNaNF16UI) == defaultNaNF16UI;
    bool v1_snan = softfloat_isSigNaNF16UI(v1);
    bool v2_snan = softfloat_isSigNaNF16UI(v2);
    if(v1_snan || v2_snan)
        softfloat_raiseFlags(softfloat_flag_invalid);
    if(v1_nan || v1_snan)
        return (v2_nan || v2_snan) ? defaultNaNF16UI : v2;
    else if(v2_nan || v2_snan)
        return v1;
    else {
        if((v1 & 0x7fff) == 0 && (v2 & 0x7fff) == 0) {
            return op == 0 ? ((v1 & 0x8000) ? v1 : v2) : ((v1 & 0x8000) ? v2 : v1);
        } else {
            float16_t v1f{v1}, v2f{v2};
            return op == 0 ? (f16_lt(v1f, v2f) ? v1 : v2) : (f16_lt(v1f, v2f) ? v2 : v1);
        }
    }
}

uint16_t fclass_h(uint16_t v1) {

    float16_t a{v1};
    union ui16_f16 uA;
    uint_fast16_t uiA;

    uA.f = a;
    uiA = uA.ui;

    bool infOrNaN = expF16UI(uiA) == 0x1F;
    bool subnormalOrZero = expF16UI(uiA) == 0;
    bool sign = signF16UI(uiA);
    bool fracZero = fracF16UI(uiA) == 0;
    bool isNaN = isNaNF16UI(uiA);
    bool isSNaN = softfloat_isSigNaNF16UI(uiA);

    return (sign && infOrNaN && fracZero) << 0 | (sign && !infOrNaN && !subnormalOrZero) << 1 |
           (sign && subnormalOrZero && !fracZero) << 2 | (sign && subnormalOrZero && fracZero) << 3 | (!sign && infOrNaN && fracZero) << 7 |
           (!sign && !infOrNaN && !subnormalOrZero) << 6 | (!sign && subnormalOrZero && !fracZero) << 5 |
           (!sign && subnormalOrZero && fracZero) << 4 | (isNaN && isSNaN) << 8 | (isNaN && !isSNaN) << 9;
}

uint16_t frsqrt7_h(uint16_t v) {
    bool subnormal = false;
    uint16_t ret_val = 0;
    if(rsqrt_check(fclass_h(v), subnormal, ret_val)) {
        return ret_val;
    }
    uint16_t sig = fracF64UI(v);
    int16_t exp = expF64UI(v);
    uint16_t sign = signF64UI(v);
    unsigned constexpr e = 5;
    unsigned constexpr s = 10;
    return frsqrt7_general(s, e, sign, exp, sig, subnormal);
}

uint16_t frec7_h(uint16_t v, uint8_t mode) {
    bool subnormal = false;
    uint64_t ret_val = 0;
    if(recip_check(fclass_h(v), subnormal, ret_val)) {
        return ret_val;
    }
    uint16_t sig = fracF16UI(v);
    int exp = expF16UI(v);
    uint16_t sign = signF16UI(v);
    unsigned constexpr e = 5;
    unsigned constexpr s = 10;
    if(frec_general(ret_val, s, e, sign, exp, sig, subnormal, mode))
        softfloat_exceptionFlags |= (softfloat_flag_inexact | softfloat_flag_overflow);
    return ret_val;
}

uint16_t unbox_h(uint8_t FLEN, uint64_t v) {
    uint64_t mask = 0;
    switch(FLEN) {
    case 32: {
        mask = std::numeric_limits<uint32_t>::max() & ~((uint64_t)std::numeric_limits<uint16_t>::max());
        break;
    }
    case 64: {
        mask = std::numeric_limits<uint64_t>::max() & ~((uint64_t)std::numeric_limits<uint16_t>::max());
        break;
    }
    default:
        break;
    }
    if((v & mask) != mask)
        return defaultNaNF16UI;
    else
        return v & std::numeric_limits<uint32_t>::max();
}

uint32_t fadd_s(uint32_t v1, uint32_t v2, uint8_t mode) {
    float32_t v1f{v1}, v2f{v2};
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float32_t r = f32_add(v1f, v2f);
    return r.v;
}

uint32_t fsub_s(uint32_t v1, uint32_t v2, uint8_t mode) {
    float32_t v1f{v1}, v2f{v2};
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float32_t r = f32_sub(v1f, v2f);
    return r.v;
}

uint32_t fmul_s(uint32_t v1, uint32_t v2, uint8_t mode) {
    float32_t v1f{v1}, v2f{v2};
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float32_t r = f32_mul(v1f, v2f);
    return r.v;
}

uint32_t fdiv_s(uint32_t v1, uint32_t v2, uint8_t mode) {
    float32_t v1f{v1}, v2f{v2};
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float32_t r = f32_div(v1f, v2f);
    return r.v;
}

uint32_t fsqrt_s(uint32_t v1, uint8_t mode) {
    float32_t v1f{v1};
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float32_t r = f32_sqrt(v1f);
    return r.v;
}

uint32_t fcmp_s(uint32_t v1, uint32_t v2, uint32_t op) {
    float32_t v1f{v1}, v2f{v2};
    softfloat_exceptionFlags = 0;
    bool nan = v1 == defaultNaNF32UI || v2 == defaultNaNF32UI;
    bool snan = softfloat_isSigNaNF32UI(v1) || softfloat_isSigNaNF32UI(v2);
    switch(op) {
    case 0:
        if(nan | snan) {
            if(snan)
                softfloat_raiseFlags(softfloat_flag_invalid);
            return 0;
        } else
            return f32_eq(v1f, v2f) ? 1 : 0;
    case 1:
        if(nan | snan) {
            softfloat_raiseFlags(softfloat_flag_invalid);
            return 0;
        } else
            return f32_le(v1f, v2f) ? 1 : 0;
    case 2:
        if(nan | snan) {
            softfloat_raiseFlags(softfloat_flag_invalid);
            return 0;
        } else
            return f32_lt(v1f, v2f) ? 1 : 0;
    default:
        break;
    }
    return -1;
}

uint32_t fmadd_s(uint32_t v1, uint32_t v2, uint32_t v3, uint32_t op, uint8_t mode) {
    uint32_t F32_SIGN = 1UL << 31;
    switch(op) {
    case 0: // FMADD_S
        break;
    case 1: // FMSUB_S
        v3 ^= F32_SIGN;
        break;
    case 2: // FNMADD_S
        v1 ^= F32_SIGN;
        v3 ^= F32_SIGN;
        break;
    case 3: // FNMSUB_S
        v1 ^= F32_SIGN;
        break;
    }
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float32_t res = softfloat_mulAddF32(v1, v2, v3, 0);
    return res.v;
}

uint32_t fsel_s(uint32_t v1, uint32_t v2, uint32_t op) {
    softfloat_exceptionFlags = 0;
    bool v1_nan = (v1 & defaultNaNF32UI) == defaultNaNF32UI;
    bool v2_nan = (v2 & defaultNaNF32UI) == defaultNaNF32UI;
    bool v1_snan = softfloat_isSigNaNF32UI(v1);
    bool v2_snan = softfloat_isSigNaNF32UI(v2);
    if(v1_snan || v2_snan)
        softfloat_raiseFlags(softfloat_flag_invalid);
    if(v1_nan || v1_snan)
        return (v2_nan || v2_snan) ? defaultNaNF32UI : v2;
    else if(v2_nan || v2_snan)
        return v1;
    else {
        if((v1 & 0x7fffffff) == 0 && (v2 & 0x7fffffff) == 0) {
            return op == 0 ? ((v1 & 0x80000000) ? v1 : v2) : ((v1 & 0x80000000) ? v2 : v1);
        } else {
            float32_t v1f{v1}, v2f{v2};
            return op == 0 ? (f32_lt(v1f, v2f) ? v1 : v2) : (f32_lt(v1f, v2f) ? v2 : v1);
        }
    }
}

uint32_t fclass_s(uint32_t v1) {

    float32_t a{v1};
    union ui32_f32 uA;
    uint_fast32_t uiA;

    uA.f = a;
    uiA = uA.ui;

    bool infOrNaN = expF32UI(uiA) == 0xFF;
    bool subnormalOrZero = expF32UI(uiA) == 0;
    bool sign = signF32UI(uiA);
    bool fracZero = fracF32UI(uiA) == 0;
    bool isNaN = isNaNF32UI(uiA);
    bool isSNaN = softfloat_isSigNaNF32UI(uiA);

    return (sign && infOrNaN && fracZero) << 0 | (sign && !infOrNaN && !subnormalOrZero) << 1 |
           (sign && subnormalOrZero && !fracZero) << 2 | (sign && subnormalOrZero && fracZero) << 3 | (!sign && infOrNaN && fracZero) << 7 |
           (!sign && !infOrNaN && !subnormalOrZero) << 6 | (!sign && subnormalOrZero && !fracZero) << 5 |
           (!sign && subnormalOrZero && fracZero) << 4 | (isNaN && isSNaN) << 8 | (isNaN && !isSNaN) << 9;
}

uint32_t frsqrt7_s(uint32_t v) {
    bool subnormal = false;
    uint32_t ret_val = 0;
    if(rsqrt_check(fclass_s(v), subnormal, ret_val)) {
        return ret_val;
    }
    uint32_t sig = fracF32UI(v);
    int exp = expF32UI(v);
    uint32_t sign = signF32UI(v);
    unsigned constexpr e = 8;
    unsigned constexpr s = 23;
    return frsqrt7_general(s, e, sign, exp, sig, subnormal);
}

uint32_t frec7_s(uint32_t v, uint8_t mode) {
    bool subnormal = false;
    uint64_t ret_val = 0;
    if(recip_check(fclass_s(v), subnormal, ret_val)) {
        return ret_val;
    }
    uint32_t sig = fracF32UI(v);
    int exp = expF32UI(v);
    uint32_t sign = signF32UI(v);
    unsigned constexpr e = 8;
    unsigned constexpr s = 23;
    if(frec_general(ret_val, s, e, sign, exp, sig, subnormal, mode))
        softfloat_exceptionFlags |= (softfloat_flag_inexact | softfloat_flag_overflow);
    return ret_val;
}

uint32_t unbox_s(uint8_t FLEN, uint64_t v) {
    uint64_t mask = 0;
    switch(FLEN) {
    case 32: {
        return v;
    }
    case 64: {
        mask = std::numeric_limits<uint64_t>::max() & ~((uint64_t)std::numeric_limits<uint32_t>::max());
        break;
    }
    default:
        break;
    }
    if((v & mask) != mask)
        return defaultNaNF32UI;
    else
        return v & std::numeric_limits<uint32_t>::max();
}

uint64_t fadd_d(uint64_t v1, uint64_t v2, uint8_t mode) {
    bool nan = v1 == defaultNaNF32UI;
    bool snan = softfloat_isSigNaNF32UI(v1);
    float64_t v1f{v1}, v2f{v2};
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float64_t r = f64_add(v1f, v2f);
    return r.v;
}

uint64_t fsub_d(uint64_t v1, uint64_t v2, uint8_t mode) {
    float64_t v1f{v1}, v2f{v2};
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float64_t r = f64_sub(v1f, v2f);
    return r.v;
}

uint64_t fmul_d(uint64_t v1, uint64_t v2, uint8_t mode) {
    float64_t v1f{v1}, v2f{v2};
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float64_t r = f64_mul(v1f, v2f);
    return r.v;
}

uint64_t fdiv_d(uint64_t v1, uint64_t v2, uint8_t mode) {
    float64_t v1f{v1}, v2f{v2};
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float64_t r = f64_div(v1f, v2f);
    return r.v;
}

uint64_t fsqrt_d(uint64_t v1, uint8_t mode) {
    float64_t v1f{v1};
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float64_t r = f64_sqrt(v1f);
    return r.v;
}

uint64_t fcmp_d(uint64_t v1, uint64_t v2, uint32_t op) {
    float64_t v1f{v1}, v2f{v2};
    softfloat_exceptionFlags = 0;
    bool nan = v1 == defaultNaNF64UI || v2 == defaultNaNF64UI;
    bool snan = softfloat_isSigNaNF64UI(v1) || softfloat_isSigNaNF64UI(v2);
    switch(op) {
    case 0:
        if(nan | snan) {
            if(snan)
                softfloat_raiseFlags(softfloat_flag_invalid);
            return 0;
        } else
            return f64_eq(v1f, v2f) ? 1 : 0;
    case 1:
        if(nan | snan) {
            softfloat_raiseFlags(softfloat_flag_invalid);
            return 0;
        } else
            return f64_le(v1f, v2f) ? 1 : 0;
    case 2:
        if(nan | snan) {
            softfloat_raiseFlags(softfloat_flag_invalid);
            return 0;
        } else
            return f64_lt(v1f, v2f) ? 1 : 0;
    default:
        break;
    }
    return -1;
}

uint64_t fmadd_d(uint64_t v1, uint64_t v2, uint64_t v3, uint32_t op, uint8_t mode) {
    uint64_t F64_SIGN = 1ULL << 63;
    switch(op) {
    case 0: // FMADD_D
        break;
    case 1: // FMSUB_D
        v3 ^= F64_SIGN;
        break;
    case 2: // FNMADD_D
        v1 ^= F64_SIGN;
        v3 ^= F64_SIGN;
        break;
    case 3: // FNMSUB_D
        v1 ^= F64_SIGN;
        break;
    }
    softfloat_roundingMode = mode;
    softfloat_exceptionFlags = 0;
    float64_t res = softfloat_mulAddF64(v1, v2, v3, 0);
    return res.v;
}

uint64_t fsel_d(uint64_t v1, uint64_t v2, uint32_t op) {
    softfloat_exceptionFlags = 0;
    bool v1_nan = (v1 & defaultNaNF64UI) == defaultNaNF64UI;
    bool v2_nan = (v2 & defaultNaNF64UI) == defaultNaNF64UI;
    bool v1_snan = softfloat_isSigNaNF64UI(v1);
    bool v2_snan = softfloat_isSigNaNF64UI(v2);
    if(v1_snan || v2_snan)
        softfloat_raiseFlags(softfloat_flag_invalid);
    if(v1_nan || v1_snan)
        return (v2_nan || v2_snan) ? defaultNaNF64UI : v2;
    else if(v2_nan || v2_snan)
        return v1;
    else {
        if((v1 & std::numeric_limits<int64_t>::max()) == 0 && (v2 & std::numeric_limits<int64_t>::max()) == 0) {
            return op == 0 ? ((v1 & std::numeric_limits<int64_t>::min()) ? v1 : v2)
                           : ((v1 & std::numeric_limits<int64_t>::min()) ? v2 : v1);
        } else {
            float64_t v1f{v1}, v2f{v2};
            return op == 0 ? (f64_lt(v1f, v2f) ? v1 : v2) : (f64_lt(v1f, v2f) ? v2 : v1);
        }
    }
}

uint64_t fclass_d(uint64_t v1) {

    float64_t a{v1};
    union ui64_f64 uA;
    uint_fast64_t uiA;

    uA.f = a;
    uiA = uA.ui;

    bool infOrNaN = expF64UI(uiA) == 0x7FF;
    bool subnormalOrZero = expF64UI(uiA) == 0;
    bool sign = signF64UI(uiA);
    bool fracZero = fracF64UI(uiA) == 0;
    bool isNaN = isNaNF64UI(uiA);
    bool isSNaN = softfloat_isSigNaNF64UI(uiA);

    return (sign && infOrNaN && fracZero) << 0 | (sign && !infOrNaN && !subnormalOrZero) << 1 |
           (sign && subnormalOrZero && !fracZero) << 2 | (sign && subnormalOrZero && fracZero) << 3 | (!sign && infOrNaN && fracZero) << 7 |
           (!sign && !infOrNaN && !subnormalOrZero) << 6 | (!sign && subnormalOrZero && !fracZero) << 5 |
           (!sign && subnormalOrZero && fracZero) << 4 | (isNaN && isSNaN) << 8 | (isNaN && !isSNaN) << 9;
}

uint64_t frsqrt7_d(uint64_t v) {
    bool subnormal = false;
    uint64_t ret_val = 0;
    if(rsqrt_check(fclass_d(v), subnormal, ret_val)) {
        return ret_val;
    }
    uint64_t sig = fracF64UI(v);
    int exp = expF64UI(v);
    uint64_t sign = signF64UI(v);
    unsigned constexpr e = 11;
    unsigned constexpr s = 52;
    return frsqrt7_general(s, e, sign, exp, sig, subnormal);
}

uint64_t frec7_d(uint64_t v, uint8_t mode) {
    bool subnormal = false;
    uint64_t ret_val = 0;
    if(recip_check(fclass_d(v), subnormal, ret_val)) {
        return ret_val;
    }
    uint64_t sig = fracF64UI(v);
    int exp = expF64UI(v);
    uint64_t sign = signF64UI(v);
    unsigned constexpr e = 11;
    unsigned constexpr s = 52;
    if(frec_general(ret_val, s, e, sign, exp, sig, subnormal, mode))
        softfloat_exceptionFlags |= (softfloat_flag_inexact | softfloat_flag_overflow);
    return ret_val;
}

uint64_t unbox_d(uint8_t FLEN, uint64_t v) {
    uint64_t mask = 0;
    switch(FLEN) {
    case 64: {
        return v;
        break;
    }
    default:
        break;
    }
    if((v & mask) != mask)
        return defaultNaNF64UI;
    else
        return v & std::numeric_limits<uint32_t>::max();
}

// conversion: float to float
uint32_t f16tof32(uint16_t val, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f16_to_f32(float16_t{val}).v;
    }
uint64_t f16tof64(uint16_t val, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f16_to_f64(float16_t{val}).v;
}

uint16_t f32tof16(uint32_t val, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f32_to_f16(float32_t{val}).v;
    }
uint64_t f32tof64(uint32_t val, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f32_to_f64(float32_t{val}).v;
    }

uint16_t f64tof16(uint64_t val, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f64_to_f16(float64_t{val}).v;
    }
uint32_t f64tof32(uint64_t val, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f64_to_f32(float64_t{val}).v;
}

// conversions: float to unsigned
uint32_t f16toui32(uint16_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f16_to_ui32(float16_t{v}, rm, true);
}
uint64_t f16toui64(uint16_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f16_to_ui64(float16_t{v}, rm, true);
}
uint32_t f32toui32(uint32_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f32_to_ui32(float32_t{v}, rm, true);
}
uint64_t f32toui64(uint32_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f32_to_ui64(float32_t{v}, rm, true);
}
uint32_t f64toui32(uint64_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f64_to_ui32(float64_t{v}, rm, true);
}
uint64_t f64toui64(uint64_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f64_to_ui64(float64_t{v}, rm, true);
}

// conversions: float to signed
uint32_t f16toi32(uint16_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f16_to_i32(float16_t{v}, rm, true);
}
uint64_t f16toi64(uint16_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f16_to_i64(float16_t{v}, rm, true);
}
uint32_t f32toi32(uint32_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f32_to_i32(float32_t{v}, rm, true);
}
uint64_t f32toi64(uint32_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f32_to_i64(float32_t{v}, rm, true);
}
uint32_t f64toi32(uint64_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f64_to_i32(float64_t{v}, rm, true);
}
uint64_t f64toi64(uint64_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return f64_to_i64(float64_t{v}, rm, true);
}

// conversions: unsigned to float
uint16_t ui32tof16(uint32_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return ui32_to_f16(v).v;
}
uint16_t ui64tof16(uint64_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return ui64_to_f16(v).v;
}
uint32_t ui32tof32(uint32_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return ui32_to_f32(v).v;
}
uint32_t ui64tof32(uint64_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return ui64_to_f32(v).v;
}
uint64_t ui32tof64(uint32_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return ui32_to_f64(v).v;
}
uint64_t ui64tof64(uint64_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return ui64_to_f64(v).v;
}

// conversions: signed to float
uint16_t i32tof16(uint32_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return i32_to_f16(v).v;
}
uint16_t i64tof16(uint64_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return i64_to_f16(v).v;
}
uint32_t i32tof32(uint32_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return i32_to_f32(v).v;
}
uint32_t i64tof32(uint64_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return i64_to_f32(v).v;
}
uint64_t i32tof64(uint32_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return i32_to_f64(v).v;
}
uint64_t i64tof64(uint64_t v, uint8_t rm) {
    softfloat_exceptionFlags = 0;
    softfloat_roundingMode = rm;
    return i64_to_f64(v).v;
}
}
