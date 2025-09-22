////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2025, MINRES Technologies GmbH
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
//       alex@minres.com - initial API and implementation
////////////////////////////////////////////////////////////////////////////////

#include "vector_functions.h"
#include "vm/aes_sbox.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <math.h>
#include <stdexcept>
#include <util/ities.h>
#include <vector>

namespace softvector {

vtype_t::vtype_t(uint32_t vtype_val) { underlying = (vtype_val & 0x8000) << 32 | (vtype_val & ~0x8000); }
vtype_t::vtype_t(uint64_t vtype_val) { underlying = vtype_val; }
bool vtype_t::vill() { return underlying >> 63; }
bool vtype_t::vma() { return (underlying >> 7) & 1; }
bool vtype_t::vta() { return (underlying >> 6) & 1; }
unsigned vtype_t::sew() {
    uint8_t vsew = (underlying >> 3) & 0b111;
    // pow(2, 3 + vsew);
    return 1 << (3 + vsew);
}
double vtype_t::lmul() {
    uint8_t vlmul = underlying & 0b111;
    assert(vlmul != 0b100); // reserved encoding
    int8_t signed_vlmul = (vlmul >> 2) ? 0b11111000 | vlmul : vlmul;
    return pow(2, signed_vlmul);
}

mask_bit_reference& mask_bit_reference::operator=(const bool new_value) {
    *start = *start & ~(1U << pos) | static_cast<unsigned>(new_value) << pos;
    return *this;
}

mask_bit_reference::mask_bit_reference(uint8_t* start, uint8_t pos)
: start(start)
, pos(pos) {
    assert(pos < 8 && "Bit reference can only be initialized for bytes");
};
mask_bit_reference::operator bool() const { return *(start) & (1U << (pos)); }

mask_bit_reference vmask_view::operator[](size_t idx) const {
    assert(idx < elem_count);
    return {start + idx / 8, static_cast<uint8_t>(idx % 8)};
}

vmask_view read_vmask(uint8_t* V, uint16_t VLEN, uint16_t elem_count, uint8_t reg_idx) {
    uint8_t* mask_start = V + VLEN / 8 * reg_idx;
    assert(mask_start + elem_count / 8 <= V + VLEN * RFS / 8);
    return {mask_start, elem_count};
}
uint8_t xt2(uint8_t x) { return (x << 1) ^ (bit_sub<7, 1>(x) ? 27 : 0); }

uint8_t xt3(uint8_t x) { return x ^ xt2(x); }

uint8_t gfmul(uint8_t x, uint8_t y) {
    return (bit_sub<0, 1>(y) ? x : 0) ^ (bit_sub<1, 1>(y) ? xt2(x) : 0) ^ (bit_sub<2, 1>(y) ? xt2(xt2(x)) : 0) ^
           (bit_sub<3, 1>(y) ? xt2(xt2(xt2(x))) : 0);
}

uint32_t aes_mixcolumn_byte_fwd(uint8_t so) {
    return ((uint32_t)gfmul(so, 3) << 24) | ((uint32_t)so << 16) | ((uint32_t)so << 8) | gfmul(so, 2);
}

uint32_t aes_mixcolumn_byte_inv(uint8_t so) {
    return ((uint32_t)gfmul(so, 11) << 24) | ((uint32_t)gfmul(so, 13) << 16) | ((uint32_t)gfmul(so, 9) << 8) | gfmul(so, 14);
}

uint32_t aes_mixcolumn_fwd(uint32_t x) {
    uint8_t s0 = bit_sub<0, 7 - 0 + 1>(x);
    uint8_t s1 = bit_sub<8, 15 - 8 + 1>(x);
    uint8_t s2 = bit_sub<16, 23 - 16 + 1>(x);
    uint8_t s3 = bit_sub<24, 31 - 24 + 1>(x);
    uint8_t b0 = xt2(s0) ^ xt3(s1) ^ (s2) ^ (s3);
    uint8_t b1 = (s0) ^ xt2(s1) ^ xt3(s2) ^ (s3);
    uint8_t b2 = (s0) ^ (s1) ^ xt2(s2) ^ xt3(s3);
    uint8_t b3 = xt3(s0) ^ (s1) ^ (s2) ^ xt2(s3);
    return ((uint32_t)b3 << 24) | ((uint32_t)b2 << 16) | ((uint32_t)b1 << 8) | b0;
}

uint32_t aes_mixcolumn_inv(uint32_t x) {
    uint8_t s0 = bit_sub<0, 7 - 0 + 1>(x);
    uint8_t s1 = bit_sub<8, 15 - 8 + 1>(x);
    uint8_t s2 = bit_sub<16, 23 - 16 + 1>(x);
    uint8_t s3 = bit_sub<24, 31 - 24 + 1>(x);
    uint8_t b0 = gfmul(s0, 14) ^ gfmul(s1, 11) ^ gfmul(s2, 13) ^ gfmul(s3, 9);
    uint8_t b1 = gfmul(s0, 9) ^ gfmul(s1, 14) ^ gfmul(s2, 11) ^ gfmul(s3, 13);
    uint8_t b2 = gfmul(s0, 13) ^ gfmul(s1, 9) ^ gfmul(s2, 14) ^ gfmul(s3, 11);
    uint8_t b3 = gfmul(s0, 11) ^ gfmul(s1, 13) ^ gfmul(s2, 9) ^ gfmul(s3, 14);
    return ((uint32_t)b3 << 24) | ((uint32_t)b2 << 16) | ((uint32_t)b1 << 8) | b0;
}

uint32_t aes_decode_rcon(uint8_t r) {
    switch(r) {
    case 0:
        return 1;
    case 1:
        return 2;
    case 2:
        return 4;
    case 3:
        return 8;
    case 4:
        return 16;
    case 5:
        return 32;
    case 6:
        return 64;
    case 7:
        return 128;
    case 8:
        return 27;
    case 9:
        return 54;
    }
    return 0;
}

uint32_t aes_subword_fwd(uint32_t x) {
    return ((uint32_t)aes_sbox_fwd(bit_sub<24, 31 - 24 + 1>(x)) << 24) | ((uint32_t)aes_sbox_fwd(bit_sub<16, 23 - 16 + 1>(x)) << 16) |
           ((uint32_t)aes_sbox_fwd(bit_sub<8, 15 - 8 + 1>(x)) << 8) | aes_sbox_fwd(bit_sub<0, 7 - 0 + 1>(x));
}

uint32_t aes_subword_inv(uint32_t x) {
    return ((uint32_t)aes_sbox_inv(bit_sub<24, 31 - 24 + 1>(x)) << 24) | ((uint32_t)aes_sbox_inv(bit_sub<16, 23 - 16 + 1>(x)) << 16) |
           ((uint32_t)aes_sbox_inv(bit_sub<8, 15 - 8 + 1>(x)) << 8) | aes_sbox_inv(bit_sub<0, 7 - 0 + 1>(x));
}

uint32_t aes_get_column(__uint128_t state, unsigned c) {
    assert(c < 4);
    return static_cast<uint32_t>(state >> (32 * c));
};

uint64_t aes_apply_fwd_sbox_to_each_byte(uint64_t x) {
    return ((uint64_t)aes_sbox_fwd(bit_sub<56, 63 - 56 + 1>(x)) << 56) | ((uint64_t)aes_sbox_fwd(bit_sub<48, 55 - 48 + 1>(x)) << 48) |
           ((uint64_t)aes_sbox_fwd(bit_sub<40, 47 - 40 + 1>(x)) << 40) | ((uint64_t)aes_sbox_fwd(bit_sub<32, 39 - 32 + 1>(x)) << 32) |
           ((uint64_t)aes_sbox_fwd(bit_sub<24, 31 - 24 + 1>(x)) << 24) | ((uint64_t)aes_sbox_fwd(bit_sub<16, 23 - 16 + 1>(x)) << 16) |
           ((uint64_t)aes_sbox_fwd(bit_sub<8, 15 - 8 + 1>(x)) << 8) | aes_sbox_fwd(bit_sub<0, 7 - 0 + 1>(x));
}

uint64_t aes_apply_inv_sbox_to_each_byte(uint64_t x) {
    return ((uint64_t)aes_sbox_inv(bit_sub<56, 63 - 56 + 1>(x)) << 56) | ((uint64_t)aes_sbox_inv(bit_sub<48, 55 - 48 + 1>(x)) << 48) |
           ((uint64_t)aes_sbox_inv(bit_sub<40, 47 - 40 + 1>(x)) << 40) | ((uint64_t)aes_sbox_inv(bit_sub<32, 39 - 32 + 1>(x)) << 32) |
           ((uint64_t)aes_sbox_inv(bit_sub<24, 31 - 24 + 1>(x)) << 24) | ((uint64_t)aes_sbox_inv(bit_sub<16, 23 - 16 + 1>(x)) << 16) |
           ((uint64_t)aes_sbox_inv(bit_sub<8, 15 - 8 + 1>(x)) << 8) | aes_sbox_inv(bit_sub<0, 7 - 0 + 1>(x));
}

uint64_t aes_rv64_shiftrows_fwd(uint64_t rs2, uint64_t rs1) {
    return ((uint64_t)bit_sub<24, 31 - 24 + 1>(rs1) << 56) | ((uint64_t)bit_sub<48, 55 - 48 + 1>(rs2) << 48) |
           ((uint64_t)bit_sub<8, 15 - 8 + 1>(rs2) << 40) | ((uint64_t)bit_sub<32, 39 - 32 + 1>(rs1) << 32) |
           ((uint64_t)bit_sub<56, 63 - 56 + 1>(rs2) << 24) | ((uint64_t)bit_sub<16, 23 - 16 + 1>(rs2) << 16) |
           ((uint64_t)bit_sub<40, 47 - 40 + 1>(rs1) << 8) | bit_sub<0, 7 - 0 + 1>(rs1);
}

uint64_t aes_rv64_shiftrows_inv(uint64_t rs2, uint64_t rs1) {
    return ((uint64_t)bit_sub<24, 31 - 24 + 1>(rs2) << 56) | ((uint64_t)bit_sub<48, 55 - 48 + 1>(rs2) << 48) |
           ((uint64_t)bit_sub<8, 15 - 8 + 1>(rs1) << 40) | ((uint64_t)bit_sub<32, 39 - 32 + 1>(rs1) << 32) |
           ((uint64_t)bit_sub<56, 63 - 56 + 1>(rs1) << 24) | ((uint64_t)bit_sub<16, 23 - 16 + 1>(rs2) << 16) |
           ((uint64_t)bit_sub<40, 47 - 40 + 1>(rs2) << 8) | bit_sub<0, 7 - 0 + 1>(rs1);
}

uint128_t aes_shift_rows_fwd(uint128_t x) {
    uint32_t ic3 = aes_get_column(x, 3);
    uint32_t ic2 = aes_get_column(x, 2);
    uint32_t ic1 = aes_get_column(x, 1);
    uint32_t ic0 = aes_get_column(x, 0);
    uint32_t oc0 = ((uint32_t)bit_sub<24, 31 - 24 + 1>(ic3) << 24) | ((uint32_t)bit_sub<16, 23 - 16 + 1>(ic2) << 16) |
                   ((uint32_t)bit_sub<8, 15 - 8 + 1>(ic1) << 8) | bit_sub<0, 7 - 0 + 1>(ic0);
    uint32_t oc1 = ((uint32_t)bit_sub<24, 31 - 24 + 1>(ic0) << 24) | ((uint32_t)bit_sub<16, 23 - 16 + 1>(ic3) << 16) |
                   ((uint32_t)bit_sub<8, 15 - 8 + 1>(ic2) << 8) | bit_sub<0, 7 - 0 + 1>(ic1);
    uint32_t oc2 = ((uint32_t)bit_sub<24, 31 - 24 + 1>(ic1) << 24) | ((uint32_t)bit_sub<16, 23 - 16 + 1>(ic0) << 16) |
                   ((uint32_t)bit_sub<8, 15 - 8 + 1>(ic3) << 8) | bit_sub<0, 7 - 0 + 1>(ic2);
    uint32_t oc3 = ((uint32_t)bit_sub<24, 31 - 24 + 1>(ic2) << 24) | ((uint32_t)bit_sub<16, 23 - 16 + 1>(ic1) << 16) |
                   ((uint32_t)bit_sub<8, 15 - 8 + 1>(ic0) << 8) | bit_sub<0, 7 - 0 + 1>(ic3);
    return ((uint128_t)oc3 << 96) | ((uint128_t)oc2 << 64) | ((uint128_t)oc1 << 32) | oc0;
}

uint128_t aes_shift_rows_inv(uint128_t x) {
    uint32_t ic3 = aes_get_column(x, 3);
    uint32_t ic2 = aes_get_column(x, 2);
    uint32_t ic1 = aes_get_column(x, 1);
    uint32_t ic0 = aes_get_column(x, 0);
    uint32_t oc0 = ((uint32_t)bit_sub<24, 31 - 24 + 1>(ic1) << 24) | ((uint32_t)bit_sub<16, 23 - 16 + 1>(ic2) << 16) |
                   ((uint32_t)bit_sub<8, 15 - 8 + 1>(ic3) << 8) | bit_sub<0, 7 - 0 + 1>(ic0);
    uint32_t oc1 = ((uint32_t)bit_sub<24, 31 - 24 + 1>(ic2) << 24) | ((uint32_t)bit_sub<16, 23 - 16 + 1>(ic3) << 16) |
                   ((uint32_t)bit_sub<8, 15 - 8 + 1>(ic0) << 8) | bit_sub<0, 7 - 0 + 1>(ic1);
    uint32_t oc2 = ((uint32_t)bit_sub<24, 31 - 24 + 1>(ic3) << 24) | ((uint32_t)bit_sub<16, 23 - 16 + 1>(ic0) << 16) |
                   ((uint32_t)bit_sub<8, 15 - 8 + 1>(ic1) << 8) | bit_sub<0, 7 - 0 + 1>(ic2);
    uint32_t oc3 = ((uint32_t)bit_sub<24, 31 - 24 + 1>(ic0) << 24) | ((uint32_t)bit_sub<16, 23 - 16 + 1>(ic1) << 16) |
                   ((uint32_t)bit_sub<8, 15 - 8 + 1>(ic2) << 8) | bit_sub<0, 7 - 0 + 1>(ic3);
    return ((uint128_t)oc3 << 96) | ((uint128_t)oc2 << 64) | ((uint128_t)oc1 << 32) | oc0;
}

uint128_t aes_subbytes_fwd(uint128_t x) {
    uint32_t oc0 = aes_subword_fwd(aes_get_column(x, 0));
    uint32_t oc1 = aes_subword_fwd(aes_get_column(x, 1));
    uint32_t oc2 = aes_subword_fwd(aes_get_column(x, 2));
    uint32_t oc3 = aes_subword_fwd(aes_get_column(x, 3));
    return ((uint128_t)oc3 << 96) | ((uint128_t)oc2 << 64) | ((uint128_t)oc1 << 32) | oc0;
}

uint128_t aes_subbytes_inv(uint128_t x) {
    uint32_t oc0 = aes_subword_inv(aes_get_column(x, 0));
    uint32_t oc1 = aes_subword_inv(aes_get_column(x, 1));
    uint32_t oc2 = aes_subword_inv(aes_get_column(x, 2));
    uint32_t oc3 = aes_subword_inv(aes_get_column(x, 3));
    return ((uint128_t)oc3 << 96) | ((uint128_t)oc2 << 64) | ((uint128_t)oc1 << 32) | oc0;
}

uint128_t aes_mixcolumns_fwd(uint128_t x) {
    uint32_t oc0 = aes_mixcolumn_fwd(aes_get_column(x, 0));
    uint32_t oc1 = aes_mixcolumn_fwd(aes_get_column(x, 1));
    uint32_t oc2 = aes_mixcolumn_fwd(aes_get_column(x, 2));
    uint32_t oc3 = aes_mixcolumn_fwd(aes_get_column(x, 3));
    return ((uint128_t)oc3 << 96) | ((uint128_t)oc2 << 64) | ((uint128_t)oc1 << 32) | oc0;
}

uint128_t aes_mixcolumns_inv(uint128_t x) {
    uint32_t oc0 = aes_mixcolumn_inv(aes_get_column(x, 0));
    uint32_t oc1 = aes_mixcolumn_inv(aes_get_column(x, 1));
    uint32_t oc2 = aes_mixcolumn_inv(aes_get_column(x, 2));
    uint32_t oc3 = aes_mixcolumn_inv(aes_get_column(x, 3));
    return ((uint128_t)oc3 << 96) | ((uint128_t)oc2 << 64) | ((uint128_t)oc1 << 32) | oc0;
}

uint32_t aes_rotword(uint32_t x) {
    uint8_t a0 = bit_sub<0, 7 - 0 + 1>(x);
    uint8_t a1 = bit_sub<8, 15 - 8 + 1>(x);
    uint8_t a2 = bit_sub<16, 23 - 16 + 1>(x);
    uint8_t a3 = bit_sub<24, 31 - 24 + 1>(x);
    return ((uint32_t)a0 << 24) | ((uint32_t)a3 << 16) | ((uint32_t)a2 << 8) | a1;
}

std::function<uint128_t(uint128_t, uint128_t, uint128_t)> get_crypto_funct(unsigned funct6, unsigned vs1) {
    switch(funct6) {
    case 0b101000: // VAES.VV
    case 0b101001: // VAES.VS
        switch(vs1) {
        case 0b00000: // VAESDM
            return [](uint128_t state, uint128_t rkey, uint128_t) {
                uint128_t sr = aes_shift_rows_inv(state);
                uint128_t sb = aes_subbytes_inv(sr);
                uint128_t ark = sb ^ rkey;
                uint128_t mix = aes_mixcolumns_inv(ark);
                return mix;
            };
        case 0b00001: // VAESDF
            return [](uint128_t state, uint128_t rkey, uint128_t) {
                uint128_t sr = aes_shift_rows_inv(state);
                uint128_t sb = aes_subbytes_inv(sr);
                uint128_t ark = sb ^ rkey;
                return ark;
            };
        case 0b00010: // VAESEM
            return [](uint128_t state, uint128_t rkey, uint128_t) {
                uint128_t sb = aes_subbytes_fwd(state);
                uint128_t sr = aes_shift_rows_fwd(sb);
                uint128_t mix = aes_mixcolumns_fwd(sr);
                uint128_t ark = mix ^ rkey;
                return ark;
            };
        case 0b00011: // VAESEF
            return [](uint128_t state, uint128_t rkey, uint128_t) {
                uint128_t sb = aes_subbytes_fwd(state);
                uint128_t sr = aes_shift_rows_fwd(sb);
                uint128_t ark = sr ^ rkey;
                return ark;
            };
        case 0b00111: // VAESZ
            return [](uint128_t state, uint128_t rkey, uint128_t) {
                uint128_t ark = state ^ rkey;
                return ark;
            };
        case 0b10000: // VSM4R
            throw new std::runtime_error("Unsupported operation in get_crypto_funct");
        case 0b10001: // VGMUL
            return [](uint128_t vd, uint128_t vs2, uint128_t) {
                uint128_t Y = brev8<uint128_t>(vd);
                uint128_t H = brev8<uint128_t>(vs2);
                uint128_t Z = 0;
                for(size_t bit = 0; bit < 128; bit++) {
                    if((Y >> bit) & 1)
                        Z ^= H;
                    bool reduce = (H >> 127) & 1;
                    H = H << 1;
                    if(reduce)
                        H ^= 0x87;
                }
                uint128_t result = brev8<uint128_t>(Z);
                return result;
            };
        default:
            throw new std::runtime_error("Unsupported operation in get_crypto_funct");
        }
    case 0b100000: // VSM3ME
    case 0b100001: // VSM4K
        throw new std::runtime_error("Unsupported operation in get_crypto_funct");
    case 0b100010: // VAESKF1
        return [](uint128_t vd, uint128_t vs2, uint128_t r) {
            auto extract_word = [](const uint128_t& value, int index) -> uint32_t {
                return static_cast<uint32_t>((value >> (32 * index)) & std::numeric_limits<uint32_t>::max());
            };

            uint32_t k0 = (vs2 >> 32 * 0) & std::numeric_limits<uint32_t>::max();
            uint32_t k1 = (vs2 >> 32 * 1) & std::numeric_limits<uint32_t>::max();
            uint32_t k2 = (vs2 >> 32 * 2) & std::numeric_limits<uint32_t>::max();
            uint32_t k3 = (vs2 >> 32 * 3) & std::numeric_limits<uint32_t>::max();
            uint32_t w0 = aes_subword_fwd(aes_rotword(k3)) ^ aes_decode_rcon(r) ^ k0;
            uint32_t w1 = w0 ^ k1;
            uint32_t w2 = w1 ^ k2;
            uint32_t w3 = w2 ^ k3;
            uint128_t result = (uint128_t(w3) << 96) | (uint128_t(w2) << 64) | (uint128_t(w1) << 32) | (uint128_t(w0));
            return result;
        };
    case 0b101010: // VAESKF2
        return [](uint128_t vd, uint128_t vs2, uint128_t r) {
            uint32_t k0 = (vs2 >> 32 * 0) & std::numeric_limits<uint32_t>::max();
            uint32_t k1 = (vs2 >> 32 * 1) & std::numeric_limits<uint32_t>::max();
            uint32_t k2 = (vs2 >> 32 * 2) & std::numeric_limits<uint32_t>::max();
            uint32_t k3 = (vs2 >> 32 * 3) & std::numeric_limits<uint32_t>::max();
            uint32_t rkb0 = (vd >> 32 * 0) & std::numeric_limits<uint32_t>::max();
            uint32_t rkb1 = (vd >> 32 * 1) & std::numeric_limits<uint32_t>::max();
            uint32_t rkb2 = (vd >> 32 * 2) & std::numeric_limits<uint32_t>::max();
            uint32_t rkb3 = (vd >> 32 * 3) & std::numeric_limits<uint32_t>::max();
            uint32_t w0 = r & 1 ? aes_subword_fwd(k3) ^ rkb0 : aes_subword_fwd(aes_rotword(k3)) ^ aes_decode_rcon((r >> 1) - 1) ^ rkb0;
            uint32_t w1 = w0 ^ rkb1;
            uint32_t w2 = w1 ^ rkb2;
            uint32_t w3 = w2 ^ rkb3;
            uint128_t result = (uint128_t(w3) << 96) | (uint128_t(w2) << 64) | (uint128_t(w1) << 32) | (uint128_t(w0));
            return result;
        };
    case 0b101011: // VSM3C
        throw new std::runtime_error("Unsupported operation in get_crypto_funct");
    case 0b101100: // VGHSH
        return [](uint128_t Y, uint128_t vs2, uint128_t X) {
            auto H = brev8<uint128_t>(vs2);
            uint128_t Z = 0;
            uint128_t S = brev8<uint128_t>(Y ^ X);
            for(size_t bit = 0; bit < 128; bit++) {
                if((S >> bit) & 1)
                    Z ^= H;
                bool reduce = (H >> 127) & 1;
                H = H << 1;
                if(reduce)
                    H ^= 0x87;
            }
            uint128_t result = brev8<uint128_t>(Z);
            return result;
        };
    case 0b101101: // VSHA2MS
    case 0b101110: // VSHA2CH
    case 0b101111: // VSHA2CL
    default:
        throw new std::runtime_error("Unknown funct6 in get_crypto_funct");
    }
}
} // namespace softvector