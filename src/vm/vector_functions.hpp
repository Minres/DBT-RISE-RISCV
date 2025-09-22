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
#pragma once
extern "C" {
#include <softfloat.h>
}
#include "softfloat_types.h"
#include "specialize.h"
#include "vm/fp_functions.h"
#include "vm/vector_functions.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#ifndef _VM_VECTOR_FUNCTIONS_H_
#error __FILE__ should only be included from vector_functions.h
#endif
#include <math.h>

#ifdef __SIZEOF_INT128__
template <> struct std::make_signed<__uint128_t> { using type = __int128_t; };
template <> struct std::make_signed<__int128_t> { using type = __int128_t; };
// helper struct to make calling twice<T> on 128-bit datatypes legal at compile time
struct poison128_t {
    poison128_t() { throw std::runtime_error("Attempt to use twice<__uint128_t>::type at runtime"); }
    poison128_t(const poison128_t&) { throw std::runtime_error("Copy of poison128_t is not allowed"); }
    template <typename U> poison128_t(U) { throw std::runtime_error("Conversion to poison128_t is not allowed"); }
    operator __uint128_t() const { throw std::runtime_error("Use of poison128_t as __uint128_t is not allowed"); }
};
template <> struct std::make_signed<poison128_t> { using type = poison128_t; };
#endif

namespace softvector {

template <typename elem_t> struct vreg_view {
    uint8_t* start;
    size_t vlmax;
    elem_t& operator[](size_t idx) {
        assert(idx < vlmax);
        return *(reinterpret_cast<elem_t*>(start) + idx);
    }
};
template <unsigned VLEN, typename elem_t> vreg_view<elem_t> get_vreg(uint8_t* V, uint8_t reg_idx, uint16_t vlmax) {
    assert(V + vlmax * sizeof(elem_t) <= V + VLEN * RFS / 8);
    return {V + VLEN / 8 * reg_idx, vlmax};
}
template <unsigned VLEN> vmask_view read_vmask(uint8_t* V, uint16_t vlmax, uint8_t reg_idx) {
    uint8_t* mask_start = V + VLEN / 8 * reg_idx;
    assert(mask_start + vlmax / 8 <= V + VLEN * RFS / 8);
    return {mask_start, vlmax};
}

template <typename elem_t> constexpr elem_t shift_mask() {
    static_assert(std::numeric_limits<elem_t>::is_integer, "shift_mask only supports integer types");
    return std::numeric_limits<elem_t>::digits - 1;
}
template <typename T> constexpr T agnostic_behavior(T val) {
#ifdef AGNOSTIC_ONES
    return std::numeric_limits<T>::max();
#else
    return val;
#endif
}

enum FUNCT3 {
    OPIVV = 0b000,
    OPFVV = 0b001,
    OPMVV = 0b010,
    OPIVI = 0b011,
    OPIVX = 0b100,
    OPFVF = 0b101,
    OPMVX = 0b110,
};
template <class, typename enable = void> struct twice;
template <> struct twice<int8_t> { using type = int16_t; };
template <> struct twice<uint8_t> { using type = uint16_t; };
template <> struct twice<int16_t> { using type = int32_t; };
template <> struct twice<uint16_t> { using type = uint32_t; };
template <> struct twice<int32_t> { using type = int64_t; };
template <> struct twice<uint32_t> { using type = uint64_t; };
#ifdef __SIZEOF_INT128__
template <> struct twice<int64_t> { using type = __int128_t; };
template <> struct twice<uint64_t> { using type = __uint128_t; };

template <> struct twice<__uint128_t> { using type = poison128_t; };
template <> struct twice<__int128_t> { using type = poison128_t; };

#endif
template <class T> using twice_t = typename twice<T>::type; // for convenience
template <typename TO, typename FROM> constexpr TO sext(FROM val) {
    return static_cast<std::make_signed_t<TO>>(static_cast<std::make_signed_t<FROM>>(val));
};

template <unsigned VLEN, typename eew_t>
uint64_t vector_load_store(void* core, std::function<bool(void*, uint64_t, uint64_t, uint8_t*)> load_store_fn, uint8_t* V, uint64_t vl,
                           uint64_t vstart, vtype_t vtype, bool vm, uint8_t vd, uint64_t rs1, uint8_t segment_size, int64_t stride,
                           bool use_stride) {
    unsigned vlmax = VLEN * vtype.lmul() / vtype.sew();
    auto emul_stride = std::max<unsigned>(vlmax, VLEN / (sizeof(eew_t) * 8));
    auto vd_view = get_vreg<VLEN, eew_t>(V, vd, emul_stride * segment_size);
    vmask_view mask_reg = read_vmask(V, VLEN, vlmax);
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active) {
            signed stride_offset = stride * idx;
            auto seg_offset = use_stride ? 0 : segment_size * sizeof(eew_t) * idx;
            for(size_t s_idx = 0; s_idx < segment_size; s_idx++) {
                eew_t* addressed_elem = &vd_view[idx + emul_stride * s_idx];
                uint64_t addr = rs1 + stride_offset + seg_offset + s_idx * sizeof(eew_t);
                if(!load_store_fn(core, addr, sizeof(eew_t), reinterpret_cast<uint8_t*>(addressed_elem)))
                    return idx;
            }
        } else if(vtype.vma())
            for(size_t s_idx = 0; s_idx < segment_size; s_idx++)
                vd_view[idx + emul_stride * s_idx] = agnostic_behavior(vd_view[idx + emul_stride * s_idx]);
    }
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++)
            for(size_t s_idx = 0; s_idx < segment_size; s_idx++)
                vd_view[idx + emul_stride * s_idx] = agnostic_behavior(vd_view[idx + emul_stride * s_idx]);
    return 0;
}
// eew for index registers, sew for data register
template <unsigned XLEN, unsigned VLEN, typename eew_t, typename sew_t>
uint64_t vector_load_store_index(void* core, std::function<bool(void*, uint64_t, uint64_t, uint8_t*)> load_store_fn, uint8_t* V,
                                 uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, uint8_t vd, uint64_t rs1, uint8_t vs2,
                                 uint8_t segment_size) {
    // All load stores are ordered in this implementation
    unsigned vlmax = VLEN * vtype.lmul() / vtype.sew();
    auto emul_stride = std::max<unsigned>(vlmax, VLEN / (sizeof(sew_t) * 8));
    auto vd_view = get_vreg<VLEN, sew_t>(V, vd, emul_stride * segment_size);
    auto vs2_view = get_vreg<VLEN, eew_t>(V, vs2, vlmax);
    vmask_view mask_reg = read_vmask(V, VLEN, vlmax);
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active) {
            uint64_t index_offset = vs2_view[idx] & std::numeric_limits<std::conditional_t<XLEN == 32, uint32_t, uint64_t>>::max();
            for(size_t s_idx = 0; s_idx < segment_size; s_idx++) {
                sew_t* addressed_elem = &vd_view[idx + emul_stride * s_idx];
                uint64_t addr = rs1 + index_offset + s_idx * sizeof(sew_t);
                if(!load_store_fn(core, addr, sizeof(sew_t), reinterpret_cast<uint8_t*>(addressed_elem)))
                    return idx;
            }
        } else if(vtype.vma())
            for(size_t s_idx = 0; s_idx < segment_size; s_idx++)
                vd_view[idx + emul_stride * s_idx] = agnostic_behavior(vd_view[idx + emul_stride * s_idx]);
    }
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++)
            for(size_t s_idx = 0; s_idx < segment_size; s_idx++)
                vd_view[idx + emul_stride * s_idx] = agnostic_behavior(vd_view[idx + emul_stride * s_idx]);
    return 0;
}
template <typename dest_elem_t, typename src2_elem_t = dest_elem_t, typename src1_elem_t = dest_elem_t>
std::function<dest_elem_t(dest_elem_t, src2_elem_t, src1_elem_t)> get_funct(unsigned funct6, unsigned funct3) {
    if(funct3 == OPIVV || funct3 == OPIVX || funct3 == OPIVI)
        switch(funct6) {
        case 0b000000: // VADD
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return vs2 + vs1; };
        case 0b000001: // VANDN
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return vs2 & ~vs1; };
        case 0b000010: // VSUB
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return vs2 - vs1; };
        case 0b000011: // VRSUB
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return vs1 - vs2; };
        case 0b000100: // VMINU
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return std::min<src2_elem_t>(vs2, vs1); };
        case 0b000101: // VMIN
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return std::min<std::make_signed_t<src2_elem_t>>(vs2, vs1); };
        case 0b000110: // VMAXU
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return std::max<src2_elem_t>(vs2, vs1); };
        case 0b000111: // VMAX
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return std::max<std::make_signed_t<src2_elem_t>>(vs2, vs1); };
        case 0b001001: // VAND
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return vs1 & vs2; };
        case 0b001010: // VOR
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return vs1 | vs2; };
        case 0b001011: // VXOR
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return vs1 ^ vs2; };
        case 0b010000: // VADC
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return vs2 + vs1; };
        case 0b010010: // VSBC
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return static_cast<std::make_signed_t<dest_elem_t>>(static_cast<std::make_signed_t<src2_elem_t>>(vs2) -
                                                                    static_cast<std::make_signed_t<src1_elem_t>>(vs1));
            };
        case 0b010100: // VROR
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                constexpr dest_elem_t bits = sizeof(src2_elem_t) * 8;
                auto shamt = vs1 & shift_mask<src1_elem_t>();
                return (vs2 >> shamt) | (vs2 << (bits - shamt));
            };
        case 0b010101: { // VROL
            if(funct3 == OPIVI)
                return get_funct<dest_elem_t>(0b010100, funct3);
            else
                return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                    constexpr dest_elem_t bits = sizeof(src2_elem_t) * 8;
                    auto shamt = vs1 & shift_mask<src1_elem_t>();
                    return (vs2 << shamt) | (vs2 >> (bits - shamt));
                };
        }
        case 0b100101: // VSLL
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return vs2 << (vs1 & shift_mask<src2_elem_t>()); };
        case 0b101000: // VSRL
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return vs2 >> (vs1 & shift_mask<src2_elem_t>()); };
        case 0b101001: // VSRA
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return static_cast<std::make_signed_t<src2_elem_t>>(vs2) >> (vs1 & shift_mask<src2_elem_t>());
            };
        case 0b101100: // VNSRL
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return vs2 >> (vs1 & shift_mask<src2_elem_t>()); };
        case 0b101101: // VNSRA
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return static_cast<std::make_signed_t<src2_elem_t>>(vs2) >> (vs1 & shift_mask<src2_elem_t>());
            };
        case 0b110101: // VWSLL
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return static_cast<dest_elem_t>(vs2) << (vs1 & (shift_mask<dest_elem_t>()));
            };
        default:
            throw new std::runtime_error("Unknown funct6 in get_funct");
        }
    else if(funct3 == OPMVV || funct3 == OPMVX)
        switch(funct6) {
        case 0b100000: // VDIVU
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) -> dest_elem_t {
                if(vs1 == 0)
                    return -1;
                else
                    return vs2 / vs1;
            };
        case 0b100001: // VDIV
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) -> dest_elem_t {
                if(vs1 == 0)
                    return -1;
                else if(vs2 == std::numeric_limits<std::make_signed_t<src2_elem_t>>::min() &&
                        static_cast<std::make_signed_t<src1_elem_t>>(vs1) == -1)
                    return vs2;
                else
                    return static_cast<std::make_signed_t<src2_elem_t>>(vs2) / static_cast<std::make_signed_t<src1_elem_t>>(vs1);
            };
        case 0b100010: // VREMU
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) -> dest_elem_t {
                if(vs1 == 0)
                    return vs2;
                else
                    return vs2 % vs1;
            };
        case 0b100011: // VREM
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) -> dest_elem_t {
                if(vs1 == 0)
                    return vs2;
                else if(vs2 == std::numeric_limits<std::make_signed_t<src2_elem_t>>::min() &&
                        static_cast<std::make_signed_t<src1_elem_t>>(vs1) == -1)
                    return 0;
                else
                    return static_cast<std::make_signed_t<src2_elem_t>>(vs2) % static_cast<std::make_signed_t<src1_elem_t>>(vs1);
            };
        case 0b100100: // VMULHU
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return (static_cast<twice_t<src2_elem_t>>(vs2) * static_cast<twice_t<src2_elem_t>>(vs1)) >> sizeof(dest_elem_t) * 8;
            };
        case 0b100101: // VMUL
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return static_cast<std::make_signed_t<src2_elem_t>>(vs2) * static_cast<std::make_signed_t<src1_elem_t>>(vs1);
            };
        case 0b100110: // VMULHSU
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return (sext<twice_t<src2_elem_t>>(vs2) * static_cast<twice_t<src2_elem_t>>(vs1)) >> sizeof(dest_elem_t) * 8;
            };
        case 0b100111: // VMULH
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return (sext<twice_t<src2_elem_t>>(vs2) * sext<twice_t<src1_elem_t>>(vs1)) >> sizeof(dest_elem_t) * 8;
            };
        case 0b101001: // VMADD
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return vs1 * vd + vs2; };
        case 0b101011: // VNMSUB
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return -1 * (vs1 * vd) + vs2; };
        case 0b101101: // VMACC
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return vs1 * vs2 + vd; };
        case 0b101111: // VNMSAC
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return -1 * (vs1 * vs2) + vd; };
        case 0b110000: // VWADDU
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return static_cast<dest_elem_t>(vs2) + static_cast<dest_elem_t>(vs1);
            };
        case 0b110001: // VWADD
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return sext<dest_elem_t>(vs2) + sext<dest_elem_t>(vs1); };
        case 0b110010: // VWSUBU
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return static_cast<dest_elem_t>(vs2) - static_cast<dest_elem_t>(vs1);
            };
        case 0b110011: // VWSUB
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return sext<dest_elem_t>(vs2) - sext<dest_elem_t>(vs1); };
        case 0b110100: // VWADDU.W
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return static_cast<dest_elem_t>(vs2) + static_cast<dest_elem_t>(vs1);
            };
        case 0b110101: // VWADD.W
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return sext<dest_elem_t>(vs2) + sext<dest_elem_t>(vs1); };
        case 0b110110: // VWSUBU.W
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return static_cast<dest_elem_t>(vs2) - static_cast<dest_elem_t>(vs1);
            };
        case 0b110111: // VWSUB.W
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return sext<dest_elem_t>(vs2) - sext<dest_elem_t>(vs1); };
        case 0b111000: // VWMULU
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return (static_cast<dest_elem_t>(vs2) * static_cast<dest_elem_t>(vs1));
            };
        case 0b111010: // VWMULSU
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return sext<dest_elem_t>(vs2) * static_cast<dest_elem_t>(vs1); };
        case 0b111011: // VWMUL
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return sext<dest_elem_t>(vs2) * sext<dest_elem_t>(vs1); };
        case 0b111100: // VWMACCU
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return static_cast<dest_elem_t>(vs1) * static_cast<dest_elem_t>(vs2) + vd;
            };
        case 0b111101: // VWMACC
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) { return sext<dest_elem_t>(vs1) * sext<dest_elem_t>(vs2) + vd; };
        case 0b111110: // VWMACCUS
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return static_cast<dest_elem_t>(vs1) * sext<dest_elem_t>(vs2) + vd;
            };
        case 0b111111: // VWMACCSU
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return sext<dest_elem_t>(vs1) * static_cast<dest_elem_t>(vs2) + vd;
            };
        case 0b001100: // VCLMUL
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t output = 0;
                for(size_t i = 0; i <= sizeof(dest_elem_t) * 8 - 1; i++) {
                    if((vs2 >> i) & 1)
                        output = output ^ (vs1 << i);
                }
                return output;
            };
        case 0b001101: // VCLMULH
            return [](dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t output = 0;
                for(size_t i = 1; i < sizeof(dest_elem_t) * 8; i++) {
                    if((vs2 >> i) & 1)
                        output = output ^ (vs1 >> (sizeof(dest_elem_t) * 8 - i));
                }
                return output;
            };
        default:
            throw new std::runtime_error("Unknown funct6 in get_funct");
        }
    else
        throw new std::runtime_error("Unknown funct3 in get_funct");
}
template <unsigned VLEN, typename dest_elem_t, typename src2_elem_t, typename src1_elem_t>
void vector_vector_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd,
                      unsigned vs2, unsigned vs1) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs1_view = get_vreg<VLEN, src1_elem_t>(V, vs1, vlmax);
    auto vs2_view = get_vreg<VLEN, src2_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, dest_elem_t>(V, vd, vlmax);
    auto fn = get_funct<dest_elem_t, src2_elem_t, src1_elem_t>(funct6, funct3);
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_view[idx] = fn(vd_view[idx], vs2_view[idx], vs1_view[idx]);
        else if(vtype.vma())
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
    }
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}
template <unsigned VLEN, typename dest_elem_t, typename src2_elem_t, typename src1_elem_t>
void vector_imm_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd,
                   unsigned vs2, typename std::make_signed<src1_elem_t>::type imm) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs2_view = get_vreg<VLEN, src2_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, dest_elem_t>(V, vd, vlmax);
    auto fn = get_funct<dest_elem_t, src2_elem_t, src1_elem_t>(funct6, funct3);
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_view[idx] = fn(vd_view[idx], vs2_view[idx], imm);
        else if(vtype.vma())
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
    }
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}
template <unsigned VLEN, typename elem_t>
void vector_vector_carry(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, unsigned vd,
                         unsigned vs2, unsigned vs1, signed carry) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs1_view = get_vreg<VLEN, elem_t>(V, vs1, vlmax);
    auto vs2_view = get_vreg<VLEN, elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, elem_t>(V, vd, vlmax);
    auto fn = get_funct<elem_t, elem_t, elem_t>(funct6, funct3);
    for(size_t idx = vstart; idx < vl; idx++)
        vd_view[idx] = fn(vd_view[idx], vs2_view[idx], vs1_view[idx]) + carry * mask_reg[idx];
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}
template <unsigned VLEN, typename elem_t>
void vector_imm_carry(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, unsigned vd, unsigned vs2,
                      typename std::make_signed<elem_t>::type imm, signed carry) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs2_view = get_vreg<VLEN, elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, elem_t>(V, vd, vlmax);
    auto fn = get_funct<elem_t, elem_t, elem_t>(funct6, funct3);
    for(size_t idx = vstart; idx < vl; idx++)
        vd_view[idx] = fn(vd_view[idx], vs2_view[idx], imm) + carry * mask_reg[idx];
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}
template <unsigned VLEN, typename scr_elem_t>
void vector_vector_merge(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2, unsigned vs1) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs1_view = get_vreg<VLEN, scr_elem_t>(V, vs1, vlmax);
    auto vs2_view = get_vreg<VLEN, scr_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, scr_elem_t>(V, vd, vlmax);
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_view[idx] = vs1_view[idx];
        else
            vd_view[idx] = vs2_view[idx];
    }
}
template <unsigned VLEN, typename scr_elem_t>
void vector_imm_merge(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs2_view = get_vreg<VLEN, scr_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, scr_elem_t>(V, vd, vlmax);
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_view[idx] = imm;
        else
            vd_view[idx] = vs2_view[idx];
    }
}
template <typename elem_t> std::function<bool(elem_t, elem_t)> get_mask_funct(unsigned funct6, unsigned funct3) {
    if(funct3 == OPIVV || funct3 == OPIVX || funct3 == OPIVI)
        switch(funct6) {
        case 0b011000: // VMSEQ
            return [](elem_t vs2, elem_t vs1) { return vs2 == vs1; };
        case 0b011001: // VMSNE
            return [](elem_t vs2, elem_t vs1) { return vs2 != vs1; };
        case 0b011010: // VMSLTU
            return [](elem_t vs2, elem_t vs1) { return vs2 < vs1; };
        case 0b011011: // VMSLT
            return [](elem_t vs2, elem_t vs1) {
                return static_cast<std::make_signed_t<elem_t>>(vs2) < static_cast<std::make_signed_t<elem_t>>(vs1);
            };
        case 0b011100: // VMSLEU
            return [](elem_t vs2, elem_t vs1) { return vs2 <= vs1; };
        case 0b011101: // VMSLE
            return [](elem_t vs2, elem_t vs1) {
                return static_cast<std::make_signed_t<elem_t>>(vs2) <= static_cast<std::make_signed_t<elem_t>>(vs1);
            };
        case 0b011110: // VMSGTU
            return [](elem_t vs2, elem_t vs1) { return vs2 > vs1; };
        case 0b011111: // VMSGT
            return [](elem_t vs2, elem_t vs1) {
                return static_cast<std::make_signed_t<elem_t>>(vs2) > static_cast<std::make_signed_t<elem_t>>(vs1);
            };

        default:
            throw new std::runtime_error("Unknown funct6 in get_mask_funct");
        }
    else if(funct3 == OPMVV || funct3 == OPMVX)
        switch(funct6) {
        case 0b011000: // VMANDN
            return [](elem_t vs2, elem_t vs1) { return vs2 & !vs1; };
        case 0b011001: // VMAND
            return [](elem_t vs2, elem_t vs1) { return vs2 & vs1; };
        case 0b011010: // VMOR
            return [](elem_t vs2, elem_t vs1) { return vs2 | vs1; };
        case 0b011011: // VMXOR
            return [](elem_t vs2, elem_t vs1) { return vs2 ^ vs1; };
        case 0b011100: // VMORN
            return [](elem_t vs2, elem_t vs1) { return vs2 | !vs1; };
        case 0b011101: // VMNAND
            return [](elem_t vs2, elem_t vs1) { return !(vs2 & vs1); };
        case 0b011110: // VMNOR
            return [](elem_t vs2, elem_t vs1) { return !(vs2 | vs1); };
        case 0b011111: // VMXNOR
            return [](elem_t vs2, elem_t vs1) { return !(vs2 ^ vs1); };
        default:
            throw new std::runtime_error("Unknown funct6 in get_mask_funct");
        }
    else
        throw new std::runtime_error("Unknown funct3 in get_mask_funct");
}
template <unsigned VLEN, typename elem_t>
void mask_vector_vector_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd,
                           unsigned vs2, unsigned vs1) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs1_view = get_vreg<VLEN, elem_t>(V, vs1, vlmax);
    auto vs2_view = get_vreg<VLEN, elem_t>(V, vs2, vlmax);
    vmask_view vd_mask_view = read_vmask<VLEN>(V, VLEN, vd);
    auto fn = get_mask_funct<elem_t>(funct6, funct3);
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_mask_view[idx] = fn(vs2_view[idx], vs1_view[idx]);
        else if(vtype.vma())
            vd_mask_view[idx] = agnostic_behavior(vd_mask_view[idx]);
    }
    if(vtype.vta())
        for(size_t idx = vl; idx < VLEN; idx++)
            vd_mask_view[idx] = agnostic_behavior(vd_mask_view[idx]);
}
template <unsigned VLEN, typename elem_t>
void mask_vector_imm_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd,
                        unsigned vs2, typename std::make_signed<elem_t>::type imm) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs2_view = get_vreg<VLEN, elem_t>(V, vs2, vlmax);
    vmask_view vd_mask_view = read_vmask<VLEN>(V, VLEN, vd);
    auto fn = get_mask_funct<elem_t>(funct6, funct3);
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_mask_view[idx] = fn(vs2_view[idx], imm);
        else if(vtype.vma())
            vd_mask_view[idx] = agnostic_behavior(vd_mask_view[idx]);
    }
    if(vtype.vta())
        for(size_t idx = vl; idx < VLEN; idx++)
            vd_mask_view[idx] = agnostic_behavior(vd_mask_view[idx]);
}
template <typename dest_elem_t, typename src2_elem_t = dest_elem_t>
std::function<dest_elem_t(src2_elem_t)> get_unary_fn(unsigned unary_op) {
    switch(unary_op) {
    case 0b00111: // VSEXT.VF2
    case 0b00101: // VSEXT.VF4
    case 0b00011: // VSEXT.VF8
        return [](src2_elem_t vs2) { return static_cast<std::make_signed_t<src2_elem_t>>(vs2); };
    case 0b00110: // VZEXT.VF2
    case 0b00100: // VZEXT.VF4
    case 0b00010: // VZEXT.VF8
        return [](src2_elem_t vs2) { return vs2; };
    case 0b01000: // VBREV8
        return [](src2_elem_t vs2) { return brev8<dest_elem_t>(vs2); };
    case 0b01001: // VREV8
        return [](src2_elem_t vs2) {
            constexpr unsigned byte_count = sizeof(src2_elem_t);
            dest_elem_t result = 0;
            for(size_t i = 0; i < byte_count; ++i) {
                result <<= 8;
                result |= (vs2 & 0xFF);
                vs2 >>= 8;
            }
            return result;
        };
    case 0b01010: // VBREV
        return [](src2_elem_t vs2) { return brev<dest_elem_t>(vs2); };
    case 0b01100: // VCLZ
        return [](src2_elem_t vs2) {
            if(std::is_same_v<src2_elem_t, unsigned int>)
                return static_cast<dest_elem_t>(__builtin_clz(vs2));
            else if(std::is_same_v<src2_elem_t, unsigned long>)
                return static_cast<dest_elem_t>(__builtin_clzl(vs2));
            else if(std::is_same_v<src2_elem_t, unsigned long long>)
                return static_cast<dest_elem_t>(__builtin_clzll(vs2));
            else {
                constexpr dest_elem_t bits = sizeof(src2_elem_t) * 8;
                if(vs2 == 0)
                    return bits;
                dest_elem_t count = 0;
                for(size_t i = bits - 1; i >= 0; --i) {
                    if((vs2 >> i) & 1)
                        break;
                    ++count;
                }
                return count;
            }
        };
    case 0b01101: // VCTZ
        return [](src2_elem_t vs2) {
            if(std::is_same_v<src2_elem_t, unsigned int>)
                return static_cast<dest_elem_t>(__builtin_ctz(vs2));
            else if(std::is_same_v<src2_elem_t, unsigned long>)
                return static_cast<dest_elem_t>(__builtin_ctzl(vs2));
            else if(std::is_same_v<src2_elem_t, unsigned long long>)
                return static_cast<dest_elem_t>(__builtin_ctzll(vs2));
            else {
                constexpr dest_elem_t bits = sizeof(src2_elem_t) * 8;
                if(vs2 == 0)
                    return bits;
                dest_elem_t count = 0;
                while((vs2 & 1) == 0) {
                    ++count;
                    vs2 >>= 1;
                }
                return count;
            }
        };
    case 0b01110: // VCPOP
        return [](src2_elem_t vs2) {
            if(std::is_same_v<src2_elem_t, unsigned int>)
                return static_cast<dest_elem_t>(__builtin_popcount(vs2));
            else if(std::is_same_v<src2_elem_t, unsigned long>)
                return static_cast<dest_elem_t>(__builtin_popcountl(vs2));
            else if(std::is_same_v<src2_elem_t, unsigned long long>)
                return static_cast<dest_elem_t>(__builtin_popcountll(vs2));
            else {
                dest_elem_t count = 0;
                while(vs2) {
                    count += vs2 & 1;
                    vs2 >>= 1;
                }
                return count;
            }
        };
    default:
        throw new std::runtime_error("Unknown funct in get_unary_fn");
    }
}
template <unsigned VLEN, typename dest_elem_t, typename src2_elem_t>
void vector_unary_op(uint8_t* V, unsigned unary_op, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs2_view = get_vreg<VLEN, src2_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, dest_elem_t>(V, vd, vlmax);
    auto fn = get_unary_fn<dest_elem_t, src2_elem_t>(unary_op);
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_view[idx] = fn(vs2_view[idx]);
        else if(vtype.vma())
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
    }
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}
template <typename elem_t> std::function<bool(elem_t, elem_t, elem_t)> get_carry_funct(unsigned funct) {
    switch(funct) {
    case 0b010001: // VMADC
        return [](elem_t vs2, elem_t vs1, elem_t carry) {
            return static_cast<elem_t>(vs2 + vs1 + carry) < std::max(vs1, vs2) || static_cast<elem_t>(vs2 + vs1) < std::max(vs1, vs2);
        };
    case 0b010011: // VMSBC
        return [](elem_t vs2, elem_t vs1, elem_t carry) {
            return vs2 < static_cast<elem_t>(vs1 + carry) || (vs1 == std::numeric_limits<elem_t>::max() && carry);
        };
    default:
        throw new std::runtime_error("Unknown funct in get_carry_funct");
    }
}
template <unsigned VLEN, typename elem_t>
void carry_vector_vector_op(uint8_t* V, unsigned funct, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                            unsigned vs1) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs1_view = get_vreg<VLEN, elem_t>(V, vs1, vlmax);
    auto vs2_view = get_vreg<VLEN, elem_t>(V, vs2, vlmax);
    vmask_view vd_mask_view = read_vmask<VLEN>(V, vlmax, vd);
    auto fn = get_carry_funct<elem_t>(funct);
    for(size_t idx = vstart; idx < vl; idx++) {
        elem_t carry = vm ? 0 : mask_reg[idx];
        vd_mask_view[idx] = fn(vs2_view[idx], vs1_view[idx], carry);
    }
    for(size_t idx = vl; idx < vlmax; idx++)
        vd_mask_view[idx] = agnostic_behavior(vd_mask_view[idx]);
}
template <unsigned VLEN, typename elem_t>
void carry_vector_imm_op(uint8_t* V, unsigned funct, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                         typename std::make_signed<elem_t>::type imm) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs2_view = get_vreg<VLEN, elem_t>(V, vs2, vlmax);
    vmask_view vd_mask_view = read_vmask<VLEN>(V, vlmax, vd);
    auto fn = get_carry_funct<elem_t>(funct);
    for(size_t idx = vstart; idx < vl; idx++) {
        elem_t carry = vm ? 0 : mask_reg[idx];
        vd_mask_view[idx] = fn(vs2_view[idx], imm, carry);
    }
    for(size_t idx = vl; idx < vlmax; idx++)
        vd_mask_view[idx] = agnostic_behavior(vd_mask_view[idx]);
}
template <typename T> bool get_rounding_increment(T v, uint64_t d, int64_t vxrm) {
    if(d == 0)
        return 0;
    switch(vxrm & 0b11) {
    case 0b00:
        return (v >> (d - 1)) & 1;
    case 0b01:
        return ((v >> (d - 1)) & 1) && (((v & ((1 << (d - 1)) - 1)) != 0) || ((v >> d) & 1));
    case 0b10:
        return false;
    case 0b11:
        return (!(v & (static_cast<T>(1) << d)) && ((v & ((static_cast<T>(1) << d) - 1)) != 0));
    }
    return false;
}
template <typename T> T roundoff(T v, uint64_t d, int64_t vxrm) {
    unsigned r = get_rounding_increment(v, d, vxrm);
    return (v >> d) + r;
}
template <typename dest_elem_t, typename src2_elem_t = dest_elem_t, typename src1_elem_t = dest_elem_t>
std::function<bool(uint64_t, vtype_t, dest_elem_t&, src2_elem_t, src1_elem_t)> get_sat_funct(unsigned funct6, unsigned funct3) {
    if(funct3 == OPIVV || funct3 == OPIVX || funct3 == OPIVI)
        switch(funct6) {
        case 0b100000: // VSADDU
            return [](uint64_t vxrm, vtype_t vtype, dest_elem_t& vd, src2_elem_t vs2, src1_elem_t vs1) {
                auto res = static_cast<twice_t<src1_elem_t>>(vs2) + static_cast<twice_t<src1_elem_t>>(vs1);
                if(res > std::numeric_limits<dest_elem_t>::max()) {
                    vd = std::numeric_limits<dest_elem_t>::max();
                    return 1;
                } else {
                    vd = res;
                    return 0;
                }
            };
        case 0b100001: // VSADD
            return [](uint64_t vxrm, vtype_t vtype, dest_elem_t& vd, src2_elem_t vs2, src1_elem_t vs1) {
                auto res = static_cast<twice_t<std::make_signed_t<src2_elem_t>>>(static_cast<std::make_signed_t<src2_elem_t>>(vs2)) +
                           static_cast<twice_t<std::make_signed_t<src1_elem_t>>>(static_cast<std::make_signed_t<src1_elem_t>>(vs1));
                if(res < std::numeric_limits<std::make_signed_t<dest_elem_t>>::min()) {
                    vd = std::numeric_limits<std::make_signed_t<dest_elem_t>>::min();
                    return 1;
                } else if(res > std::numeric_limits<std::make_signed_t<dest_elem_t>>::max()) {
                    vd = std::numeric_limits<std::make_signed_t<dest_elem_t>>::max();
                    return 1;
                } else {
                    vd = res;
                    return 0;
                }
            };
        case 0b100010: // VSSUBU
            return [](uint64_t vxrm, vtype_t vtype, dest_elem_t& vd, src2_elem_t vs2, src1_elem_t vs1) {
                if(vs2 < vs1) {
                    vd = 0;
                    return 1;
                } else {
                    vd = vs2 - vs1;
                    return 0;
                }
            };
        case 0b100011: // VSSUB
            return [](uint64_t vxrm, vtype_t vtype, dest_elem_t& vd, src2_elem_t vs2, src1_elem_t vs1) {
                auto res = static_cast<twice_t<std::make_signed_t<src2_elem_t>>>(static_cast<std::make_signed_t<src2_elem_t>>(vs2)) -
                           static_cast<twice_t<std::make_signed_t<src1_elem_t>>>(static_cast<std::make_signed_t<src1_elem_t>>(vs1));
                if(res < std::numeric_limits<std::make_signed_t<dest_elem_t>>::min()) {
                    vd = std::numeric_limits<std::make_signed_t<dest_elem_t>>::min();
                    return 1;
                } else if(res > std::numeric_limits<std::make_signed_t<dest_elem_t>>::max()) {
                    vd = std::numeric_limits<std::make_signed_t<dest_elem_t>>::max();
                    return 1;
                } else {
                    vd = res;
                    return 0;
                }
            };
        case 0b100111: // VSMUL
            return [](uint64_t vxrm, vtype_t vtype, dest_elem_t& vd, src2_elem_t vs2, src1_elem_t vs1) {
                auto big_val = static_cast<twice_t<std::make_signed_t<src2_elem_t>>>(static_cast<std::make_signed_t<src2_elem_t>>(vs2)) *
                               static_cast<twice_t<std::make_signed_t<src1_elem_t>>>(static_cast<std::make_signed_t<src1_elem_t>>(vs1));
                auto res = roundoff(big_val, vtype.sew() - 1, vxrm);
                if(res < std::numeric_limits<std::make_signed_t<dest_elem_t>>::min()) {
                    vd = std::numeric_limits<std::make_signed_t<dest_elem_t>>::min();
                    return 1;
                } else if(res > std::numeric_limits<std::make_signed_t<dest_elem_t>>::max()) {
                    vd = std::numeric_limits<std::make_signed_t<dest_elem_t>>::max();
                    return 1;
                } else {
                    vd = res;
                    return 0;
                }
            };
        case 0b101010: // VSSRL
            return [](uint64_t vxrm, vtype_t vtype, dest_elem_t& vd, src2_elem_t vs2, src1_elem_t vs1) {
                vd = roundoff(vs2, vs1 & shift_mask<src1_elem_t>(), vxrm);
                return 0;
            };
        case 0b101011: // VSSRA
            return [](uint64_t vxrm, vtype_t vtype, dest_elem_t& vd, src2_elem_t vs2, src1_elem_t vs1) {
                vd = roundoff(static_cast<std::make_signed_t<src2_elem_t>>(vs2), vs1 & shift_mask<src1_elem_t>(), vxrm);
                return 0;
            };
        case 0b101110: // VNCLIPU
            return [](uint64_t vxrm, vtype_t vtype, dest_elem_t& vd, src2_elem_t vs2, src1_elem_t vs1) {
                auto res = roundoff(vs2, vs1 & shift_mask<src2_elem_t>(), vxrm);
                if(res > std::numeric_limits<dest_elem_t>::max()) {
                    vd = std::numeric_limits<dest_elem_t>::max();
                    return 1;
                } else {
                    vd = res;
                    return 0;
                }
            };
        case 0b101111: // VNCLIP
            return [](uint64_t vxrm, vtype_t vtype, dest_elem_t& vd, src2_elem_t vs2, src1_elem_t vs1) {
                auto res = roundoff(static_cast<std::make_signed_t<src2_elem_t>>(vs2), vs1 & shift_mask<src2_elem_t>(), vxrm);
                if(res < std::numeric_limits<std::make_signed_t<dest_elem_t>>::min()) {
                    vd = std::numeric_limits<std::make_signed_t<dest_elem_t>>::min();
                    return 1;
                } else if(res > std::numeric_limits<std::make_signed_t<dest_elem_t>>::max()) {
                    vd = std::numeric_limits<std::make_signed_t<dest_elem_t>>::max();
                    return 1;
                } else {
                    vd = res;
                    return 0;
                }
            };
        default:
            throw new std::runtime_error("Unknown funct6 in get_sat_funct");
        }
    else if(funct3 == OPMVV || funct3 == OPMVX)
        switch(funct6) {
        case 0b001000: // VAADDU
            return [](uint64_t vxrm, vtype_t vtype, dest_elem_t& vd, src2_elem_t vs2, src1_elem_t vs1) {
                auto res = static_cast<dest_elem_t>(vs2) + static_cast<twice_t<src1_elem_t>>(vs1);
                vd = roundoff(res, 1, vxrm);
                return 0;
            };
        case 0b001001: // VAADD
            return [](uint64_t vxrm, vtype_t vtype, dest_elem_t& vd, src2_elem_t vs2, src1_elem_t vs1) {
                auto res = sext<twice_t<src2_elem_t>>(vs2) + sext<twice_t<src1_elem_t>>(vs1);
                vd = roundoff(res, 1, vxrm);
                return 0;
            };
        case 0b001010: // VASUBU
            return [](uint64_t vxrm, vtype_t vtype, dest_elem_t& vd, src2_elem_t vs2, src1_elem_t vs1) {
                auto res = static_cast<dest_elem_t>(vs2) - static_cast<twice_t<src1_elem_t>>(vs1);
                vd = roundoff(res, 1, vxrm);
                return 0;
            };
        case 0b001011: // VASUB
            return [](uint64_t vxrm, vtype_t vtype, dest_elem_t& vd, src2_elem_t vs2, src1_elem_t vs1) {
                auto res = sext<twice_t<src2_elem_t>>(vs2) - sext<twice_t<src1_elem_t>>(vs1);
                vd = roundoff(res, 1, vxrm);
                return 0;
            };
        default:
            throw new std::runtime_error("Unknown funct6 in get_sat_funct");
        }
    else
        throw new std::runtime_error("Unknown funct3 in get_sat_funct");
}
template <unsigned VLEN, typename dest_elem_t, typename src2_elem_t, typename src1_elem_t>
bool sat_vector_vector_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, int64_t vxrm, bool vm,
                          unsigned vd, unsigned vs2, unsigned vs1) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    bool saturated = false;
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs1_view = get_vreg<VLEN, src1_elem_t>(V, vs1, vlmax);
    auto vs2_view = get_vreg<VLEN, src2_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, dest_elem_t>(V, vd, vlmax);
    auto fn = get_sat_funct<dest_elem_t, src2_elem_t, src1_elem_t>(funct6, funct3);
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            saturated |= fn(vxrm, vtype, vd_view[idx], vs2_view[idx], vs1_view[idx]);
        else if(vtype.vma())
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
    }
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++) {
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
        }
    return saturated;
}
template <unsigned VLEN, typename dest_elem_t, typename src2_elem_t, typename src1_elem_t>
bool sat_vector_imm_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, int64_t vxrm, bool vm,
                       unsigned vd, unsigned vs2, typename std::make_signed<src1_elem_t>::type imm) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    bool saturated = false;
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs2_view = get_vreg<VLEN, src2_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, dest_elem_t>(V, vd, vlmax);
    auto fn = get_sat_funct<dest_elem_t, src2_elem_t, src1_elem_t>(funct6, funct3);
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            saturated |= fn(vxrm, vtype, vd_view[idx], vs2_view[idx], imm);
        else if(vtype.vma())
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
    }
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++) {
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
        }
    return saturated;
}
template <typename dest_elem_t, typename src_elem_t>
std::function<void(dest_elem_t&, src_elem_t)> get_red_funct(unsigned funct6, unsigned funct3) {
    if(funct3 == OPIVV || funct3 == OPIVX || funct3 == OPIVI)
        switch(funct6) {
        case 0b110000: // VWREDSUMU
            return [](dest_elem_t& running_total, src_elem_t vs2) { return running_total += static_cast<dest_elem_t>(vs2); };
        case 0b110001: // VWREDSUM
            return [](dest_elem_t& running_total, src_elem_t vs2) {
                // cast the signed vs2 elem to unsigned to enable wrap around on overflow
                return running_total += static_cast<dest_elem_t>(sext<dest_elem_t>(vs2));
            };
        default:
            throw new std::runtime_error("Unknown funct6 in get_red_funct");
        }
    else if(funct3 == OPMVV || funct3 == OPMVX)
        switch(funct6) {
        case 0b000000: // VREDSUM
            return [](dest_elem_t& running_total, src_elem_t vs2) { return running_total += vs2; };
        case 0b000001: // VREDAND
            return [](dest_elem_t& running_total, src_elem_t vs2) { return running_total &= vs2; };
        case 0b000010: // VREDOR
            return [](dest_elem_t& running_total, src_elem_t vs2) { return running_total |= vs2; };
        case 0b000011: // VREDXOR
            return [](dest_elem_t& running_total, src_elem_t vs2) { running_total ^= vs2; };
        case 0b000100: // VREDMINU
            return [](dest_elem_t& running_total, src_elem_t vs2) { running_total = std::min<dest_elem_t>(running_total, vs2); };
        case 0b000101: // VREDMIN
            return [](dest_elem_t& running_total, src_elem_t vs2) {
                running_total = std::min(static_cast<std::make_signed_t<dest_elem_t>>(running_total),
                                         static_cast<std::make_signed_t<dest_elem_t>>(static_cast<std::make_signed_t<src_elem_t>>(vs2)));
            };
        case 0b000110: // VREDMAXU
            return
                [](dest_elem_t& running_total, src_elem_t vs2) { running_total = std::max(running_total, static_cast<dest_elem_t>(vs2)); };
        case 0b000111: // VREDMAX
            return [](dest_elem_t& running_total, src_elem_t vs2) {
                running_total = std::max(static_cast<std::make_signed_t<dest_elem_t>>(running_total),
                                         static_cast<std::make_signed_t<dest_elem_t>>(static_cast<std::make_signed_t<src_elem_t>>(vs2)));
            };
        default:
            throw new std::runtime_error("Unknown funct6 in get_red_funct");
        }
    else
        throw new std::runtime_error("Unknown funct3 in get_red_funct");
}
template <unsigned VLEN, typename dest_elem_t, typename src_elem_t>
void vector_red_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd,
                   unsigned vs2, unsigned vs1) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs1_elem = get_vreg<VLEN, dest_elem_t>(V, vs1, vlmax)[0];
    auto vs2_view = get_vreg<VLEN, src_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, dest_elem_t>(V, vd, vlmax);
    auto fn = get_red_funct<dest_elem_t, src_elem_t>(funct6, funct3);
    dest_elem_t& running_total = vd_view[0] = vs1_elem;
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active) {
            fn(running_total, vs2_view[idx]);
        }
    }
    // the tail is all elements of the destination register beyond the first one
    if(vtype.vta())
        for(size_t idx = 1; idx < VLEN / vtype.sew(); idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}

// might be that these exist somewhere in softfloat
template <typename src_elem_t> constexpr bool isNaN(src_elem_t x);
template <> constexpr bool isNaN<uint16_t>(uint16_t x) { return ((x & 0x7C00) == 0x7C00) && ((x & 0x03FF) != 0); }
template <> constexpr bool isNaN<uint32_t>(uint32_t x) { return ((x & 0x7F800000) == 0x7F800000) && ((x & 0x007FFFFF) != 0); }
template <> constexpr bool isNaN<uint64_t>(uint64_t x) {
    return ((x & 0x7FF0000000000000) == 0x7FF0000000000000) && ((x & 0x000FFFFFFFFFFFFF) != 0);
}
template <typename src_elem_t> constexpr bool isNegZero(src_elem_t x);
template <> constexpr bool isNegZero<uint16_t>(uint16_t x) { return x == 0x8000; }
template <> constexpr bool isNegZero<uint32_t>(uint32_t x) { return x == 0x80000000; }
template <> constexpr bool isNegZero<uint64_t>(uint64_t x) { return x == 0x8000000000000000; }
template <typename src_elem_t> constexpr bool isPosZero(src_elem_t x);
template <> constexpr bool isPosZero<uint16_t>(uint16_t x) { return x == 0x0000; }
template <> constexpr bool isPosZero<uint32_t>(uint32_t x) { return x == 0x00000000; }
template <> constexpr bool isPosZero<uint64_t>(uint64_t x) { return x == 0x0000000000000000; }

template <typename dest_elem_t, typename src_elem_t> dest_elem_t widen_float(src_elem_t val) {
    throw new std::runtime_error("Trying to widen a weird 'float'");
}
template <> inline uint64_t widen_float<uint64_t, uint32_t>(uint32_t val) { return f32_to_f64(float32_t{val}).v; }

template <typename elem_size_t> elem_size_t fp_add(uint8_t, elem_size_t, elem_size_t);
template <> inline uint16_t fp_add<uint16_t>(uint8_t mode, uint16_t v2, uint16_t v1) { return fadd_h(v2, v1, mode); }
template <> inline uint32_t fp_add<uint32_t>(uint8_t mode, uint32_t v2, uint32_t v1) { return fadd_s(v2, v1, mode); }
template <> inline uint64_t fp_add<uint64_t>(uint8_t mode, uint64_t v2, uint64_t v1) { return fadd_d(v2, v1, mode); }
template <typename elem_size_t> elem_size_t fp_sub(uint8_t, elem_size_t, elem_size_t);
template <> inline uint16_t fp_sub<uint16_t>(uint8_t mode, uint16_t v2, uint16_t v1) { return fsub_h(v2, v1, mode); }
template <> inline uint32_t fp_sub<uint32_t>(uint8_t mode, uint32_t v2, uint32_t v1) { return fsub_s(v2, v1, mode); }
template <> inline uint64_t fp_sub<uint64_t>(uint8_t mode, uint64_t v2, uint64_t v1) { return fsub_d(v2, v1, mode); }
template <typename elem_size_t> elem_size_t fp_mul(uint8_t, elem_size_t, elem_size_t);
template <> inline uint16_t fp_mul<uint16_t>(uint8_t mode, uint16_t v2, uint16_t v1) { return fmul_h(v2, v1, mode); }
template <> inline uint32_t fp_mul<uint32_t>(uint8_t mode, uint32_t v2, uint32_t v1) { return fmul_s(v2, v1, mode); }
template <> inline uint64_t fp_mul<uint64_t>(uint8_t mode, uint64_t v2, uint64_t v1) { return fmul_d(v2, v1, mode); }
template <typename elem_size_t> elem_size_t fp_div(uint8_t, elem_size_t, elem_size_t);
template <> inline uint16_t fp_div<uint16_t>(uint8_t mode, uint16_t v2, uint16_t v1) { return fdiv_h(v2, v1, mode); }
template <> inline uint32_t fp_div<uint32_t>(uint8_t mode, uint32_t v2, uint32_t v1) { return fdiv_s(v2, v1, mode); }
template <> inline uint64_t fp_div<uint64_t>(uint8_t mode, uint64_t v2, uint64_t v1) { return fdiv_d(v2, v1, mode); }
template <typename elem_size_t> elem_size_t fp_madd(uint8_t, elem_size_t, elem_size_t, elem_size_t);
template <> inline uint16_t fp_madd<uint16_t>(uint8_t mode, uint16_t v2, uint16_t v1, uint16_t v3) { return fmadd_h(v1, v2, v3, 0, mode); }
template <> inline uint32_t fp_madd<uint32_t>(uint8_t mode, uint32_t v2, uint32_t v1, uint32_t v3) { return fmadd_s(v1, v2, v3, 0, mode); }
template <> inline uint64_t fp_madd<uint64_t>(uint8_t mode, uint64_t v2, uint64_t v1, uint64_t v3) { return fmadd_d(v1, v2, v3, 0, mode); }
template <typename elem_size_t> elem_size_t fp_nmadd(uint8_t, elem_size_t, elem_size_t, elem_size_t);
template <> inline uint16_t fp_nmadd<uint16_t>(uint8_t mode, uint16_t v2, uint16_t v1, uint16_t v3) { return fmadd_h(v1, v2, v3, 2, mode); }
template <> inline uint32_t fp_nmadd<uint32_t>(uint8_t mode, uint32_t v2, uint32_t v1, uint32_t v3) { return fmadd_s(v1, v2, v3, 2, mode); }
template <> inline uint64_t fp_nmadd<uint64_t>(uint8_t mode, uint64_t v2, uint64_t v1, uint64_t v3) { return fmadd_d(v1, v2, v3, 2, mode); }
template <typename elem_size_t> elem_size_t fp_msub(uint8_t, elem_size_t, elem_size_t, elem_size_t);
template <> inline uint16_t fp_msub<uint16_t>(uint8_t mode, uint16_t v2, uint16_t v1, uint16_t v3) { return fmadd_h(v1, v2, v3, 1, mode); }
template <> inline uint32_t fp_msub<uint32_t>(uint8_t mode, uint32_t v2, uint32_t v1, uint32_t v3) { return fmadd_s(v1, v2, v3, 1, mode); }
template <> inline uint64_t fp_msub<uint64_t>(uint8_t mode, uint64_t v2, uint64_t v1, uint64_t v3) { return fmadd_d(v1, v2, v3, 1, mode); }
template <typename elem_size_t> elem_size_t fp_nmsub(uint8_t, elem_size_t, elem_size_t, elem_size_t);
template <> inline uint16_t fp_nmsub<uint16_t>(uint8_t mode, uint16_t v2, uint16_t v1, uint16_t v3) { return fmadd_h(v1, v2, v3, 3, mode); }
template <> inline uint32_t fp_nmsub<uint32_t>(uint8_t mode, uint32_t v2, uint32_t v1, uint32_t v3) { return fmadd_s(v1, v2, v3, 3, mode); }
template <> inline uint64_t fp_nmsub<uint64_t>(uint8_t mode, uint64_t v2, uint64_t v1, uint64_t v3) { return fmadd_d(v1, v2, v3, 3, mode); }
template <typename elem_size_t> elem_size_t fp_min(elem_size_t, elem_size_t);
template <> inline uint16_t fp_min<uint16_t>(uint16_t v2, uint16_t v1) {
    if(isNaN(v1) && isNaN(v2))
        return defaultNaNF16UI;
    else if(isNaN(v1))
        return v2;
    else if(isNaN(v2))
        return v1;
    else if(isNegZero(v1) && isNegZero(v2))
        return v1;
    else if(isNegZero(v2) && isNegZero(v1))
        return v2;
    else if(fcmp_h(v1, v2, 2))
        return v1;
    else
        return v2;
}
template <> inline uint32_t fp_min<uint32_t>(uint32_t v2, uint32_t v1) {
    if(isNaN(v1) && isNaN(v2))
        return defaultNaNF32UI;
    else if(isNaN(v1))
        return v2;
    else if(isNaN(v2))
        return v1;
    else if(isNegZero(v1) && isNegZero(v2))
        return v1;
    else if(isNegZero(v2) && isNegZero(v1))
        return v2;
    else if(fcmp_s(v1, v2, 2))
        return v1;
    else
        return v2;
}
template <> inline uint64_t fp_min<uint64_t>(uint64_t v2, uint64_t v1) {
    if(isNaN(v1) && isNaN(v2))
        return defaultNaNF64UI;
    else if(isNaN(v1))
        return v2;
    else if(isNaN(v2))
        return v1;
    else if(isNegZero(v1) && isNegZero(v2))
        return v1;
    else if(isNegZero(v2) && isNegZero(v1))
        return v2;
    else if(fcmp_d(v1, v2, 2))
        return v1;
    else
        return v2;
}
template <typename elem_size_t> elem_size_t fp_max(elem_size_t, elem_size_t);
template <> inline uint16_t fp_max<uint16_t>(uint16_t v2, uint16_t v1) {
    if(isNaN(v1) && isNaN(v2))
        return defaultNaNF16UI;
    else if(isNaN(v1))
        return v2;
    else if(isNaN(v2))
        return v1;
    else if(isNegZero(v1) && isNegZero(v2))
        return v2;
    else if(isNegZero(v2) && isNegZero(v1))
        return v1;
    else if(fcmp_h(v1, v2, 2))
        return v2;
    else
        return v1;
}
template <> inline uint32_t fp_max<uint32_t>(uint32_t v2, uint32_t v1) {
    if(isNaN(v1) && isNaN(v2))
        return defaultNaNF32UI;
    else if(isNaN(v1))
        return v2;
    else if(isNaN(v2))
        return v1;
    else if(isNegZero(v1) && isNegZero(v2))
        return v2;
    else if(isNegZero(v2) && isNegZero(v1))
        return v1;
    else if(fcmp_s(v1, v2, 2))
        return v2;
    else
        return v1;
}
template <> inline uint64_t fp_max<uint64_t>(uint64_t v2, uint64_t v1) {
    if(isNaN(v1) && isNaN(v2))
        return defaultNaNF64UI;
    else if(isNaN(v1))
        return v2;
    else if(isNaN(v2))
        return v1;
    else if(isNegZero(v1) && isNegZero(v2))
        return v2;
    else if(isNegZero(v2) && isNegZero(v1))
        return v1;
    else if(fcmp_d(v1, v2, 2))
        return v2;
    else
        return v1;
}

template <typename dest_elem_t, typename src2_elem_t = dest_elem_t, typename src1_elem_t = dest_elem_t>
std::function<dest_elem_t(uint8_t, uint8_t&, dest_elem_t, src2_elem_t, src1_elem_t)> get_fp_funct(unsigned funct6, unsigned funct3) {
    if(funct3 == OPFVV || funct3 == OPFVF)
        switch(funct6) {
        case 0b000000: // VFADD
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_add<src2_elem_t>(rm, vs2, vs1);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b000010: // VFSUB
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_sub<src2_elem_t>(rm, vs2, vs1);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b000100: // VFMIN
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return fp_min<src2_elem_t>(vs2, vs1);
            };
        case 0b000110: // VFMAX
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                return fp_max<src2_elem_t>(vs2, vs1);
            };
        case 0b100000: // VFDIV
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_div<src2_elem_t>(rm, vs2, vs1);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b100001: // VFRDIV
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_div<src2_elem_t>(rm, vs1, vs2);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b100100: // VFMUL
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_mul<src2_elem_t>(rm, vs2, vs1);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b100111: // VFRSUB
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_sub<src2_elem_t>(rm, vs1, vs2);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b101000: // VFMADD
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_madd<src2_elem_t>(rm, vs1, vd, vs2);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b101001: // VFNMADD
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_nmadd<src2_elem_t>(rm, vs1, vd, vs2);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b101010: // VFMSUB
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_msub<src2_elem_t>(rm, vs1, vd, vs2);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b101011: // VFNMSUB
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_nmsub<src2_elem_t>(rm, vs1, vd, vs2);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b101100: // VFMACC
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_madd<src2_elem_t>(rm, vs1, vs2, vd);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b101101: // VFNMAC
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_nmadd<src2_elem_t>(rm, vs1, vs2, vd);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b101110: // VFMSAC
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_msub<src2_elem_t>(rm, vs1, vs2, vd);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b101111: // VFNMSAC
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_nmsub<src2_elem_t>(rm, vs1, vs2, vd);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b110000: // VFWADD
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_add<dest_elem_t>(rm, widen_float<dest_elem_t>(vs2), widen_float<dest_elem_t>(vs1));
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b110010: // VFWSUB
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_sub<dest_elem_t>(rm, widen_float<dest_elem_t>(vs2), widen_float<dest_elem_t>(vs1));
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b110100: // VFWADD.W
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_add<dest_elem_t>(rm, vs2, widen_float<dest_elem_t>(vs1));
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b110110: // VFWSUB.W
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_sub<dest_elem_t>(rm, vs2, widen_float<dest_elem_t>(vs1));
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b111000: // VFWMUL
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_mul<dest_elem_t>(rm, widen_float<dest_elem_t>(vs2), widen_float<dest_elem_t>(vs1));
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b111100: // VFWMACC
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_madd<dest_elem_t>(rm, widen_float<dest_elem_t>(vs1), widen_float<dest_elem_t>(vs2), vd);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b111101: // VFWNMACC
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_nmadd<dest_elem_t>(rm, widen_float<dest_elem_t>(vs1), widen_float<dest_elem_t>(vs2), vd);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b111110: // VFWMSAC
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_msub<dest_elem_t>(rm, widen_float<dest_elem_t>(vs1), widen_float<dest_elem_t>(vs2), vd);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b111111: // VFWNMSAC
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t val = fp_nmsub<dest_elem_t>(rm, widen_float<dest_elem_t>(vs1), widen_float<dest_elem_t>(vs2), vd);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b001000: // VFSGNJ
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t mask = std::numeric_limits<dest_elem_t>::max() >> 1;
                dest_elem_t sign_mask = std::numeric_limits<std::make_signed_t<dest_elem_t>>::min();
                return (vs2 & mask) | (vs1 & sign_mask);
            };
        case 0b001001: // VFSGNJN
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t mask = std::numeric_limits<dest_elem_t>::max() >> 1;
                dest_elem_t sign_mask = std::numeric_limits<std::make_signed_t<dest_elem_t>>::min();
                return (vs2 & mask) | (~vs1 & sign_mask);
            };
        case 0b001010: // VFSGNJX
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t vd, src2_elem_t vs2, src1_elem_t vs1) {
                dest_elem_t mask = std::numeric_limits<dest_elem_t>::max() >> 1;
                dest_elem_t sign_mask = std::numeric_limits<std::make_signed_t<dest_elem_t>>::min();
                return vs2 ^ (vs1 & sign_mask);
            };
        default:
            throw new std::runtime_error("Unknown funct6 in get_fp_funct");
        }
    else
        throw new std::runtime_error("Unknown funct3 in get_fp_funct");
}
template <unsigned VLEN, typename dest_elem_t, typename src2_elem_t, typename src1_elem_t>
void fp_vector_vector_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd,
                         unsigned vs2, unsigned vs1, uint8_t rm) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs1_view = get_vreg<VLEN, src1_elem_t>(V, vs1, vlmax);
    auto vs2_view = get_vreg<VLEN, src2_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, dest_elem_t>(V, vd, vlmax);
    auto fn = get_fp_funct<dest_elem_t, src2_elem_t, src1_elem_t>(funct6, funct3);
    uint8_t accrued_flags = 0;
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_view[idx] = fn(rm, accrued_flags, vd_view[idx], vs2_view[idx], vs1_view[idx]);
        else if(vtype.vma())
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
    }
    softfloat_exceptionFlags = accrued_flags;
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++) {
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
        }
}
template <unsigned VLEN, typename dest_elem_t, typename src2_elem_t, typename src1_elem_t>
void fp_vector_imm_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd,
                      unsigned vs2, src1_elem_t imm, uint8_t rm) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs2_view = get_vreg<VLEN, src2_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, dest_elem_t>(V, vd, vlmax);
    auto fn = get_fp_funct<dest_elem_t, src2_elem_t, src1_elem_t>(funct6, funct3);
    uint8_t accrued_flags = 0;
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_view[idx] = fn(rm, accrued_flags, vd_view[idx], vs2_view[idx], imm);
        else if(vtype.vma())
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
    }
    softfloat_exceptionFlags = accrued_flags;
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}
template <typename dest_elem_t, typename src_elem_t>
std::function<void(uint8_t, uint8_t&, dest_elem_t&, src_elem_t)> get_fp_red_funct(unsigned funct6, unsigned funct3) {
    if(funct3 == OPFVV || funct3 == OPFVF)
        switch(funct6) {
        case 0b000001: // VFREDUSUM
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t& running_total, src_elem_t vs2) {
                running_total = fp_add<dest_elem_t>(rm, running_total, vs2);
                accrued_flags |= softfloat_exceptionFlags;
            };
        case 0b000011: // VFREDOSUM
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t& running_total, src_elem_t vs2) {
                running_total = fp_add<dest_elem_t>(rm, running_total, vs2);
                accrued_flags |= softfloat_exceptionFlags;
            };
        case 0b000101: // VFREDMIN
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t& running_total, src_elem_t vs2) {
                running_total = fp_min<dest_elem_t>(running_total, vs2);
            };
        case 0b000111: // VFREDMAX
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t& running_total, src_elem_t vs2) {
                running_total = fp_max<dest_elem_t>(running_total, vs2);
            };
        case 0b110001: // VFWREDUSUM
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t& running_total, src_elem_t vs2) {
                running_total = fp_add<dest_elem_t>(rm, running_total, widen_float<dest_elem_t>(vs2));
                accrued_flags |= softfloat_exceptionFlags;
            };
        case 0b110011: // VFWREDOSUM
            return [](uint8_t rm, uint8_t& accrued_flags, dest_elem_t& running_total, src_elem_t vs2) {
                running_total = fp_add<dest_elem_t>(rm, running_total, widen_float<dest_elem_t>(vs2));
                accrued_flags |= softfloat_exceptionFlags;
            };
        default:
            throw new std::runtime_error("Unknown funct6 in get_fp_red_funct");
        }
    else
        throw new std::runtime_error("Unknown funct3 in get_fp_red_funct");
}
template <unsigned VLEN, typename dest_elem_t, typename src_elem_t>
void fp_vector_red_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd,
                      unsigned vs2, unsigned vs1, uint8_t rm) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs1_elem = get_vreg<VLEN, dest_elem_t>(V, vs1, vlmax)[0];
    auto vs2_view = get_vreg<VLEN, src_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, dest_elem_t>(V, vd, vlmax);
    auto fn = get_fp_red_funct<dest_elem_t, src_elem_t>(funct6, funct3);
    dest_elem_t& running_total = vd_view[0] = vs1_elem;
    uint8_t accrued_flags = 0;
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            fn(rm, accrued_flags, running_total, vs2_view[idx]);
    }
    softfloat_exceptionFlags = accrued_flags;
    // the tail is all elements of the destination register beyond the first one
    if(vtype.vta())
        for(size_t idx = 1; idx < VLEN / vtype.sew(); idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}
template <typename elem_size_t> elem_size_t fp_sqrt(uint8_t, elem_size_t);
template <> inline uint16_t fp_sqrt<uint16_t>(uint8_t mode, uint16_t v2) { return fsqrt_h(v2, mode); }
template <> inline uint32_t fp_sqrt<uint32_t>(uint8_t mode, uint32_t v2) { return fsqrt_s(v2, mode); }
template <> inline uint64_t fp_sqrt<uint64_t>(uint8_t mode, uint64_t v2) { return fsqrt_d(v2, mode); }
template <typename elem_size_t> elem_size_t fp_rsqrt7(elem_size_t);
template <> inline uint16_t fp_rsqrt7<uint16_t>(uint16_t v2) { return frsqrt7_h(v2); }
template <> inline uint32_t fp_rsqrt7<uint32_t>(uint32_t v2) { return frsqrt7_s(v2); }
template <> inline uint64_t fp_rsqrt7<uint64_t>(uint64_t v2) { return frsqrt7_d(v2); }
template <typename elem_size_t> elem_size_t fp_rec7(uint8_t, elem_size_t);
template <> inline uint16_t fp_rec7<uint16_t>(uint8_t mode, uint16_t v2) { return frec7_h(v2, mode); }
template <> inline uint32_t fp_rec7<uint32_t>(uint8_t mode, uint32_t v2) { return frec7_s(v2, mode); }
template <> inline uint64_t fp_rec7<uint64_t>(uint8_t mode, uint64_t v2) { return frec7_d(v2, mode); }
template <typename elem_size_t> elem_size_t fp_fclass(elem_size_t);
template <> inline uint16_t fp_fclass<uint16_t>(uint16_t v2) { return fclass_h(v2); }
template <> inline uint32_t fp_fclass<uint32_t>(uint32_t v2) { return fclass_s(v2); }
template <> inline uint64_t fp_fclass<uint64_t>(uint64_t v2) { return fclass_d(v2); }

template <typename dest_elem_size_t, typename src_elem_size_t> dest_elem_size_t fp_f_to_ui(uint8_t, src_elem_size_t);
template <typename dest_elem_size_t, typename src_elem_size_t> dest_elem_size_t fp_f_to_i(uint8_t, src_elem_size_t);
template <typename dest_elem_size_t, typename src_elem_size_t> dest_elem_size_t fp_ui_to_f(uint8_t, src_elem_size_t);
template <typename dest_elem_size_t, typename src_elem_size_t> dest_elem_size_t fp_i_to_f(uint8_t, src_elem_size_t);
template <typename dest_elem_t, typename src_elem_t> dest_elem_t fp_f_to_f(uint8_t rm, src_elem_t val);

template <> inline uint16_t fp_f_to_ui<uint16_t, uint16_t>(uint8_t rm, uint16_t v2) { return f16toui32(v2, rm); }
template <> inline uint32_t fp_f_to_ui<uint32_t, uint32_t>(uint8_t rm, uint32_t v2) { return f32toui32(v2, rm); }
template <> inline uint64_t fp_f_to_ui<uint64_t, uint64_t>(uint8_t rm, uint64_t v2) { return f64toui64(v2, rm); }

template <> inline uint16_t fp_f_to_i<uint16_t, uint16_t>(uint8_t rm, uint16_t v2) { return f16toi32(v2, rm); }
template <> inline uint32_t fp_f_to_i<uint32_t, uint32_t>(uint8_t rm, uint32_t v2) { return f32toi32(v2, rm); }
template <> inline uint64_t fp_f_to_i<uint64_t, uint64_t>(uint8_t rm, uint64_t v2) { return f64toi64(v2, rm); }

template <> inline uint16_t fp_ui_to_f<uint16_t, uint16_t>(uint8_t rm, uint16_t v2) { return ui32tof16(v2, rm); }
template <> inline uint32_t fp_ui_to_f<uint32_t, uint32_t>(uint8_t rm, uint32_t v2) { return ui32tof32(v2, rm); }
template <> inline uint64_t fp_ui_to_f<uint64_t, uint64_t>(uint8_t rm, uint64_t v2) { return ui64tof64(v2, rm); }

template <> inline uint16_t fp_i_to_f<uint16_t, uint16_t>(uint8_t rm, uint16_t v2) { return i32tof16(v2, rm); }
template <> inline uint32_t fp_i_to_f<uint32_t, uint32_t>(uint8_t rm, uint32_t v2) { return i32tof32(v2, rm); }
template <> inline uint64_t fp_i_to_f<uint64_t, uint64_t>(uint8_t rm, uint64_t v2) { return i64tof64(v2, rm); }

template <typename elem_t> std::function<elem_t(uint8_t, uint8_t&, elem_t)> get_fp_unary_fn(unsigned encoding_space, unsigned unary_op) {
    if(encoding_space == 0b010011) // VFUNARY1
        switch(unary_op) {
        case 0b00000: // VFSQRT
            return [](uint8_t rm, uint8_t& accrued_flags, elem_t vs2) {
                elem_t val = fp_sqrt(rm, vs2);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b00100: // VFRSQRT7
            return [](uint8_t rm, uint8_t& accrued_flags, elem_t vs2) {
                elem_t val = fp_rsqrt7(vs2);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b00101: // VFREC7
            return [](uint8_t rm, uint8_t& accrued_flags, elem_t vs2) {
                elem_t val = fp_rec7(rm, vs2);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b10000: // VFCLASS
            return [](uint8_t rm, uint8_t& accrued_flags, elem_t vs2) {
                elem_t val = fp_fclass(vs2);
                return val;
            };
        default:
            throw new std::runtime_error("Unknown funct in get_fp_unary_fn");
        }
    else if(encoding_space == 0b010010) // VFUNARY0
        switch(unary_op) {
        case 0b00000: // VFCVT.XU.F.V
        case 0b00110: // VFCVT.RTZ.XU.F.V
            return [](uint8_t rm, uint8_t& accrued_flags, elem_t vs2) {
                elem_t val = fp_f_to_ui<elem_t, elem_t>(rm, vs2);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b00001: // VFCVT.X.F.V
        case 0b00111: // VFCVT.RTZ.X.F.V
            return [](uint8_t rm, uint8_t& accrued_flags, elem_t vs2) {
                elem_t val = fp_f_to_i<elem_t, elem_t>(rm, vs2);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b00010: // VFCVT.F.XU.V
            return [](uint8_t rm, uint8_t& accrued_flags, elem_t vs2) {
                elem_t val = fp_ui_to_f<elem_t, elem_t>(rm, vs2);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        case 0b00011: // VFCVT.F.X.V
            return [](uint8_t rm, uint8_t& accrued_flags, elem_t vs2) {
                elem_t val = fp_i_to_f<elem_t, elem_t>(rm, vs2);
                accrued_flags |= softfloat_exceptionFlags;
                return val;
            };
        default:
            throw new std::runtime_error("Unknown funct in get_fp_unary_fn");
        }
    else
        throw new std::runtime_error("Unknown funct in get_fp_unary_fn");
}
template <unsigned VLEN, typename elem_t>
void fp_vector_unary_op(uint8_t* V, unsigned encoding_space, unsigned unary_op, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm,
                        unsigned vd, unsigned vs2, uint8_t rm) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs2_view = get_vreg<VLEN, elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, elem_t>(V, vd, vlmax);
    auto fn = get_fp_unary_fn<elem_t>(encoding_space, unary_op);
    uint8_t accrued_flags = 0;
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_view[idx] = fn(rm, accrued_flags, vs2_view[idx]);
        else if(vtype.vma())
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
    }
    softfloat_exceptionFlags = accrued_flags;
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}

template <> inline uint16_t fp_f_to_ui<uint16_t, uint8_t>(uint8_t rm, uint8_t v2) {
    throw new std::runtime_error("Attempting illegal widening conversion");
}
template <> inline uint32_t fp_f_to_ui<uint32_t, uint16_t>(uint8_t rm, uint16_t v2) { return f16toui32(v2, rm); }
template <> inline uint64_t fp_f_to_ui<uint64_t, uint32_t>(uint8_t rm, uint32_t v2) { return f32toui64(v2, rm); }

template <> inline uint16_t fp_f_to_i<uint16_t, uint8_t>(uint8_t rm, uint8_t v2) {
    throw new std::runtime_error("Attempting illegal widening conversion");
}
template <> inline uint32_t fp_f_to_i<uint32_t, uint16_t>(uint8_t rm, uint16_t v2) { return f16toi32(v2, rm); }
template <> inline uint64_t fp_f_to_i<uint64_t, uint32_t>(uint8_t rm, uint32_t v2) { return f32toi64(v2, rm); }

template <> inline uint16_t fp_ui_to_f<uint16_t, uint8_t>(uint8_t rm, uint8_t v2) { return ui32tof16(v2, rm); }
template <> inline uint32_t fp_ui_to_f<uint32_t, uint16_t>(uint8_t rm, uint16_t v2) { return ui32tof32(v2, rm); }
template <> inline uint64_t fp_ui_to_f<uint64_t, uint32_t>(uint8_t rm, uint32_t v2) { return ui32tof64(v2, rm); }

template <> inline uint16_t fp_i_to_f<uint16_t, uint8_t>(uint8_t rm, uint8_t v2) { return i32tof16(v2, rm); }
template <> inline uint32_t fp_i_to_f<uint32_t, uint16_t>(uint8_t rm, uint16_t v2) { return i32tof32(v2, rm); }
template <> inline uint64_t fp_i_to_f<uint64_t, uint32_t>(uint8_t rm, uint32_t v2) { return i32tof64(v2, rm); }

template <> inline uint16_t fp_f_to_f<uint16_t, uint8_t>(uint8_t rm, uint8_t val) {
    throw new std::runtime_error("Attempting illegal widening conversion");
}
template <> inline uint32_t fp_f_to_f<uint32_t, uint16_t>(uint8_t rm, uint16_t val) { return f16tof32(val, rm); }
template <> inline uint64_t fp_f_to_f<uint64_t, uint32_t>(uint8_t rm, uint32_t val) { return f32tof64(val, rm); }

template <typename dest_elem_t, typename src_elem_t>
std::function<dest_elem_t(uint8_t, uint8_t&, src_elem_t)> get_fp_widening_fn(unsigned unary_op) {
    switch(unary_op) {
    case 0b01000: // VFWCVT.XU.F.V
    case 0b01110: // VFWCVT.RTZ.XU.F.V
        return [](uint8_t rm, uint8_t& accrued_flags, src_elem_t vs2) {
            dest_elem_t val = fp_f_to_ui<dest_elem_t, src_elem_t>(rm, vs2);
            accrued_flags |= softfloat_exceptionFlags;
            return val;
        };
    case 0b01001: // VFWCVT.X.F.V
    case 0b01111: // VFWCVT.RTZ.X.F.V
        return [](uint8_t rm, uint8_t& accrued_flags, src_elem_t vs2) {
            dest_elem_t val = fp_f_to_i<dest_elem_t, src_elem_t>(rm, vs2);
            accrued_flags |= softfloat_exceptionFlags;
            return val;
        };
    case 0b01010: // VFWCVT.F.XU.V
        return [](uint8_t rm, uint8_t& accrued_flags, src_elem_t vs2) {
            dest_elem_t val = fp_ui_to_f<dest_elem_t, src_elem_t>(rm, vs2);
            accrued_flags |= softfloat_exceptionFlags;
            return val;
        };
    case 0b01011: // VFWCVT.F.X.V
        return [](uint8_t rm, uint8_t& accrued_flags, src_elem_t vs2) {
            dest_elem_t val = fp_i_to_f<dest_elem_t, src_elem_t>(rm, vs2);
            accrued_flags |= softfloat_exceptionFlags;
            return val;
        };
    case 0b01100: // VFWCVT.F.F.V
        return [](uint8_t rm, uint8_t& accrued_flags, src_elem_t vs2) {
            dest_elem_t val = fp_f_to_f<dest_elem_t, src_elem_t>(rm, vs2);
            accrued_flags |= softfloat_exceptionFlags;
            return val;
        };
    default:
        throw new std::runtime_error("Unknown funct in get_fp_unary_fn");
    }
}
template <unsigned VLEN, typename dest_elem_t, typename src_elem_t>
void fp_vector_unary_w(uint8_t* V, unsigned unary_op, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                       uint8_t rm) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs2_view = get_vreg<VLEN, src_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, dest_elem_t>(V, vd, vlmax);
    auto fn = get_fp_widening_fn<dest_elem_t, src_elem_t>(unary_op);
    uint8_t accrued_flags = 0;
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_view[idx] = fn(rm, accrued_flags, vs2_view[idx]);
        else if(vtype.vma())
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
    }
    softfloat_exceptionFlags = accrued_flags;
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}

template <> inline uint8_t fp_f_to_ui<uint8_t, uint16_t>(uint8_t rm, uint16_t v2) { return f16toui32(v2, rm); }
template <> inline uint16_t fp_f_to_ui<uint16_t, uint32_t>(uint8_t rm, uint32_t v2) { return f32toui32(v2, rm); }
template <> inline uint32_t fp_f_to_ui<uint32_t, uint64_t>(uint8_t rm, uint64_t v2) { return f64toui32(v2, rm); }

template <> inline uint8_t fp_f_to_i<uint8_t, uint16_t>(uint8_t rm, uint16_t v2) { return f16toi32(v2, rm); }
template <> inline uint16_t fp_f_to_i<uint16_t, uint32_t>(uint8_t rm, uint32_t v2) { return f32toi32(v2, rm); }
template <> inline uint32_t fp_f_to_i<uint32_t, uint64_t>(uint8_t rm, uint64_t v2) { return f64toi32(v2, rm); }

template <> inline uint8_t fp_ui_to_f<uint8_t, uint16_t>(uint8_t rm, uint16_t v2) {
    throw new std::runtime_error("Attempting illegal narrowing conversion");
}
template <> inline uint16_t fp_ui_to_f<uint16_t, uint32_t>(uint8_t rm, uint32_t v2) { return ui32tof16(v2, rm); }
template <> inline uint32_t fp_ui_to_f<uint32_t, uint64_t>(uint8_t rm, uint64_t v2) { return ui64tof32(v2, rm); }

template <> inline uint8_t fp_i_to_f<uint8_t, uint16_t>(uint8_t rm, uint16_t v2) {
    throw new std::runtime_error("Attempting illegal narrowing conversion");
}
template <> inline uint16_t fp_i_to_f<uint16_t, uint32_t>(uint8_t rm, uint32_t v2) { return i32tof16(v2, rm); }
template <> inline uint32_t fp_i_to_f<uint32_t, uint64_t>(uint8_t rm, uint64_t v2) { return i64tof32(v2, rm); }

template <> inline uint8_t fp_f_to_f<uint8_t, uint16_t>(uint8_t rm, uint16_t val) {
    throw new std::runtime_error("Attempting illegal narrowing conversion");
}
template <> inline uint16_t fp_f_to_f<uint16_t, uint32_t>(uint8_t rm, uint32_t val) { return f32tof16(val, rm); }
template <> inline uint32_t fp_f_to_f<uint32_t, uint64_t>(uint8_t rm, uint64_t val) { return f64tof32(val, rm); }
template <typename dest_elem_t, typename src_elem_t>
std::function<dest_elem_t(uint8_t, uint8_t&, src_elem_t)> get_fp_narrowing_fn(unsigned unary_op) {
    switch(unary_op) {
    case 0b10000: // VFNCVT.XU.F.W
    case 0b10110: // VFNCVT.RTZ.XU.F.W
        return [](uint8_t rm, uint8_t& accrued_flags, src_elem_t vs2) {
            dest_elem_t val = fp_f_to_ui<dest_elem_t, src_elem_t>(rm, vs2);
            accrued_flags |= softfloat_exceptionFlags;
            return val;
        };
    case 0b10001: // VFNCVT.X.F.W
    case 0b10111: // VFNCVT.RTZ.X.F.W
        return [](uint8_t rm, uint8_t& accrued_flags, src_elem_t vs2) {
            dest_elem_t val = fp_f_to_i<dest_elem_t, src_elem_t>(rm, vs2);
            accrued_flags |= softfloat_exceptionFlags;
            return val;
        };
    case 0b10010: // VFNCVT.F.XU.W
        return [](uint8_t rm, uint8_t& accrued_flags, src_elem_t vs2) {
            dest_elem_t val = fp_ui_to_f<dest_elem_t, src_elem_t>(rm, vs2);
            accrued_flags |= softfloat_exceptionFlags;
            return val;
        };
    case 0b10011: // VFNCVT.F.X.W
        return [](uint8_t rm, uint8_t& accrued_flags, src_elem_t vs2) {
            dest_elem_t val = fp_i_to_f<dest_elem_t, src_elem_t>(rm, vs2);
            accrued_flags |= softfloat_exceptionFlags;
            return val;
        };
    case 0b10100: // VFNCVT.F.F.W
    case 0b10101: // VFNCVT.ROD.F.F.W
        return [](uint8_t rm, uint8_t& accrued_flags, src_elem_t vs2) {
            dest_elem_t val = fp_f_to_f<dest_elem_t, src_elem_t>(rm, vs2);
            accrued_flags |= softfloat_exceptionFlags;
            return val;
        };
    default:
        throw new std::runtime_error("Unknown funct in get_fp_narrowing_fn");
    }
}
template <unsigned VLEN, typename dest_elem_t, typename src_elem_t>
void fp_vector_unary_n(uint8_t* V, unsigned unary_op, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                       uint8_t rm) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs2_view = get_vreg<VLEN, src_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, dest_elem_t>(V, vd, vlmax);
    auto fn = get_fp_narrowing_fn<dest_elem_t, src_elem_t>(unary_op);
    uint8_t accrued_flags = 0;
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_view[idx] = fn(rm, accrued_flags, vs2_view[idx]);
        else if(vtype.vma())
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
    }
    softfloat_exceptionFlags = accrued_flags;
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}
template <typename elem_size_t> bool fp_eq(elem_size_t, elem_size_t);
template <> inline bool fp_eq<uint16_t>(uint16_t v2, uint16_t v1) { return fcmp_h(v2, v1, 0); }
template <> inline bool fp_eq<uint32_t>(uint32_t v2, uint32_t v1) { return fcmp_s(v2, v1, 0); }
template <> inline bool fp_eq<uint64_t>(uint64_t v2, uint64_t v1) { return fcmp_d(v2, v1, 0); }
template <typename elem_size_t> bool fp_le(elem_size_t, elem_size_t);
template <> inline bool fp_le<uint16_t>(uint16_t v2, uint16_t v1) { return fcmp_h(v2, v1, 1); }
template <> inline bool fp_le<uint32_t>(uint32_t v2, uint32_t v1) { return fcmp_s(v2, v1, 1); }
template <> inline bool fp_le<uint64_t>(uint64_t v2, uint64_t v1) { return fcmp_d(v2, v1, 1); }
template <typename elem_size_t> bool fp_lt(elem_size_t, elem_size_t);
template <> inline bool fp_lt<uint16_t>(uint16_t v2, uint16_t v1) { return fcmp_h(v2, v1, 2); }
template <> inline bool fp_lt<uint32_t>(uint32_t v2, uint32_t v1) { return fcmp_s(v2, v1, 2); }
template <> inline bool fp_lt<uint64_t>(uint64_t v2, uint64_t v1) { return fcmp_d(v2, v1, 2); }
template <typename elem_t> std::function<bool(uint8_t, uint8_t&, elem_t, elem_t)> get_fp_mask_funct(unsigned funct6) {
    switch(funct6) {
    case 0b011000: // VMFEQ
        return [](uint8_t rm, uint8_t& accrued_flags, elem_t vs2, elem_t vs1) {
            elem_t val = fp_eq(vs2, vs1);
            accrued_flags |= softfloat_exceptionFlags;
            return val;
        };
    case 0b011001: // VMFLE
        return [](uint8_t rm, uint8_t& accrued_flags, elem_t vs2, elem_t vs1) {
            elem_t val = fp_le(vs2, vs1);
            accrued_flags |= softfloat_exceptionFlags;
            return val;
        };
    case 0b011011: // VMFLT
        return [](uint8_t rm, uint8_t& accrued_flags, elem_t vs2, elem_t vs1) {
            elem_t val = fp_lt(vs2, vs1);
            accrued_flags |= softfloat_exceptionFlags;
            return val;
        };
    case 0b011100: // VMFNE
        return [](uint8_t rm, uint8_t& accrued_flags, elem_t vs2, elem_t vs1) {
            elem_t val = !fp_eq(vs2, vs1);
            accrued_flags |= softfloat_exceptionFlags;
            return val;
        };
    case 0b011101: // VMFGT
        return [](uint8_t rm, uint8_t& accrued_flags, elem_t vs2, elem_t vs1) {
            elem_t val = fp_lt(vs1, vs2);
            accrued_flags |= softfloat_exceptionFlags;
            return val;
        };
    case 0b011111: // VMFGE
        return [](uint8_t rm, uint8_t& accrued_flags, elem_t vs2, elem_t vs1) {
            elem_t val = fp_le(vs1, vs2);
            accrued_flags |= softfloat_exceptionFlags;
            return val;
        };
    default:
        throw new std::runtime_error("Unknown funct6 in get_fp_mask_funct");
    }
}
template <unsigned VLEN, typename elem_t>
void mask_fp_vector_vector_op(uint8_t* V, unsigned funct6, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                              unsigned vs1, uint8_t rm) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs1_view = get_vreg<VLEN, elem_t>(V, vs1, vlmax);
    auto vs2_view = get_vreg<VLEN, elem_t>(V, vs2, vlmax);
    vmask_view vd_mask_view = read_vmask<VLEN>(V, VLEN, vd);
    auto fn = get_fp_mask_funct<elem_t>(funct6);
    uint8_t accrued_flags = 0;
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_mask_view[idx] = fn(rm, accrued_flags, vs2_view[idx], vs1_view[idx]);
        else if(vtype.vma())
            vd_mask_view[idx] = agnostic_behavior(vd_mask_view[idx]);
    }
    softfloat_exceptionFlags = accrued_flags;
    if(vtype.vta())
        for(size_t idx = vl; idx < VLEN; idx++)
            vd_mask_view[idx] = agnostic_behavior(vd_mask_view[idx]);
}
template <unsigned VLEN, typename elem_t>
void mask_fp_vector_imm_op(uint8_t* V, unsigned funct6, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                           elem_t imm, uint8_t rm) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs2_view = get_vreg<VLEN, elem_t>(V, vs2, vlmax);
    vmask_view vd_mask_view = read_vmask<VLEN>(V, VLEN, vd);
    auto fn = get_fp_mask_funct<elem_t>(funct6);
    uint8_t accrued_flags = 0;
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_mask_view[idx] = fn(rm, accrued_flags, vs2_view[idx], imm);
        else if(vtype.vma())
            vd_mask_view[idx] = agnostic_behavior(vd_mask_view[idx]);
    }
    softfloat_exceptionFlags = accrued_flags;
    if(vtype.vta())
        for(size_t idx = vl; idx < VLEN; idx++)
            vd_mask_view[idx] = agnostic_behavior(vd_mask_view[idx]);
}
template <unsigned VLEN>
void mask_mask_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, unsigned vd, unsigned vs2, unsigned vs1) {
    uint64_t vlmax = VLEN;
    auto vs1_view = read_vmask<VLEN>(V, vlmax, vs1);
    auto vs2_view = read_vmask<VLEN>(V, vlmax, vs2);
    auto vd_view = read_vmask<VLEN>(V, vlmax, vd);
    auto fn = get_mask_funct<unsigned>(funct6, funct3); // could be bool, but would break the make_signed_t in get_mask_funct
    for(size_t idx = vstart; idx < vl; idx++)
        vd_view[idx] = fn(vs2_view[idx], vs1_view[idx]);

    // the tail is all elements of the destination register beyond the first one
    for(size_t idx = 1; idx < VLEN; idx++)
        vd_view[idx] = agnostic_behavior(vd_view[idx]);
}
template <unsigned VLEN> uint64_t vcpop(uint8_t* V, uint64_t vl, uint64_t vstart, bool vm, unsigned vs2) {
    uint64_t vlmax = VLEN;
    auto vs2_view = read_vmask<VLEN>(V, vlmax, vs2);
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    unsigned running_total = 0;
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active && vs2_view[idx])
            running_total += 1;
    }
    return running_total;
}
template <unsigned VLEN> uint64_t vfirst(uint8_t* V, uint64_t vl, uint64_t vstart, bool vm, unsigned vs2) {
    uint64_t vlmax = VLEN;
    auto vs2_view = read_vmask<VLEN>(V, vlmax, vs2);
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active && vs2_view[idx])
            return idx;
    }
    return -1;
}
inline std::function<bool(bool&, bool)> get_mask_set_funct(unsigned enc) {
    switch(enc) {
    case 0b00001: // VMSBF
        return [](bool& marker, bool vs2) {
            if(marker)
                return 0;
            if(vs2) {
                marker = true;
                return 0;
            } else
                return 1;
        };
    case 0b00010: // VMSOF
        return [](bool& marker, bool vs2) {
            if(marker)
                return 0;
            if(vs2) {
                marker = true;
                return 1;
            } else
                return 0;
        };
    case 0b00011: // VMSIF
        return [](bool& marker, bool vs2) {
            if(marker)
                return 0;
            if(vs2) {
                marker = true;
                return 1;
            } else
                return 1;
        };
    default:
        throw new std::runtime_error("Unknown enc in get_mask_set_funct");
    }
}
template <unsigned VLEN> void mask_set_op(uint8_t* V, unsigned enc, uint64_t vl, uint64_t vstart, bool vm, unsigned vd, unsigned vs2) {
    uint64_t vlmax = VLEN;
    auto vs2_view = read_vmask<VLEN>(V, vlmax, vs2);
    auto vd_view = read_vmask<VLEN>(V, vlmax, vd);
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto fn = get_mask_set_funct(enc);
    bool marker = false;
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_view[idx] = fn(marker, vs2_view[idx]);
    }
    // the tail is all elements of the destination register beyond the first one
    for(size_t idx = 1; idx < VLEN; idx++)
        vd_view[idx] = agnostic_behavior(vd_view[idx]);
}
template <unsigned VLEN, typename src_elem_t>
void viota(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    auto vs2_view = read_vmask<VLEN>(V, vlmax, vs2);
    auto vd_view = get_vreg<VLEN, src_elem_t>(V, vd, vlmax);
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    unsigned current = 0;
    for(size_t idx = vstart; idx < std::min(vl, vlmax); idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active) {
            vd_view[idx] = current;
            if(vs2_view[idx])
                current += 1;
        }
    }
}
template <unsigned VLEN, typename src_elem_t> void vid(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    auto vd_view = get_vreg<VLEN, src_elem_t>(V, vd, vlmax);
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    for(size_t idx = vstart; idx < std::min(vl, vlmax); idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_view[idx] = idx;
    }
}
template <unsigned VLEN, typename src_elem_t> uint64_t scalar_move(uint8_t* V, vtype_t vtype, unsigned vd, uint64_t val, bool to_vector) {
    unsigned vlmax = VLEN * vtype.lmul() / vtype.sew();
    auto vd_view = get_vreg<VLEN, src_elem_t>(V, vd, vlmax);
    if(to_vector) {
        vd_view[0] = val;
        if(vtype.vta())
            for(size_t idx = 1; idx < vlmax; idx++)
                vd_view[idx] = agnostic_behavior(vd_view[idx]);
    }
    return static_cast<int64_t>(static_cast<std::make_signed_t<src_elem_t>>(vd_view[0]));
}
template <unsigned VLEN, typename src_elem_t>
void vector_slideup(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm) {
    uint64_t vlmax = VLEN * vtype.lmul() / (sizeof(src_elem_t) * 8);
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs2_view = get_vreg<VLEN, src_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, src_elem_t>(V, vd, vlmax);
    for(size_t idx = std::max(vstart, imm); idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_view[idx] = idx - imm < vlmax ? vs2_view[idx - imm] : 0;
        else if(vtype.vma())
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
    }
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}
template <unsigned VLEN, typename src_elem_t>
void vector_slidedown(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm) {
    uint64_t vlmax = VLEN * vtype.lmul() / (sizeof(src_elem_t) * 8);
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs2_view = get_vreg<VLEN, src_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, src_elem_t>(V, vd, vlmax);
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_view[idx] = std::numeric_limits<uint64_t>::max() - idx > imm && idx + imm < vlmax ? vs2_view[idx + imm] : 0;
        else if(vtype.vma())
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
    }
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}
template <unsigned VLEN, typename src_elem_t>
void vector_slide1up(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm) {
    vector_slideup<VLEN, src_elem_t>(V, vl, vstart, vtype, vm, vd, vs2, 1);
    vmask_view mask_reg = read_vmask<VLEN>(V, 1);
    auto vd_view = get_vreg<VLEN, src_elem_t>(V, vd, 1);
    if(vm || mask_reg[0])
        vd_view[0] = imm;
    else if(vtype.vma())
        vd_view[0] = agnostic_behavior(vd_view[0]);
}
template <unsigned VLEN, typename src_elem_t>
void vector_slide1down(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm) {
    vector_slidedown<VLEN, src_elem_t>(V, vl, vstart, vtype, vm, vd, vs2, 1);

    vmask_view mask_reg = read_vmask<VLEN>(V, vl);
    auto vd_view = get_vreg<VLEN, src_elem_t>(V, vd, vl);
    if(vm || mask_reg[vl - 1])
        vd_view[vl - 1] = imm;
    else if(vtype.vma())
        vd_view[0] = agnostic_behavior(vd_view[0]);
}
template <unsigned VLEN, typename dest_elem_t, typename scr_elem_t>
void vector_vector_gather(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2, unsigned vs1) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs1_view = get_vreg<VLEN, scr_elem_t>(V, vs1, vlmax);
    auto vs2_view = get_vreg<VLEN, dest_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, dest_elem_t>(V, vd, vlmax);
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_view[idx] = (vs1_view[idx] >= vlmax) ? 0 : vs2_view[vs1_view[idx]];
        else if(vtype.vma())
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
    }
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}
template <unsigned VLEN, typename scr_elem_t>
void vector_imm_gather(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax);
    auto vs2_view = get_vreg<VLEN, scr_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, scr_elem_t>(V, vd, vlmax);
    for(size_t idx = vstart; idx < vl; idx++) {
        bool mask_active = vm ? 1 : mask_reg[idx];
        if(mask_active)
            vd_view[idx] = (imm >= vlmax) ? 0 : vs2_view[imm];
        else if(vtype.vma())
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
    }
    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}
template <unsigned VLEN, typename scr_elem_t>
void vector_compress(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, unsigned vd, unsigned vs2, unsigned vs1) {
    uint64_t vlmax = VLEN * vtype.lmul() / vtype.sew();
    vmask_view mask_reg = read_vmask<VLEN>(V, vlmax, vs1);
    auto vs2_view = get_vreg<VLEN, scr_elem_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, scr_elem_t>(V, vd, vlmax);
    unsigned current_pos = 0;
    for(size_t idx = vstart; idx < vl; idx++)
        if(mask_reg[idx]) {
            vd_view[current_pos] = vs2_view[idx];
            current_pos += 1;
        }

    if(vtype.vta())
        for(size_t idx = vl; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}
template <unsigned VLEN> void vector_whole_move(uint8_t* V, unsigned vd, unsigned vs2, unsigned count) {
    auto vd_view = get_vreg<VLEN, uint8_t>(V, vd, 1);
    auto vs2_view = get_vreg<VLEN, uint8_t>(V, vs2, 1);
    memcpy(vd_view.start, vs2_view.start, VLEN / 8 * count);
}

template <unsigned VLEN, unsigned EGS>
void vector_vector_crypto(uint8_t* V, unsigned funct6, uint64_t eg_len, uint64_t eg_start, vtype_t vtype, unsigned vd, unsigned vs2,
                          unsigned vs1) {
    uint64_t vlmax = VLEN * vtype.lmul() / (vtype.sew() * EGS);
    auto vs1_view = get_vreg<VLEN, uint128_t>(V, vs1, vlmax);
    auto vs2_view = get_vreg<VLEN, uint128_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, uint128_t>(V, vd, vlmax);
    auto fn = get_crypto_funct(funct6, vs1);
    for(size_t idx = eg_start; idx < eg_len; idx++) {
        vd_view[idx] = fn(vd_view[idx], vs2_view[idx], vs1_view[idx]);
    }
    if(vtype.vta())
        for(size_t idx = eg_len; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}
template <unsigned VLEN, unsigned EGS>
void vector_scalar_crypto(uint8_t* V, unsigned funct6, uint64_t eg_len, uint64_t eg_start, vtype_t vtype, unsigned vd, unsigned vs2,
                          unsigned vs1) {
    uint64_t vlmax = VLEN * vtype.lmul() / (vtype.sew() * EGS);
    auto vs2_val = get_vreg<VLEN, uint128_t>(V, vs2, vlmax)[0];
    auto vd_view = get_vreg<VLEN, uint128_t>(V, vd, vlmax);
    auto fn = get_crypto_funct(funct6, vs1);
    for(size_t idx = eg_start; idx < eg_len; idx++) {
        vd_view[idx] = fn(vd_view[idx], vs2_val, -1);
    }
    if(vtype.vta())
        for(size_t idx = eg_len; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}

template <unsigned VLEN, unsigned EGS>
void vector_imm_crypto(uint8_t* V, unsigned funct6, uint64_t eg_len, uint64_t eg_start, vtype_t vtype, unsigned vd, unsigned vs2,
                       uint8_t imm) {
    uint64_t vlmax = VLEN * vtype.lmul() / (vtype.sew() * EGS);
    auto vs2_view = get_vreg<VLEN, uint128_t>(V, vs2, vlmax);
    auto vd_view = get_vreg<VLEN, uint128_t>(V, vd, vlmax);
    auto fn = get_crypto_funct(funct6, -1);
    for(size_t idx = eg_start; idx < eg_len; idx++) {
        vd_view[idx] = fn(vd_view[idx], vs2_view[idx], imm);
    }
    if(vtype.vta())
        for(size_t idx = eg_len; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
}
template <typename T> T rotr(T x, unsigned n) {
    assert(n < sizeof(T) * 8);
    return (x >> n) | (x << (sizeof(T) * 8 - n));
}
template <typename T> T shr(T x, unsigned n) {
    assert(n < sizeof(T) * 8);
    return (x >> n);
}
template <typename T> T sum0(T);
template <> inline uint32_t sum0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
template <> inline uint64_t sum0(uint64_t x) { return rotr(x, 28) ^ rotr(x, 34) ^ rotr(x, 39); }
template <typename T> T sum1(T);
template <> inline uint32_t sum1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
template <> inline uint64_t sum1(uint64_t x) { return rotr(x, 14) ^ rotr(x, 18) ^ rotr(x, 41); }
template <typename T> T ch(T x, T y, T z) { return ((x & y) ^ ((~x) & z)); }
template <typename T> T maj(T x, T y, T z) { return ((x & y) ^ (x & z) ^ (y & z)); }
template <typename T> T sig0(T);
template <> inline uint32_t sig0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ shr(x, 3); }
template <> inline uint64_t sig0(uint64_t x) { return rotr(x, 1) ^ rotr(x, 8) ^ shr(x, 7); }
template <typename T> T sig1(T);
template <> inline uint32_t sig1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ shr(x, 10); }
template <> inline uint64_t sig1(uint64_t x) { return rotr(x, 19) ^ rotr(x, 61) ^ shr(x, 6); }
template <typename T> std::function<void(vreg_view<T>&, vreg_view<T>&, vreg_view<T>&)> get_crypto_funct(unsigned int funct6) {
    switch(funct6) {
    case 0b101110: // VSHA2CH
        return [](vreg_view<T>& vd_view, vreg_view<T>& vs2_view, vreg_view<T>& vs1_view) {
            T a = vs2_view[3];
            T b = vs2_view[2];
            T c = vd_view[3];
            T d = vd_view[2];
            T e = vs2_view[1];
            T f = vs2_view[0];
            T g = vd_view[1];
            T h = vd_view[0];
            T W0 = vs1_view[2];
            T W1 = vs1_view[3];
            T T1 = h + sum1(e) + ch(e, f, g) + W0;
            T T2 = sum0(a) + maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + T1;
            d = c;
            c = b;
            b = a;
            a = T1 + T2;
            T1 = h + sum1(e) + ch(e, f, g) + W1;
            T2 = sum0(a) + maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + T1;
            d = c;
            c = b;
            b = a;
            a = T1 + T2;
            vd_view[0] = f;
            vd_view[1] = e;
            vd_view[2] = b;
            vd_view[3] = a;
        };
    case 0b101111: // VSHA2CL
        return [](vreg_view<T>& vd_view, vreg_view<T>& vs2_view, vreg_view<T>& vs1_view) {
            T a = vs2_view[3];
            T b = vs2_view[2];
            T c = vd_view[3];
            T d = vd_view[2];
            T e = vs2_view[1];
            T f = vs2_view[0];
            T g = vd_view[1];
            T h = vd_view[0];
            T W0 = vs1_view[0];
            T W1 = vs1_view[1];
            T T1 = h + sum1(e) + ch(e, f, g) + W0;
            T T2 = sum0(a) + maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + T1;
            d = c;
            c = b;
            b = a;
            a = T1 + T2;
            T1 = h + sum1(e) + ch(e, f, g) + W1;
            T2 = sum0(a) + maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + T1;
            d = c;
            c = b;
            b = a;
            a = T1 + T2;
            vd_view[0] = f;
            vd_view[1] = e;
            vd_view[2] = b;
            vd_view[3] = a;
        };
    case 0b101101: // VSHA2MS
        return [](vreg_view<T>& vd_view, vreg_view<T>& vs2_view, vreg_view<T>& vs1_view) {
            T W0 = vd_view[0];
            T W1 = vd_view[1];
            T W2 = vd_view[2];
            T W3 = vd_view[3];

            T W4 = vs2_view[0];
            T W9 = vs2_view[1];
            T W10 = vs2_view[2];
            T W11 = vs2_view[3];

            T W12 = vs1_view[0];
            T W13 = vs1_view[1];
            T W14 = vs1_view[2];
            T W15 = vs1_view[3];

            T W16 = sig1(W14) + W9 + sig0(W1) + W0;
            T W17 = sig1(W15) + W10 + sig0(W2) + W1;
            T W18 = sig1(W16) + W11 + sig0(W3) + W2;
            T W19 = sig1(W17) + W12 + sig0(W4) + W3;

            vd_view[0] = W16;
            vd_view[1] = W17;
            vd_view[2] = W18;
            vd_view[3] = W19;
        };
    default:
        throw new std::runtime_error("Unsupported operation in get_crypto_funct");
    }
}
template <unsigned VLEN, unsigned EGS, typename elem_type_t>
void vector_crypto(uint8_t* V, unsigned funct6, uint64_t eg_len, uint64_t eg_start, vtype_t vtype, unsigned vd, unsigned vs2,
                   unsigned vs1) {
    auto fn = get_crypto_funct<elem_type_t>(funct6);
    auto vd_view = get_vreg<VLEN, elem_type_t>(V, vd, EGS);
    auto vs2_view = get_vreg<VLEN, elem_type_t>(V, vs2, EGS);
    auto vs1_view = get_vreg<VLEN, elem_type_t>(V, vs1, EGS);
    for(size_t idx = eg_start; idx < eg_len; idx++) {
        fn(vd_view, vs2_view, vs1_view);
        // We cannot use views in case EGW < VLEN, as views can only address the start of a register
        vd_view.start += EGS * sizeof(elem_type_t);
        vs2_view.start += EGS * sizeof(elem_type_t);
        vs1_view.start += EGS * sizeof(elem_type_t);
    }
    if(vtype.vta()) {
        uint64_t vlmax = VLEN * vtype.lmul() / (vtype.sew());
        auto vd_view = get_vreg<VLEN, elem_type_t>(V, vd, vlmax);
        for(size_t idx = eg_len * EGS; idx < vlmax; idx++)
            vd_view[idx] = agnostic_behavior(vd_view[idx]);
    }
}

template <typename dest_elem_t, typename src_elem_t> dest_elem_t brev(src_elem_t vs2) {
    constexpr dest_elem_t bits = sizeof(src_elem_t) * 8;
    dest_elem_t result = 0;
    for(size_t i = 0; i < bits; ++i) {
        result <<= 1;
        result |= (vs2 & 1);
        vs2 >>= 1;
    }
    return result;
};
template <typename dest_elem_t, typename src_elem_t> dest_elem_t brev8(src_elem_t vs2) {
    constexpr unsigned byte_count = sizeof(src_elem_t);
    dest_elem_t result = 0;
    for(size_t i = 0; i < byte_count; ++i) {
        dest_elem_t byte = (vs2 >> (i * 8)) & 0xFF;
        byte = ((byte & 0xF0) >> 4) | ((byte & 0x0F) << 4);
        byte = ((byte & 0xCC) >> 2) | ((byte & 0x33) << 2);
        byte = ((byte & 0xAA) >> 1) | ((byte & 0x55) << 1);
        result |= byte << (i * 8);
    }
    return result;
};
} // namespace softvector