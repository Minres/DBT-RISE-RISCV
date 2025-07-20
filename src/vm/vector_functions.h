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

#ifndef _VM_VECTOR_FUNCTIONS_H_
#define _VM_VECTOR_FUNCTIONS_H_

#include "iss/arch_if.h"
#include "iss/vm_types.h"
#include <cstdint>
#include <functional>
#include <stdint.h>
namespace softvector {
#ifndef _MSC_VER
using int128_t = __int128;
using uint128_t = unsigned __int128;
#endif
const unsigned RFS = 32;

struct vtype_t {
    uint64_t underlying;
    vtype_t(uint32_t vtype_val);
    vtype_t(uint64_t vtype_val);
    unsigned sew();
    double lmul();
    bool vill();
    bool vma();
    bool vta();
};
class mask_bit_reference {
    uint8_t* start;
    uint8_t pos;

public:
    mask_bit_reference& operator=(const bool new_value);
    mask_bit_reference(uint8_t* start, uint8_t pos);
    operator bool() const;
};

struct vmask_view {
    uint8_t* start;
    size_t elem_count;
    mask_bit_reference operator[](size_t) const;
};
vmask_view read_vmask(uint8_t* V, uint16_t VLEN, uint16_t elem_count, uint8_t reg_idx = 0);
template <unsigned VLEN> vmask_view read_vmask(uint8_t* V, uint16_t elem_count, uint8_t reg_idx = 0);
std::function<uint128_t(uint128_t, uint128_t, uint128_t)> get_crypto_funct(unsigned funct6, unsigned vs1);

template <typename dest_elem_t, typename src_elem_t = dest_elem_t> dest_elem_t brev(src_elem_t vs2);
template <typename dest_elem_t, typename src_elem_t = dest_elem_t> dest_elem_t brev8(src_elem_t vs2);

bool softvec_read(void* core, uint64_t addr, uint64_t length, uint8_t* data);
bool softvec_write(void* core, uint64_t addr, uint64_t length, uint8_t* data);

template <unsigned VLEN, typename eew_t>
uint64_t vector_load_store(void* core, std::function<bool(void*, uint64_t, uint64_t, uint8_t*)> load_store_fn, uint8_t* V, uint64_t vl,
                           uint64_t vstart, vtype_t vtype, bool vm, uint8_t vd, uint64_t rs1, uint8_t segment_size, int64_t stride = 0,
                           bool use_stride = false);
template <unsigned XLEN, unsigned VLEN, typename eew_t, typename sew_t>
uint64_t vector_load_store_index(void* core, std::function<bool(void*, uint64_t, uint64_t, uint8_t*)> load_store_fn, uint8_t* V,
                                 uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, uint8_t vd, uint64_t rs1, uint8_t vs2,
                                 uint8_t segment_size);
template <unsigned VLEN, typename dest_elem_t, typename src2_elem_t = dest_elem_t, typename src1_elem_t = src2_elem_t>
void vector_vector_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd,
                      unsigned vs2, unsigned vs1);
template <unsigned VLEN, typename dest_elem_t, typename src2_elem_t = dest_elem_t, typename src1_elem_t = src2_elem_t>
void vector_imm_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd,
                   unsigned vs2, typename std::make_signed<src1_elem_t>::type imm);
template <unsigned VLEN, typename elem_t>
void vector_vector_carry(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, unsigned vd,
                         unsigned vs2, unsigned vs1, signed carry);
template <unsigned VLEN, typename elem_t>
void vector_imm_carry(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, unsigned vd, unsigned vs2,
                      typename std::make_signed<elem_t>::type imm, signed carry);
template <unsigned VLEN, typename scr_elem_t>
void vector_vector_merge(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2, unsigned vs1);
template <unsigned VLEN, typename scr_elem_t>
void vector_imm_merge(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm);
template <unsigned VLEN, typename dest_elem_t, typename src2_elem_t = dest_elem_t>
void vector_unary_op(uint8_t* V, unsigned unary_op, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2);
template <unsigned VLEN, typename elem_t>
void mask_vector_vector_op(uint8_t* V, unsigned funct, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd,
                           unsigned vs2, unsigned vs1);
template <unsigned VLEN, typename elem_t>
void mask_vector_imm_op(uint8_t* V, unsigned funct, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd,
                        unsigned vs2, typename std::make_signed<elem_t>::type imm);
void carry_vector_vector_op(uint8_t* V, unsigned funct, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                            unsigned vs1);
template <unsigned VLEN, typename elem_t>
void carry_vector_imm_op(uint8_t* V, unsigned funct, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                         typename std::make_signed<elem_t>::type imm);
template <unsigned VLEN, typename dest_elem_t, typename src2_elem_t = dest_elem_t, typename src1_elem_t = dest_elem_t>
bool sat_vector_vector_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, int64_t vxrm, bool vm,
                          unsigned vd, unsigned vs2, unsigned vs1);
template <unsigned VLEN, typename dest_elem_t, typename src2_elem_t = dest_elem_t, typename src1_elem_t = dest_elem_t>
bool sat_vector_imm_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, int64_t vxrm, bool vm,
                       unsigned vd, unsigned vs2, typename std::make_signed<src1_elem_t>::type imm);
template <unsigned VLEN, typename dest_elem_t, typename src_elem_t = dest_elem_t>
void vector_red_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd,
                   unsigned vs2, unsigned vs1);
template <unsigned VLEN>
void mask_mask_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, unsigned vd, unsigned vs2, unsigned vs1);
template <unsigned VLEN> uint64_t vcpop(uint8_t* V, uint64_t vl, uint64_t vstart, bool vm, unsigned vs2);
template <unsigned VLEN> uint64_t vfirst(uint8_t* V, uint64_t vl, uint64_t vstart, bool vm, unsigned vs2);
template <unsigned VLEN> void mask_set_op(uint8_t* V, unsigned enc, uint64_t vl, uint64_t vstart, bool vm, unsigned vd, unsigned vs2);
template <unsigned VLEN, typename src_elem_t>
void viota(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2);
template <unsigned VLEN, typename src_elem_t> void vid(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd);
template <unsigned VLEN, typename src_elem_t> uint64_t scalar_move(uint8_t* V, vtype_t vtype, unsigned vd, uint64_t val, bool to_vector);
template <unsigned VLEN, typename src_elem_t>
void vector_slideup(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm);
template <unsigned VLEN, typename src_elem_t>
void vector_slidedown(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm);
template <unsigned VLEN, typename src_elem_t>
void vector_slide1up(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm);
template <unsigned VLEN, typename src_elem_t>
void vector_slide1down(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm);
template <unsigned VLEN, typename dest_elem_t, typename scr_elem_t = dest_elem_t>
void vector_vector_gather(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2, unsigned vs1);
template <unsigned VLEN, typename scr_elem_t>
void vector_imm_gather(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm);
template <unsigned VLEN, typename scr_elem_t>
void vector_compress(uint8_t* V, uint64_t vl, uint64_t vstart, vtype_t vtype, unsigned vd, unsigned vs2, unsigned vs1);
template <unsigned VLEN> void vector_whole_move(uint8_t* V, unsigned vd, unsigned vs2, unsigned count);
template <unsigned VLEN, typename dest_elem_t, typename src_elem_t = dest_elem_t>
void fp_vector_red_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd,
                      unsigned vs2, unsigned vs1, uint8_t rm);
template <unsigned VLEN, typename dest_elem_t, typename src2_elem_t = dest_elem_t, typename src1_elem_t = src2_elem_t>
void fp_vector_vector_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd,
                         unsigned vs2, unsigned vs1, uint8_t rm);
template <unsigned VLEN, typename dest_elem_t, typename src2_elem_t = dest_elem_t, typename src1_elem_t = src2_elem_t>
void fp_vector_imm_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd,
                      unsigned vs2, src1_elem_t imm, uint8_t rm);
template <unsigned VLEN, typename elem_t>
void fp_vector_unary_op(uint8_t* V, unsigned encoding_space, unsigned unary_op, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm,
                        unsigned vd, unsigned vs2, uint8_t rm);
template <unsigned VLEN, typename dest_elem_t, typename src_elem_t>
void fp_vector_unary_w(uint8_t* V, unsigned unary_op, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                       uint8_t rm);
template <unsigned VLEN, typename dest_elem_t, typename src_elem_t>
void fp_vector_unary_n(uint8_t* V, unsigned unary_op, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                       uint8_t rm);
template <unsigned VLEN, typename elem_t>
void mask_fp_vector_vector_op(uint8_t* V, unsigned funct6, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                              unsigned vs1, uint8_t rm);
template <unsigned VLEN, typename elem_t>
void mask_fp_vector_imm_op(uint8_t* V, unsigned funct6, uint64_t vl, uint64_t vstart, vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                           elem_t imm, uint8_t rm);
template <unsigned VLEN, unsigned EGS>
void vector_vector_crypto(uint8_t* V, unsigned funct6, uint64_t eg_len, uint64_t eg_start, vtype_t vtype, unsigned vd, unsigned vs2,
                          unsigned vs1);
template <unsigned VLEN, unsigned EGS>
void vector_scalar_crypto(uint8_t* V, unsigned funct6, uint64_t eg_len, uint64_t eg_start, vtype_t vtype, unsigned vd, unsigned vs2,
                          unsigned vs1);
template <unsigned VLEN, unsigned EGS>
void vector_imm_crypto(uint8_t* V, unsigned funct6, uint64_t eg_len, uint64_t eg_start, vtype_t vtype, unsigned vd, unsigned vs2,
                       uint8_t imm);
template <unsigned VLEN, unsigned EGS, typename elem_type_t>
void vector_crypto(uint8_t* V, unsigned funct6, uint64_t eg_len, uint64_t eg_start, vtype_t vtype, unsigned vd, unsigned vs2, unsigned vs1);
} // namespace softvector
#include "vm/vector_functions.hpp"
#endif /* _VM_VECTOR_FUNCTIONS_H_ */
