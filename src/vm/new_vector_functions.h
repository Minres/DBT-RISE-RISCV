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

#ifndef RISCV_SRC_VM_VECTOR_FUNCTIONS_H_
#define RISCV_SRC_VM_VECTOR_FUNCTIONS_H_
#include <cstdint>
#include <vm/vector_functions.h>

// To make the language server happy
#ifndef CUR_VLEN
#define CUR_VLEN 0
#endif
#ifndef CUR_XLEN
#define CUR_XLEN 0
#endif
extern "C" {
uint64_t vlseg(void* core, uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint64_t rs1_val,
               uint8_t width_val, uint8_t segment_size);
uint64_t vsseg(void* core, uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint64_t rs1_val,
               uint8_t width_val, uint8_t segment_size);
uint64_t vlsseg(void* core, uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint64_t rs1_val,
                uint8_t width_val, uint8_t segment_size, int64_t stride);
uint64_t vssseg(void* core, uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint64_t rs1_val,
                uint8_t width_val, uint8_t segment_size, int64_t stride);
uint64_t vlxseg(void* core, uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint64_t rs1_val,
                uint8_t vs2, uint8_t segment_size, uint8_t index_byte_size, uint8_t data_byte_size, bool ordered);
uint64_t vsxseg(void* core, uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vs3, uint64_t rs1_val,
                uint8_t vs2, uint8_t segment_size, uint8_t index_byte_size, uint8_t data_byte_size, bool ordered);
void vector_vector_op(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t sew_val);
void vector_imm_op(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                   uint8_t vs2, int64_t imm, uint8_t sew_val);
void vector_vector_wv(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t sew_val);
void vector_imm_wv(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                   uint8_t vs2, int64_t imm, uint8_t sew_val);
void vector_vector_ww(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t sew_val);
void vector_imm_ww(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                   uint8_t vs2, int64_t imm, uint8_t sew_val);
void vector_extend(uint8_t* V, uint8_t unary_op, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t vs2,
                   uint8_t target_sew_pow, uint8_t frac_pow);
void vector_vector_carry(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, uint8_t vd,
                         uint8_t vs2, uint8_t vs1, uint8_t sew_val, int8_t carry);
void vector_imm_carry(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, uint8_t vd,
                      uint8_t vs2, int64_t imm, uint8_t sew_val, int8_t carry);
void carry_vector_vector_op(uint8_t* V, unsigned funct6, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, unsigned vd,
                            unsigned vs2, unsigned vs1, uint8_t sew_val);
void carry_vector_imm_op(uint8_t* V, unsigned funct6, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, unsigned vd,
                         unsigned vs2, int64_t imm, uint8_t sew_val);
void mask_vector_vector_op(uint8_t* V, unsigned funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                           unsigned vd, unsigned vs2, unsigned vs1, uint8_t sew_val);
void mask_vector_imm_op(uint8_t* V, unsigned funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                        unsigned vd, unsigned vs2, int64_t imm, uint8_t sew_val);
void vector_vector_vw(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t sew_val);
void vector_imm_vw(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                   uint8_t vs2, int64_t imm, uint8_t sew_val);
void vector_vector_merge(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t vs2, uint8_t vs1,
                         uint8_t sew_val);
void vector_imm_merge(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t vs2, int64_t imm,
                      uint8_t sew_val);
bool sat_vector_vector_op(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype,
                          uint64_t vxrm, bool vm, uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t sew_val);
bool sat_vector_imm_op(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, uint64_t vxrm,
                       bool vm, uint8_t vd, uint8_t vs2, int64_t imm, uint8_t sew_val);
bool sat_vector_vector_vw(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype,
                          uint64_t vxrm, bool vm, uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t sew_val);
bool sat_vector_imm_vw(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, uint64_t vxrm,
                       bool vm, uint8_t vd, uint8_t vs2, int64_t imm, uint8_t sew_val);
void vector_red_op(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                   uint8_t vs2, uint8_t vs1, uint8_t sew_val);
void vector_red_wv(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                   uint8_t vs2, uint8_t vs1, uint8_t sew_val);
void mask_mask_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, unsigned vd, unsigned vs2, unsigned vs1);
uint64_t vcpop(uint8_t* V, uint64_t vl, uint64_t vstart, bool vm, unsigned vs2);
int64_t vfirst(uint8_t* V, uint64_t vl, uint64_t vstart, bool vm, unsigned vs2);
void mask_set_op(uint8_t* V, unsigned enc, uint64_t vl, uint64_t vstart, bool vm, unsigned vd, unsigned vs2);
void viota(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t vs2, uint8_t sew_val);
void vid(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t sew_val);
void scalar_to_vector(uint8_t* V, softvector::vtype_t vtype, unsigned vd, uint64_t val, uint8_t sew_val);
uint64_t scalar_from_vector(uint8_t* V, softvector::vtype_t vtype, unsigned vd, uint8_t sew_val);
void vector_slideup(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm,
                    uint8_t sew_val);
void vector_slidedown(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm,
                      uint8_t sew_val);
void vector_slide1up(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm,
                     uint8_t sew_val);
void vector_slide1down(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                       uint64_t imm, uint8_t sew_val);
void vector_vector_gather(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t vs2,
                          uint8_t vs1, uint8_t sew_val);
void vector_vector_gatherei16(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t vs2,
                              uint8_t vs1, uint8_t sew_val);
void vector_imm_gather(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t vs2, uint64_t imm,
                       uint8_t sew_val);
void vector_compress(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, uint8_t vd, uint8_t vs2, uint8_t vs1,
                     uint8_t sew_val);
void vector_whole_move(uint8_t* V, uint8_t vd, uint8_t vs2, uint8_t count);
uint64_t fp_scalar_from_vector(uint8_t* V, softvector::vtype_t vtype, unsigned vd, uint8_t sew_val);
void fp_vector_slide1up(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                        uint64_t imm, uint8_t sew_val);
void fp_vector_slide1down(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                          uint64_t imm, uint8_t sew_val);
void fp_vector_red_op(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t rm, uint8_t sew_val);
void fp_vector_red_wv(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t rm, uint8_t sew_val);
void fp_vector_vector_op(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                         uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t rm, uint8_t sew_val);
void fp_vector_imm_op(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint64_t imm, uint8_t rm, uint8_t sew_val);
void fp_vector_vector_wv(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                         uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t rm, uint8_t sew_val);
void fp_vector_imm_wv(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint64_t imm, uint8_t rm, uint8_t sew_val);
void fp_vector_vector_ww(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                         uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t rm, uint8_t sew_val);
void fp_vector_imm_ww(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint64_t imm, uint8_t rm, uint8_t sew_val);
void fp_vector_unary_op(uint8_t* V, uint8_t encoding_space, uint8_t unary_op, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype,
                        bool vm, uint8_t vd, uint8_t vs2, uint8_t rm, uint8_t sew_val);
void mask_fp_vector_vector_op(uint8_t* V, uint8_t funct6, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                              uint8_t vs2, uint8_t vs1, uint8_t rm, uint8_t sew_val);
void mask_fp_vector_imm_op(uint8_t* V, uint8_t funct6, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                           uint8_t vs2, uint64_t imm, uint8_t rm, uint8_t sew_val);
void fp_vector_imm_merge(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t vs2,
                         uint64_t imm, uint8_t sew_val);
void fp_vector_unary_w(uint8_t* V, uint8_t unary_op, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                       uint8_t vs2, uint8_t rm, uint8_t sew_val);
void fp_vector_unary_n(uint8_t* V, uint8_t unary_op, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                       uint8_t vs2, uint8_t rm, uint8_t sew_val);
void vector_unary_op(uint8_t* V, uint8_t unary_op, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                     uint8_t vs2, uint8_t sew_val);
}
#endif /* RISCV_SRC_VM_VECTOR_FUNCTIONS_H_ */
