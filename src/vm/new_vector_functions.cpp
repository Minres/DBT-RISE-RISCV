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

#include <limits>
#include <vm/new_vector_functions.h>

#include "iss/arch_if.h"
#include "iss/vm_types.h"
#include <cstdint>
#include <vm/vector_functions.h>

using indexed_load_store_t = std::function<uint64_t(void*, std::function<bool(void*, uint64_t, uint64_t, uint8_t*)>, uint8_t*, uint64_t,
                                                    uint64_t, softvector::vtype_t, bool, uint8_t, uint64_t, uint8_t, uint8_t)>;
template <typename T1, typename T2> indexed_load_store_t getFunction() {
    return [](void* core, std::function<uint64_t(void*, uint64_t, uint64_t, uint8_t*)> load_store_fn, uint8_t* V, uint64_t vl,
              uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint64_t rs1, uint8_t vs2, uint8_t segment_size) {
        return softvector::vector_load_store_index<CUR_XLEN, CUR_VLEN, T1, T2>(core, load_store_fn, V, vl, vstart, vtype, vm, vd, rs1, vs2,
                                                                               segment_size);
    };
}

const std::array<std::array<indexed_load_store_t, 4>, 4> functionTable = {
    {{getFunction<uint8_t, uint8_t>(), getFunction<uint8_t, uint16_t>(), getFunction<uint8_t, uint32_t>(),
      getFunction<uint8_t, uint64_t>()},
     {getFunction<uint16_t, uint8_t>(), getFunction<uint16_t, uint16_t>(), getFunction<uint16_t, uint32_t>(),
      getFunction<uint16_t, uint64_t>()},
     {getFunction<uint32_t, uint8_t>(), getFunction<uint32_t, uint16_t>(), getFunction<uint32_t, uint32_t>(),
      getFunction<uint32_t, uint64_t>()},
     {getFunction<uint64_t, uint8_t>(), getFunction<uint64_t, uint16_t>(), getFunction<uint64_t, uint32_t>(),
      getFunction<uint64_t, uint64_t>()}}};
constexpr size_t map_index_size[9] = {0,
                                      std::numeric_limits<size_t>::max(),
                                      1,
                                      std::numeric_limits<size_t>::max(),
                                      2,
                                      std::numeric_limits<size_t>::max(),
                                      std::numeric_limits<size_t>::max(),
                                      std::numeric_limits<size_t>::max(),
                                      3};

bool softvec_read(void* core, uint64_t addr, uint64_t length, uint8_t* data) {
    // Read length bytes from addr into *data
    iss::status status = static_cast<iss::arch_if*>(core)->read(iss::address_type::PHYSICAL, iss::access_type::READ,
                                                                0 /*traits<ARCH>::MEM*/, addr, length, data);
    return status == iss::Ok;
}
bool softvec_write(void* core, uint64_t addr, uint64_t length, uint8_t* data) {
    // Write length bytes from addr into *data
    iss::status status = static_cast<iss::arch_if*>(core)->write(iss::address_type::PHYSICAL, iss::access_type::READ,
                                                                 0 /*traits<ARCH>::MEM*/, addr, length, data);
    return status == iss::Ok;
}
extern "C" {
uint64_t vlseg(void* core, uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint64_t rs1_val,
               uint8_t width_val, uint8_t segment_size) {
    switch(width_val) {
    case 0b000:
        return softvector::vector_load_store<CUR_VLEN, uint8_t>(core, softvec_read, V, vl, vstart, vtype, vm, vd, rs1_val, segment_size);
    case 0b101:
        return softvector::vector_load_store<CUR_VLEN, uint16_t>(core, softvec_read, V, vl, vstart, vtype, vm, vd, rs1_val, segment_size);
    case 0b110:
        return softvector::vector_load_store<CUR_VLEN, uint32_t>(core, softvec_read, V, vl, vstart, vtype, vm, vd, rs1_val, segment_size);
    case 0b111:
        return softvector::vector_load_store<CUR_VLEN, uint64_t>(core, softvec_read, V, vl, vstart, vtype, vm, vd, rs1_val, segment_size);
    default:
        throw new std::runtime_error("Unsupported width bit value");
    }
}
uint64_t vsseg(void* core, uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint64_t rs1_val,
               uint8_t width_val, uint8_t segment_size) {
    switch(width_val) {
    case 0b000:
        return softvector::vector_load_store<CUR_VLEN, uint8_t>(core, softvec_write, V, vl, vstart, vtype, vm, vd, rs1_val, segment_size);
    case 0b101:
        return softvector::vector_load_store<CUR_VLEN, uint16_t>(core, softvec_write, V, vl, vstart, vtype, vm, vd, rs1_val, segment_size);
    case 0b110:
        return softvector::vector_load_store<CUR_VLEN, uint32_t>(core, softvec_write, V, vl, vstart, vtype, vm, vd, rs1_val, segment_size);
    case 0b111:
        return softvector::vector_load_store<CUR_VLEN, uint64_t>(core, softvec_write, V, vl, vstart, vtype, vm, vd, rs1_val, segment_size);
    default:
        throw new std::runtime_error("Unsupported width bit value");
    }
}
uint64_t vlsseg(void* core, uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint64_t rs1_val,
                uint8_t width_val, uint8_t segment_size, int64_t stride) {
    switch(width_val) {
    case 0b000:
        return softvector::vector_load_store<CUR_VLEN, uint8_t>(core, softvec_read, V, vl, vstart, vtype, vm, vd, rs1_val, segment_size,
                                                                stride, true);
    case 0b101:
        return softvector::vector_load_store<CUR_VLEN, uint16_t>(core, softvec_read, V, vl, vstart, vtype, vm, vd, rs1_val, segment_size,
                                                                 stride, true);
    case 0b110:
        return softvector::vector_load_store<CUR_VLEN, uint32_t>(core, softvec_read, V, vl, vstart, vtype, vm, vd, rs1_val, segment_size,
                                                                 stride, true);
    case 0b111:
        return softvector::vector_load_store<CUR_VLEN, uint64_t>(core, softvec_read, V, vl, vstart, vtype, vm, vd, rs1_val, segment_size,
                                                                 stride, true);
    default:
        throw new std::runtime_error("Unsupported width bit value");
    }
}
uint64_t vssseg(void* core, uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint64_t rs1_val,
                uint8_t width_val, uint8_t segment_size, int64_t stride) {
    switch(width_val) {
    case 0b000:
        return softvector::vector_load_store<CUR_VLEN, uint8_t>(core, softvec_write, V, vl, vstart, vtype, vm, vd, rs1_val, segment_size,
                                                                stride, true);
    case 0b101:
        return softvector::vector_load_store<CUR_VLEN, uint16_t>(core, softvec_write, V, vl, vstart, vtype, vm, vd, rs1_val, segment_size,
                                                                 stride, true);
    case 0b110:
        return softvector::vector_load_store<CUR_VLEN, uint32_t>(core, softvec_write, V, vl, vstart, vtype, vm, vd, rs1_val, segment_size,
                                                                 stride, true);
    case 0b111:
        return softvector::vector_load_store<CUR_VLEN, uint64_t>(core, softvec_write, V, vl, vstart, vtype, vm, vd, rs1_val, segment_size,
                                                                 stride, true);
    default:
        throw new std::runtime_error("Unsupported width bit value");
    }
}
uint64_t vlxseg(void* core, uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint64_t rs1_val,
                uint8_t vs2, uint8_t segment_size, uint8_t index_byte_size, uint8_t data_byte_size, bool ordered) {
    return functionTable[map_index_size[index_byte_size]][data_byte_size](core, softvec_read, V, vl, vstart, vtype, vm, vd, rs1_val, vs2,
                                                                          segment_size);
}
uint64_t vsxseg(void* core, uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vs3, uint64_t rs1_val,
                uint8_t vs2, uint8_t segment_size, uint8_t index_byte_size, uint8_t data_byte_size, bool ordered) {
    return functionTable[map_index_size[index_byte_size]][data_byte_size](core, softvec_write, V, vl, vstart, vtype, vm, vs3, rs1_val, vs2,
                                                                          segment_size);
}
void vector_vector_op(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_vector_op<CUR_VLEN, uint8_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b001:
        return softvector::vector_vector_op<CUR_VLEN, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b010:
        return softvector::vector_vector_op<CUR_VLEN, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b011:
        return softvector::vector_vector_op<CUR_VLEN, uint64_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_imm_op(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                   uint8_t vs2, int64_t imm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_imm_op<CUR_VLEN, uint8_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b001:
        return softvector::vector_imm_op<CUR_VLEN, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b010:
        return softvector::vector_imm_op<CUR_VLEN, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b011:
        return softvector::vector_imm_op<CUR_VLEN, uint64_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_vector_wv(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_vector_op<CUR_VLEN, uint16_t, uint8_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b001:
        return softvector::vector_vector_op<CUR_VLEN, uint32_t, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b010:
        return softvector::vector_vector_op<CUR_VLEN, uint64_t, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b011: // would widen to 128 bits
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_imm_wv(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                   uint8_t vs2, int64_t imm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_imm_op<CUR_VLEN, uint16_t, uint8_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b001:
        return softvector::vector_imm_op<CUR_VLEN, uint32_t, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b010:
        return softvector::vector_imm_op<CUR_VLEN, uint64_t, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b011: // would widen to 128 bits
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_vector_ww(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_vector_op<CUR_VLEN, uint16_t, uint16_t, uint8_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b001:
        return softvector::vector_vector_op<CUR_VLEN, uint32_t, uint32_t, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b010:
        return softvector::vector_vector_op<CUR_VLEN, uint64_t, uint64_t, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b011: // would widen to 128 bits
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_imm_ww(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                   uint8_t vs2, int64_t imm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_imm_op<CUR_VLEN, uint16_t, uint16_t, uint8_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b001:
        return softvector::vector_imm_op<CUR_VLEN, uint32_t, uint32_t, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b010:
        return softvector::vector_imm_op<CUR_VLEN, uint64_t, uint64_t, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b011: // would widen to 128 bits
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_extend(uint8_t* V, uint8_t unary_op, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t vs2,
                   uint8_t target_sew_pow, uint8_t frac_pow) {
    switch(target_sew_pow) {
    case 4: // uint16_t target
        if(frac_pow != 1)
            throw new std::runtime_error("Unsupported frac_pow");
        return softvector::vector_unary_op<CUR_VLEN, uint16_t, uint8_t>(V, unary_op, vl, vstart, vtype, vm, vd, vs2);
    case 5: // uint32_t target
        switch(frac_pow) {
        case 1:
            return softvector::vector_unary_op<CUR_VLEN, uint32_t, uint16_t>(V, unary_op, vl, vstart, vtype, vm, vd, vs2);
        case 2:
            return softvector::vector_unary_op<CUR_VLEN, uint32_t, uint8_t>(V, unary_op, vl, vstart, vtype, vm, vd, vs2);
        default:
            throw new std::runtime_error("Unsupported frac_pow");
        }
    case 6: // uint64_t target
        switch(frac_pow) {
        case 1:
            return softvector::vector_unary_op<CUR_VLEN, uint64_t, uint32_t>(V, unary_op, vl, vstart, vtype, vm, vd, vs2);
        case 2:
            return softvector::vector_unary_op<CUR_VLEN, uint64_t, uint16_t>(V, unary_op, vl, vstart, vtype, vm, vd, vs2);
        case 3:
            return softvector::vector_unary_op<CUR_VLEN, uint64_t, uint8_t>(V, unary_op, vl, vstart, vtype, vm, vd, vs2);
        default:
            throw new std::runtime_error("Unsupported frac_pow");
        }
    default:
        throw new std::runtime_error("Unsupported target_sew_pow");
    }
}
void vector_vector_carry(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, uint8_t vd,
                         uint8_t vs2, uint8_t vs1, uint8_t sew_val, int8_t carry) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_vector_carry<CUR_VLEN, uint8_t>(V, funct6, funct3, vl, vstart, vtype, vd, vs2, vs1, carry);
    case 0b001:
        return softvector::vector_vector_carry<CUR_VLEN, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vd, vs2, vs1, carry);
    case 0b010:
        return softvector::vector_vector_carry<CUR_VLEN, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vd, vs2, vs1, carry);
    case 0b011:
        return softvector::vector_vector_carry<CUR_VLEN, uint64_t>(V, funct6, funct3, vl, vstart, vtype, vd, vs2, vs1, carry);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_imm_carry(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, uint8_t vd,
                      uint8_t vs2, int64_t imm, uint8_t sew_val, int8_t carry) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_imm_carry<CUR_VLEN, uint8_t>(V, funct6, funct3, vl, vstart, vtype, vd, vs2, imm, carry);
    case 0b001:
        return softvector::vector_imm_carry<CUR_VLEN, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vd, vs2, imm, carry);
    case 0b010:
        return softvector::vector_imm_carry<CUR_VLEN, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vd, vs2, imm, carry);
    case 0b011:
        return softvector::vector_imm_carry<CUR_VLEN, uint64_t>(V, funct6, funct3, vl, vstart, vtype, vd, vs2, imm, carry);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void carry_vector_vector_op(uint8_t* V, unsigned funct6, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, unsigned vd,
                            unsigned vs2, unsigned vs1, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::carry_vector_vector_op<CUR_VLEN, uint8_t>(V, funct6, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b001:
        return softvector::carry_vector_vector_op<CUR_VLEN, uint16_t>(V, funct6, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b010:
        return softvector::carry_vector_vector_op<CUR_VLEN, uint32_t>(V, funct6, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b011:
        return softvector::carry_vector_vector_op<CUR_VLEN, uint64_t>(V, funct6, vl, vstart, vtype, vm, vd, vs2, vs1);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void carry_vector_imm_op(uint8_t* V, unsigned funct6, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, unsigned vd,
                         unsigned vs2, int64_t imm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::carry_vector_imm_op<CUR_VLEN, uint8_t>(V, funct6, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b001:
        return softvector::carry_vector_imm_op<CUR_VLEN, uint16_t>(V, funct6, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b010:
        return softvector::carry_vector_imm_op<CUR_VLEN, uint32_t>(V, funct6, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b011:
        return softvector::carry_vector_imm_op<CUR_VLEN, uint64_t>(V, funct6, vl, vstart, vtype, vm, vd, vs2, imm);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void mask_vector_vector_op(uint8_t* V, unsigned funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                           unsigned vd, unsigned vs2, unsigned vs1, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::mask_vector_vector_op<CUR_VLEN, uint8_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b001:
        return softvector::mask_vector_vector_op<CUR_VLEN, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b010:
        return softvector::mask_vector_vector_op<CUR_VLEN, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b011:
        return softvector::mask_vector_vector_op<CUR_VLEN, uint64_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void mask_vector_imm_op(uint8_t* V, unsigned funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                        unsigned vd, unsigned vs2, int64_t imm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::mask_vector_imm_op<CUR_VLEN, uint8_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b001:
        return softvector::mask_vector_imm_op<CUR_VLEN, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b010:
        return softvector::mask_vector_imm_op<CUR_VLEN, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b011:
        return softvector::mask_vector_imm_op<CUR_VLEN, uint64_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_vector_vw(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_vector_op<CUR_VLEN, uint8_t, uint16_t, uint8_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b001:
        return softvector::vector_vector_op<CUR_VLEN, uint16_t, uint32_t, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b010:
        return softvector::vector_vector_op<CUR_VLEN, uint32_t, uint64_t, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b011: // would require 128 bits vs2 value
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_imm_vw(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                   uint8_t vs2, int64_t imm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_imm_op<CUR_VLEN, uint8_t, uint16_t, uint8_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b001:
        return softvector::vector_imm_op<CUR_VLEN, uint16_t, uint32_t, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b010:
        return softvector::vector_imm_op<CUR_VLEN, uint32_t, uint64_t, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b011: // would require 128 bits vs2 value
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_vector_merge(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t vs2, uint8_t vs1,
                         uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_vector_merge<CUR_VLEN, uint8_t>(V, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b001:
        return softvector::vector_vector_merge<CUR_VLEN, uint16_t>(V, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b010:
        return softvector::vector_vector_merge<CUR_VLEN, uint32_t>(V, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b011:
        return softvector::vector_vector_merge<CUR_VLEN, uint64_t>(V, vl, vstart, vtype, vm, vd, vs2, vs1);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_imm_merge(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t vs2, int64_t imm,
                      uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_imm_merge<CUR_VLEN, uint8_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b001:
        return softvector::vector_imm_merge<CUR_VLEN, uint16_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b010:
        return softvector::vector_imm_merge<CUR_VLEN, uint32_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b011:
        return softvector::vector_imm_merge<CUR_VLEN, uint64_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
bool sat_vector_vector_op(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype,
                          uint64_t vxrm, bool vm, uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::sat_vector_vector_op<CUR_VLEN, uint8_t>(V, funct6, funct3, vl, vstart, vtype, vxrm, vm, vd, vs2, vs1);
    case 0b001:
        return softvector::sat_vector_vector_op<CUR_VLEN, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vxrm, vm, vd, vs2, vs1);
    case 0b010:
        return softvector::sat_vector_vector_op<CUR_VLEN, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vxrm, vm, vd, vs2, vs1);
    case 0b011:
        return softvector::sat_vector_vector_op<CUR_VLEN, uint64_t>(V, funct6, funct3, vl, vstart, vtype, vxrm, vm, vd, vs2, vs1);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
bool sat_vector_imm_op(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, uint64_t vxrm,
                       bool vm, uint8_t vd, uint8_t vs2, int64_t imm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::sat_vector_imm_op<CUR_VLEN, uint8_t>(V, funct6, funct3, vl, vstart, vtype, vxrm, vm, vd, vs2, imm);
    case 0b001:
        return softvector::sat_vector_imm_op<CUR_VLEN, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vxrm, vm, vd, vs2, imm);
    case 0b010:
        return softvector::sat_vector_imm_op<CUR_VLEN, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vxrm, vm, vd, vs2, imm);
    case 0b011:
        return softvector::sat_vector_imm_op<CUR_VLEN, uint64_t>(V, funct6, funct3, vl, vstart, vtype, vxrm, vm, vd, vs2, imm);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
bool sat_vector_vector_vw(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype,
                          uint64_t vxrm, bool vm, uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::sat_vector_vector_op<CUR_VLEN, uint8_t, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vxrm, vm, vd, vs2, vs1);
    case 0b001:
        return softvector::sat_vector_vector_op<CUR_VLEN, uint16_t, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vxrm, vm, vd, vs2, vs1);
    case 0b010:
        return softvector::sat_vector_vector_op<CUR_VLEN, uint32_t, uint64_t>(V, funct6, funct3, vl, vstart, vtype, vxrm, vm, vd, vs2, vs1);
    case 0b011: // would require 128 bits vs2 value
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
bool sat_vector_imm_vw(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, uint64_t vxrm,
                       bool vm, uint8_t vd, uint8_t vs2, int64_t imm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::sat_vector_imm_op<CUR_VLEN, uint8_t, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vxrm, vm, vd, vs2, imm);
    case 0b001:
        return softvector::sat_vector_imm_op<CUR_VLEN, uint16_t, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vxrm, vm, vd, vs2, imm);
    case 0b010:
        return softvector::sat_vector_imm_op<CUR_VLEN, uint32_t, uint64_t>(V, funct6, funct3, vl, vstart, vtype, vxrm, vm, vd, vs2, imm);
    case 0b011: // would require 128 bits vs2 value
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_red_op(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                   uint8_t vs2, uint8_t vs1, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_red_op<CUR_VLEN, uint8_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b001:
        return softvector::vector_red_op<CUR_VLEN, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b010:
        return softvector::vector_red_op<CUR_VLEN, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b011:
        return softvector::vector_red_op<CUR_VLEN, uint64_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_red_wv(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                   uint8_t vs2, uint8_t vs1, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_red_op<CUR_VLEN, uint16_t, uint8_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b001:
        return softvector::vector_red_op<CUR_VLEN, uint32_t, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b010:
        return softvector::vector_red_op<CUR_VLEN, uint64_t, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b011: // would require 128 bits vs2 value
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void mask_mask_op(uint8_t* V, unsigned funct6, unsigned funct3, uint64_t vl, uint64_t vstart, unsigned vd, unsigned vs2, unsigned vs1) {
    return softvector::mask_mask_op<CUR_VLEN>(V, funct6, funct3, vl, vstart, vd, vs2, vs1);
}
uint64_t vcpop(uint8_t* V, uint64_t vl, uint64_t vstart, bool vm, unsigned vs2) {
    return softvector::vcpop<CUR_VLEN>(V, vl, vstart, vm, vs2);
}
int64_t vfirst(uint8_t* V, uint64_t vl, uint64_t vstart, bool vm, unsigned vs2) {
    return softvector::vfirst<CUR_VLEN>(V, vl, vstart, vm, vs2);
}
void mask_set_op(uint8_t* V, unsigned enc, uint64_t vl, uint64_t vstart, bool vm, unsigned vd, unsigned vs2) {
    return softvector::mask_set_op<CUR_VLEN>(V, enc, vl, vstart, vm, vd, vs2);
}
void viota(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t vs2, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::viota<CUR_VLEN, uint8_t>(V, vl, vstart, vtype, vm, vd, vs2);
    case 0b001:
        return softvector::viota<CUR_VLEN, uint16_t>(V, vl, vstart, vtype, vm, vd, vs2);
    case 0b010:
        return softvector::viota<CUR_VLEN, uint32_t>(V, vl, vstart, vtype, vm, vd, vs2);
    case 0b011:
        return softvector::viota<CUR_VLEN, uint64_t>(V, vl, vstart, vtype, vm, vd, vs2);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vid(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vid<CUR_VLEN, uint8_t>(V, vl, vstart, vtype, vm, vd);
    case 0b001:
        return softvector::vid<CUR_VLEN, uint16_t>(V, vl, vstart, vtype, vm, vd);
    case 0b010:
        return softvector::vid<CUR_VLEN, uint32_t>(V, vl, vstart, vtype, vm, vd);
    case 0b011:
        return softvector::vid<CUR_VLEN, uint64_t>(V, vl, vstart, vtype, vm, vd);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void scalar_to_vector(uint8_t* V, softvector::vtype_t vtype, unsigned vd, uint64_t val, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        softvector::scalar_move<CUR_VLEN, uint8_t>(V, vtype, vd, val, true);
        break;
    case 0b001:
        softvector::scalar_move<CUR_VLEN, uint16_t>(V, vtype, vd, val, true);
        break;
    case 0b010:
        softvector::scalar_move<CUR_VLEN, uint32_t>(V, vtype, vd, val, true);
        break;
    case 0b011:
        softvector::scalar_move<CUR_VLEN, uint64_t>(V, vtype, vd, val, true);
        break;
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
uint64_t scalar_from_vector(uint8_t* V, softvector::vtype_t vtype, unsigned vd, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::scalar_move<CUR_VLEN, uint8_t>(V, vtype, vd, 0, false);
    case 0b001:
        return softvector::scalar_move<CUR_VLEN, uint16_t>(V, vtype, vd, 0, false);
    case 0b010:
        return softvector::scalar_move<CUR_VLEN, uint32_t>(V, vtype, vd, 0, false);
    case 0b011:
        return softvector::scalar_move<CUR_VLEN, uint64_t>(V, vtype, vd, 0, false);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_slideup(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm,
                    uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_slideup<CUR_VLEN, uint8_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b001:
        return softvector::vector_slideup<CUR_VLEN, uint16_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b010:
        return softvector::vector_slideup<CUR_VLEN, uint32_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b011:
        return softvector::vector_slideup<CUR_VLEN, uint64_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_slidedown(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm,
                      uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_slidedown<CUR_VLEN, uint8_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b001:
        return softvector::vector_slidedown<CUR_VLEN, uint16_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b010:
        return softvector::vector_slidedown<CUR_VLEN, uint32_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b011:
        return softvector::vector_slidedown<CUR_VLEN, uint64_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_slide1up(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, unsigned vd, unsigned vs2, uint64_t imm,
                     uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_slide1up<CUR_VLEN, uint8_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b001:
        return softvector::vector_slide1up<CUR_VLEN, uint16_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b010:
        return softvector::vector_slide1up<CUR_VLEN, uint32_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b011:
        return softvector::vector_slide1up<CUR_VLEN, uint64_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_slide1down(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                       uint64_t imm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_slide1down<CUR_VLEN, uint8_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b001:
        return softvector::vector_slide1down<CUR_VLEN, uint16_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b010:
        return softvector::vector_slide1down<CUR_VLEN, uint32_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b011:
        return softvector::vector_slide1down<CUR_VLEN, uint64_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_vector_gather(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t vs2,
                          uint8_t vs1, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_vector_gather<CUR_VLEN, uint8_t>(V, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b001:
        return softvector::vector_vector_gather<CUR_VLEN, uint16_t>(V, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b010:
        return softvector::vector_vector_gather<CUR_VLEN, uint32_t>(V, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b011:
        return softvector::vector_vector_gather<CUR_VLEN, uint64_t>(V, vl, vstart, vtype, vm, vd, vs2, vs1);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_vector_gatherei16(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t vs2,
                              uint8_t vs1, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_vector_gather<CUR_VLEN, uint8_t, uint16_t>(V, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b001:
        return softvector::vector_vector_gather<CUR_VLEN, uint16_t, uint16_t>(V, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b010:
        return softvector::vector_vector_gather<CUR_VLEN, uint32_t, uint16_t>(V, vl, vstart, vtype, vm, vd, vs2, vs1);
    case 0b011:
        return softvector::vector_vector_gather<CUR_VLEN, uint64_t, uint16_t>(V, vl, vstart, vtype, vm, vd, vs2, vs1);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_imm_gather(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t vs2, uint64_t imm,
                       uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_imm_gather<CUR_VLEN, uint8_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b001:
        return softvector::vector_imm_gather<CUR_VLEN, uint16_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b010:
        return softvector::vector_imm_gather<CUR_VLEN, uint32_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    case 0b011:
        return softvector::vector_imm_gather<CUR_VLEN, uint64_t>(V, vl, vstart, vtype, vm, vd, vs2, imm);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_compress(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, uint8_t vd, uint8_t vs2, uint8_t vs1,
                     uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_compress<CUR_VLEN, uint8_t>(V, vl, vstart, vtype, vd, vs2, vs1);
    case 0b001:
        return softvector::vector_compress<CUR_VLEN, uint16_t>(V, vl, vstart, vtype, vd, vs2, vs1);
    case 0b010:
        return softvector::vector_compress<CUR_VLEN, uint32_t>(V, vl, vstart, vtype, vd, vs2, vs1);
    case 0b011:
        return softvector::vector_compress<CUR_VLEN, uint64_t>(V, vl, vstart, vtype, vd, vs2, vs1);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_whole_move(uint8_t* V, uint8_t vd, uint8_t vs2, uint8_t count) {
    return softvector::vector_whole_move<CUR_VLEN>(V, vd, vs2, count);
}
uint64_t fp_scalar_from_vector(uint8_t* V, softvector::vtype_t vtype, unsigned vd, uint8_t sew_val) {
    return scalar_from_vector(V, vtype, vd, sew_val);
}
void fp_vector_slide1up(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                        uint64_t imm, uint8_t sew_val) {
    return vector_slide1up(V, vl, vstart, vtype, vm, vd, vs2, imm, sew_val);
}
void fp_vector_slide1down(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, unsigned vd, unsigned vs2,
                          uint64_t imm, uint8_t sew_val) {
    return vector_slide1down(V, vl, vstart, vtype, vm, vd, vs2, imm, sew_val);
}
void fp_vector_red_op(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t rm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        throw new std::runtime_error("Unsupported sew bit value");
    case 0b001:
        return softvector::fp_vector_red_op<CUR_VLEN, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1, rm);
    case 0b010:
        return softvector::fp_vector_red_op<CUR_VLEN, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1, rm);
    case 0b011:
        return softvector::fp_vector_red_op<CUR_VLEN, uint64_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1, rm);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void fp_vector_red_wv(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t rm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::fp_vector_red_op<CUR_VLEN, uint16_t, uint8_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1, rm);
    case 0b001:
        return softvector::fp_vector_red_op<CUR_VLEN, uint32_t, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1, rm);
    case 0b010:
        return softvector::fp_vector_red_op<CUR_VLEN, uint64_t, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1, rm);
    case 0b011: // would require 128 bits vs2 value
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void fp_vector_vector_op(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                         uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t rm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        throw new std::runtime_error("Unsupported sew bit value");
    case 0b001:
        return softvector::fp_vector_vector_op<CUR_VLEN, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1, rm);
    case 0b010:
        return softvector::fp_vector_vector_op<CUR_VLEN, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1, rm);
    case 0b011:
        return softvector::fp_vector_vector_op<CUR_VLEN, uint64_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1, rm);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void fp_vector_imm_op(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint64_t imm, uint8_t rm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        throw new std::runtime_error("Unsupported sew bit value");
    case 0b001:
        return softvector::fp_vector_imm_op<CUR_VLEN, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm, rm);
    case 0b010:
        return softvector::fp_vector_imm_op<CUR_VLEN, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm, rm);
    case 0b011:
        return softvector::fp_vector_imm_op<CUR_VLEN, uint64_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm, rm);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void fp_vector_vector_wv(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                         uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t rm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        throw new std::runtime_error("Unsupported sew bit value");
    case 0b001:
        return softvector::fp_vector_vector_op<CUR_VLEN, uint32_t, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1, rm);
    case 0b010:
        return softvector::fp_vector_vector_op<CUR_VLEN, uint64_t, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, vs1, rm);
    case 0b011: // would widen to 128 bits
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void fp_vector_imm_wv(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint64_t imm, uint8_t rm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        throw new std::runtime_error("Unsupported sew bit value");
    case 0b001:
        return softvector::fp_vector_imm_op<CUR_VLEN, uint32_t, uint16_t, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm,
                                                                                    rm);
    case 0b010:
        return softvector::fp_vector_imm_op<CUR_VLEN, uint64_t, uint32_t, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm,
                                                                                    rm);
    case 0b011: // would widen to 128 bits
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void fp_vector_vector_ww(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                         uint8_t vd, uint8_t vs2, uint8_t vs1, uint8_t rm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        throw new std::runtime_error("Unsupported sew bit value");
    case 0b001:
        return softvector::fp_vector_vector_op<CUR_VLEN, uint32_t, uint32_t, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2,
                                                                                       vs1, rm);
    case 0b010:
        return softvector::fp_vector_vector_op<CUR_VLEN, uint64_t, uint64_t, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2,
                                                                                       vs1, rm);
    case 0b011: // would widen to 128 bits
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void fp_vector_imm_ww(uint8_t* V, uint8_t funct6, uint8_t funct3, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm,
                      uint8_t vd, uint8_t vs2, uint64_t imm, uint8_t rm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        throw new std::runtime_error("Unsupported sew bit value");
    case 0b001:
        return softvector::fp_vector_imm_op<CUR_VLEN, uint32_t, uint32_t, uint16_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm,
                                                                                    rm);
    case 0b010:
        return softvector::fp_vector_imm_op<CUR_VLEN, uint64_t, uint64_t, uint32_t>(V, funct6, funct3, vl, vstart, vtype, vm, vd, vs2, imm,
                                                                                    rm);
    case 0b011: // would widen to 128 bits
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void fp_vector_unary_op(uint8_t* V, uint8_t encoding_space, uint8_t unary_op, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype,
                        bool vm, uint8_t vd, uint8_t vs2, uint8_t rm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        throw new std::runtime_error("Unsupported sew bit value");
    case 0b001:
        return softvector::fp_vector_unary_op<CUR_VLEN, uint16_t>(V, encoding_space, unary_op, vl, vstart, vtype, vm, vd, vs2, rm);
    case 0b010:
        return softvector::fp_vector_unary_op<CUR_VLEN, uint32_t>(V, encoding_space, unary_op, vl, vstart, vtype, vm, vd, vs2, rm);
    case 0b011:
        return softvector::fp_vector_unary_op<CUR_VLEN, uint64_t>(V, encoding_space, unary_op, vl, vstart, vtype, vm, vd, vs2, rm);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void mask_fp_vector_vector_op(uint8_t* V, uint8_t funct6, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                              uint8_t vs2, uint8_t vs1, uint8_t rm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        throw new std::runtime_error("Unsupported sew bit value");
    case 0b001:
        return softvector::mask_fp_vector_vector_op<CUR_VLEN, uint16_t>(V, funct6, vl, vstart, vtype, vm, vd, vs2, vs1, rm);
    case 0b010:
        return softvector::mask_fp_vector_vector_op<CUR_VLEN, uint32_t>(V, funct6, vl, vstart, vtype, vm, vd, vs2, vs1, rm);
    case 0b011:
        return softvector::mask_fp_vector_vector_op<CUR_VLEN, uint64_t>(V, funct6, vl, vstart, vtype, vm, vd, vs2, vs1, rm);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void mask_fp_vector_imm_op(uint8_t* V, uint8_t funct6, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                           uint8_t vs2, uint64_t imm, uint8_t rm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        throw new std::runtime_error("Unsupported sew bit value");
    case 0b001:
        return softvector::mask_fp_vector_imm_op<CUR_VLEN, uint16_t>(V, funct6, vl, vstart, vtype, vm, vd, vs2, imm, rm);
    case 0b010:
        return softvector::mask_fp_vector_imm_op<CUR_VLEN, uint32_t>(V, funct6, vl, vstart, vtype, vm, vd, vs2, imm, rm);
    case 0b011:
        return softvector::mask_fp_vector_imm_op<CUR_VLEN, uint64_t>(V, funct6, vl, vstart, vtype, vm, vd, vs2, imm, rm);
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void fp_vector_imm_merge(uint8_t* V, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd, uint8_t vs2,
                         uint64_t imm, uint8_t sew_val) {
    vector_imm_merge(V, vl, vstart, vtype, vm, vd, vs2, imm, sew_val);
}
void fp_vector_unary_w(uint8_t* V, uint8_t unary_op, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                       uint8_t vs2, uint8_t rm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::fp_vector_unary_w<CUR_VLEN, uint16_t, uint8_t>(V, unary_op, vl, vstart, vtype, vm, vd, vs2, rm);
    case 0b001:
        return softvector::fp_vector_unary_w<CUR_VLEN, uint32_t, uint16_t>(V, unary_op, vl, vstart, vtype, vm, vd, vs2, rm);
    case 0b010:
        return softvector::fp_vector_unary_w<CUR_VLEN, uint64_t, uint32_t>(V, unary_op, vl, vstart, vtype, vm, vd, vs2, rm);
    case 0b011: // would widen to 128 bits
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void fp_vector_unary_n(uint8_t* V, uint8_t unary_op, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                       uint8_t vs2, uint8_t rm, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::fp_vector_unary_n<CUR_VLEN, uint8_t, uint16_t>(V, unary_op, vl, vstart, vtype, vm, vd, vs2, rm);
    case 0b001:
        return softvector::fp_vector_unary_n<CUR_VLEN, uint16_t, uint32_t>(V, unary_op, vl, vstart, vtype, vm, vd, vs2, rm);
    case 0b010:
        return softvector::fp_vector_unary_n<CUR_VLEN, uint32_t, uint64_t>(V, unary_op, vl, vstart, vtype, vm, vd, vs2, rm);
    case 0b011: // would require 128 bit value to narrow
    default:
        throw new std::runtime_error("Unsupported sew bit value");
    }
}
void vector_unary_op(uint8_t* V, uint8_t unary_op, uint64_t vl, uint64_t vstart, softvector::vtype_t vtype, bool vm, uint8_t vd,
                     uint8_t vs2, uint8_t sew_val) {
    switch(sew_val) {
    case 0b000:
        return softvector::vector_unary_op<CUR_VLEN, uint8_t>(V, unary_op, vl, vstart, vtype, vm, vd, vs2);
    case 0b001:
        return softvector::vector_unary_op<CUR_VLEN, uint16_t>(V, unary_op, vl, vstart, vtype, vm, vd, vs2);
    case 0b010:
        return softvector::vector_unary_op<CUR_VLEN, uint32_t>(V, unary_op, vl, vstart, vtype, vm, vd, vs2);
    case 0b011:
        return softvector::vector_unary_op<CUR_VLEN, uint64_t>(V, unary_op, vl, vstart, vtype, vm, vd, vs2);
    default:
        throw new std::runtime_error("Unsupported sew_val");
    }
}
}