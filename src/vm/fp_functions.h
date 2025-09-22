////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2020, MINRES Technologies GmbH
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

#ifndef RISCV_SRC_VM_FP_FUNCTIONS_H_
#define RISCV_SRC_VM_FP_FUNCTIONS_H_

#include <stdint.h>

extern "C" {
uint32_t fget_flags();

// half precision
uint16_t fadd_h(uint16_t v1, uint16_t v2, uint8_t mode);
uint16_t fsub_h(uint16_t v1, uint16_t v2, uint8_t mode);
uint16_t fmul_h(uint16_t v1, uint16_t v2, uint8_t mode);
uint16_t fdiv_h(uint16_t v1, uint16_t v2, uint8_t mode);
uint16_t fsqrt_h(uint16_t v1, uint8_t mode);
uint16_t fcmp_h(uint16_t v1, uint16_t v2, uint16_t op);
uint16_t fmadd_h(uint16_t v1, uint16_t v2, uint16_t v3, uint16_t op, uint8_t mode);
uint16_t fsel_h(uint16_t v1, uint16_t v2, uint16_t op);
uint16_t fclass_h(uint16_t v1);
uint16_t frsqrt7_h(uint16_t v);
uint16_t frec7_h(uint16_t v, uint8_t mode);
uint16_t unbox_h(uint8_t FLEN, uint64_t v);

// single precision
uint32_t fadd_s(uint32_t v1, uint32_t v2, uint8_t mode);
uint32_t fsub_s(uint32_t v1, uint32_t v2, uint8_t mode);
uint32_t fmul_s(uint32_t v1, uint32_t v2, uint8_t mode);
uint32_t fdiv_s(uint32_t v1, uint32_t v2, uint8_t mode);
uint32_t fsqrt_s(uint32_t v1, uint8_t mode);
uint32_t fcmp_s(uint32_t v1, uint32_t v2, uint32_t op);
uint32_t fmadd_s(uint32_t v1, uint32_t v2, uint32_t v3, uint32_t op, uint8_t mode);
uint32_t fsel_s(uint32_t v1, uint32_t v2, uint32_t op);
uint32_t fclass_s(uint32_t v1);
uint32_t frsqrt7_s(uint32_t v);
uint32_t frec7_s(uint32_t v, uint8_t mode);
uint32_t unbox_s(uint8_t FLEN, uint64_t v);

// double precision
uint64_t fadd_d(uint64_t v1, uint64_t v2, uint8_t mode);
uint64_t fsub_d(uint64_t v1, uint64_t v2, uint8_t mode);
uint64_t fmul_d(uint64_t v1, uint64_t v2, uint8_t mode);
uint64_t fdiv_d(uint64_t v1, uint64_t v2, uint8_t mode);
uint64_t fsqrt_d(uint64_t v1, uint8_t mode);
uint64_t fcmp_d(uint64_t v1, uint64_t v2, uint32_t op);
uint64_t fmadd_d(uint64_t v1, uint64_t v2, uint64_t v3, uint32_t op, uint8_t mode);
uint64_t fsel_d(uint64_t v1, uint64_t v2, uint32_t op);
uint64_t fclass_d(uint64_t v1);
uint64_t frsqrt7_d(uint64_t v);
uint64_t frec7_d(uint64_t v, uint8_t mode);
uint64_t unbox_d(uint8_t FLEN, uint64_t v);

// conversion: float to float
uint32_t f16tof32(uint16_t val, uint8_t rm);
uint64_t f16tof64(uint16_t val, uint8_t rm);
uint16_t f32tof16(uint32_t val, uint8_t rm);
uint64_t f32tof64(uint32_t val, uint8_t rm);
uint16_t f64tof16(uint64_t val, uint8_t rm);
uint32_t f64tof32(uint64_t val, uint8_t rm);

// conversions: float to unsigned
uint32_t f16toui32(uint16_t v, uint8_t rm);
uint64_t f16toui64(uint16_t v, uint8_t rm);
uint32_t f32toui32(uint32_t v, uint8_t rm);
uint64_t f32toui64(uint32_t v, uint8_t rm);
uint32_t f64toui32(uint64_t v, uint8_t rm);
uint64_t f64toui64(uint64_t v, uint8_t rm);

// conversions: float to signed
uint32_t f16toi32(uint16_t v, uint8_t rm);
uint64_t f16toi64(uint16_t v, uint8_t rm);
uint32_t f32toi32(uint32_t v, uint8_t rm);
uint64_t f32toi64(uint32_t v, uint8_t rm);
uint32_t f64toi32(uint64_t v, uint8_t rm);
uint64_t f64toi64(uint64_t v, uint8_t rm);

// conversions: unsigned to float
uint16_t ui32tof16(uint32_t v, uint8_t rm);
uint16_t ui64tof16(uint64_t v, uint8_t rm);
uint32_t ui32tof32(uint32_t v, uint8_t rm);
uint32_t ui64tof32(uint64_t v, uint8_t rm);
uint64_t ui32tof64(uint32_t v, uint8_t rm);
uint64_t ui64tof64(uint64_t v, uint8_t rm);

// conversions: signed to float
uint16_t i32tof16(uint32_t v, uint8_t rm);
uint16_t i64tof16(uint64_t v, uint8_t rm);
uint32_t i32tof32(uint32_t v, uint8_t rm);
uint32_t i64tof32(uint64_t v, uint8_t rm);
uint64_t i32tof64(uint32_t v, uint8_t rm);
uint64_t i64tof64(uint64_t v, uint8_t rm);
}
#endif /* RISCV_SRC_VM_FP_FUNCTIONS_H_ */
