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

#ifndef _VM_FP_FUNCTIONS_H_
#define _VM_FP_FUNCTIONS_H_

#include <stdint.h>

extern "C" {
uint32_t fget_flags();
uint32_t fadd_s(uint32_t v1, uint32_t v2, uint8_t mode);
uint32_t fsub_s(uint32_t v1, uint32_t v2, uint8_t mode);
uint32_t fmul_s(uint32_t v1, uint32_t v2, uint8_t mode);
uint32_t fdiv_s(uint32_t v1, uint32_t v2, uint8_t mode);
uint32_t fsqrt_s(uint32_t v1, uint8_t mode);
uint32_t fcmp_s(uint32_t v1, uint32_t v2, uint32_t op);
uint32_t fcvt_s(uint32_t v1, uint32_t op, uint8_t mode);
uint32_t fmadd_s(uint32_t v1, uint32_t v2, uint32_t v3, uint32_t op, uint8_t mode);
uint32_t fsel_s(uint32_t v1, uint32_t v2, uint32_t op);
uint32_t fclass_s(uint32_t v1);
uint32_t fconv_d2f(uint64_t v1, uint8_t mode);
uint64_t fconv_f2d(uint32_t v1, uint8_t mode);
uint64_t fadd_d(uint64_t v1, uint64_t v2, uint8_t mode);
uint64_t fsub_d(uint64_t v1, uint64_t v2, uint8_t mode);
uint64_t fmul_d(uint64_t v1, uint64_t v2, uint8_t mode);
uint64_t fdiv_d(uint64_t v1, uint64_t v2, uint8_t mode);
uint64_t fsqrt_d(uint64_t v1, uint8_t mode);
uint64_t fcmp_d(uint64_t v1, uint64_t v2, uint32_t op);
uint64_t fcvt_d(uint64_t v1, uint32_t op, uint8_t mode);
uint64_t fmadd_d(uint64_t v1, uint64_t v2, uint64_t v3, uint32_t op, uint8_t mode);
uint64_t fsel_d(uint64_t v1, uint64_t v2, uint32_t op);
uint64_t fclass_d(uint64_t v1);
uint64_t fcvt_32_64(uint32_t v1, uint32_t op, uint8_t mode);
uint32_t fcvt_64_32(uint64_t v1, uint32_t op, uint8_t mode);
uint32_t unbox_s(uint64_t v);
}
#endif /* RISCV_SRC_VM_FP_FUNCTIONS_H_ */
