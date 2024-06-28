/*******************************************************************************
 * Copyright (C) 2017 - 2020 MINRES Technologies GmbH
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 *******************************************************************************/

// clang-format off
#include "rv32i.h"
#include "util/ities.h"
#include <util/logging.h>
#include <cstdio>
#include <cstring>
#include <fstream>

using namespace iss::arch;

constexpr std::array<const char*, 36>    iss::arch::traits<iss::arch::rv32i>::reg_names;
constexpr std::array<const char*, 36>    iss::arch::traits<iss::arch::rv32i>::reg_aliases;
constexpr std::array<const uint32_t, 43> iss::arch::traits<iss::arch::rv32i>::reg_bit_widths;
constexpr std::array<const uint32_t, 43> iss::arch::traits<iss::arch::rv32i>::reg_byte_offsets;

rv32i::rv32i()  = default;

rv32i::~rv32i() = default;

void rv32i::reset(uint64_t address) {
    auto base_ptr = reinterpret_cast<traits<rv32i>::reg_t*>(get_regs_base_ptr());
    for(size_t i=0; i<traits<rv32i>::NUM_REGS; ++i)
        *(base_ptr+i)=0;
    reg.PC=address;
    reg.NEXT_PC=reg.PC;
    reg.PRIV=0x3;
    reg.trap_state=0;
    reg.icount=0;
}

uint8_t *rv32i::get_regs_base_ptr() {
	return reinterpret_cast<uint8_t*>(&reg);
}

rv32i::phys_addr_t rv32i::virt2phys(const iss::addr_t &addr) {
    return phys_addr_t(addr.access, addr.space, addr.val&traits<rv32i>::addr_mask);
}
// clang-format on
