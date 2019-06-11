/*******************************************************************************
 * Copyright (C) 2017, 2018 MINRES Technologies GmbH
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
 
#include "util/ities.h"
#include <util/logging.h>

#include <elfio/elfio.hpp>
#include <iss/arch/rv64i.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <ihex.h>
#ifdef __cplusplus
}
#endif
#include <cstdio>
#include <cstring>
#include <fstream>

using namespace iss::arch;

constexpr std::array<const char*, 33>    iss::arch::traits<iss::arch::rv64i>::reg_names;
constexpr std::array<const char*, 33>    iss::arch::traits<iss::arch::rv64i>::reg_aliases;
constexpr std::array<const uint32_t, 39> iss::arch::traits<iss::arch::rv64i>::reg_bit_widths;
constexpr std::array<const uint32_t, 40> iss::arch::traits<iss::arch::rv64i>::reg_byte_offsets;

rv64i::rv64i() {
    reg.icount = 0;
}

rv64i::~rv64i() = default;

void rv64i::reset(uint64_t address) {
    for(size_t i=0; i<traits<rv64i>::NUM_REGS; ++i) set_reg(i, std::vector<uint8_t>(sizeof(traits<rv64i>::reg_t),0));
    reg.PC=address;
    reg.NEXT_PC=reg.PC;
    reg.trap_state=0;
    reg.machine_state=0x0;
    reg.icount=0;
}

uint8_t *rv64i::get_regs_base_ptr() {
	return reinterpret_cast<uint8_t*>(&reg);
}

rv64i::phys_addr_t rv64i::virt2phys(const iss::addr_t &pc) {
    return phys_addr_t(pc); // change logical address to physical address
}

