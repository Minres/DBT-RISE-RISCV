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
// Created on: Tue Aug 29 16:45:20 CEST 2017
//             *      rv32imac.cpp Author: <CoreDSL Generator>
//
////////////////////////////////////////////////////////////////////////////////

#include "util/ities.h"
#include <easylogging++.h>

#include <elfio/elfio.hpp>
#include <iss/arch/rv32imac.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <ihex.h>
#ifdef __cplusplus
}
#endif
#include <fstream>
#include <cstdio>
#include <cstring>

using namespace iss::arch;

rv32imac::rv32imac() {
    reg.icount=0;
}

rv32imac::~rv32imac(){
}

void rv32imac::reset(uint64_t address) {
    for(size_t i=0; i<traits<rv32imac>::NUM_REGS; ++i) set_reg(i, std::vector<uint8_t>(sizeof(traits<rv32imac>::reg_t),0));
    reg.PC=address;
    reg.NEXT_PC=reg.PC;
    reg.trap_state=0;
    reg.machine_state=0x3;
}

uint8_t* rv32imac::get_regs_base_ptr(){
    return reinterpret_cast<uint8_t*>(&reg);
}

rv32imac::phys_addr_t rv32imac::v2p(const iss::addr_t& pc) {
    return phys_addr_t(pc); //change logical address to physical address
}
