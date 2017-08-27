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
// Contributors:
//       eyck@minres.com - initial API and implementation
////////////////////////////////////////////////////////////////////////////////

#include "util/ities.h"
#include <easylogging++.h>

#include <elfio/elfio.hpp>
#include <iss/arch/minrv_ima.h>

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

minrv_ima::minrv_ima() {
    reg.icount=0;
}

minrv_ima::~minrv_ima(){
}

void minrv_ima::reset(uint64_t address) {
    for(size_t i=0; i<traits<minrv_ima>::NUM_REGS; ++i) set_reg(i, std::vector<uint8_t>(sizeof(traits<minrv_ima>::reg_t),0));
    reg.PC=address;
    reg.NEXT_PC=reg.PC;
    reg.trap_state=0;
    reg.machine_state=0x3;
}

uint8_t* minrv_ima::get_regs_base_ptr(){
    return reinterpret_cast<uint8_t*>(&reg);
}

void minrv_ima::get_reg(short idx, std::vector<uint8_t>& value) {
    if(idx<traits<minrv_ima>::NUM_REGS){
        value.resize(traits<minrv_ima>::reg_byte_offset(idx+1)-traits<minrv_ima>::reg_byte_offset(idx));
        uint8_t* r_ptr= ((uint8_t*)&reg)+traits<minrv_ima>::reg_byte_offset(idx);
        std::copy(r_ptr, r_ptr+sizeof(traits<minrv_ima>::reg_t), value.begin());
    }
}

void minrv_ima::set_reg(short idx, const std::vector<uint8_t>& value) {
    if(idx < traits<minrv_ima>::NUM_REGS){
        uint8_t* r_ptr= ((uint8_t*)&reg)+traits<minrv_ima>::reg_byte_offset(idx);
        std::copy(value.begin(), value.end(), r_ptr);
    }
}

bool minrv_ima::get_flag(int flag){
    return false;
}

void minrv_ima::set_flag(int flag, bool value){
}

void minrv_ima::update_flags(operations op, uint64_t r0, uint64_t r1){
}

minrv_ima::phys_addr_t minrv_ima::v2p(const iss::addr_t& pc) {
    return phys_addr_t(pc); //change logical address to physical address
}

using namespace ELFIO;

/*
void minrv_ima::loadFile(std::string name, int type) {
    FILE* fp = fopen(name.c_str(), "r");
    if(fp){
        char buf[5];
        auto n = fread(buf, 1,4,fp);
        if(n!=4) throw std::runtime_error("input file has insufficient size");
        buf[4]=0;
        if(strcmp(buf+1, "ELF")==0){
            fclose(fp);
            //Create elfio reader
            elfio reader;
            // Load ELF data
            if ( !reader.load( name ) ) throw std::runtime_error("could not process elf file");
            // check elf properties
            //TODO: fix ELFCLASS like:
            // if ( reader.get_class() != ELFCLASS32 ) throw std::runtime_error("wrong elf class in file");
            if ( reader.get_type() != ET_EXEC ) throw std::runtime_error("wrong elf type in file");
            //TODO: fix machine type like:
            // if ( reader.get_machine() != EM_RISCV ) throw std::runtime_error("wrong elf machine in file");
            auto sec_num = reader.sections.size();
            auto seg_num = reader.segments.size();
            for ( int i = 0; i < seg_num; ++i ) {
                const auto pseg = reader.segments[i];
                const auto fsize=pseg->get_file_size();         // 0x42c/0x0
                const auto seg_data=pseg->get_data();
                if(fsize>0){
                    this->write(typed_addr_t<PHYSICAL>(iss::DEBUG_WRITE, traits<minrv_ima>::MEM, pseg->get_virtual_address()), fsize, reinterpret_cast<const uint8_t* const>(seg_data));
                }
            }
        } else {
            fseek(fp, 0, SEEK_SET);
            if(type<0) throw std::runtime_error("a memory type needs to be specified for IHEX files");
            IHexRecord irec;
            while (Read_IHexRecord(&irec, fp) == IHEX_OK) {
                this->write(typed_addr_t<PHYSICAL>(iss::DEBUG_WRITE, type, irec.address), irec.dataLen, irec.data);
            }
        }
    } else {
        LOG(ERROR)<<"Could not open input file "<<name;
        throw std::runtime_error("Could not open input file");
    }
}
*/
