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
// Created on: Tue Feb 06 17:18:49 UTC 2018
//             *      rv32imac.h Author: <CoreDSL Generator>
//
////////////////////////////////////////////////////////////////////////////////

#ifndef _RV32IMAC_H_
#define _RV32IMAC_H_

#include <iss/arch_if.h>
#include <iss/vm_if.h>
#include <iss/arch/traits.h>
#include <array>

namespace iss {
namespace arch {

struct rv32imac;

template<>
struct traits<rv32imac> {

    enum constants {XLEN=32,XLEN2=64,XLEN_BIT_MASK=31,PCLEN=32,fence=0,fencei=1,fencevmal=2,fencevmau=3,MISA_VAL=1075056897,PGSIZE=4096,PGMASK=4095};

    enum reg_e {
        X0,
        X1,
        X2,
        X3,
        X4,
        X5,
        X6,
        X7,
        X8,
        X9,
        X10,
        X11,
        X12,
        X13,
        X14,
        X15,
        X16,
        X17,
        X18,
        X19,
        X20,
        X21,
        X22,
        X23,
        X24,
        X25,
        X26,
        X27,
        X28,
        X29,
        X30,
        X31,
        PC,
        NUM_REGS,
        NEXT_PC=NUM_REGS,
        TRAP_STATE,
        PENDING_TRAP,
        MACHINE_STATE,
        ICOUNT
    };

    using reg_t = uint32_t;

    using addr_t = uint32_t;

    using code_word_t = uint32_t; //TODO: check removal

    using virt_addr_t = iss::typed_addr_t<iss::address_type::VIRTUAL>;

    using phys_addr_t = iss::typed_addr_t<iss::address_type::PHYSICAL>;

    constexpr static unsigned reg_bit_width(unsigned r) {
        constexpr std::array<const uint32_t, 38> RV32IMAC_reg_size{{32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,64}};
        return RV32IMAC_reg_size[r];
    }

    constexpr static unsigned reg_byte_offset(unsigned r) {
    	constexpr std::array<const uint32_t, 39> RV32IMAC_reg_byte_offset{{0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,152,160}};
        return RV32IMAC_reg_byte_offset[r];
    }

    static const uint64_t addr_mask = (reg_t(1) << (XLEN - 1)) | ((reg_t(1) << (XLEN - 1)) - 1);

    enum sreg_flag_e {FLAGS};

    enum mem_type_e {MEM,CSR,FENCE,RES};

};

struct rv32imac: public arch_if {

    using virt_addr_t = typename traits<rv32imac>::virt_addr_t;
    using phys_addr_t = typename traits<rv32imac>::phys_addr_t;
    using reg_t =  typename traits<rv32imac>::reg_t;
    using addr_t = typename traits<rv32imac>::addr_t;

    rv32imac();
    ~rv32imac();

    void reset(uint64_t address=0) override;

    uint8_t* get_regs_base_ptr() override;
    /// deprecated
    void get_reg(short idx, std::vector<uint8_t>& value) override {}
    void set_reg(short idx, const std::vector<uint8_t>& value) override {}
    /// deprecated
    bool get_flag(int flag) override {return false;}
    void set_flag(int, bool value) override {};
    /// deprecated
    void update_flags(operations op, uint64_t opr1, uint64_t opr2) override {};

    uint64_t get_icount() { return reg.icount;}

    inline phys_addr_t v2p(const iss::addr_t& addr){
        if(addr.space != traits<rv32imac>::MEM ||
                addr.type == iss::address_type::PHYSICAL ||
                addr_mode[static_cast<uint16_t>(addr.access)&0x3]==address_type::PHYSICAL){
            return phys_addr_t(addr.access, addr.space, addr.val&traits<rv32imac>::addr_mask);
        } else
            return virt2phys(addr);
    }

    virtual phys_addr_t virt2phys(const iss::addr_t& addr);

    virtual iss::sync_type needed_sync() const { return iss::NO_SYNC; }

protected:
    struct RV32IMAC_regs {
        uint32_t X0 = 0;
        uint32_t X1 = 0;
        uint32_t X2 = 0;
        uint32_t X3 = 0;
        uint32_t X4 = 0;
        uint32_t X5 = 0;
        uint32_t X6 = 0;
        uint32_t X7 = 0;
        uint32_t X8 = 0;
        uint32_t X9 = 0;
        uint32_t X10 = 0;
        uint32_t X11 = 0;
        uint32_t X12 = 0;
        uint32_t X13 = 0;
        uint32_t X14 = 0;
        uint32_t X15 = 0;
        uint32_t X16 = 0;
        uint32_t X17 = 0;
        uint32_t X18 = 0;
        uint32_t X19 = 0;
        uint32_t X20 = 0;
        uint32_t X21 = 0;
        uint32_t X22 = 0;
        uint32_t X23 = 0;
        uint32_t X24 = 0;
        uint32_t X25 = 0;
        uint32_t X26 = 0;
        uint32_t X27 = 0;
        uint32_t X28 = 0;
        uint32_t X29 = 0;
        uint32_t X30 = 0;
        uint32_t X31 = 0;
        uint32_t PC = 0;
        uint32_t NEXT_PC = 0;
        uint32_t trap_state = 0, pending_trap = 0, machine_state = 0;
        uint64_t icount = 0;
    } reg;

    std::array<address_type, 4> addr_mode;

    uint64_t cycles = 0;

};

}
}            
#endif /* _RV32IMAC_H_ */
