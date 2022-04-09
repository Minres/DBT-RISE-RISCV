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


#ifndef _RV64GC_H_
#define _RV64GC_H_

#include <array>
#include <iss/arch/traits.h>
#include <iss/arch_if.h>
#include <iss/vm_if.h>

namespace iss {
namespace arch {

struct rv64gc;

template <> struct traits<rv64gc> {

	constexpr static char const* const core_type = "RV64GC";
    
  	static constexpr std::array<const char*, 66> reg_names{
 		{"x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30", "x31", "pc", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31", "fcsr"}};
 
  	static constexpr std::array<const char*, 66> reg_aliases{
 		{"zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6", "pc", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31", "fcsr"}};

    enum constants {XLEN=64, FLEN=64, PCLEN=64, MUL_LEN=128, MISA_VAL=0b1000000000101000001000100101101, PGSIZE=0x1000, PGMASK=0xfff};

    constexpr static unsigned FP_REGS_SIZE = 64;

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
        F0,
        F1,
        F2,
        F3,
        F4,
        F5,
        F6,
        F7,
        F8,
        F9,
        F10,
        F11,
        F12,
        F13,
        F14,
        F15,
        F16,
        F17,
        F18,
        F19,
        F20,
        F21,
        F22,
        F23,
        F24,
        F25,
        F26,
        F27,
        F28,
        F29,
        F30,
        F31,
        FCSR,
        NUM_REGS,
        NEXT_PC=NUM_REGS,
        TRAP_STATE,
        PENDING_TRAP,
        MACHINE_STATE,
        LAST_BRANCH,
        ICOUNT,
        ZERO = X0,
        RA = X1,
        SP = X2,
        GP = X3,
        TP = X4,
        T0 = X5,
        T1 = X6,
        T2 = X7,
        S0 = X8,
        S1 = X9,
        A0 = X10,
        A1 = X11,
        A2 = X12,
        A3 = X13,
        A4 = X14,
        A5 = X15,
        A6 = X16,
        A7 = X17,
        S2 = X18,
        S3 = X19,
        S4 = X20,
        S5 = X21,
        S6 = X22,
        S7 = X23,
        S8 = X24,
        S9 = X25,
        S10 = X26,
        S11 = X27,
        T3 = X28,
        T4 = X29,
        T5 = X30,
        T6 = X31
    };

    using reg_t = uint64_t;

    using addr_t = uint64_t;

    using code_word_t = uint64_t; //TODO: check removal

    using virt_addr_t = iss::typed_addr_t<iss::address_type::VIRTUAL>;

    using phys_addr_t = iss::typed_addr_t<iss::address_type::PHYSICAL>;

 	static constexpr std::array<const uint32_t, 72> reg_bit_widths{
 		{64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,32,64,32,32,32,32,64}};

    static constexpr std::array<const uint32_t, 73> reg_byte_offsets{
    	{0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,264,272,280,288,296,304,312,320,328,336,344,352,360,368,376,384,392,400,408,416,424,432,440,448,456,464,472,480,488,496,504,512,520,528,536,540,544,548,552,560}};

    static const uint64_t addr_mask = (reg_t(1) << (XLEN - 1)) | ((reg_t(1) << (XLEN - 1)) - 1);

    enum sreg_flag_e { FLAGS };

    enum mem_type_e { MEM, CSR, FENCE, RES };
};

struct rv64gc: public arch_if {

    using virt_addr_t = typename traits<rv64gc>::virt_addr_t;
    using phys_addr_t = typename traits<rv64gc>::phys_addr_t;
    using reg_t =  typename traits<rv64gc>::reg_t;
    using addr_t = typename traits<rv64gc>::addr_t;

    rv64gc();
    ~rv64gc();

    void reset(uint64_t address=0) override;

    uint8_t* get_regs_base_ptr() override;

    inline uint64_t get_icount() { return reg.icount; }

    inline bool should_stop() { return interrupt_sim; }

    inline uint64_t stop_code() { return interrupt_sim; }

    inline phys_addr_t v2p(const iss::addr_t& addr){
        if (addr.space != traits<rv64gc>::MEM || addr.type == iss::address_type::PHYSICAL ||
                addr_mode[static_cast<uint16_t>(addr.access)&0x3]==address_type::PHYSICAL) {
            return phys_addr_t(addr.access, addr.space, addr.val&traits<rv64gc>::addr_mask);
        } else
            return virt2phys(addr);
    }

    virtual phys_addr_t virt2phys(const iss::addr_t& addr);

    virtual iss::sync_type needed_sync() const { return iss::NO_SYNC; }

    inline uint32_t get_last_branch() { return reg.last_branch; }

protected:
    struct RV64GC_regs {
        uint64_t X0 = 0;
        uint64_t X1 = 0;
        uint64_t X2 = 0;
        uint64_t X3 = 0;
        uint64_t X4 = 0;
        uint64_t X5 = 0;
        uint64_t X6 = 0;
        uint64_t X7 = 0;
        uint64_t X8 = 0;
        uint64_t X9 = 0;
        uint64_t X10 = 0;
        uint64_t X11 = 0;
        uint64_t X12 = 0;
        uint64_t X13 = 0;
        uint64_t X14 = 0;
        uint64_t X15 = 0;
        uint64_t X16 = 0;
        uint64_t X17 = 0;
        uint64_t X18 = 0;
        uint64_t X19 = 0;
        uint64_t X20 = 0;
        uint64_t X21 = 0;
        uint64_t X22 = 0;
        uint64_t X23 = 0;
        uint64_t X24 = 0;
        uint64_t X25 = 0;
        uint64_t X26 = 0;
        uint64_t X27 = 0;
        uint64_t X28 = 0;
        uint64_t X29 = 0;
        uint64_t X30 = 0;
        uint64_t X31 = 0;
        uint64_t PC = 0;
        uint64_t F0 = 0;
        uint64_t F1 = 0;
        uint64_t F2 = 0;
        uint64_t F3 = 0;
        uint64_t F4 = 0;
        uint64_t F5 = 0;
        uint64_t F6 = 0;
        uint64_t F7 = 0;
        uint64_t F8 = 0;
        uint64_t F9 = 0;
        uint64_t F10 = 0;
        uint64_t F11 = 0;
        uint64_t F12 = 0;
        uint64_t F13 = 0;
        uint64_t F14 = 0;
        uint64_t F15 = 0;
        uint64_t F16 = 0;
        uint64_t F17 = 0;
        uint64_t F18 = 0;
        uint64_t F19 = 0;
        uint64_t F20 = 0;
        uint64_t F21 = 0;
        uint64_t F22 = 0;
        uint64_t F23 = 0;
        uint64_t F24 = 0;
        uint64_t F25 = 0;
        uint64_t F26 = 0;
        uint64_t F27 = 0;
        uint64_t F28 = 0;
        uint64_t F29 = 0;
        uint64_t F30 = 0;
        uint64_t F31 = 0;
        uint32_t FCSR = 0;
        uint64_t NEXT_PC = 0;
        uint32_t trap_state = 0, pending_trap = 0, machine_state = 0, last_branch = 0;
        uint64_t icount = 0;
    } reg;

    std::array<address_type, 4> addr_mode;
    
    uint64_t interrupt_sim=0;

	uint32_t get_fcsr(){return reg.FCSR;}
	void set_fcsr(uint32_t val){reg.FCSR = val;}		

};

}
}            
#endif /* _RV64GC_H_ */
