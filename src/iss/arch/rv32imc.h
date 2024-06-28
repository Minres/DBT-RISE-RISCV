/*******************************************************************************
 * Copyright (C) 2017 - 2021 MINRES Technologies GmbH
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

#ifndef _rv32imc_H_
#define _rv32imc_H_
// clang-format off
#include <array>
#include <iss/arch/traits.h>
#include <iss/arch_if.h>
#include <iss/vm_if.h>

namespace iss {
namespace arch {

struct rv32imc;

template <> struct traits<rv32imc> {

    constexpr static char const* const core_type = "rv32imc";
    
    static constexpr std::array<const char*, 36> reg_names{
        {"x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30", "x31", "pc", "next_pc", "priv", "dpc"}};
 
    static constexpr std::array<const char*, 36> reg_aliases{
        {"zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6", "pc", "next_pc", "priv", "dpc"}};

    enum constants {MISA_VAL=1073746180ULL, MARCHID_VAL=2147483651ULL, CLIC_NUM_IRQ=0ULL, XLEN=32ULL, INSTR_ALIGNMENT=2ULL, RFS=32ULL, fence=0ULL, fencei=1ULL, fencevmal=2ULL, fencevmau=3ULL, CSR_SIZE=4096ULL, MUL_LEN=64ULL};

    constexpr static unsigned FP_REGS_SIZE = 0;

    enum reg_e {
        X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30, X31, PC, NEXT_PC, PRIV, DPC, NUM_REGS, TRAP_STATE=NUM_REGS, PENDING_TRAP, ICOUNT, CYCLE, INSTRET, INSTRUCTION, LAST_BRANCH
    };

    using reg_t = uint32_t;

    using addr_t = uint32_t;

    using code_word_t = uint32_t; //TODO: check removal

    using virt_addr_t = iss::typed_addr_t<iss::address_type::VIRTUAL>;

    using phys_addr_t = iss::typed_addr_t<iss::address_type::PHYSICAL>;

    static constexpr std::array<const uint32_t, 43> reg_bit_widths{
        {32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,8,32,32,32,64,64,64,32,32}};

    static constexpr std::array<const uint32_t, 43> reg_byte_offsets{
        {0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,137,141,145,149,157,165,173,177}};

    static const uint64_t addr_mask = (reg_t(1) << (XLEN - 1)) | ((reg_t(1) << (XLEN - 1)) - 1);

    enum sreg_flag_e { FLAGS };

    enum mem_type_e { MEM, FENCE, RES, CSR, IMEM = MEM };
    
    enum class opcode_e {
        LUI = 0,
        AUIPC = 1,
        JAL = 2,
        JALR = 3,
        BEQ = 4,
        BNE = 5,
        BLT = 6,
        BGE = 7,
        BLTU = 8,
        BGEU = 9,
        LB = 10,
        LH = 11,
        LW = 12,
        LBU = 13,
        LHU = 14,
        SB = 15,
        SH = 16,
        SW = 17,
        ADDI = 18,
        SLTI = 19,
        SLTIU = 20,
        XORI = 21,
        ORI = 22,
        ANDI = 23,
        SLLI = 24,
        SRLI = 25,
        SRAI = 26,
        ADD = 27,
        SUB = 28,
        SLL = 29,
        SLT = 30,
        SLTU = 31,
        XOR = 32,
        SRL = 33,
        SRA = 34,
        OR = 35,
        AND = 36,
        FENCE = 37,
        ECALL = 38,
        EBREAK = 39,
        MRET = 40,
        WFI = 41,
        CSRRW = 42,
        CSRRS = 43,
        CSRRC = 44,
        CSRRWI = 45,
        CSRRSI = 46,
        CSRRCI = 47,
        FENCE_I = 48,
        MUL = 49,
        MULH = 50,
        MULHSU = 51,
        MULHU = 52,
        DIV = 53,
        DIVU = 54,
        REM = 55,
        REMU = 56,
        C__ADDI4SPN = 57,
        C__LW = 58,
        C__SW = 59,
        C__ADDI = 60,
        C__NOP = 61,
        C__JAL = 62,
        C__LI = 63,
        C__LUI = 64,
        C__ADDI16SP = 65,
        __reserved_clui = 66,
        C__SRLI = 67,
        C__SRAI = 68,
        C__ANDI = 69,
        C__SUB = 70,
        C__XOR = 71,
        C__OR = 72,
        C__AND = 73,
        C__J = 74,
        C__BEQZ = 75,
        C__BNEZ = 76,
        C__SLLI = 77,
        C__LWSP = 78,
        C__MV = 79,
        C__JR = 80,
        __reserved_cmv = 81,
        C__ADD = 82,
        C__JALR = 83,
        C__EBREAK = 84,
        C__SWSP = 85,
        DII = 86,
        MAX_OPCODE
    };
};

struct rv32imc: public arch_if {

    using virt_addr_t = typename traits<rv32imc>::virt_addr_t;
    using phys_addr_t = typename traits<rv32imc>::phys_addr_t;
    using reg_t =  typename traits<rv32imc>::reg_t;
    using addr_t = typename traits<rv32imc>::addr_t;

    rv32imc();
    ~rv32imc();

    void reset(uint64_t address=0) override;

    uint8_t* get_regs_base_ptr() override;

    inline uint64_t get_icount() { return reg.icount; }

    inline bool should_stop() { return interrupt_sim; }

    inline uint64_t stop_code() { return interrupt_sim; }

    virtual phys_addr_t virt2phys(const iss::addr_t& addr);

    virtual iss::sync_type needed_sync() const { return iss::NO_SYNC; }

    inline uint32_t get_last_branch() { return reg.last_branch; }


#pragma pack(push, 1)
    struct rv32imc_regs { 
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
        uint8_t PRIV = 0; 
        uint32_t DPC = 0;
        uint32_t trap_state = 0, pending_trap = 0;
        uint64_t icount = 0;
        uint64_t cycle = 0;
        uint64_t instret = 0;
        uint32_t instruction = 0;
        uint32_t last_branch = 0;
    } reg;
#pragma pack(pop)
    std::array<address_type, 4> addr_mode;
    
    uint64_t interrupt_sim=0;

    uint32_t get_fcsr(){return 0;}
    void set_fcsr(uint32_t val){}

};

}
}            
#endif /* _rv32imc_H_ */
// clang-format on
