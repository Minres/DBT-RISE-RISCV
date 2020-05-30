/*******************************************************************************
 * Copyright (C) 2020 MINRES Technologies GmbH
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

#include "../fp_functions.h"
#include <iss/arch/rv64i.h>
#include <iss/arch/riscv_hart_msu_vp.h>
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/server.h>
#include <iss/iss.h>
#include <iss/interp/vm_base.h>
#include <util/logging.h>
#include <sstream>

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#include <array>
#include <iss/debugger/riscv_target_adapter.h>

namespace iss {
namespace interp {
namespace rv64i {
using namespace iss::arch;
using namespace iss::debugger;

template <typename ARCH> class vm_impl : public iss::interp::vm_base<ARCH> {
public:
    using super = typename iss::interp::vm_base<ARCH>;
    using virt_addr_t = typename super::virt_addr_t;
    using phys_addr_t = typename super::phys_addr_t;
    using code_word_t = typename super::code_word_t;
    using addr_t = typename super::addr_t;
    using reg_t = typename traits<ARCH>::reg_t;
    using iss::interp::vm_base<ARCH>::get_reg;

    vm_impl();

    vm_impl(ARCH &core, unsigned core_id = 0, unsigned cluster_id = 0);

    void enableDebug(bool enable) { super::sync_exec = super::ALL_SYNC; }

    target_adapter_if *accquire_target_adapter(server_if *srv) override {
        debugger_if::dbg_enabled = true;
        if (super::tgt_adapter == nullptr)
            super::tgt_adapter = new riscv_target_adapter<ARCH>(srv, this->get_arch());
        return super::tgt_adapter;
    }

protected:
    using this_class = vm_impl<ARCH>;
    using compile_ret_t = virt_addr_t;
    using compile_func = compile_ret_t (this_class::*)(virt_addr_t &pc, code_word_t instr);

    inline const char *name(size_t index){return traits<ARCH>::reg_aliases.at(index);}

    virt_addr_t execute_inst(virt_addr_t start, std::function<bool(void)> pred) override;

    // some compile time constants
    // enum { MASK16 = 0b1111110001100011, MASK32 = 0b11111111111100000111000001111111 };
    enum { MASK16 = 0b1111111111111111, MASK32 = 0b11111111111100000111000001111111 };
    enum { EXTR_MASK16 = MASK16 >> 2, EXTR_MASK32 = MASK32 >> 2 };
    enum { LUT_SIZE = 1 << util::bit_count(EXTR_MASK32), LUT_SIZE_C = 1 << util::bit_count(EXTR_MASK16) };

    std::array<compile_func, LUT_SIZE> lut;

    std::array<compile_func, LUT_SIZE_C> lut_00, lut_01, lut_10;
    std::array<compile_func, LUT_SIZE> lut_11;

    std::array<compile_func *, 4> qlut;

    std::array<const uint32_t, 4> lutmasks = {{EXTR_MASK16, EXTR_MASK16, EXTR_MASK16, EXTR_MASK32}};

    void expand_bit_mask(int pos, uint32_t mask, uint32_t value, uint32_t valid, uint32_t idx, compile_func lut[],
                         compile_func f) {
        if (pos < 0) {
            lut[idx] = f;
        } else {
            auto bitmask = 1UL << pos;
            if ((mask & bitmask) == 0) {
                expand_bit_mask(pos - 1, mask, value, valid, idx, lut, f);
            } else {
                if ((valid & bitmask) == 0) {
                    expand_bit_mask(pos - 1, mask, value, valid, (idx << 1), lut, f);
                    expand_bit_mask(pos - 1, mask, value, valid, (idx << 1) + 1, lut, f);
                } else {
                    auto new_val = idx << 1;
                    if ((value & bitmask) != 0) new_val++;
                    expand_bit_mask(pos - 1, mask, value, valid, new_val, lut, f);
                }
            }
        }
    }

    inline uint32_t extract_fields(uint32_t val) { return extract_fields(29, val >> 2, lutmasks[val & 0x3], 0); }

    uint32_t extract_fields(int pos, uint32_t val, uint32_t mask, uint32_t lut_val) {
        if (pos >= 0) {
            auto bitmask = 1UL << pos;
            if ((mask & bitmask) == 0) {
                lut_val = extract_fields(pos - 1, val, mask, lut_val);
            } else {
                auto new_val = lut_val << 1;
                if ((val & bitmask) != 0) new_val++;
                lut_val = extract_fields(pos - 1, val, mask, new_val);
            }
        }
        return lut_val;
    }

    void raise_trap(uint16_t trap_id, uint16_t cause){
        auto trap_val =  0x80ULL << 24 | (cause << 16) | trap_id;
        this->template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE) = trap_val;
        this->template get_reg<uint32_t>(arch::traits<ARCH>::NEXT_PC) = std::numeric_limits<uint32_t>::max();
    }

    void leave_trap(unsigned lvl){
        this->core.leave_trap(lvl);
        auto pc_val = super::template read_mem<reg_t>(traits<ARCH>::CSR, (lvl << 8) + 0x41);
        this->template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = pc_val;
        this->template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH) = std::numeric_limits<uint32_t>::max();
    }

    void wait(unsigned type){
        this->core.wait_until(type);
    }


private:
    /****************************************************************************
     * start opcode definitions
     ****************************************************************************/
    struct InstructionDesriptor {
        size_t length;
        uint32_t value;
        uint32_t mask;
        compile_func op;
    };

    const std::array<InstructionDesriptor, 64> instr_descr = {{
         /* entries are: size, valid value, valid mask, function ptr */
        /* instruction LUI */
        {32, 0b00000000000000000000000000110111, 0b00000000000000000000000001111111, &this_class::__lui},
        /* instruction AUIPC */
        {32, 0b00000000000000000000000000010111, 0b00000000000000000000000001111111, &this_class::__auipc},
        /* instruction JAL */
        {32, 0b00000000000000000000000001101111, 0b00000000000000000000000001111111, &this_class::__jal},
        /* instruction JALR */
        {32, 0b00000000000000000000000001100111, 0b00000000000000000111000001111111, &this_class::__jalr},
        /* instruction BEQ */
        {32, 0b00000000000000000000000001100011, 0b00000000000000000111000001111111, &this_class::__beq},
        /* instruction BNE */
        {32, 0b00000000000000000001000001100011, 0b00000000000000000111000001111111, &this_class::__bne},
        /* instruction BLT */
        {32, 0b00000000000000000100000001100011, 0b00000000000000000111000001111111, &this_class::__blt},
        /* instruction BGE */
        {32, 0b00000000000000000101000001100011, 0b00000000000000000111000001111111, &this_class::__bge},
        /* instruction BLTU */
        {32, 0b00000000000000000110000001100011, 0b00000000000000000111000001111111, &this_class::__bltu},
        /* instruction BGEU */
        {32, 0b00000000000000000111000001100011, 0b00000000000000000111000001111111, &this_class::__bgeu},
        /* instruction LB */
        {32, 0b00000000000000000000000000000011, 0b00000000000000000111000001111111, &this_class::__lb},
        /* instruction LH */
        {32, 0b00000000000000000001000000000011, 0b00000000000000000111000001111111, &this_class::__lh},
        /* instruction LW */
        {32, 0b00000000000000000010000000000011, 0b00000000000000000111000001111111, &this_class::__lw},
        /* instruction LBU */
        {32, 0b00000000000000000100000000000011, 0b00000000000000000111000001111111, &this_class::__lbu},
        /* instruction LHU */
        {32, 0b00000000000000000101000000000011, 0b00000000000000000111000001111111, &this_class::__lhu},
        /* instruction SB */
        {32, 0b00000000000000000000000000100011, 0b00000000000000000111000001111111, &this_class::__sb},
        /* instruction SH */
        {32, 0b00000000000000000001000000100011, 0b00000000000000000111000001111111, &this_class::__sh},
        /* instruction SW */
        {32, 0b00000000000000000010000000100011, 0b00000000000000000111000001111111, &this_class::__sw},
        /* instruction ADDI */
        {32, 0b00000000000000000000000000010011, 0b00000000000000000111000001111111, &this_class::__addi},
        /* instruction SLTI */
        {32, 0b00000000000000000010000000010011, 0b00000000000000000111000001111111, &this_class::__slti},
        /* instruction SLTIU */
        {32, 0b00000000000000000011000000010011, 0b00000000000000000111000001111111, &this_class::__sltiu},
        /* instruction XORI */
        {32, 0b00000000000000000100000000010011, 0b00000000000000000111000001111111, &this_class::__xori},
        /* instruction ORI */
        {32, 0b00000000000000000110000000010011, 0b00000000000000000111000001111111, &this_class::__ori},
        /* instruction ANDI */
        {32, 0b00000000000000000111000000010011, 0b00000000000000000111000001111111, &this_class::__andi},
        /* instruction SLLI */
        {32, 0b00000000000000000001000000010011, 0b11111100000000000111000001111111, &this_class::__slli},
        /* instruction SRLI */
        {32, 0b00000000000000000101000000010011, 0b11111100000000000111000001111111, &this_class::__srli},
        /* instruction SRAI */
        {32, 0b01000000000000000101000000010011, 0b11111100000000000111000001111111, &this_class::__srai},
        /* instruction ADD */
        {32, 0b00000000000000000000000000110011, 0b11111110000000000111000001111111, &this_class::__add},
        /* instruction SUB */
        {32, 0b01000000000000000000000000110011, 0b11111110000000000111000001111111, &this_class::__sub},
        /* instruction SLL */
        {32, 0b00000000000000000001000000110011, 0b11111110000000000111000001111111, &this_class::__sll},
        /* instruction SLT */
        {32, 0b00000000000000000010000000110011, 0b11111110000000000111000001111111, &this_class::__slt},
        /* instruction SLTU */
        {32, 0b00000000000000000011000000110011, 0b11111110000000000111000001111111, &this_class::__sltu},
        /* instruction XOR */
        {32, 0b00000000000000000100000000110011, 0b11111110000000000111000001111111, &this_class::__xor},
        /* instruction SRL */
        {32, 0b00000000000000000101000000110011, 0b11111110000000000111000001111111, &this_class::__srl},
        /* instruction SRA */
        {32, 0b01000000000000000101000000110011, 0b11111110000000000111000001111111, &this_class::__sra},
        /* instruction OR */
        {32, 0b00000000000000000110000000110011, 0b11111110000000000111000001111111, &this_class::__or},
        /* instruction AND */
        {32, 0b00000000000000000111000000110011, 0b11111110000000000111000001111111, &this_class::__and},
        /* instruction FENCE */
        {32, 0b00000000000000000000000000001111, 0b11110000000000000111000001111111, &this_class::__fence},
        /* instruction FENCE_I */
        {32, 0b00000000000000000001000000001111, 0b00000000000000000111000001111111, &this_class::__fence_i},
        /* instruction ECALL */
        {32, 0b00000000000000000000000001110011, 0b11111111111111111111111111111111, &this_class::__ecall},
        /* instruction EBREAK */
        {32, 0b00000000000100000000000001110011, 0b11111111111111111111111111111111, &this_class::__ebreak},
        /* instruction URET */
        {32, 0b00000000001000000000000001110011, 0b11111111111111111111111111111111, &this_class::__uret},
        /* instruction SRET */
        {32, 0b00010000001000000000000001110011, 0b11111111111111111111111111111111, &this_class::__sret},
        /* instruction MRET */
        {32, 0b00110000001000000000000001110011, 0b11111111111111111111111111111111, &this_class::__mret},
        /* instruction WFI */
        {32, 0b00010000010100000000000001110011, 0b11111111111111111111111111111111, &this_class::__wfi},
        /* instruction SFENCE.VMA */
        {32, 0b00010010000000000000000001110011, 0b11111110000000000111111111111111, &this_class::__sfence_vma},
        /* instruction CSRRW */
        {32, 0b00000000000000000001000001110011, 0b00000000000000000111000001111111, &this_class::__csrrw},
        /* instruction CSRRS */
        {32, 0b00000000000000000010000001110011, 0b00000000000000000111000001111111, &this_class::__csrrs},
        /* instruction CSRRC */
        {32, 0b00000000000000000011000001110011, 0b00000000000000000111000001111111, &this_class::__csrrc},
        /* instruction CSRRWI */
        {32, 0b00000000000000000101000001110011, 0b00000000000000000111000001111111, &this_class::__csrrwi},
        /* instruction CSRRSI */
        {32, 0b00000000000000000110000001110011, 0b00000000000000000111000001111111, &this_class::__csrrsi},
        /* instruction CSRRCI */
        {32, 0b00000000000000000111000001110011, 0b00000000000000000111000001111111, &this_class::__csrrci},
        /* instruction LWU */
        {32, 0b00000000000000000110000000000011, 0b00000000000000000111000001111111, &this_class::__lwu},
        /* instruction LD */
        {32, 0b00000000000000000011000000000011, 0b00000000000000000111000001111111, &this_class::__ld},
        /* instruction SD */
        {32, 0b00000000000000000011000000100011, 0b00000000000000000111000001111111, &this_class::__sd},
        /* instruction ADDIW */
        {32, 0b00000000000000000000000000011011, 0b00000000000000000111000001111111, &this_class::__addiw},
        /* instruction SLLIW */
        {32, 0b00000000000000000001000000011011, 0b11111110000000000111000001111111, &this_class::__slliw},
        /* instruction SRLIW */
        {32, 0b00000000000000000101000000011011, 0b11111110000000000111000001111111, &this_class::__srliw},
        /* instruction SRAIW */
        {32, 0b01000000000000000101000000011011, 0b11111110000000000111000001111111, &this_class::__sraiw},
        /* instruction ADDW */
        {32, 0b00000000000000000000000000111011, 0b11111110000000000111000001111111, &this_class::__addw},
        /* instruction SUBW */
        {32, 0b01000000000000000000000000111011, 0b11111110000000000111000001111111, &this_class::__subw},
        /* instruction SLLW */
        {32, 0b00000000000000000001000000111011, 0b11111110000000000111000001111111, &this_class::__sllw},
        /* instruction SRLW */
        {32, 0b00000000000000000101000000111011, 0b11111110000000000111000001111111, &this_class::__srlw},
        /* instruction SRAW */
        {32, 0b01000000000000000101000000111011, 0b11111110000000000111000001111111, &this_class::__sraw},
    }};
 
    /* instruction definitions */
    /* instruction 0: LUI */
    compile_ret_t __lui(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 0);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        int32_t imm = signextend<int32_t,32>((bit_sub<12,20>(instr) << 12));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "lui"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (imm);
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 0);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 1: AUIPC */
    compile_ret_t __auipc(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 1);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        int32_t imm = signextend<int32_t,32>((bit_sub<12,20>(instr) << 12));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#08x}", fmt::arg("mnemonic", "auipc"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (static_cast<int64_t>(cur_pc_val) + (imm));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 1);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 2: JAL */
    compile_ret_t __jal(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 2);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        int32_t imm = signextend<int32_t,21>((bit_sub<12,8>(instr) << 12) | (bit_sub<20,1>(instr) << 11) | (bit_sub<21,10>(instr) << 1) | (bit_sub<31,1>(instr) << 20));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#0x}", fmt::arg("mnemonic", "jal"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (cur_pc_val + 4);
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto PC_val = (static_cast<int64_t>(cur_pc_val) + (imm));
        super::template get_reg(traits<ARCH>::NEXT_PC) = PC_val;
        auto is_cont_v = PC_val !=pc.val;
        super::template get_reg(traits<ARCH>::LAST_BRANCH) = is_cont_v?1:0;
        this->do_sync(POST_SYNC, 2);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 3: JALR */
    compile_ret_t __jalr(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 3);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm:#0x}", fmt::arg("mnemonic", "jalr"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto new_pc_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
        auto align_val = (new_pc_val & 0x2);
        {
            if((align_val != 0)) {
                raise_trap(0, 0);
            }
            else {
                if(rd != 0){
                    auto Xtmp0_val = (cur_pc_val + 4);
                    super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
                }
                auto PC_val = (new_pc_val & ~(0x1));
                super::template get_reg(traits<ARCH>::NEXT_PC) = PC_val;
                super::template get_reg(traits<ARCH>::LAST_BRANCH) = std::numeric_limits<uint32_t>::max();
            }
        }
        this->do_sync(POST_SYNC, 3);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 4: BEQ */
    compile_ret_t __beq(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 4);
        
        int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "beq"),
            	fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto PC_val = (super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0) == super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0))?
            (static_cast<int64_t>(cur_pc_val) + (imm)):
            (cur_pc_val + 4);
        super::template get_reg(traits<ARCH>::NEXT_PC) = PC_val;
        auto is_cont_v = PC_val !=pc.val;
        super::template get_reg(traits<ARCH>::LAST_BRANCH) = is_cont_v?1:0;
        this->do_sync(POST_SYNC, 4);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 5: BNE */
    compile_ret_t __bne(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 5);
        
        int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bne"),
            	fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto PC_val = (super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0) != super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0))?
            (static_cast<int64_t>(cur_pc_val) + (imm)):
            (cur_pc_val + 4);
        super::template get_reg(traits<ARCH>::NEXT_PC) = PC_val;
        auto is_cont_v = PC_val !=pc.val;
        super::template get_reg(traits<ARCH>::LAST_BRANCH) = is_cont_v?1:0;
        this->do_sync(POST_SYNC, 5);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 6: BLT */
    compile_ret_t __blt(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 6);
        
        int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "blt"),
            	fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto PC_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) < static_cast<int64_t>(super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)))?
            (static_cast<int64_t>(cur_pc_val) + (imm)):
            (cur_pc_val + 4);
        super::template get_reg(traits<ARCH>::NEXT_PC) = PC_val;
        auto is_cont_v = PC_val !=pc.val;
        super::template get_reg(traits<ARCH>::LAST_BRANCH) = is_cont_v?1:0;
        this->do_sync(POST_SYNC, 6);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 7: BGE */
    compile_ret_t __bge(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 7);
        
        int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bge"),
            	fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto PC_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) >= static_cast<int64_t>(super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)))?
            (static_cast<int64_t>(cur_pc_val) + (imm)):
            (cur_pc_val + 4);
        super::template get_reg(traits<ARCH>::NEXT_PC) = PC_val;
        auto is_cont_v = PC_val !=pc.val;
        super::template get_reg(traits<ARCH>::LAST_BRANCH) = is_cont_v?1:0;
        this->do_sync(POST_SYNC, 7);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 8: BLTU */
    compile_ret_t __bltu(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 8);
        
        int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bltu"),
            	fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto PC_val = (super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0) < super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0))?
            (static_cast<int64_t>(cur_pc_val) + (imm)):
            (cur_pc_val + 4);
        super::template get_reg(traits<ARCH>::NEXT_PC) = PC_val;
        auto is_cont_v = PC_val !=pc.val;
        super::template get_reg(traits<ARCH>::LAST_BRANCH) = is_cont_v?1:0;
        this->do_sync(POST_SYNC, 8);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 9: BGEU */
    compile_ret_t __bgeu(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 9);
        
        int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bgeu"),
            	fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto PC_val = (super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0) >= super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0))?
            (static_cast<int64_t>(cur_pc_val) + (imm)):
            (cur_pc_val + 4);
        super::template get_reg(traits<ARCH>::NEXT_PC) = PC_val;
        auto is_cont_v = PC_val !=pc.val;
        super::template get_reg(traits<ARCH>::LAST_BRANCH) = is_cont_v?1:0;
        this->do_sync(POST_SYNC, 9);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 10: LB */
    compile_ret_t __lb(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 10);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lb"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
        if(rd != 0){
            auto Xtmp0_val = super::template sext<int64_t>(super::template read_mem<uint8_t>(traits<ARCH>::MEM, offs_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 10);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 11: LH */
    compile_ret_t __lh(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 11);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lh"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
        if(rd != 0){
            auto Xtmp0_val = super::template sext<int64_t>(super::template read_mem<uint16_t>(traits<ARCH>::MEM, offs_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 11);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 12: LW */
    compile_ret_t __lw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 12);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lw"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
        if(rd != 0){
            auto Xtmp0_val = super::template sext<int64_t>(super::template read_mem<uint32_t>(traits<ARCH>::MEM, offs_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 12);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 13: LBU */
    compile_ret_t __lbu(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 13);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lbu"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
        if(rd != 0){
            auto Xtmp0_val = super::template zext<uint64_t>(super::template read_mem<uint8_t>(traits<ARCH>::MEM, offs_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 13);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 14: LHU */
    compile_ret_t __lhu(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 14);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lhu"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
        if(rd != 0){
            auto Xtmp0_val = super::template zext<uint64_t>(super::template read_mem<uint16_t>(traits<ARCH>::MEM, offs_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 14);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 15: SB */
    compile_ret_t __sb(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 15);
        
        int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sb"),
            	fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
        auto MEMtmp0_val = super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0);
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint8_t>(MEMtmp0_val));
        this->do_sync(POST_SYNC, 15);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 16: SH */
    compile_ret_t __sh(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 16);
        
        int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sh"),
            	fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
        auto MEMtmp0_val = super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0);
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint16_t>(MEMtmp0_val));
        this->do_sync(POST_SYNC, 16);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 17: SW */
    compile_ret_t __sw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 17);
        
        int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sw"),
            	fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
        auto MEMtmp0_val = super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0);
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint32_t>(MEMtmp0_val));
        this->do_sync(POST_SYNC, 17);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 18: ADDI */
    compile_ret_t __addi(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 18);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "addi"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 18);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 19: SLTI */
    compile_ret_t __slti(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 19);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "slti"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) < (imm))?
                1:
                0;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 19);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 20: SLTIU */
    compile_ret_t __sltiu(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 20);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "sltiu"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        int64_t full_imm_val = imm;
        if(rd != 0){
            auto Xtmp0_val = (super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0) < full_imm_val)?
                1:
                0;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 20);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 21: XORI */
    compile_ret_t __xori(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 21);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "xori"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) ^ (imm));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 21);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 22: ORI */
    compile_ret_t __ori(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 22);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "ori"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) | (imm));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 22);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 23: ANDI */
    compile_ret_t __andi(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 23);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "andi"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) & (imm));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 23);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 24: SLLI */
    compile_ret_t __slli(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 24);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,6>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "slli"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)<<(shamt));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 24);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 25: SRLI */
    compile_ret_t __srli(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 25);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,6>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srli"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (static_cast<uint64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0))>>(shamt));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 25);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 26: SRAI */
    compile_ret_t __srai(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 26);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,6>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srai"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0))>>(shamt));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 26);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 27: ADD */
    compile_ret_t __add(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 27);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "add"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0) + super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 27);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 28: SUB */
    compile_ret_t __sub(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 28);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sub"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0) - super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 28);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 29: SLL */
    compile_ret_t __sll(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 29);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sll"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)<<(super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0) & ((64) - 1)));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 29);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 30: SLT */
    compile_ret_t __slt(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 30);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "slt"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) < static_cast<int64_t>(super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)))?
                1:
                0;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 30);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 31: SLTU */
    compile_ret_t __sltu(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 31);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sltu"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (super::template zext<uint64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) < super::template zext<uint64_t>(super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)))?
                1:
                0;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 31);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 32: XOR */
    compile_ret_t __xor(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 32);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "xor"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0) ^ super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 32);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 33: SRL */
    compile_ret_t __srl(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 33);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "srl"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (static_cast<uint64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0))>>(super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0) & ((64) - 1)));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 33);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 34: SRA */
    compile_ret_t __sra(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 34);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sra"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0))>>(super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0) & ((64) - 1)));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 34);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 35: OR */
    compile_ret_t __or(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 35);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "or"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0) | super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 35);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 36: AND */
    compile_ret_t __and(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 36);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "and"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = (super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0) & super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 36);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 37: FENCE */
    compile_ret_t __fence(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 37);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t succ = ((bit_sub<20,4>(instr)));
        uint8_t pred = ((bit_sub<24,4>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            this->core.disass_output(pc.val, "fence");
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto FENCEtmp0_val = (((pred) << 4) | (succ));
        super::write_mem(traits<ARCH>::FENCE, (0), static_cast<uint64_t>(FENCEtmp0_val));
        this->do_sync(POST_SYNC, 37);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 38: FENCE_I */
    compile_ret_t __fence_i(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 38);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            this->core.disass_output(pc.val, "fence_i");
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto FENCEtmp0_val = (imm);
        super::write_mem(traits<ARCH>::FENCE, (1), static_cast<uint64_t>(FENCEtmp0_val));
        this->do_sync(POST_SYNC, 38);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 39: ECALL */
    compile_ret_t __ecall(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 39);
        
        if(this->disass_enabled){
            /* generate console output when executing the command */
            this->core.disass_output(pc.val, "ecall");
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        raise_trap(0, 11);
        this->do_sync(POST_SYNC, 39);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 40: EBREAK */
    compile_ret_t __ebreak(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 40);
        
        if(this->disass_enabled){
            /* generate console output when executing the command */
            this->core.disass_output(pc.val, "ebreak");
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        raise_trap(0, 3);
        this->do_sync(POST_SYNC, 40);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 41: URET */
    compile_ret_t __uret(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 41);
        
        if(this->disass_enabled){
            /* generate console output when executing the command */
            this->core.disass_output(pc.val, "uret");
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        leave_trap(0);
        this->do_sync(POST_SYNC, 41);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 42: SRET */
    compile_ret_t __sret(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 42);
        
        if(this->disass_enabled){
            /* generate console output when executing the command */
            this->core.disass_output(pc.val, "sret");
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        leave_trap(1);
        this->do_sync(POST_SYNC, 42);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 43: MRET */
    compile_ret_t __mret(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 43);
        
        if(this->disass_enabled){
            /* generate console output when executing the command */
            this->core.disass_output(pc.val, "mret");
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        leave_trap(3);
        this->do_sync(POST_SYNC, 43);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 44: WFI */
    compile_ret_t __wfi(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 44);
        
        if(this->disass_enabled){
            /* generate console output when executing the command */
            this->core.disass_output(pc.val, "wfi");
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        wait(1);
        this->do_sync(POST_SYNC, 44);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 45: SFENCE.VMA */
    compile_ret_t __sfence_vma(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 45);
        
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            this->core.disass_output(pc.val, "sfence.vma");
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto FENCEtmp0_val = (rs1);
        super::write_mem(traits<ARCH>::FENCE, (2), static_cast<uint64_t>(FENCEtmp0_val));
        auto FENCEtmp1_val = (rs2);
        super::write_mem(traits<ARCH>::FENCE, (3), static_cast<uint64_t>(FENCEtmp1_val));
        this->do_sync(POST_SYNC, 45);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 46: CSRRW */
    compile_ret_t __csrrw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 46);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {rs1}", fmt::arg("mnemonic", "csrrw"),
            	fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto rs_val_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        if(rd != 0){
            auto csr_val_val = super::template read_mem<uint64_t>(traits<ARCH>::CSR, (csr));
            auto CSRtmp0_val = rs_val_val;
            super::write_mem(traits<ARCH>::CSR, (csr), static_cast<uint64_t>(CSRtmp0_val));
            auto Xtmp1_val = csr_val_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp1_val;
        } else {
            auto CSRtmp2_val = rs_val_val;
            super::write_mem(traits<ARCH>::CSR, (csr), static_cast<uint64_t>(CSRtmp2_val));
        }
        this->do_sync(POST_SYNC, 46);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 47: CSRRS */
    compile_ret_t __csrrs(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 47);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {rs1}", fmt::arg("mnemonic", "csrrs"),
            	fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto xrd_val = super::template read_mem<uint64_t>(traits<ARCH>::CSR, (csr));
        auto xrs1_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        if(rd != 0){
            auto Xtmp0_val = xrd_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        if(rs1 != 0){
            auto CSRtmp1_val = (xrd_val | xrs1_val);
            super::write_mem(traits<ARCH>::CSR, (csr), static_cast<uint64_t>(CSRtmp1_val));
        }
        this->do_sync(POST_SYNC, 47);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 48: CSRRC */
    compile_ret_t __csrrc(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 48);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {rs1}", fmt::arg("mnemonic", "csrrc"),
            	fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto xrd_val = super::template read_mem<uint64_t>(traits<ARCH>::CSR, (csr));
        auto xrs1_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        if(rd != 0){
            auto Xtmp0_val = xrd_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        if(rs1 != 0){
            auto CSRtmp1_val = (xrd_val & ~(xrs1_val));
            super::write_mem(traits<ARCH>::CSR, (csr), static_cast<uint64_t>(CSRtmp1_val));
        }
        this->do_sync(POST_SYNC, 48);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 49: CSRRWI */
    compile_ret_t __csrrwi(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 49);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t zimm = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {zimm:#0x}", fmt::arg("mnemonic", "csrrwi"),
            	fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("zimm", zimm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = super::template read_mem<uint64_t>(traits<ARCH>::CSR, (csr));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto CSRtmp1_val = super::template zext<uint64_t>((zimm));
        super::write_mem(traits<ARCH>::CSR, (csr), static_cast<uint64_t>(CSRtmp1_val));
        this->do_sync(POST_SYNC, 49);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 50: CSRRSI */
    compile_ret_t __csrrsi(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 50);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t zimm = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {zimm:#0x}", fmt::arg("mnemonic", "csrrsi"),
            	fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("zimm", zimm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = super::template read_mem<uint64_t>(traits<ARCH>::CSR, (csr));
        if(zimm != 0){
            auto CSRtmp0_val = (res_val | super::template zext<uint64_t>((zimm)));
            super::write_mem(traits<ARCH>::CSR, (csr), static_cast<uint64_t>(CSRtmp0_val));
        }
        if(rd != 0){
            auto Xtmp1_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp1_val;
        }
        this->do_sync(POST_SYNC, 50);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 51: CSRRCI */
    compile_ret_t __csrrci(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 51);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t zimm = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {zimm:#0x}", fmt::arg("mnemonic", "csrrci"),
            	fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("zimm", zimm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = super::template read_mem<uint64_t>(traits<ARCH>::CSR, (csr));
        if(rd != 0){
            auto Xtmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        if(zimm != 0){
            auto CSRtmp1_val = (res_val & ~(super::template zext<uint64_t>((zimm))));
            super::write_mem(traits<ARCH>::CSR, (csr), static_cast<uint64_t>(CSRtmp1_val));
        }
        this->do_sync(POST_SYNC, 51);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 52: LWU */
    compile_ret_t __lwu(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 52);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lwu"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
        if(rd != 0){
            auto Xtmp0_val = super::template zext<uint64_t>(super::template read_mem<uint32_t>(traits<ARCH>::MEM, offs_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 52);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 53: LD */
    compile_ret_t __ld(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 53);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "ld"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
        if(rd != 0){
            auto Xtmp0_val = super::template sext<int64_t>(super::template read_mem<uint64_t>(traits<ARCH>::MEM, offs_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 53);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 54: SD */
    compile_ret_t __sd(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 54);
        
        int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sd"),
            	fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
        auto MEMtmp0_val = super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0);
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint64_t>(MEMtmp0_val));
        this->do_sync(POST_SYNC, 54);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 55: ADDIW */
    compile_ret_t __addiw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 55);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "addiw"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto res_val = (static_cast<int32_t>(static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
            )) + (imm));
            auto Xtmp0_val = super::template sext<int64_t>(res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 55);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 56: SLLIW */
    compile_ret_t __slliw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 56);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "slliw"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto sh_val_val = (static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
            )<<(shamt));
            auto Xtmp0_val = super::template sext<int64_t>(sh_val_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 56);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 57: SRLIW */
    compile_ret_t __srliw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 57);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srliw"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto sh_val_val = (static_cast<uint32_t>(static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
            ))>>(shamt));
            auto Xtmp0_val = super::template sext<int64_t>(sh_val_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 57);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 58: SRAIW */
    compile_ret_t __sraiw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 58);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "sraiw"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto sh_val_val = (static_cast<int32_t>(static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
            ))>>(shamt));
            auto Xtmp0_val = super::template sext<int64_t>(sh_val_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 58);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 59: ADDW */
    compile_ret_t __addw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 59);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            this->core.disass_output(pc.val, "addw");
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto res_val = (static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
            ) + static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)
            ));
            auto Xtmp0_val = super::template sext<int64_t>(res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 59);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 60: SUBW */
    compile_ret_t __subw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 60);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            this->core.disass_output(pc.val, "subw");
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto res_val = (static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
            ) - static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)
            ));
            auto Xtmp0_val = super::template sext<int64_t>(res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 60);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 61: SLLW */
    compile_ret_t __sllw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 61);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sllw"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            uint32_t mask_val = 0x1f;
            auto count_val = (static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)
            ) & mask_val);
            auto sh_val_val = (static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
            )<<count_val);
            auto Xtmp0_val = super::template sext<int64_t>(sh_val_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 61);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 62: SRLW */
    compile_ret_t __srlw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 62);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "srlw"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            uint32_t mask_val = 0x1f;
            auto count_val = (static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)
            ) & mask_val);
            auto sh_val_val = (static_cast<uint32_t>(static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
            ))>>count_val);
            auto Xtmp0_val = super::template sext<int64_t>(sh_val_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 62);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 63: SRAW */
    compile_ret_t __sraw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 63);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sraw"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            uint32_t mask_val = 0x1f;
            auto count_val = (static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)
            ) & mask_val);
            auto sh_val_val = (static_cast<int32_t>(static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
            ))>>count_val);
            auto Xtmp0_val = super::template sext<int64_t>(sh_val_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 63);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /****************************************************************************
     * end opcode definitions
     ****************************************************************************/
    compile_ret_t illegal_intruction(virt_addr_t &pc, code_word_t instr) {
        pc = pc + ((instr & 3) == 3 ? 4 : 2);
        return pc;
    }
};

template <typename CODE_WORD> void debug_fn(CODE_WORD insn) {
    volatile CODE_WORD x = insn;
    insn = 2 * x;
}

template <typename ARCH> vm_impl<ARCH>::vm_impl() { this(new ARCH()); }

template <typename ARCH>
vm_impl<ARCH>::vm_impl(ARCH &core, unsigned core_id, unsigned cluster_id)
: vm_base<ARCH>(core, core_id, cluster_id) {
    qlut[0] = lut_00.data();
    qlut[1] = lut_01.data();
    qlut[2] = lut_10.data();
    qlut[3] = lut_11.data();
    for (auto instr : instr_descr) {
        auto quantrant = instr.value & 0x3;
        expand_bit_mask(29, lutmasks[quantrant], instr.value >> 2, instr.mask >> 2, 0, qlut[quantrant], instr.op);
    }
}

template <typename ARCH>
typename vm_base<ARCH>::virt_addr_t vm_impl<ARCH>::execute_inst(virt_addr_t start, std::function<bool(void)> pred) {
    // we fetch at max 4 byte, alignment is 2
    enum {TRAP_ID=1<<16};
    const typename traits<ARCH>::addr_t upper_bits = ~traits<ARCH>::PGMASK;
    code_word_t insn = 0;
    auto *const data = (uint8_t *)&insn;
    auto pc=start;
    while(pred){
        auto paddr = this->core.v2p(pc);
        if ((pc.val & upper_bits) != ((pc.val + 2) & upper_bits)) { // we may cross a page boundary
            if (this->core.read(paddr, 2, data) != iss::Ok) throw trap_access(TRAP_ID, pc.val);
            if ((insn & 0x3) == 0x3) // this is a 32bit instruction
                if (this->core.read(this->core.v2p(pc + 2), 2, data + 2) != iss::Ok) throw trap_access(TRAP_ID, pc.val);
        } else {
            if (this->core.read(paddr, 4, data) != iss::Ok) throw trap_access(TRAP_ID, pc.val);
        }
        if (insn == 0x0000006f || (insn&0xffff)==0xa001) throw simulation_stopped(0); // 'J 0' or 'C.J 0'
        auto lut_val = extract_fields(insn);
        auto f = qlut[insn & 0x3][lut_val];
        if (!f)
            f = &this_class::illegal_intruction;
        pc = (this->*f)(pc, insn);
    }
    return pc;
}

} // namespace mnrv32

template <>
std::unique_ptr<vm_if> create<arch::rv64i>(arch::rv64i *core, unsigned short port, bool dump) {
    auto ret = new rv64i::vm_impl<arch::rv64i>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret, port);
    return std::unique_ptr<vm_if>(ret);
}
} // namespace interp
} // namespace iss
