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

#include <iss/arch/mnrv32.h>
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
namespace mnrv32 {
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


    virt_addr_t execute_single_inst(virt_addr_t pc) override;

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

    const std::array<InstructionDesriptor, 52> instr_descr = {{
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
        {32, 0b00000000000000000001000000010011, 0b11111110000000000111000001111111, &this_class::__slli},
        /* instruction SRLI */
        {32, 0b00000000000000000101000000010011, 0b11111110000000000111000001111111, &this_class::__srli},
        /* instruction SRAI */
        {32, 0b01000000000000000101000000010011, 0b11111110000000000111000001111111, &this_class::__srai},
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
    }};
 
    /* instruction definitions */
    /* instruction 0: LUI */
    compile_ret_t __lui(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 1: AUIPC */
    compile_ret_t __auipc(virt_addr_t& pc, code_word_t instr){
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
        auto cur_pc = pc.val;
        pc=pc+4;
        if(rd != 0){
            auto& rs = this->template get_reg<reg_t>(traits<ARCH>::X0 + rd);
            rs=pc.val;
        }
        auto& pc_reg = this->template get_reg<reg_t>(arch::traits<ARCH>::PC);
        pc_reg = cur_pc+imm;
        this->do_sync(POST_SYNC, 2);
        auto& trap_state = this->template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        if(trap_state!=0){
            auto& last_br = this->template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            this->core.enter_trap(trap_state, cur_pc);
            pc.val=this->template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        }
        return pc;
    }
    
    /* instruction 3: JALR */
    compile_ret_t __jalr(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 4: BEQ */
    compile_ret_t __beq(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 5: BNE */
    compile_ret_t __bne(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 6: BLT */
    compile_ret_t __blt(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 7: BGE */
    compile_ret_t __bge(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 8: BLTU */
    compile_ret_t __bltu(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 9: BGEU */
    compile_ret_t __bgeu(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 10: LB */
    compile_ret_t __lb(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 11: LH */
    compile_ret_t __lh(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 12: LW */
    compile_ret_t __lw(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 13: LBU */
    compile_ret_t __lbu(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 14: LHU */
    compile_ret_t __lhu(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 15: SB */
    compile_ret_t __sb(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 16: SH */
    compile_ret_t __sh(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 17: SW */
    compile_ret_t __sw(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 18: ADDI */
    compile_ret_t __addi(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 19: SLTI */
    compile_ret_t __slti(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 20: SLTIU */
    compile_ret_t __sltiu(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 21: XORI */
    compile_ret_t __xori(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 22: ORI */
    compile_ret_t __ori(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 23: ANDI */
    compile_ret_t __andi(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 24: SLLI */
    compile_ret_t __slli(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 25: SRLI */
    compile_ret_t __srli(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 26: SRAI */
    compile_ret_t __srai(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 27: ADD */
    compile_ret_t __add(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 28: SUB */
    compile_ret_t __sub(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 29: SLL */
    compile_ret_t __sll(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 30: SLT */
    compile_ret_t __slt(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 31: SLTU */
    compile_ret_t __sltu(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 32: XOR */
    compile_ret_t __xor(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 33: SRL */
    compile_ret_t __srl(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 34: SRA */
    compile_ret_t __sra(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 35: OR */
    compile_ret_t __or(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 36: AND */
    compile_ret_t __and(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 37: FENCE */
    compile_ret_t __fence(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 38: FENCE_I */
    compile_ret_t __fence_i(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 39: ECALL */
    compile_ret_t __ecall(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 40: EBREAK */
    compile_ret_t __ebreak(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 41: URET */
    compile_ret_t __uret(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 42: SRET */
    compile_ret_t __sret(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 43: MRET */
    compile_ret_t __mret(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 44: WFI */
    compile_ret_t __wfi(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 45: SFENCE.VMA */
    compile_ret_t __sfence_vma(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 46: CSRRW */
    compile_ret_t __csrrw(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 47: CSRRS */
    compile_ret_t __csrrs(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 48: CSRRC */
    compile_ret_t __csrrc(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 49: CSRRWI */
    compile_ret_t __csrrwi(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 50: CSRRSI */
    compile_ret_t __csrrsi(virt_addr_t& pc, code_word_t instr){
    }
    
    /* instruction 51: CSRRCI */
    compile_ret_t __csrrci(virt_addr_t& pc, code_word_t instr){
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
typename vm_base<ARCH>::virt_addr_t vm_impl<ARCH>::execute_single_inst(virt_addr_t pc) {
    // we fetch at max 4 byte, alignment is 2
    enum {TRAP_ID=1<<16};
    code_word_t insn = 0;
    const typename traits<ARCH>::addr_t upper_bits = ~traits<ARCH>::PGMASK;
    phys_addr_t paddr(pc);
    auto *const data = (uint8_t *)&insn;
    paddr = this->core.v2p(pc);
    if ((pc.val & upper_bits) != ((pc.val + 2) & upper_bits)) { // we may cross a page boundary
        auto res = this->core.read(paddr, 2, data);
        if (res != iss::Ok) throw trap_access(TRAP_ID, pc.val);
        if ((insn & 0x3) == 0x3) { // this is a 32bit instruction
            res = this->core.read(this->core.v2p(pc + 2), 2, data + 2);
        }
    } else {
        auto res = this->core.read(paddr, 4, data);
        if (res != iss::Ok) throw trap_access(TRAP_ID, pc.val);
    }
    if (insn == 0x0000006f || (insn&0xffff)==0xa001) throw simulation_stopped(0); // 'J 0' or 'C.J 0'
    // curr pc on stack
    auto lut_val = extract_fields(insn);
    auto f = qlut[insn & 0x3][lut_val];
    if (f == nullptr) {
        f = &this_class::illegal_intruction;
    }
    return (this->*f)(pc, insn);
}

} // namespace mnrv32

template <>
std::unique_ptr<vm_if> create<arch::mnrv32>(arch::mnrv32 *core, unsigned short port, bool dump) {
    auto ret = new mnrv32::vm_impl<arch::mnrv32>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret, port);
    return std::unique_ptr<vm_if>(ret);
}
} // namespace interp
} // namespace iss
