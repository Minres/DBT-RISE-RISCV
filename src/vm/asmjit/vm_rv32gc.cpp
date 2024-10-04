/*******************************************************************************
 * Copyright (C) 2024 MINRES Technologies GmbH
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
#include <iss/arch/rv32gc.h>
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/server.h>
#include <iss/iss.h>
#include <iss/asmjit/vm_base.h>
#include <asmjit/asmjit.h>
#include <util/logging.h>
#include <iss/instruction_decoder.h>

#include <vm/fp_functions.h>
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#include <array>
#include <iss/debugger/riscv_target_adapter.h>

namespace iss {
namespace asmjit {


namespace rv32gc {
using namespace ::asmjit;
using namespace iss::arch;
using namespace iss::debugger;

template <typename ARCH> class vm_impl : public iss::asmjit::vm_base<ARCH> {
public:
    using traits = arch::traits<ARCH>;
    using super = typename iss::asmjit::vm_base<ARCH>;
    using virt_addr_t = typename super::virt_addr_t;
    using phys_addr_t = typename super::phys_addr_t;
    using code_word_t = typename super::code_word_t;
    using mem_type_e = typename super::mem_type_e;
    using addr_t = typename super::addr_t;

    vm_impl();

    vm_impl(ARCH &core, unsigned core_id = 0, unsigned cluster_id = 0);

    void enableDebug(bool enable) { super::sync_exec = super::ALL_SYNC; }

    target_adapter_if *accquire_target_adapter(server_if *srv) override {
        debugger_if::dbg_enabled = true;
        if (vm_base<ARCH>::tgt_adapter == nullptr)
            vm_base<ARCH>::tgt_adapter = new riscv_target_adapter<ARCH>(srv, this->get_arch());
        return vm_base<ARCH>::tgt_adapter;
    }

protected:
    using super::get_ptr_for;
    using super::get_reg_for;
    using super::get_reg_for_Gp;
    using super::load_reg_from_mem;
    using super::load_reg_from_mem_Gp;
    using super::write_reg_to_mem;
    using super::gen_read_mem;
    using super::gen_write_mem;
    using super::gen_leave;
    using super::gen_sync;
   
    using this_class = vm_impl<ARCH>;
    using compile_func = continuation_e (this_class::*)(virt_addr_t&, code_word_t, jit_holder&);

    continuation_e gen_single_inst_behavior(virt_addr_t&, unsigned int &, jit_holder&) override;
    enum globals_e {TVAL = 0, GLOBALS_SIZE};
    void gen_block_prologue(jit_holder& jh) override;
    void gen_block_epilogue(jit_holder& jh) override;
    inline const char *name(size_t index){return traits::reg_aliases.at(index);}

    inline const char *fname(size_t index){return index < 32?name(index+traits::F0):"illegal";}   

    void gen_instr_prologue(jit_holder& jh);
    void gen_instr_epilogue(jit_holder& jh);
    inline void gen_raise(jit_holder& jh, uint16_t trap_id, uint16_t cause);
    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type> void gen_set_tval(jit_holder& jh, T new_tval) ;
    void gen_set_tval(jit_holder& jh, x86_reg_t _new_tval) ;

    template<unsigned W, typename U, typename S = typename std::make_signed<U>::type>
    inline S sext(U from) {
        auto mask = (1ULL<<W) - 1;
        auto sign_mask = 1ULL<<(W-1);
        return (from & mask) | ((from & sign_mask) ? ~mask : 0);
    }

    x86_reg_t get_rm(jit_holder& jh , uint8_t get_rm_rm){
        x86::Compiler& cc = jh.cc;
        x86_reg_t rm = get_reg(cc, 8, false);
        mov(cc, rm, get_rm_rm);
        auto label_then378 = cc.newLabel();
        auto label_merge378 = cc.newLabel();
        auto tmp_reg378 = get_reg(cc, 8, false);
        cmp(cc, rm, 7);
        cc.je(label_then378);
        mov(cc, tmp_reg378,rm);
        cc.jmp(label_merge378);
        cc.bind(label_then378);
        mov(cc, tmp_reg378, gen_slice(jh.cc, load_reg_from_mem(jh, traits::FCSR), 5, 7-5+1));
        cc.bind(label_merge378);
        auto rm_eff = tmp_reg378
        ;
        {
        auto label_merge = cc.newLabel();
        cmp(cc, gen_operation(cc, gtu, rm_eff, 4)
        ,0);
        cc.je(label_merge);
        {
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        cc.bind(label_merge);
        }
        return rm_eff;
    }

private:
    /****************************************************************************
     * start opcode definitions
     ****************************************************************************/
    struct instruction_descriptor {
        uint32_t length;
        uint32_t value;
        uint32_t mask;
        compile_func op;
    };

    const std::array<instruction_descriptor, 158> instr_descr = {{
         /* entries are: size, valid value, valid mask, function ptr */
        /* instruction LUI, encoding '0b00000000000000000000000000110111' */
        {32, 0b00000000000000000000000000110111, 0b00000000000000000000000001111111, &this_class::__lui},
        /* instruction AUIPC, encoding '0b00000000000000000000000000010111' */
        {32, 0b00000000000000000000000000010111, 0b00000000000000000000000001111111, &this_class::__auipc},
        /* instruction JAL, encoding '0b00000000000000000000000001101111' */
        {32, 0b00000000000000000000000001101111, 0b00000000000000000000000001111111, &this_class::__jal},
        /* instruction JALR, encoding '0b00000000000000000000000001100111' */
        {32, 0b00000000000000000000000001100111, 0b00000000000000000111000001111111, &this_class::__jalr},
        /* instruction BEQ, encoding '0b00000000000000000000000001100011' */
        {32, 0b00000000000000000000000001100011, 0b00000000000000000111000001111111, &this_class::__beq},
        /* instruction BNE, encoding '0b00000000000000000001000001100011' */
        {32, 0b00000000000000000001000001100011, 0b00000000000000000111000001111111, &this_class::__bne},
        /* instruction BLT, encoding '0b00000000000000000100000001100011' */
        {32, 0b00000000000000000100000001100011, 0b00000000000000000111000001111111, &this_class::__blt},
        /* instruction BGE, encoding '0b00000000000000000101000001100011' */
        {32, 0b00000000000000000101000001100011, 0b00000000000000000111000001111111, &this_class::__bge},
        /* instruction BLTU, encoding '0b00000000000000000110000001100011' */
        {32, 0b00000000000000000110000001100011, 0b00000000000000000111000001111111, &this_class::__bltu},
        /* instruction BGEU, encoding '0b00000000000000000111000001100011' */
        {32, 0b00000000000000000111000001100011, 0b00000000000000000111000001111111, &this_class::__bgeu},
        /* instruction LB, encoding '0b00000000000000000000000000000011' */
        {32, 0b00000000000000000000000000000011, 0b00000000000000000111000001111111, &this_class::__lb},
        /* instruction LH, encoding '0b00000000000000000001000000000011' */
        {32, 0b00000000000000000001000000000011, 0b00000000000000000111000001111111, &this_class::__lh},
        /* instruction LW, encoding '0b00000000000000000010000000000011' */
        {32, 0b00000000000000000010000000000011, 0b00000000000000000111000001111111, &this_class::__lw},
        /* instruction LBU, encoding '0b00000000000000000100000000000011' */
        {32, 0b00000000000000000100000000000011, 0b00000000000000000111000001111111, &this_class::__lbu},
        /* instruction LHU, encoding '0b00000000000000000101000000000011' */
        {32, 0b00000000000000000101000000000011, 0b00000000000000000111000001111111, &this_class::__lhu},
        /* instruction SB, encoding '0b00000000000000000000000000100011' */
        {32, 0b00000000000000000000000000100011, 0b00000000000000000111000001111111, &this_class::__sb},
        /* instruction SH, encoding '0b00000000000000000001000000100011' */
        {32, 0b00000000000000000001000000100011, 0b00000000000000000111000001111111, &this_class::__sh},
        /* instruction SW, encoding '0b00000000000000000010000000100011' */
        {32, 0b00000000000000000010000000100011, 0b00000000000000000111000001111111, &this_class::__sw},
        /* instruction ADDI, encoding '0b00000000000000000000000000010011' */
        {32, 0b00000000000000000000000000010011, 0b00000000000000000111000001111111, &this_class::__addi},
        /* instruction SLTI, encoding '0b00000000000000000010000000010011' */
        {32, 0b00000000000000000010000000010011, 0b00000000000000000111000001111111, &this_class::__slti},
        /* instruction SLTIU, encoding '0b00000000000000000011000000010011' */
        {32, 0b00000000000000000011000000010011, 0b00000000000000000111000001111111, &this_class::__sltiu},
        /* instruction XORI, encoding '0b00000000000000000100000000010011' */
        {32, 0b00000000000000000100000000010011, 0b00000000000000000111000001111111, &this_class::__xori},
        /* instruction ORI, encoding '0b00000000000000000110000000010011' */
        {32, 0b00000000000000000110000000010011, 0b00000000000000000111000001111111, &this_class::__ori},
        /* instruction ANDI, encoding '0b00000000000000000111000000010011' */
        {32, 0b00000000000000000111000000010011, 0b00000000000000000111000001111111, &this_class::__andi},
        /* instruction SLLI, encoding '0b00000000000000000001000000010011' */
        {32, 0b00000000000000000001000000010011, 0b11111110000000000111000001111111, &this_class::__slli},
        /* instruction SRLI, encoding '0b00000000000000000101000000010011' */
        {32, 0b00000000000000000101000000010011, 0b11111110000000000111000001111111, &this_class::__srli},
        /* instruction SRAI, encoding '0b01000000000000000101000000010011' */
        {32, 0b01000000000000000101000000010011, 0b11111110000000000111000001111111, &this_class::__srai},
        /* instruction ADD, encoding '0b00000000000000000000000000110011' */
        {32, 0b00000000000000000000000000110011, 0b11111110000000000111000001111111, &this_class::__add},
        /* instruction SUB, encoding '0b01000000000000000000000000110011' */
        {32, 0b01000000000000000000000000110011, 0b11111110000000000111000001111111, &this_class::__sub},
        /* instruction SLL, encoding '0b00000000000000000001000000110011' */
        {32, 0b00000000000000000001000000110011, 0b11111110000000000111000001111111, &this_class::__sll},
        /* instruction SLT, encoding '0b00000000000000000010000000110011' */
        {32, 0b00000000000000000010000000110011, 0b11111110000000000111000001111111, &this_class::__slt},
        /* instruction SLTU, encoding '0b00000000000000000011000000110011' */
        {32, 0b00000000000000000011000000110011, 0b11111110000000000111000001111111, &this_class::__sltu},
        /* instruction XOR, encoding '0b00000000000000000100000000110011' */
        {32, 0b00000000000000000100000000110011, 0b11111110000000000111000001111111, &this_class::__xor},
        /* instruction SRL, encoding '0b00000000000000000101000000110011' */
        {32, 0b00000000000000000101000000110011, 0b11111110000000000111000001111111, &this_class::__srl},
        /* instruction SRA, encoding '0b01000000000000000101000000110011' */
        {32, 0b01000000000000000101000000110011, 0b11111110000000000111000001111111, &this_class::__sra},
        /* instruction OR, encoding '0b00000000000000000110000000110011' */
        {32, 0b00000000000000000110000000110011, 0b11111110000000000111000001111111, &this_class::__or},
        /* instruction AND, encoding '0b00000000000000000111000000110011' */
        {32, 0b00000000000000000111000000110011, 0b11111110000000000111000001111111, &this_class::__and},
        /* instruction FENCE, encoding '0b00000000000000000000000000001111' */
        {32, 0b00000000000000000000000000001111, 0b00000000000000000111000001111111, &this_class::__fence},
        /* instruction ECALL, encoding '0b00000000000000000000000001110011' */
        {32, 0b00000000000000000000000001110011, 0b11111111111111111111111111111111, &this_class::__ecall},
        /* instruction EBREAK, encoding '0b00000000000100000000000001110011' */
        {32, 0b00000000000100000000000001110011, 0b11111111111111111111111111111111, &this_class::__ebreak},
        /* instruction MRET, encoding '0b00110000001000000000000001110011' */
        {32, 0b00110000001000000000000001110011, 0b11111111111111111111111111111111, &this_class::__mret},
        /* instruction WFI, encoding '0b00010000010100000000000001110011' */
        {32, 0b00010000010100000000000001110011, 0b11111111111111111111111111111111, &this_class::__wfi},
        /* instruction CSRRW, encoding '0b00000000000000000001000001110011' */
        {32, 0b00000000000000000001000001110011, 0b00000000000000000111000001111111, &this_class::__csrrw},
        /* instruction CSRRS, encoding '0b00000000000000000010000001110011' */
        {32, 0b00000000000000000010000001110011, 0b00000000000000000111000001111111, &this_class::__csrrs},
        /* instruction CSRRC, encoding '0b00000000000000000011000001110011' */
        {32, 0b00000000000000000011000001110011, 0b00000000000000000111000001111111, &this_class::__csrrc},
        /* instruction CSRRWI, encoding '0b00000000000000000101000001110011' */
        {32, 0b00000000000000000101000001110011, 0b00000000000000000111000001111111, &this_class::__csrrwi},
        /* instruction CSRRSI, encoding '0b00000000000000000110000001110011' */
        {32, 0b00000000000000000110000001110011, 0b00000000000000000111000001111111, &this_class::__csrrsi},
        /* instruction CSRRCI, encoding '0b00000000000000000111000001110011' */
        {32, 0b00000000000000000111000001110011, 0b00000000000000000111000001111111, &this_class::__csrrci},
        /* instruction FENCE_I, encoding '0b00000000000000000001000000001111' */
        {32, 0b00000000000000000001000000001111, 0b00000000000000000111000001111111, &this_class::__fence_i},
        /* instruction MUL, encoding '0b00000010000000000000000000110011' */
        {32, 0b00000010000000000000000000110011, 0b11111110000000000111000001111111, &this_class::__mul},
        /* instruction MULH, encoding '0b00000010000000000001000000110011' */
        {32, 0b00000010000000000001000000110011, 0b11111110000000000111000001111111, &this_class::__mulh},
        /* instruction MULHSU, encoding '0b00000010000000000010000000110011' */
        {32, 0b00000010000000000010000000110011, 0b11111110000000000111000001111111, &this_class::__mulhsu},
        /* instruction MULHU, encoding '0b00000010000000000011000000110011' */
        {32, 0b00000010000000000011000000110011, 0b11111110000000000111000001111111, &this_class::__mulhu},
        /* instruction DIV, encoding '0b00000010000000000100000000110011' */
        {32, 0b00000010000000000100000000110011, 0b11111110000000000111000001111111, &this_class::__div},
        /* instruction DIVU, encoding '0b00000010000000000101000000110011' */
        {32, 0b00000010000000000101000000110011, 0b11111110000000000111000001111111, &this_class::__divu},
        /* instruction REM, encoding '0b00000010000000000110000000110011' */
        {32, 0b00000010000000000110000000110011, 0b11111110000000000111000001111111, &this_class::__rem},
        /* instruction REMU, encoding '0b00000010000000000111000000110011' */
        {32, 0b00000010000000000111000000110011, 0b11111110000000000111000001111111, &this_class::__remu},
        /* instruction LRW, encoding '0b00010000000000000010000000101111' */
        {32, 0b00010000000000000010000000101111, 0b11111001111100000111000001111111, &this_class::__lrw},
        /* instruction SCW, encoding '0b00011000000000000010000000101111' */
        {32, 0b00011000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__scw},
        /* instruction AMOSWAPW, encoding '0b00001000000000000010000000101111' */
        {32, 0b00001000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoswapw},
        /* instruction AMOADDW, encoding '0b00000000000000000010000000101111' */
        {32, 0b00000000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoaddw},
        /* instruction AMOXORW, encoding '0b00100000000000000010000000101111' */
        {32, 0b00100000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoxorw},
        /* instruction AMOANDW, encoding '0b01100000000000000010000000101111' */
        {32, 0b01100000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoandw},
        /* instruction AMOORW, encoding '0b01000000000000000010000000101111' */
        {32, 0b01000000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoorw},
        /* instruction AMOMINW, encoding '0b10000000000000000010000000101111' */
        {32, 0b10000000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amominw},
        /* instruction AMOMAXW, encoding '0b10100000000000000010000000101111' */
        {32, 0b10100000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amomaxw},
        /* instruction AMOMINUW, encoding '0b11000000000000000010000000101111' */
        {32, 0b11000000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amominuw},
        /* instruction AMOMAXUW, encoding '0b11100000000000000010000000101111' */
        {32, 0b11100000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amomaxuw},
        /* instruction C__ADDI4SPN, encoding '0b0000000000000000' */
        {16, 0b0000000000000000, 0b1110000000000011, &this_class::__c__addi4spn},
        /* instruction C__LW, encoding '0b0100000000000000' */
        {16, 0b0100000000000000, 0b1110000000000011, &this_class::__c__lw},
        /* instruction C__SW, encoding '0b1100000000000000' */
        {16, 0b1100000000000000, 0b1110000000000011, &this_class::__c__sw},
        /* instruction C__ADDI, encoding '0b0000000000000001' */
        {16, 0b0000000000000001, 0b1110000000000011, &this_class::__c__addi},
        /* instruction C__NOP, encoding '0b0000000000000001' */
        {16, 0b0000000000000001, 0b1110111110000011, &this_class::__c__nop},
        /* instruction C__JAL, encoding '0b0010000000000001' */
        {16, 0b0010000000000001, 0b1110000000000011, &this_class::__c__jal},
        /* instruction C__LI, encoding '0b0100000000000001' */
        {16, 0b0100000000000001, 0b1110000000000011, &this_class::__c__li},
        /* instruction C__LUI, encoding '0b0110000000000001' */
        {16, 0b0110000000000001, 0b1110000000000011, &this_class::__c__lui},
        /* instruction C__ADDI16SP, encoding '0b0110000100000001' */
        {16, 0b0110000100000001, 0b1110111110000011, &this_class::__c__addi16sp},
        /* instruction __reserved_clui, encoding '0b0110000000000001' */
        {16, 0b0110000000000001, 0b1111000001111111, &this_class::____reserved_clui},
        /* instruction C__SRLI, encoding '0b1000000000000001' */
        {16, 0b1000000000000001, 0b1111110000000011, &this_class::__c__srli},
        /* instruction C__SRAI, encoding '0b1000010000000001' */
        {16, 0b1000010000000001, 0b1111110000000011, &this_class::__c__srai},
        /* instruction C__ANDI, encoding '0b1000100000000001' */
        {16, 0b1000100000000001, 0b1110110000000011, &this_class::__c__andi},
        /* instruction C__SUB, encoding '0b1000110000000001' */
        {16, 0b1000110000000001, 0b1111110001100011, &this_class::__c__sub},
        /* instruction C__XOR, encoding '0b1000110000100001' */
        {16, 0b1000110000100001, 0b1111110001100011, &this_class::__c__xor},
        /* instruction C__OR, encoding '0b1000110001000001' */
        {16, 0b1000110001000001, 0b1111110001100011, &this_class::__c__or},
        /* instruction C__AND, encoding '0b1000110001100001' */
        {16, 0b1000110001100001, 0b1111110001100011, &this_class::__c__and},
        /* instruction C__J, encoding '0b1010000000000001' */
        {16, 0b1010000000000001, 0b1110000000000011, &this_class::__c__j},
        /* instruction C__BEQZ, encoding '0b1100000000000001' */
        {16, 0b1100000000000001, 0b1110000000000011, &this_class::__c__beqz},
        /* instruction C__BNEZ, encoding '0b1110000000000001' */
        {16, 0b1110000000000001, 0b1110000000000011, &this_class::__c__bnez},
        /* instruction C__SLLI, encoding '0b0000000000000010' */
        {16, 0b0000000000000010, 0b1111000000000011, &this_class::__c__slli},
        /* instruction C__LWSP, encoding '0b0100000000000010' */
        {16, 0b0100000000000010, 0b1110000000000011, &this_class::__c__lwsp},
        /* instruction C__MV, encoding '0b1000000000000010' */
        {16, 0b1000000000000010, 0b1111000000000011, &this_class::__c__mv},
        /* instruction C__JR, encoding '0b1000000000000010' */
        {16, 0b1000000000000010, 0b1111000001111111, &this_class::__c__jr},
        /* instruction __reserved_cmv, encoding '0b1000000000000010' */
        {16, 0b1000000000000010, 0b1111111111111111, &this_class::____reserved_cmv},
        /* instruction C__ADD, encoding '0b1001000000000010' */
        {16, 0b1001000000000010, 0b1111000000000011, &this_class::__c__add},
        /* instruction C__JALR, encoding '0b1001000000000010' */
        {16, 0b1001000000000010, 0b1111000001111111, &this_class::__c__jalr},
        /* instruction C__EBREAK, encoding '0b1001000000000010' */
        {16, 0b1001000000000010, 0b1111111111111111, &this_class::__c__ebreak},
        /* instruction C__SWSP, encoding '0b1100000000000010' */
        {16, 0b1100000000000010, 0b1110000000000011, &this_class::__c__swsp},
        /* instruction DII, encoding '0b0000000000000000' */
        {16, 0b0000000000000000, 0b1111111111111111, &this_class::__dii},
        /* instruction FLW, encoding '0b00000000000000000010000000000111' */
        {32, 0b00000000000000000010000000000111, 0b00000000000000000111000001111111, &this_class::__flw},
        /* instruction FSW, encoding '0b00000000000000000010000000100111' */
        {32, 0b00000000000000000010000000100111, 0b00000000000000000111000001111111, &this_class::__fsw},
        /* instruction FMADD__S, encoding '0b00000000000000000000000001000011' */
        {32, 0b00000000000000000000000001000011, 0b00000110000000000000000001111111, &this_class::__fmadd__s},
        /* instruction FMSUB__S, encoding '0b00000000000000000000000001000111' */
        {32, 0b00000000000000000000000001000111, 0b00000110000000000000000001111111, &this_class::__fmsub__s},
        /* instruction FNMADD__S, encoding '0b00000000000000000000000001001111' */
        {32, 0b00000000000000000000000001001111, 0b00000110000000000000000001111111, &this_class::__fnmadd__s},
        /* instruction FNMSUB__S, encoding '0b00000000000000000000000001001011' */
        {32, 0b00000000000000000000000001001011, 0b00000110000000000000000001111111, &this_class::__fnmsub__s},
        /* instruction FADD__S, encoding '0b00000000000000000000000001010011' */
        {32, 0b00000000000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fadd__s},
        /* instruction FSUB__S, encoding '0b00001000000000000000000001010011' */
        {32, 0b00001000000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fsub__s},
        /* instruction FMUL__S, encoding '0b00010000000000000000000001010011' */
        {32, 0b00010000000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fmul__s},
        /* instruction FDIV__S, encoding '0b00011000000000000000000001010011' */
        {32, 0b00011000000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fdiv__s},
        /* instruction FSQRT__S, encoding '0b01011000000000000000000001010011' */
        {32, 0b01011000000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fsqrt__s},
        /* instruction FSGNJ__S, encoding '0b00100000000000000000000001010011' */
        {32, 0b00100000000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnj__s},
        /* instruction FSGNJN__S, encoding '0b00100000000000000001000001010011' */
        {32, 0b00100000000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnjn__s},
        /* instruction FSGNJX__S, encoding '0b00100000000000000010000001010011' */
        {32, 0b00100000000000000010000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnjx__s},
        /* instruction FMIN__S, encoding '0b00101000000000000000000001010011' */
        {32, 0b00101000000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fmin__s},
        /* instruction FMAX__S, encoding '0b00101000000000000001000001010011' */
        {32, 0b00101000000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__fmax__s},
        /* instruction FCVT__W__S, encoding '0b11000000000000000000000001010011' */
        {32, 0b11000000000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__w__s},
        /* instruction FCVT__WU__S, encoding '0b11000000000100000000000001010011' */
        {32, 0b11000000000100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__wu__s},
        /* instruction FEQ__S, encoding '0b10100000000000000010000001010011' */
        {32, 0b10100000000000000010000001010011, 0b11111110000000000111000001111111, &this_class::__feq__s},
        /* instruction FLT__S, encoding '0b10100000000000000001000001010011' */
        {32, 0b10100000000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__flt__s},
        /* instruction FLE__S, encoding '0b10100000000000000000000001010011' */
        {32, 0b10100000000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fle__s},
        /* instruction FCLASS__S, encoding '0b11100000000000000001000001010011' */
        {32, 0b11100000000000000001000001010011, 0b11111111111100000111000001111111, &this_class::__fclass__s},
        /* instruction FCVT__S__W, encoding '0b11010000000000000000000001010011' */
        {32, 0b11010000000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__s__w},
        /* instruction FCVT__S__WU, encoding '0b11010000000100000000000001010011' */
        {32, 0b11010000000100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__s__wu},
        /* instruction FMV__X__W, encoding '0b11100000000000000000000001010011' */
        {32, 0b11100000000000000000000001010011, 0b11111111111100000111000001111111, &this_class::__fmv__x__w},
        /* instruction FMV__W__X, encoding '0b11110000000000000000000001010011' */
        {32, 0b11110000000000000000000001010011, 0b11111111111100000111000001111111, &this_class::__fmv__w__x},
        /* instruction C__FLW, encoding '0b0110000000000000' */
        {16, 0b0110000000000000, 0b1110000000000011, &this_class::__c__flw},
        /* instruction C__FSW, encoding '0b1110000000000000' */
        {16, 0b1110000000000000, 0b1110000000000011, &this_class::__c__fsw},
        /* instruction C__FLWSP, encoding '0b0110000000000010' */
        {16, 0b0110000000000010, 0b1110000000000011, &this_class::__c__flwsp},
        /* instruction C__FSWSP, encoding '0b1110000000000010' */
        {16, 0b1110000000000010, 0b1110000000000011, &this_class::__c__fswsp},
        /* instruction FLD, encoding '0b00000000000000000011000000000111' */
        {32, 0b00000000000000000011000000000111, 0b00000000000000000111000001111111, &this_class::__fld},
        /* instruction FSD, encoding '0b00000000000000000011000000100111' */
        {32, 0b00000000000000000011000000100111, 0b00000000000000000111000001111111, &this_class::__fsd},
        /* instruction FMADD_D, encoding '0b00000010000000000000000001000011' */
        {32, 0b00000010000000000000000001000011, 0b00000110000000000000000001111111, &this_class::__fmadd_d},
        /* instruction FMSUB_D, encoding '0b00000010000000000000000001000111' */
        {32, 0b00000010000000000000000001000111, 0b00000110000000000000000001111111, &this_class::__fmsub_d},
        /* instruction FNMADD_D, encoding '0b00000010000000000000000001001111' */
        {32, 0b00000010000000000000000001001111, 0b00000110000000000000000001111111, &this_class::__fnmadd_d},
        /* instruction FNMSUB_D, encoding '0b00000010000000000000000001001011' */
        {32, 0b00000010000000000000000001001011, 0b00000110000000000000000001111111, &this_class::__fnmsub_d},
        /* instruction FADD_D, encoding '0b00000010000000000000000001010011' */
        {32, 0b00000010000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fadd_d},
        /* instruction FSUB_D, encoding '0b00001010000000000000000001010011' */
        {32, 0b00001010000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fsub_d},
        /* instruction FMUL_D, encoding '0b00010010000000000000000001010011' */
        {32, 0b00010010000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fmul_d},
        /* instruction FDIV_D, encoding '0b00011010000000000000000001010011' */
        {32, 0b00011010000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fdiv_d},
        /* instruction FSQRT_D, encoding '0b01011010000000000000000001010011' */
        {32, 0b01011010000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fsqrt_d},
        /* instruction FSGNJ_D, encoding '0b00100010000000000000000001010011' */
        {32, 0b00100010000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnj_d},
        /* instruction FSGNJN_D, encoding '0b00100010000000000001000001010011' */
        {32, 0b00100010000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnjn_d},
        /* instruction FSGNJX_D, encoding '0b00100010000000000010000001010011' */
        {32, 0b00100010000000000010000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnjx_d},
        /* instruction FMIN_D, encoding '0b00101010000000000000000001010011' */
        {32, 0b00101010000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fmin_d},
        /* instruction FMAX_D, encoding '0b00101010000000000001000001010011' */
        {32, 0b00101010000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__fmax_d},
        /* instruction FCVT_S_D, encoding '0b01000000000100000000000001010011' */
        {32, 0b01000000000100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_s_d},
        /* instruction FCVT_D_S, encoding '0b01000010000000000000000001010011' */
        {32, 0b01000010000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_d_s},
        /* instruction FEQ_D, encoding '0b10100010000000000010000001010011' */
        {32, 0b10100010000000000010000001010011, 0b11111110000000000111000001111111, &this_class::__feq_d},
        /* instruction FLT_D, encoding '0b10100010000000000001000001010011' */
        {32, 0b10100010000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__flt_d},
        /* instruction FLE_D, encoding '0b10100010000000000000000001010011' */
        {32, 0b10100010000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fle_d},
        /* instruction FCLASS_D, encoding '0b11100010000000000001000001010011' */
        {32, 0b11100010000000000001000001010011, 0b11111111111100000111000001111111, &this_class::__fclass_d},
        /* instruction FCVT_W_D, encoding '0b11000010000000000000000001010011' */
        {32, 0b11000010000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_w_d},
        /* instruction FCVT_WU_D, encoding '0b11000010000100000000000001010011' */
        {32, 0b11000010000100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_wu_d},
        /* instruction FCVT_D_W, encoding '0b11010010000000000000000001010011' */
        {32, 0b11010010000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_d_w},
        /* instruction FCVT_D_WU, encoding '0b11010010000100000000000001010011' */
        {32, 0b11010010000100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_d_wu},
        /* instruction C__FLD, encoding '0b0010000000000000' */
        {16, 0b0010000000000000, 0b1110000000000011, &this_class::__c__fld},
        /* instruction C__FSD, encoding '0b1010000000000000' */
        {16, 0b1010000000000000, 0b1110000000000011, &this_class::__c__fsd},
        /* instruction C__FLDSP, encoding '0b0010000000000010' */
        {16, 0b0010000000000010, 0b1110000000000011, &this_class::__c__fldsp},
        /* instruction C__FSDSP, encoding '0b1010000000000010' */
        {16, 0b1010000000000010, 0b1110000000000011, &this_class::__c__fsdsp},
    }};

    //needs to be declared after instr_descr
    decoder instr_decoder;

    /* instruction definitions */
    /* instruction 0: LUI */
    continuation_e __lui(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint32_t imm = ((bit_sub<12,20>(instr) << 12));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "lui"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("LUI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 0);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      (uint32_t)((int32_t)imm));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 0);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 1: AUIPC */
    continuation_e __auipc(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint32_t imm = ((bit_sub<12,20>(instr) << 12));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#08x}", fmt::arg("mnemonic", "auipc"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("AUIPC_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 1);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      (uint32_t)(PC+(int32_t)imm));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 1);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 2: JAL */
    continuation_e __jal(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint32_t imm = ((bit_sub<12,8>(instr) << 12) | (bit_sub<20,1>(instr) << 11) | (bit_sub<21,10>(instr) << 1) | (bit_sub<31,1>(instr) << 20));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#0x}", fmt::arg("mnemonic", "jal"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("JAL_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 2);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(jh.cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto new_pc = (uint32_t)(PC+(int32_t)sext<21>(imm));
            if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                gen_set_tval(jh, new_pc);
                gen_raise(jh, 0, 0);
            }
            else{
                if(rd!=0){
                    mov(cc, get_ptr_for(jh, traits::X0+ rd),
                          (uint32_t)(PC+4));
                }
                auto PC_val_v = new_pc;
                mov(cc, jh.next_pc, PC_val_v);
                mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(KNOWN_JUMP));
            }
        }
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 2);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 3: JALR */
    continuation_e __jalr(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm:#0x}", fmt::arg("mnemonic", "jalr"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("JALR_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 3);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(jh.cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto addr_mask = (uint32_t)- 2;
            auto new_pc = gen_ext(cc, 
                (gen_operation(cc, band, (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), addr_mask)
                ), 32, true);
            {
            auto label_merge = cc.newLabel();
            cmp(cc, gen_operation(cc, urem, new_pc, static_cast<uint32_t>(traits::INSTR_ALIGNMENT))
            ,0);
            auto label_else = cc.newLabel();
            cc.je(label_else);
            {
                gen_set_tval(jh, new_pc);
                gen_raise(jh, 0, 0);
            }
            cc.jmp(label_merge);
            cc.bind(label_else);
                {
                    if(rd!=0){
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                              (uint32_t)(PC+4));
                    }
                    auto PC_val_v = new_pc;
                    mov(cc, jh.next_pc, PC_val_v);
                    mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(UNKNOWN_JUMP));
                }
            cc.bind(label_merge);
            }
        }
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 3);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 4: BEQ */
    continuation_e __beq(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "beq"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("BEQ_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 4);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(jh.cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto label_merge = cc.newLabel();
            cmp(cc, gen_operation(cc, eq, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
            ,0);
            cc.je(label_merge);
            {
                auto new_pc = (uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_set_tval(jh, new_pc);
                    gen_raise(jh, 0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    mov(cc, jh.next_pc, PC_val_v);
                    mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(KNOWN_JUMP));
                }
            }
            cc.bind(label_merge);
            }
        }
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 4);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 5: BNE */
    continuation_e __bne(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bne"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("BNE_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 5);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(jh.cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto label_merge = cc.newLabel();
            cmp(cc, gen_operation(cc, ne, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
            ,0);
            cc.je(label_merge);
            {
                auto new_pc = (uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_set_tval(jh, new_pc);
                    gen_raise(jh, 0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    mov(cc, jh.next_pc, PC_val_v);
                    mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(KNOWN_JUMP));
                }
            }
            cc.bind(label_merge);
            }
        }
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 5);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 6: BLT */
    continuation_e __blt(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "blt"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("BLT_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 6);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(jh.cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto label_merge = cc.newLabel();
            cmp(cc, gen_operation(cc, lt, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs1), 32, false), gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false))
            ,0);
            cc.je(label_merge);
            {
                auto new_pc = (uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_set_tval(jh, new_pc);
                    gen_raise(jh, 0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    mov(cc, jh.next_pc, PC_val_v);
                    mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(KNOWN_JUMP));
                }
            }
            cc.bind(label_merge);
            }
        }
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 6);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 7: BGE */
    continuation_e __bge(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bge"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("BGE_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 7);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(jh.cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto label_merge = cc.newLabel();
            cmp(cc, gen_operation(cc, gte, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs1), 32, false), gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false))
            ,0);
            cc.je(label_merge);
            {
                auto new_pc = (uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_set_tval(jh, new_pc);
                    gen_raise(jh, 0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    mov(cc, jh.next_pc, PC_val_v);
                    mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(KNOWN_JUMP));
                }
            }
            cc.bind(label_merge);
            }
        }
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 7);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 8: BLTU */
    continuation_e __bltu(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bltu"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("BLTU_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 8);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(jh.cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto label_merge = cc.newLabel();
            cmp(cc, gen_operation(cc, ltu, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
            ,0);
            cc.je(label_merge);
            {
                auto new_pc = (uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_set_tval(jh, new_pc);
                    gen_raise(jh, 0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    mov(cc, jh.next_pc, PC_val_v);
                    mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(KNOWN_JUMP));
                }
            }
            cc.bind(label_merge);
            }
        }
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 8);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 9: BGEU */
    continuation_e __bgeu(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bgeu"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("BGEU_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 9);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(jh.cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto label_merge = cc.newLabel();
            cmp(cc, gen_operation(cc, gteu, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
            ,0);
            cc.je(label_merge);
            {
                auto new_pc = (uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_set_tval(jh, new_pc);
                    gen_raise(jh, 0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    mov(cc, jh.next_pc, PC_val_v);
                    mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(KNOWN_JUMP));
                }
            }
            cc.bind(label_merge);
            }
        }
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 9);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 10: LB */
    continuation_e __lb(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lb"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("LB_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 10);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto load_address = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            auto res = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, load_address, 1), 8, false);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 10);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 11: LH */
    continuation_e __lh(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lh"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("LH_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 11);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto load_address = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            auto res = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, load_address, 2), 16, false);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 11);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 12: LW */
    continuation_e __lw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lw"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("LW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 12);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto load_address = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            auto res = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, load_address, 4), 32, false);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 12);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 13: LBU */
    continuation_e __lbu(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lbu"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("LBU_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 13);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto load_address = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            auto res = gen_read_mem(jh, traits::MEM, load_address, 1);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 32, false));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 13);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 14: LHU */
    continuation_e __lhu(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lhu"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("LHU_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 14);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto load_address = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            auto res = gen_read_mem(jh, traits::MEM, load_address, 2);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 32, false));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 14);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 15: SB */
    continuation_e __sb(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sb"),
                fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("SB_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 15);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto store_address = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            gen_write_mem(jh, traits::MEM, store_address, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 8, false), 1);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 15);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 16: SH */
    continuation_e __sh(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sh"),
                fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("SH_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 16);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto store_address = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            gen_write_mem(jh, traits::MEM, store_address, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 16, false), 2);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 16);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 17: SW */
    continuation_e __sw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sw"),
                fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("SW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 17);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto store_address = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            gen_write_mem(jh, traits::MEM, store_address, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false), 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 17);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 18: ADDI */
    continuation_e __addi(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "addi"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("ADDI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 18);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                          ), 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 18);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 19: SLTI */
    continuation_e __slti(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "slti"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("SLTI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 19);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                auto label_then253 = cc.newLabel();
                auto label_merge253 = cc.newLabel();
                auto tmp_reg253 = get_reg(cc, 8, false);
                cmp(cc, gen_ext(cc, 
                    load_reg_from_mem(jh, traits::X0 + rs1), 32, true), (int16_t)sext<12>(imm));
                cc.jl(label_then253);
                mov(cc, tmp_reg253,0);
                cc.jmp(label_merge253);
                cc.bind(label_then253);
                mov(cc, tmp_reg253, 1);
                cc.bind(label_merge253);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, tmp_reg253
                      , 32, false)
                );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 19);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 20: SLTIU */
    continuation_e __sltiu(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "sltiu"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("SLTIU_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 20);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                auto label_then254 = cc.newLabel();
                auto label_merge254 = cc.newLabel();
                auto tmp_reg254 = get_reg(cc, 8, false);
                cmp(cc, load_reg_from_mem(jh, traits::X0 + rs1), (uint32_t)((int16_t)sext<12>(imm)));
                cc.jb(label_then254);
                mov(cc, tmp_reg254,0);
                cc.jmp(label_merge254);
                cc.bind(label_then254);
                mov(cc, tmp_reg254, 1);
                cc.bind(label_merge254);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, tmp_reg254
                      , 32, false)
                );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 20);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 21: XORI */
    continuation_e __xori(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "xori"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("XORI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 21);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, bxor, load_reg_from_mem(jh, traits::X0 + rs1), (uint32_t)((int16_t)sext<12>(imm)))
                      );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 21);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 22: ORI */
    continuation_e __ori(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "ori"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("ORI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 22);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, bor, load_reg_from_mem(jh, traits::X0 + rs1), (uint32_t)((int16_t)sext<12>(imm)))
                      );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 22);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 23: ANDI */
    continuation_e __andi(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "andi"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("ANDI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 23);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, band, load_reg_from_mem(jh, traits::X0 + rs1), (uint32_t)((int16_t)sext<12>(imm)))
                      );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 23);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 24: SLLI */
    continuation_e __slli(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "slli"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("SLLI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 24);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, shl, load_reg_from_mem(jh, traits::X0 + rs1), shamt)
                      );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 24);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 25: SRLI */
    continuation_e __srli(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srli"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("SRLI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 25);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, shr, load_reg_from_mem(jh, traits::X0 + rs1), shamt)
                      );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 25);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 26: SRAI */
    continuation_e __srai(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srai"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("SRAI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 26);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_operation(cc, sar, gen_ext(cc, 
                              load_reg_from_mem(jh, traits::X0 + rs1), 32, true), shamt)
                          ), 32, false));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 26);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 27: ADD */
    continuation_e __add(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "add"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("ADD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 27);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
                          ), 32, false));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 27);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 28: SUB */
    continuation_e __sub(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sub"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("SUB_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 28);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_operation(cc, sub, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
                          ), 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 28);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 29: SLL */
    continuation_e __sll(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sll"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("SLL_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 29);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, shl, load_reg_from_mem(jh, traits::X0 + rs1), (gen_operation(cc, band, load_reg_from_mem(jh, traits::X0 + rs2), (static_cast<uint32_t>(traits::XLEN)-1))
                      ))
                      );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 29);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 30: SLT */
    continuation_e __slt(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "slt"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("SLT_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 30);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                auto label_then255 = cc.newLabel();
                auto label_merge255 = cc.newLabel();
                auto tmp_reg255 = get_reg(cc, 8, false);
                cmp(cc, gen_ext(cc, 
                    load_reg_from_mem(jh, traits::X0 + rs1), 32, true), gen_ext(cc, 
                    load_reg_from_mem(jh, traits::X0 + rs2), 32, true));
                cc.jl(label_then255);
                mov(cc, tmp_reg255,0);
                cc.jmp(label_merge255);
                cc.bind(label_then255);
                mov(cc, tmp_reg255, 1);
                cc.bind(label_merge255);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, tmp_reg255
                      , 32, false)
                );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 30);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 31: SLTU */
    continuation_e __sltu(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sltu"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("SLTU_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 31);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                auto label_then256 = cc.newLabel();
                auto label_merge256 = cc.newLabel();
                auto tmp_reg256 = get_reg(cc, 8, false);
                cmp(cc, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2));
                cc.jb(label_then256);
                mov(cc, tmp_reg256,0);
                cc.jmp(label_merge256);
                cc.bind(label_then256);
                mov(cc, tmp_reg256, 1);
                cc.bind(label_merge256);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, tmp_reg256
                      , 32, false)
                );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 31);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 32: XOR */
    continuation_e __xor(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "xor"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("XOR_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 32);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, bxor, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
                      );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 32);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 33: SRL */
    continuation_e __srl(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "srl"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("SRL_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 33);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, shr, load_reg_from_mem(jh, traits::X0 + rs1), (gen_operation(cc, band, load_reg_from_mem(jh, traits::X0 + rs2), (static_cast<uint32_t>(traits::XLEN)-1))
                      ))
                      );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 33);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 34: SRA */
    continuation_e __sra(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sra"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("SRA_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 34);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_operation(cc, sar, gen_ext(cc, 
                              load_reg_from_mem(jh, traits::X0 + rs1), 32, true), (gen_operation(cc, band, load_reg_from_mem(jh, traits::X0 + rs2), (static_cast<uint32_t>(traits::XLEN)-1))
                          ))
                          ), 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 34);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 35: OR */
    continuation_e __or(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "or"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("OR_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 35);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, bor, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
                      );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 35);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 36: AND */
    continuation_e __and(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "and"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("AND_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 36);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, band, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
                      );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 36);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 37: FENCE */
    continuation_e __fence(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t succ = ((bit_sub<20,4>(instr)));
        uint8_t pred = ((bit_sub<24,4>(instr)));
        uint8_t fm = ((bit_sub<28,4>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {pred}, {succ} ({fm} , {rs1}, {rd})", fmt::arg("mnemonic", "fence"),
                fmt::arg("pred", pred), fmt::arg("succ", succ), fmt::arg("fm", fm), fmt::arg("rs1", name(rs1)), fmt::arg("rd", name(rd)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FENCE_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 37);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        gen_write_mem(jh, traits::FENCE, static_cast<uint32_t>(traits::fence), (uint8_t)pred<<4|succ, 4);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 37);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 38: ECALL */
    continuation_e __ecall(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate disass */
            
            //No disass specified, using instruction name
            std::string mnemonic = "ecall";
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("ECALL_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 38);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(jh.cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        gen_raise(jh, 0, 11);
        auto returnValue = TRAP;
        
        gen_sync(jh, POST_SYNC, 38);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 39: EBREAK */
    continuation_e __ebreak(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate disass */
            
            //No disass specified, using instruction name
            std::string mnemonic = "ebreak";
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("EBREAK_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 39);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(jh.cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        gen_raise(jh, 0, 3);
        auto returnValue = TRAP;
        
        gen_sync(jh, POST_SYNC, 39);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 40: MRET */
    continuation_e __mret(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate disass */
            
            //No disass specified, using instruction name
            std::string mnemonic = "mret";
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("MRET_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 40);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(jh.cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        gen_leave(jh, 3);
        auto returnValue = TRAP;
        
        gen_sync(jh, POST_SYNC, 40);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 41: WFI */
    continuation_e __wfi(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate disass */
            
            //No disass specified, using instruction name
            std::string mnemonic = "wfi";
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("WFI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 41);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_wait_257;
        jh.cc.comment("//call_wait");
        jh.cc.invoke(&call_wait_257, &wait, FuncSignature::build<void, int32_t>());
        setArg(call_wait_257, 0, 1);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 41);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 42: CSRRW */
    continuation_e __csrrw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {rs1}", fmt::arg("mnemonic", "csrrw"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("CSRRW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 42);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrs1 = load_reg_from_mem(jh, traits::X0 + rs1);
            if(rd!=0){
                auto xrd = gen_read_mem(jh, traits::CSR, csr, 4);
                gen_write_mem(jh, traits::CSR, csr, xrs1, 4);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      xrd);
            }
            else{
                gen_write_mem(jh, traits::CSR, csr, xrs1, 4);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 42);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 43: CSRRS */
    continuation_e __csrrs(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {rs1}", fmt::arg("mnemonic", "csrrs"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("CSRRS_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 43);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 4);
            auto xrs1 = load_reg_from_mem(jh, traits::X0 + rs1);
            if(rs1!=0){
                gen_write_mem(jh, traits::CSR, csr, gen_operation(cc, bor, xrd, xrs1)
                , 4);
            }
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      xrd);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 43);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 44: CSRRC */
    continuation_e __csrrc(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {rs1}", fmt::arg("mnemonic", "csrrc"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("CSRRC_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 44);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 4);
            auto xrs1 = load_reg_from_mem(jh, traits::X0 + rs1);
            if(rs1!=0){
                gen_write_mem(jh, traits::CSR, csr, gen_operation(cc, band, xrd, gen_operation(cc, bnot, xrs1))
                , 4);
            }
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      xrd);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 44);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 45: CSRRWI */
    continuation_e __csrrwi(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t zimm = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {zimm:#0x}", fmt::arg("mnemonic", "csrrwi"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("zimm", zimm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("CSRRWI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 45);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 4);
            gen_write_mem(jh, traits::CSR, csr, (uint32_t)zimm, 4);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      xrd);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 45);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 46: CSRRSI */
    continuation_e __csrrsi(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t zimm = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {zimm:#0x}", fmt::arg("mnemonic", "csrrsi"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("zimm", zimm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("CSRRSI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 46);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 4);
            if(zimm!=0){
                gen_write_mem(jh, traits::CSR, csr, gen_operation(cc, bor, xrd, (uint32_t)zimm)
                , 4);
            }
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      xrd);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 46);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 47: CSRRCI */
    continuation_e __csrrci(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t zimm = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {zimm:#0x}", fmt::arg("mnemonic", "csrrci"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("zimm", zimm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("CSRRCI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 47);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 4);
            if(zimm!=0){
                gen_write_mem(jh, traits::CSR, csr, gen_operation(cc, band, xrd, ~ ((uint32_t)zimm))
                , 4);
            }
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      xrd);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 47);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 48: FENCE_I */
    continuation_e __fence_i(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rd}, {imm}", fmt::arg("mnemonic", "fence_i"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FENCE_I_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 48);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        gen_write_mem(jh, traits::FENCE, static_cast<uint32_t>(traits::fencei), imm, 4);
        auto returnValue = FLUSH;
        
        gen_sync(jh, POST_SYNC, 48);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 49: MUL */
    continuation_e __mul(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mul"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("MUL_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 49);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto res = gen_operation(cc, smul, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs1), 32, true), gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, true))
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 49);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 50: MULH */
    continuation_e __mulh(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mulh"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("MULH_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 50);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto res = gen_operation(cc, smul, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs1), 32, true), gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, true))
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_operation(cc, sar, res, static_cast<uint32_t>(traits::XLEN))
                          ), 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 50);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 51: MULHSU */
    continuation_e __mulhsu(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mulhsu"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("MULHSU_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 51);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto res = gen_operation(cc, sumul, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs1), 32, true), load_reg_from_mem(jh, traits::X0 + rs2))
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_operation(cc, sar, res, static_cast<uint32_t>(traits::XLEN))
                          ), 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 51);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 52: MULHU */
    continuation_e __mulhu(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mulhu"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("MULHU_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 52);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto res = gen_operation(cc, umul, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_operation(cc, shr, res, static_cast<uint32_t>(traits::XLEN))
                          ), 32, false));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 52);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 53: DIV */
    continuation_e __div(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "div"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("DIV_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 53);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto dividend = gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs1), 32, true);
            auto divisor = gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, true);
            if(rd!=0){
                {
                auto label_merge = cc.newLabel();
                cmp(cc, gen_operation(cc, ne, divisor, 0)
                ,0);
                auto label_else = cc.newLabel();
                cc.je(label_else);
                {
                    auto MMIN = ((uint32_t)1)<<(static_cast<uint32_t>(traits::XLEN)-1);
                    {
                    auto label_merge = cc.newLabel();
                    cmp(cc, gen_operation(cc, land, gen_operation(cc, eq, load_reg_from_mem(jh, traits::X0 + rs1), MMIN)
                    , gen_operation(cc, eq, divisor, - 1)
                    )
                    ,0);
                    auto label_else = cc.newLabel();
                    cc.je(label_else);
                    {
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                              MMIN);
                    }
                    cc.jmp(label_merge);
                    cc.bind(label_else);
                        {
                            mov(cc, get_ptr_for(jh, traits::X0+ rd),
                                  gen_ext(cc, 
                                      (gen_operation(cc, sdiv, dividend, divisor)
                                      ), 32, true));
                        }
                    cc.bind(label_merge);
                    }
                }
                cc.jmp(label_merge);
                cc.bind(label_else);
                    {
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                              (uint32_t)- 1);
                    }
                cc.bind(label_merge);
                }
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 53);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 54: DIVU */
    continuation_e __divu(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "divu"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("DIVU_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 54);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto label_merge = cc.newLabel();
            cmp(cc, gen_operation(cc, ne, load_reg_from_mem(jh, traits::X0 + rs2), 0)
            ,0);
            auto label_else = cc.newLabel();
            cc.je(label_else);
            {
                if(rd!=0){
                    mov(cc, get_ptr_for(jh, traits::X0+ rd),
                          gen_operation(cc, udiv, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
                          );
                }
            }
            cc.jmp(label_merge);
            cc.bind(label_else);
                {
                    if(rd!=0){
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                              (uint32_t)- 1);
                    }
                }
            cc.bind(label_merge);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 54);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 55: REM */
    continuation_e __rem(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "rem"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("REM_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 55);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto label_merge = cc.newLabel();
            cmp(cc, gen_operation(cc, ne, load_reg_from_mem(jh, traits::X0 + rs2), 0)
            ,0);
            auto label_else = cc.newLabel();
            cc.je(label_else);
            {
                auto MMIN = (uint32_t)1<<(static_cast<uint32_t>(traits::XLEN)-1);
                {
                auto label_merge = cc.newLabel();
                cmp(cc, gen_operation(cc, land, gen_operation(cc, eq, load_reg_from_mem(jh, traits::X0 + rs1), MMIN)
                , gen_operation(cc, eq, gen_ext(cc, 
                    load_reg_from_mem(jh, traits::X0 + rs2), 32, false), - 1)
                )
                ,0);
                auto label_else = cc.newLabel();
                cc.je(label_else);
                {
                    if(rd!=0){
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                              gen_ext(cc, 0, 32, false)
                        );
                    }
                }
                cc.jmp(label_merge);
                cc.bind(label_else);
                    {
                        if(rd!=0){
                            mov(cc, get_ptr_for(jh, traits::X0+ rd),
                                  gen_ext(cc, 
                                      (gen_operation(cc, srem, gen_ext(cc, 
                                          load_reg_from_mem(jh, traits::X0 + rs1), 32, true), gen_ext(cc, 
                                          load_reg_from_mem(jh, traits::X0 + rs2), 32, true))
                                      ), 32, false));
                        }
                    }
                cc.bind(label_merge);
                }
            }
            cc.jmp(label_merge);
            cc.bind(label_else);
                {
                    if(rd!=0){
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                              load_reg_from_mem(jh, traits::X0 + rs1));
                    }
                }
            cc.bind(label_merge);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 55);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 56: REMU */
    continuation_e __remu(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "remu"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("REMU_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 56);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto label_merge = cc.newLabel();
            cmp(cc, gen_operation(cc, ne, load_reg_from_mem(jh, traits::X0 + rs2), 0)
            ,0);
            auto label_else = cc.newLabel();
            cc.je(label_else);
            {
                if(rd!=0){
                    mov(cc, get_ptr_for(jh, traits::X0+ rd),
                          gen_operation(cc, urem, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
                          );
                }
            }
            cc.jmp(label_merge);
            cc.bind(label_else);
                {
                    if(rd!=0){
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                              load_reg_from_mem(jh, traits::X0 + rs1));
                    }
                }
            cc.bind(label_merge);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 56);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 57: LRW */
    continuation_e __lrw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {aq}, {rl}", fmt::arg("mnemonic", "lrw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("aq", name(aq)), fmt::arg("rl", name(rl)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("LRW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 57);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                auto offs = load_reg_from_mem(jh, traits::X0 + rs1);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_ext(cc, 
                              gen_read_mem(jh, traits::MEM, offs, 4), 32, false)), 32, true));
                gen_write_mem(jh, traits::RES, offs, (uint8_t)- 1, 1);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 57);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 58: SCW */
    continuation_e __scw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {aq}, {rl}", fmt::arg("mnemonic", "scw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", name(aq)), fmt::arg("rl", name(rl)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("SCW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 58);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs = load_reg_from_mem(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::RES, offs, 1);
            {
            auto label_merge = cc.newLabel();
            cmp(cc, gen_operation(cc, ne, res1, 0)
            ,0);
            cc.je(label_merge);
            {
                gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
                    load_reg_from_mem(jh, traits::X0 + rs2), 32, false), 4);
            }
            cc.bind(label_merge);
            }
            if(rd!=0){
                auto label_then258 = cc.newLabel();
                auto label_merge258 = cc.newLabel();
                auto tmp_reg258 = get_reg(cc, 8, false);
                cmp(cc, res1, 0);
                cc.jne(label_then258);
                mov(cc, tmp_reg258,1);
                cc.jmp(label_merge258);
                cc.bind(label_then258);
                mov(cc, tmp_reg258, 0);
                cc.bind(label_merge258);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, tmp_reg258
                      , 32, false)
                );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 58);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 59: AMOSWAPW */
    continuation_e __amoswapw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoswapw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("AMOSWAPW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 59);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs = load_reg_from_mem(jh, traits::X0 + rs1);
            auto res = load_reg_from_mem(jh, traits::X0 + rs2);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_ext(cc, 
                              gen_read_mem(jh, traits::MEM, offs, 4), 32, false)), 32, true));
            }
            gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
                res, 32, false), 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 59);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 60: AMOADDW */
    continuation_e __amoaddw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoaddw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("AMOADDW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 60);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs = load_reg_from_mem(jh, traits::X0 + rs1);
            auto res1 = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, offs, 4), 32, false);
            auto res2 = gen_operation(cc, add, res1, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false))
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res1, 32, true));
            }
            gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
                res2, 32, true), 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 60);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 61: AMOXORW */
    continuation_e __amoxorw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoxorw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("AMOXORW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 61);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs = load_reg_from_mem(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::MEM, offs, 4);
            auto res2 = gen_operation(cc, bxor, res1, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false))
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          gen_ext(cc, 
                              gen_ext(cc, 
                                  res1, 32, true), 32, true), 32, false));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 61);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 62: AMOANDW */
    continuation_e __amoandw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoandw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("AMOANDW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 62);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs = load_reg_from_mem(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::MEM, offs, 4);
            auto res2 = gen_operation(cc, band, res1, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false))
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          gen_ext(cc, 
                              gen_ext(cc, 
                                  res1, 32, true), 32, true), 32, false));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 62);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 63: AMOORW */
    continuation_e __amoorw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoorw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("AMOORW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 63);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs = load_reg_from_mem(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::MEM, offs, 4);
            auto res2 = gen_operation(cc, bor, res1, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false))
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          gen_ext(cc, 
                              gen_ext(cc, 
                                  res1, 32, true), 32, true), 32, false));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 63);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 64: AMOMINW */
    continuation_e __amominw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amominw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("AMOMINW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 64);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs = load_reg_from_mem(jh, traits::X0 + rs1);
            auto res1 = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, offs, 4), 32, false);
            auto label_then259 = cc.newLabel();
            auto label_merge259 = cc.newLabel();
            auto tmp_reg259 = get_reg(cc, 32, false);
            cmp(cc, res1, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false));
            cc.jg(label_then259);
            mov(cc, tmp_reg259,gen_ext(cc, 
                res1, 32, false));
            cc.jmp(label_merge259);
            cc.bind(label_then259);
            mov(cc, tmp_reg259, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false));
            cc.bind(label_merge259);
            auto res2 = tmp_reg259
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res1, 32, true));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 64);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 65: AMOMAXW */
    continuation_e __amomaxw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amomaxw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("AMOMAXW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 65);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs = load_reg_from_mem(jh, traits::X0 + rs1);
            auto res1 = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, offs, 4), 32, false);
            auto label_then260 = cc.newLabel();
            auto label_merge260 = cc.newLabel();
            auto tmp_reg260 = get_reg(cc, 32, false);
            cmp(cc, res1, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false));
            cc.jl(label_then260);
            mov(cc, tmp_reg260,gen_ext(cc, 
                res1, 32, false));
            cc.jmp(label_merge260);
            cc.bind(label_then260);
            mov(cc, tmp_reg260, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false));
            cc.bind(label_merge260);
            auto res2 = tmp_reg260
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res1, 32, true));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 65);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 66: AMOMINUW */
    continuation_e __amominuw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amominuw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("AMOMINUW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 66);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs = load_reg_from_mem(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::MEM, offs, 4);
            auto label_then261 = cc.newLabel();
            auto label_merge261 = cc.newLabel();
            auto tmp_reg261 = get_reg(cc, 32, false);
            cmp(cc, res1, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false));
            cc.ja(label_then261);
            mov(cc, tmp_reg261,res1);
            cc.jmp(label_merge261);
            cc.bind(label_then261);
            mov(cc, tmp_reg261, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false));
            cc.bind(label_merge261);
            auto res2 = tmp_reg261
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          gen_ext(cc, 
                              gen_ext(cc, 
                                  res1, 32, true), 32, true), 32, false));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 66);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 67: AMOMAXUW */
    continuation_e __amomaxuw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amomaxuw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("AMOMAXUW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 67);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs = load_reg_from_mem(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::MEM, offs, 4);
            auto label_then262 = cc.newLabel();
            auto label_merge262 = cc.newLabel();
            auto tmp_reg262 = get_reg(cc, 32, false);
            cmp(cc, res1, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false));
            cc.jb(label_then262);
            mov(cc, tmp_reg262,res1);
            cc.jmp(label_merge262);
            cc.bind(label_then262);
            mov(cc, tmp_reg262, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false));
            cc.bind(label_merge262);
            auto res2 = tmp_reg262
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          gen_ext(cc, 
                              gen_ext(cc, 
                                  res1, 32, true), 32, true), 32, false));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 67);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 68: C__ADDI4SPN */
    continuation_e __c__addi4spn(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<2,3>(instr)));
        uint16_t imm = ((bit_sub<5,1>(instr) << 3) | (bit_sub<6,1>(instr) << 2) | (bit_sub<7,4>(instr) << 6) | (bit_sub<11,2>(instr) << 4));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c.addi4spn"),
                fmt::arg("rd", name(8+rd)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__ADDI4SPN_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 68);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(imm){
            mov(cc, get_ptr_for(jh, traits::X0+ rd+8),
                  gen_ext(cc, 
                      (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + 2), imm)
                      ), 32, false));
        }
        else{
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 68);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 69: C__LW */
    continuation_e __c__lw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {uimm:#05x}({rs1})", fmt::arg("mnemonic", "c.lw"),
                fmt::arg("rd", name(8+rd)), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__LW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 69);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1+8), uimm)
            ), 32, false);
        mov(cc, get_ptr_for(jh, traits::X0+ rd+8),
              gen_ext(cc, 
                  gen_ext(cc, 
                      gen_read_mem(jh, traits::MEM, offs, 4), 32, false), 32, true));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 69);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 70: C__SW */
    continuation_e __c__sw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {uimm:#05x}({rs1})", fmt::arg("mnemonic", "c.sw"),
                fmt::arg("rs2", name(8+rs2)), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__SW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 70);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1+8), uimm)
            ), 32, false);
        gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
            load_reg_from_mem(jh, traits::X0 + rs2+8), 32, false), 4);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 70);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 71: C__ADDI */
    continuation_e __c__addi(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t imm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.addi"),
                fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__ADDI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 71);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rs1!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rs1),
                      gen_ext(cc, 
                          (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1), (int8_t)sext<6>(imm))
                          ), 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 71);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 72: C__NOP */
    continuation_e __c__nop(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t nzimm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        if(this->disass_enabled){
            /* generate disass */
            
            //No disass specified, using instruction name
            std::string mnemonic = "c.nop";
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__NOP_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 72);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 72);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 73: C__JAL */
    continuation_e __c__jal(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,3>(instr) << 1) | (bit_sub<6,1>(instr) << 7) | (bit_sub<7,1>(instr) << 6) | (bit_sub<8,1>(instr) << 10) | (bit_sub<9,2>(instr) << 8) | (bit_sub<11,1>(instr) << 4) | (bit_sub<12,1>(instr) << 11));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {imm:#05x}", fmt::arg("mnemonic", "c.jal"),
                fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__JAL_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 73);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(jh.cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        mov(cc, get_ptr_for(jh, traits::X0+ 1),
              (uint32_t)(PC+2));
        auto PC_val_v = (uint32_t)(PC+(int16_t)sext<12>(imm));
        mov(cc, jh.next_pc, PC_val_v);
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(KNOWN_JUMP));
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 73);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 74: C__LI */
    continuation_e __c__li(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t imm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c.li"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__LI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 74);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      (uint32_t)((int8_t)sext<6>(imm)));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 74);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 75: C__LUI */
    continuation_e __c__lui(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint32_t imm = ((bit_sub<2,5>(instr) << 12) | (bit_sub<12,1>(instr) << 17));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c.lui"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__LUI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 75);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(imm==0||rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        if(rd!=0){
            mov(cc, get_ptr_for(jh, traits::X0+ rd),
                  (uint32_t)((int32_t)sext<18>(imm)));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 75);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 76: C__ADDI16SP */
    continuation_e __c__addi16sp(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t nzimm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 7) | (bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 4) | (bit_sub<12,1>(instr) << 9));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {nzimm:#05x}", fmt::arg("mnemonic", "c.addi16sp"),
                fmt::arg("nzimm", nzimm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__ADDI16SP_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 76);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(nzimm){
            mov(cc, get_ptr_for(jh, traits::X0+ 2),
                  gen_ext(cc, 
                      (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + 2), (int16_t)sext<10>(nzimm))
                      ), 32, true));
        }
        else{
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 76);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 77: __reserved_clui */
    continuation_e ____reserved_clui(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            //No disass specified, using instruction name
            std::string mnemonic = ".reserved_clui";
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("__reserved_clui_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 77);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 77);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 78: C__SRLI */
    continuation_e __c__srli(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t shamt = ((bit_sub<2,5>(instr)));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {shamt}", fmt::arg("mnemonic", "c.srli"),
                fmt::arg("rs1", name(8+rs1)), fmt::arg("shamt", shamt));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__SRLI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 78);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::X0+ rs1+8),
              gen_operation(cc, shr, load_reg_from_mem(jh, traits::X0 + rs1+8), shamt)
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 78);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 79: C__SRAI */
    continuation_e __c__srai(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t shamt = ((bit_sub<2,5>(instr)));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {shamt}", fmt::arg("mnemonic", "c.srai"),
                fmt::arg("rs1", name(8+rs1)), fmt::arg("shamt", shamt));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__SRAI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 79);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(shamt){
            mov(cc, get_ptr_for(jh, traits::X0+ rs1+8),
                  gen_ext(cc, 
                      (gen_operation(cc, sar, (gen_ext(cc, 
                          load_reg_from_mem(jh, traits::X0 + rs1+8), 32, false)), shamt)
                      ), 32, true));
        }
        else{
            if(static_cast<uint32_t>(traits::XLEN)==128){
                mov(cc, get_ptr_for(jh, traits::X0+ rs1+8),
                      gen_ext(cc, 
                          (gen_operation(cc, sar, (gen_ext(cc, 
                              load_reg_from_mem(jh, traits::X0 + rs1+8), 32, false)), 64)
                          ), 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 79);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 80: C__ANDI */
    continuation_e __c__andi(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t imm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.andi"),
                fmt::arg("rs1", name(8+rs1)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__ANDI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 80);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::X0+ rs1+8),
              gen_ext(cc, 
                  (gen_operation(cc, band, load_reg_from_mem(jh, traits::X0 + rs1+8), (int8_t)sext<6>(imm))
                  ), 32, true));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 80);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 81: C__SUB */
    continuation_e __c__sub(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.sub"),
                fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__SUB_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 81);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::X0+ rd+8),
              gen_ext(cc, 
                  (gen_operation(cc, sub, load_reg_from_mem(jh, traits::X0 + rd+8), load_reg_from_mem(jh, traits::X0 + rs2+8))
                  ), 32, true));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 81);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 82: C__XOR */
    continuation_e __c__xor(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.xor"),
                fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__XOR_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 82);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::X0+ rd+8),
              gen_operation(cc, bxor, load_reg_from_mem(jh, traits::X0 + rd+8), load_reg_from_mem(jh, traits::X0 + rs2+8))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 82);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 83: C__OR */
    continuation_e __c__or(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.or"),
                fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__OR_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 83);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::X0+ rd+8),
              gen_operation(cc, bor, load_reg_from_mem(jh, traits::X0 + rd+8), load_reg_from_mem(jh, traits::X0 + rs2+8))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 83);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 84: C__AND */
    continuation_e __c__and(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.and"),
                fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__AND_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 84);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::X0+ rd+8),
              gen_operation(cc, band, load_reg_from_mem(jh, traits::X0 + rd+8), load_reg_from_mem(jh, traits::X0 + rs2+8))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 84);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 85: C__J */
    continuation_e __c__j(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,3>(instr) << 1) | (bit_sub<6,1>(instr) << 7) | (bit_sub<7,1>(instr) << 6) | (bit_sub<8,1>(instr) << 10) | (bit_sub<9,2>(instr) << 8) | (bit_sub<11,1>(instr) << 4) | (bit_sub<12,1>(instr) << 11));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {imm:#05x}", fmt::arg("mnemonic", "c.j"),
                fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__J_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 85);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(jh.cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        auto PC_val_v = (uint32_t)(PC+(int16_t)sext<12>(imm));
        mov(cc, jh.next_pc, PC_val_v);
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(KNOWN_JUMP));
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 85);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 86: C__BEQZ */
    continuation_e __c__beqz(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 1) | (bit_sub<5,2>(instr) << 6) | (bit_sub<10,2>(instr) << 3) | (bit_sub<12,1>(instr) << 8));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.beqz"),
                fmt::arg("rs1", name(8+rs1)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__BEQZ_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 86);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(jh.cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        {
        auto label_merge = cc.newLabel();
        cmp(cc, gen_operation(cc, eq, load_reg_from_mem(jh, traits::X0 + rs1+8), 0)
        ,0);
        cc.je(label_merge);
        {
            auto PC_val_v = (uint32_t)(PC+(int16_t)sext<9>(imm));
            mov(cc, jh.next_pc, PC_val_v);
            mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(KNOWN_JUMP));
        }
        cc.bind(label_merge);
        }
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 86);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 87: C__BNEZ */
    continuation_e __c__bnez(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 1) | (bit_sub<5,2>(instr) << 6) | (bit_sub<10,2>(instr) << 3) | (bit_sub<12,1>(instr) << 8));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.bnez"),
                fmt::arg("rs1", name(8+rs1)), fmt::arg("imm", imm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__BNEZ_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 87);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(jh.cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        {
        auto label_merge = cc.newLabel();
        cmp(cc, gen_operation(cc, ne, load_reg_from_mem(jh, traits::X0 + rs1+8), 0)
        ,0);
        cc.je(label_merge);
        {
            auto PC_val_v = (uint32_t)(PC+(int16_t)sext<9>(imm));
            mov(cc, jh.next_pc, PC_val_v);
            mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(KNOWN_JUMP));
        }
        cc.bind(label_merge);
        }
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 87);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 88: C__SLLI */
    continuation_e __c__slli(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t nzuimm = ((bit_sub<2,5>(instr)));
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {nzuimm}", fmt::arg("mnemonic", "c.slli"),
                fmt::arg("rs1", name(rs1)), fmt::arg("nzuimm", nzuimm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__SLLI_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 88);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rs1!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rs1),
                      gen_operation(cc, shl, load_reg_from_mem(jh, traits::X0 + rs1), nzuimm)
                      );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 88);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 89: C__LWSP */
    continuation_e __c__lwsp(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t uimm = ((bit_sub<2,2>(instr) << 6) | (bit_sub<4,3>(instr) << 2) | (bit_sub<12,1>(instr) << 5));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, sp, {uimm:#05x}", fmt::arg("mnemonic", "c.lwsp"),
                fmt::arg("rd", name(rd)), fmt::arg("uimm", uimm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__LWSP_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 89);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rd==0){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + 2), uimm)
                ), 32, false);
            mov(cc, get_ptr_for(jh, traits::X0+ rd),
                  gen_ext(cc, 
                      gen_ext(cc, 
                          gen_read_mem(jh, traits::MEM, offs, 4), 32, false), 32, true));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 89);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 90: C__MV */
    continuation_e __c__mv(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.mv"),
                fmt::arg("rd", name(rd)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__MV_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 90);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      load_reg_from_mem(jh, traits::X0 + rs2));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 90);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 91: C__JR */
    continuation_e __c__jr(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}", fmt::arg("mnemonic", "c.jr"),
                fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__JR_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 91);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(jh.cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rs1&&rs1<static_cast<uint32_t>(traits::RFS)){
            auto addr_mask = (uint32_t)- 2;
            auto PC_val_v = gen_operation(cc, band, load_reg_from_mem(jh, traits::X0 + rs1%static_cast<uint32_t>(traits::RFS)), addr_mask)
            ;
            mov(cc, jh.next_pc, PC_val_v);
            mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(UNKNOWN_JUMP));
        }
        else{
            gen_raise(jh, 0, 2);
        }
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 91);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 92: __reserved_cmv */
    continuation_e ____reserved_cmv(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate disass */
            
            //No disass specified, using instruction name
            std::string mnemonic = ".reserved_cmv";
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("__reserved_cmv_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 92);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        gen_raise(jh, 0, 2);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 92);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 93: C__ADD */
    continuation_e __c__add(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.add"),
                fmt::arg("rd", name(rd)), fmt::arg("rs2", name(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__ADD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 93);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rd), load_reg_from_mem(jh, traits::X0 + rs2))
                          ), 32, false));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 93);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 94: C__JALR */
    continuation_e __c__jalr(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}", fmt::arg("mnemonic", "c.jalr"),
                fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__JALR_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 94);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(jh.cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto addr_mask = (uint32_t)- 2;
            auto new_pc = load_reg_from_mem(jh, traits::X0 + rs1);
            mov(cc, get_ptr_for(jh, traits::X0+ 1),
                  (uint32_t)(PC+2));
            auto PC_val_v = gen_operation(cc, band, new_pc, addr_mask)
            ;
            mov(cc, jh.next_pc, PC_val_v);
            mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(UNKNOWN_JUMP));
        }
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 94);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 95: C__EBREAK */
    continuation_e __c__ebreak(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate disass */
            
            //No disass specified, using instruction name
            std::string mnemonic = "c.ebreak";
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__EBREAK_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 95);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        gen_raise(jh, 0, 3);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 95);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 96: C__SWSP */
    continuation_e __c__swsp(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t uimm = ((bit_sub<7,2>(instr) << 6) | (bit_sub<9,4>(instr) << 2));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {uimm:#05x}(sp)", fmt::arg("mnemonic", "c.swsp"),
                fmt::arg("rs2", name(rs2)), fmt::arg("uimm", uimm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__SWSP_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 96);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + 2), uimm)
                ), 32, false);
            gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false), 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 96);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 97: DII */
    continuation_e __dii(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate disass */
            
            //No disass specified, using instruction name
            std::string mnemonic = "dii";
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("DII_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 97);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 97);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 98: FLW */
    continuation_e __flw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "flw"),
                fmt::arg("rd", fname(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FLW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 98);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            auto res = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, offs, 4), 32, false);
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  gen_operation(cc, bor, gen_ext(cc, 
                      res, 64, false), (uint64_t)((int64_t)- 1<<32))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 98);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 99: FSW */
    continuation_e __fsw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "fsw"),
                fmt::arg("rs2", fname(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FSW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 99);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
                load_reg_from_mem(jh, traits::F0 + rs2), 32, false), 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 99);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 100: FMADD__S */
    continuation_e __fmadd__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rs3 = ((bit_sub<27,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}, {rs3}", fmt::arg("mnemonic", "fmadd.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FMADD__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 100);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_unbox_s_264;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_264_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_264 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_264, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_265;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_265_arg0 = load_reg_from_mem(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_265 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_265, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_266;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_266_arg0 = load_reg_from_mem(jh, traits::F0 + rs3);
        x86::Gp ret_val_unbox_s_266 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_266, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        InvokeNode* call_fmadd_s_263;
        jh.cc.comment("//call_fmadd_s");
        auto fmadd_s_263_arg0 = ret_val_unbox_s_264;
        auto fmadd_s_263_arg1 = ret_val_unbox_s_265;
        auto fmadd_s_263_arg2 = ret_val_unbox_s_266;
        auto fmadd_s_263_arg4 = get_rm(jh, (uint8_t)rm);
        x86::Gp ret_val_fmadd_s_263 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fmadd_s_263, &fmadd_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint8_t>());
        auto res = ret_val_fmadd_s_263;
        setArg(call_unbox_s_264, 0, unbox_s_264_arg0);
        setRet(call_unbox_s_264, 0, ret_val_unbox_s_264);setArg(call_unbox_s_265, 0, unbox_s_265_arg0);
        setRet(call_unbox_s_265, 0, ret_val_unbox_s_265);setArg(call_unbox_s_266, 0, unbox_s_266_arg0);
        setRet(call_unbox_s_266, 0, ret_val_unbox_s_266);setArg(call_fmadd_s_263, 0, fmadd_s_263_arg0);
        setArg(call_fmadd_s_263, 1, fmadd_s_263_arg1);
        setArg(call_fmadd_s_263, 2, fmadd_s_263_arg2);
        setArg(call_fmadd_s_263, 3, 0);
        setArg(call_fmadd_s_263, 4, fmadd_s_263_arg4);
        setRet(call_fmadd_s_263, 0, ret_val_fmadd_s_263);
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_operation(cc, bor, gen_ext(cc, 
                  res, 64, false), (uint64_t)((int64_t)- 1<<32))
              );
        InvokeNode* call_fget_flags_267;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_267 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_267, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_267;
        setRet(call_fget_flags_267, 0, ret_val_fget_flags_267);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 100);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 101: FMSUB__S */
    continuation_e __fmsub__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rs3 = ((bit_sub<27,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}, {rs3}", fmt::arg("mnemonic", "fmsub.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FMSUB__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 101);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_unbox_s_269;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_269_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_269 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_269, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_270;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_270_arg0 = load_reg_from_mem(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_270 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_270, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_271;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_271_arg0 = load_reg_from_mem(jh, traits::F0 + rs3);
        x86::Gp ret_val_unbox_s_271 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_271, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        InvokeNode* call_fmadd_s_268;
        jh.cc.comment("//call_fmadd_s");
        auto fmadd_s_268_arg0 = ret_val_unbox_s_269;
        auto fmadd_s_268_arg1 = ret_val_unbox_s_270;
        auto fmadd_s_268_arg2 = ret_val_unbox_s_271;
        auto fmadd_s_268_arg4 = get_rm(jh, (uint8_t)rm);
        x86::Gp ret_val_fmadd_s_268 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fmadd_s_268, &fmadd_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint8_t>());
        auto res = ret_val_fmadd_s_268;
        setArg(call_unbox_s_269, 0, unbox_s_269_arg0);
        setRet(call_unbox_s_269, 0, ret_val_unbox_s_269);setArg(call_unbox_s_270, 0, unbox_s_270_arg0);
        setRet(call_unbox_s_270, 0, ret_val_unbox_s_270);setArg(call_unbox_s_271, 0, unbox_s_271_arg0);
        setRet(call_unbox_s_271, 0, ret_val_unbox_s_271);setArg(call_fmadd_s_268, 0, fmadd_s_268_arg0);
        setArg(call_fmadd_s_268, 1, fmadd_s_268_arg1);
        setArg(call_fmadd_s_268, 2, fmadd_s_268_arg2);
        setArg(call_fmadd_s_268, 3, 1);
        setArg(call_fmadd_s_268, 4, fmadd_s_268_arg4);
        setRet(call_fmadd_s_268, 0, ret_val_fmadd_s_268);
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_operation(cc, bor, gen_ext(cc, 
                  res, 64, false), (uint64_t)((int64_t)- 1<<32))
              );
        InvokeNode* call_fget_flags_272;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_272 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_272, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_272;
        setRet(call_fget_flags_272, 0, ret_val_fget_flags_272);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 101);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 102: FNMADD__S */
    continuation_e __fnmadd__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rs3 = ((bit_sub<27,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, name(rd), {rs1}, {rs2}, {rs3}", fmt::arg("mnemonic", "fnmadd.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FNMADD__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 102);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_unbox_s_273;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_273_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_273 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_273, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs1 = ret_val_unbox_s_273;
        setArg(call_unbox_s_273, 0, unbox_s_273_arg0);
        setRet(call_unbox_s_273, 0, ret_val_unbox_s_273);
        InvokeNode* call_unbox_s_274;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_274_arg0 = load_reg_from_mem(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_274 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_274, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs2 = ret_val_unbox_s_274;
        setArg(call_unbox_s_274, 0, unbox_s_274_arg0);
        setRet(call_unbox_s_274, 0, ret_val_unbox_s_274);
        InvokeNode* call_unbox_s_275;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_275_arg0 = load_reg_from_mem(jh, traits::F0 + rs3);
        x86::Gp ret_val_unbox_s_275 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_275, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs3 = ret_val_unbox_s_275;
        setArg(call_unbox_s_275, 0, unbox_s_275_arg0);
        setRet(call_unbox_s_275, 0, ret_val_unbox_s_275);
        InvokeNode* call_fmadd_s_276;
        jh.cc.comment("//call_fmadd_s");
        auto fmadd_s_276_arg0 = frs1;
        auto fmadd_s_276_arg1 = frs2;
        auto fmadd_s_276_arg2 = frs3;
        auto fmadd_s_276_arg4 = get_rm(jh, (uint8_t)rm);
        x86::Gp ret_val_fmadd_s_276 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fmadd_s_276, &fmadd_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint8_t>());
        auto res = ret_val_fmadd_s_276;
        setArg(call_fmadd_s_276, 0, fmadd_s_276_arg0);
        setArg(call_fmadd_s_276, 1, fmadd_s_276_arg1);
        setArg(call_fmadd_s_276, 2, fmadd_s_276_arg2);
        setArg(call_fmadd_s_276, 3, 2);
        setArg(call_fmadd_s_276, 4, fmadd_s_276_arg4);
        setRet(call_fmadd_s_276, 0, ret_val_fmadd_s_276);
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_operation(cc, bor, gen_ext(cc, 
                  res, 64, false), (uint64_t)((int64_t)- 1<<32))
              );
        InvokeNode* call_fget_flags_277;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_277 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_277, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_277;
        setRet(call_fget_flags_277, 0, ret_val_fget_flags_277);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 102);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 103: FNMSUB__S */
    continuation_e __fnmsub__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rs3 = ((bit_sub<27,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}, {rs3}", fmt::arg("mnemonic", "fnmsub.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FNMSUB__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 103);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_unbox_s_278;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_278_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_278 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_278, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs1 = ret_val_unbox_s_278;
        setArg(call_unbox_s_278, 0, unbox_s_278_arg0);
        setRet(call_unbox_s_278, 0, ret_val_unbox_s_278);
        InvokeNode* call_unbox_s_279;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_279_arg0 = load_reg_from_mem(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_279 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_279, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs2 = ret_val_unbox_s_279;
        setArg(call_unbox_s_279, 0, unbox_s_279_arg0);
        setRet(call_unbox_s_279, 0, ret_val_unbox_s_279);
        InvokeNode* call_unbox_s_280;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_280_arg0 = load_reg_from_mem(jh, traits::F0 + rs3);
        x86::Gp ret_val_unbox_s_280 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_280, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs3 = ret_val_unbox_s_280;
        setArg(call_unbox_s_280, 0, unbox_s_280_arg0);
        setRet(call_unbox_s_280, 0, ret_val_unbox_s_280);
        InvokeNode* call_fmadd_s_281;
        jh.cc.comment("//call_fmadd_s");
        auto fmadd_s_281_arg0 = frs1;
        auto fmadd_s_281_arg1 = frs2;
        auto fmadd_s_281_arg2 = frs3;
        auto fmadd_s_281_arg4 = get_rm(jh, (uint8_t)rm);
        x86::Gp ret_val_fmadd_s_281 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fmadd_s_281, &fmadd_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint8_t>());
        auto res = ret_val_fmadd_s_281;
        setArg(call_fmadd_s_281, 0, fmadd_s_281_arg0);
        setArg(call_fmadd_s_281, 1, fmadd_s_281_arg1);
        setArg(call_fmadd_s_281, 2, fmadd_s_281_arg2);
        setArg(call_fmadd_s_281, 3, 3);
        setArg(call_fmadd_s_281, 4, fmadd_s_281_arg4);
        setRet(call_fmadd_s_281, 0, ret_val_fmadd_s_281);
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_operation(cc, bor, gen_ext(cc, 
                  res, 64, false), (uint64_t)((int64_t)- 1<<32))
              );
        InvokeNode* call_fget_flags_282;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_282 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_282, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_282;
        setRet(call_fget_flags_282, 0, ret_val_fget_flags_282);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 103);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 104: FADD__S */
    continuation_e __fadd__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fadd.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FADD__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 104);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_unbox_s_283;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_283_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_283 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_283, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs1 = ret_val_unbox_s_283;
        setArg(call_unbox_s_283, 0, unbox_s_283_arg0);
        setRet(call_unbox_s_283, 0, ret_val_unbox_s_283);
        InvokeNode* call_unbox_s_284;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_284_arg0 = load_reg_from_mem(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_284 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_284, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs2 = ret_val_unbox_s_284;
        setArg(call_unbox_s_284, 0, unbox_s_284_arg0);
        setRet(call_unbox_s_284, 0, ret_val_unbox_s_284);
        InvokeNode* call_fadd_s_285;
        jh.cc.comment("//call_fadd_s");
        auto fadd_s_285_arg0 = frs1;
        auto fadd_s_285_arg1 = frs2;
        auto fadd_s_285_arg2 = get_rm(jh, (uint8_t)rm);
        x86::Gp ret_val_fadd_s_285 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fadd_s_285, &fadd_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint8_t>());
        auto res = ret_val_fadd_s_285;
        setArg(call_fadd_s_285, 0, fadd_s_285_arg0);
        setArg(call_fadd_s_285, 1, fadd_s_285_arg1);
        setArg(call_fadd_s_285, 2, fadd_s_285_arg2);
        setRet(call_fadd_s_285, 0, ret_val_fadd_s_285);
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_operation(cc, bor, gen_ext(cc, 
                  res, 64, false), (uint64_t)((int64_t)- 1<<32))
              );
        InvokeNode* call_fget_flags_286;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_286 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_286, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_286;
        setRet(call_fget_flags_286, 0, ret_val_fget_flags_286);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 104);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 105: FSUB__S */
    continuation_e __fsub__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsub.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FSUB__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 105);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_unbox_s_287;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_287_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_287 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_287, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs1 = ret_val_unbox_s_287;
        setArg(call_unbox_s_287, 0, unbox_s_287_arg0);
        setRet(call_unbox_s_287, 0, ret_val_unbox_s_287);
        InvokeNode* call_unbox_s_288;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_288_arg0 = load_reg_from_mem(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_288 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_288, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs2 = ret_val_unbox_s_288;
        setArg(call_unbox_s_288, 0, unbox_s_288_arg0);
        setRet(call_unbox_s_288, 0, ret_val_unbox_s_288);
        InvokeNode* call_fsub_s_289;
        jh.cc.comment("//call_fsub_s");
        auto fsub_s_289_arg0 = frs1;
        auto fsub_s_289_arg1 = frs2;
        auto fsub_s_289_arg2 = get_rm(jh, (uint8_t)rm);
        x86::Gp ret_val_fsub_s_289 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fsub_s_289, &fsub_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint8_t>());
        auto res = ret_val_fsub_s_289;
        setArg(call_fsub_s_289, 0, fsub_s_289_arg0);
        setArg(call_fsub_s_289, 1, fsub_s_289_arg1);
        setArg(call_fsub_s_289, 2, fsub_s_289_arg2);
        setRet(call_fsub_s_289, 0, ret_val_fsub_s_289);
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_operation(cc, bor, gen_ext(cc, 
                  res, 64, false), (uint64_t)((int64_t)- 1<<32))
              );
        InvokeNode* call_fget_flags_290;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_290 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_290, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_290;
        setRet(call_fget_flags_290, 0, ret_val_fget_flags_290);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 105);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 106: FMUL__S */
    continuation_e __fmul__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fmul.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FMUL__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 106);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_unbox_s_291;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_291_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_291 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_291, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs1 = ret_val_unbox_s_291;
        setArg(call_unbox_s_291, 0, unbox_s_291_arg0);
        setRet(call_unbox_s_291, 0, ret_val_unbox_s_291);
        InvokeNode* call_unbox_s_292;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_292_arg0 = load_reg_from_mem(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_292 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_292, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs2 = ret_val_unbox_s_292;
        setArg(call_unbox_s_292, 0, unbox_s_292_arg0);
        setRet(call_unbox_s_292, 0, ret_val_unbox_s_292);
        InvokeNode* call_fmul_s_293;
        jh.cc.comment("//call_fmul_s");
        auto fmul_s_293_arg0 = frs1;
        auto fmul_s_293_arg1 = frs2;
        auto fmul_s_293_arg2 = get_rm(jh, (uint8_t)rm);
        x86::Gp ret_val_fmul_s_293 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fmul_s_293, &fmul_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint8_t>());
        auto res = ret_val_fmul_s_293;
        setArg(call_fmul_s_293, 0, fmul_s_293_arg0);
        setArg(call_fmul_s_293, 1, fmul_s_293_arg1);
        setArg(call_fmul_s_293, 2, fmul_s_293_arg2);
        setRet(call_fmul_s_293, 0, ret_val_fmul_s_293);
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_operation(cc, bor, gen_ext(cc, 
                  res, 64, false), (uint64_t)((int64_t)- 1<<32))
              );
        InvokeNode* call_fget_flags_294;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_294 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_294, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_294;
        setRet(call_fget_flags_294, 0, ret_val_fget_flags_294);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 106);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 107: FDIV__S */
    continuation_e __fdiv__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fdiv.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FDIV__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 107);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_unbox_s_295;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_295_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_295 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_295, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs1 = ret_val_unbox_s_295;
        setArg(call_unbox_s_295, 0, unbox_s_295_arg0);
        setRet(call_unbox_s_295, 0, ret_val_unbox_s_295);
        InvokeNode* call_unbox_s_296;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_296_arg0 = load_reg_from_mem(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_296 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_296, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs2 = ret_val_unbox_s_296;
        setArg(call_unbox_s_296, 0, unbox_s_296_arg0);
        setRet(call_unbox_s_296, 0, ret_val_unbox_s_296);
        InvokeNode* call_fdiv_s_297;
        jh.cc.comment("//call_fdiv_s");
        auto fdiv_s_297_arg0 = frs1;
        auto fdiv_s_297_arg1 = frs2;
        auto fdiv_s_297_arg2 = get_rm(jh, (uint8_t)rm);
        x86::Gp ret_val_fdiv_s_297 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fdiv_s_297, &fdiv_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint8_t>());
        auto res = ret_val_fdiv_s_297;
        setArg(call_fdiv_s_297, 0, fdiv_s_297_arg0);
        setArg(call_fdiv_s_297, 1, fdiv_s_297_arg1);
        setArg(call_fdiv_s_297, 2, fdiv_s_297_arg2);
        setRet(call_fdiv_s_297, 0, ret_val_fdiv_s_297);
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_operation(cc, bor, gen_ext(cc, 
                  res, 64, false), (uint64_t)((int64_t)- 1<<32))
              );
        InvokeNode* call_fget_flags_298;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_298 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_298, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_298;
        setRet(call_fget_flags_298, 0, ret_val_fget_flags_298);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 107);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 108: FSQRT__S */
    continuation_e __fsqrt__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fsqrt.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FSQRT__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 108);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_unbox_s_299;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_299_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_299 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_299, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs1 = ret_val_unbox_s_299;
        setArg(call_unbox_s_299, 0, unbox_s_299_arg0);
        setRet(call_unbox_s_299, 0, ret_val_unbox_s_299);
        InvokeNode* call_fsqrt_s_300;
        jh.cc.comment("//call_fsqrt_s");
        auto fsqrt_s_300_arg0 = frs1;
        auto fsqrt_s_300_arg1 = get_rm(jh, (uint8_t)rm);
        x86::Gp ret_val_fsqrt_s_300 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fsqrt_s_300, &fsqrt_s, FuncSignature::build<uint32_t, uint32_t, uint8_t>());
        auto res = ret_val_fsqrt_s_300;
        setArg(call_fsqrt_s_300, 0, fsqrt_s_300_arg0);
        setArg(call_fsqrt_s_300, 1, fsqrt_s_300_arg1);
        setRet(call_fsqrt_s_300, 0, ret_val_fsqrt_s_300);
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_operation(cc, bor, gen_ext(cc, 
                  res, 64, false), (uint64_t)((int64_t)- 1<<32))
              );
        InvokeNode* call_fget_flags_301;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_301 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_301, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_301;
        setRet(call_fget_flags_301, 0, ret_val_fget_flags_301);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 108);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 109: FSGNJ__S */
    continuation_e __fsgnj__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnj.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FSGNJ__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 109);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_unbox_s_302;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_302_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_302 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_302, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs1 = ret_val_unbox_s_302;
        setArg(call_unbox_s_302, 0, unbox_s_302_arg0);
        setRet(call_unbox_s_302, 0, ret_val_unbox_s_302);
        InvokeNode* call_unbox_s_303;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_303_arg0 = load_reg_from_mem(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_303 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_303, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs2 = ret_val_unbox_s_303;
        setArg(call_unbox_s_303, 0, unbox_s_303_arg0);
        setRet(call_unbox_s_303, 0, ret_val_unbox_s_303);
        auto res = gen_operation(jh.cc, bor, gen_ext(jh.cc, gen_operation(jh.cc, shl, gen_slice(jh.cc, frs2, 31, 31-31+1), 31), 32, false), gen_slice(jh.cc, frs1, 0, 30-0+1));
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_operation(cc, bor, gen_ext(cc, 
                  res, 64, false), (uint64_t)(- 1<<32))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 109);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 110: FSGNJN__S */
    continuation_e __fsgnjn__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnjn.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FSGNJN__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 110);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_unbox_s_304;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_304_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_304 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_304, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs1 = ret_val_unbox_s_304;
        setArg(call_unbox_s_304, 0, unbox_s_304_arg0);
        setRet(call_unbox_s_304, 0, ret_val_unbox_s_304);
        InvokeNode* call_unbox_s_305;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_305_arg0 = load_reg_from_mem(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_305 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_305, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs2 = ret_val_unbox_s_305;
        setArg(call_unbox_s_305, 0, unbox_s_305_arg0);
        setRet(call_unbox_s_305, 0, ret_val_unbox_s_305);
        auto res = gen_operation(jh.cc, bor, gen_ext(jh.cc, gen_operation(jh.cc, shl, gen_operation(cc, bnot, gen_slice(jh.cc, frs2, 31, 31-31+1)), 31), 32, false), gen_slice(jh.cc, frs1, 0, 30-0+1));
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_operation(cc, bor, gen_ext(cc, 
                  res, 64, false), (uint64_t)(- 1<<32))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 110);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 111: FSGNJX__S */
    continuation_e __fsgnjx__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnjx.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FSGNJX__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 111);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_unbox_s_306;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_306_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_306 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_306, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs1 = ret_val_unbox_s_306;
        setArg(call_unbox_s_306, 0, unbox_s_306_arg0);
        setRet(call_unbox_s_306, 0, ret_val_unbox_s_306);
        InvokeNode* call_unbox_s_307;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_307_arg0 = load_reg_from_mem(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_307 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_307, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs2 = ret_val_unbox_s_307;
        setArg(call_unbox_s_307, 0, unbox_s_307_arg0);
        setRet(call_unbox_s_307, 0, ret_val_unbox_s_307);
        auto res = gen_operation(cc, bxor, frs1, (gen_operation(cc, band, frs2, (uint32_t)2147483648)
        ))
        ;
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_operation(cc, bor, gen_ext(cc, 
                  res, 64, false), (uint64_t)((int64_t)- 1<<32))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 111);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 112: FMIN__S */
    continuation_e __fmin__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fmin.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FMIN__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 112);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_unbox_s_308;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_308_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_308 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_308, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs1 = ret_val_unbox_s_308;
        setArg(call_unbox_s_308, 0, unbox_s_308_arg0);
        setRet(call_unbox_s_308, 0, ret_val_unbox_s_308);
        InvokeNode* call_unbox_s_309;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_309_arg0 = load_reg_from_mem(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_309 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_309, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs2 = ret_val_unbox_s_309;
        setArg(call_unbox_s_309, 0, unbox_s_309_arg0);
        setRet(call_unbox_s_309, 0, ret_val_unbox_s_309);
        InvokeNode* call_fsel_s_310;
        jh.cc.comment("//call_fsel_s");
        auto fsel_s_310_arg0 = frs1;
        auto fsel_s_310_arg1 = frs2;
        x86::Gp ret_val_fsel_s_310 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fsel_s_310, &fsel_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t>());
        auto res = ret_val_fsel_s_310;
        setArg(call_fsel_s_310, 0, fsel_s_310_arg0);
        setArg(call_fsel_s_310, 1, fsel_s_310_arg1);
        setArg(call_fsel_s_310, 2, 0);
        setRet(call_fsel_s_310, 0, ret_val_fsel_s_310);
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_operation(cc, bor, gen_ext(cc, 
                  res, 64, false), (uint64_t)((int64_t)- 1<<32))
              );
        InvokeNode* call_fget_flags_311;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_311 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_311, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_311;
        setRet(call_fget_flags_311, 0, ret_val_fget_flags_311);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 112);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 113: FMAX__S */
    continuation_e __fmax__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fmax.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FMAX__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 113);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_unbox_s_312;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_312_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_312 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_312, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs1 = ret_val_unbox_s_312;
        setArg(call_unbox_s_312, 0, unbox_s_312_arg0);
        setRet(call_unbox_s_312, 0, ret_val_unbox_s_312);
        InvokeNode* call_unbox_s_313;
        jh.cc.comment("//call_unbox_s");
        auto unbox_s_313_arg0 = load_reg_from_mem(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_313 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_unbox_s_313, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
        auto frs2 = ret_val_unbox_s_313;
        setArg(call_unbox_s_313, 0, unbox_s_313_arg0);
        setRet(call_unbox_s_313, 0, ret_val_unbox_s_313);
        InvokeNode* call_fsel_s_314;
        jh.cc.comment("//call_fsel_s");
        auto fsel_s_314_arg0 = frs1;
        auto fsel_s_314_arg1 = frs2;
        x86::Gp ret_val_fsel_s_314 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fsel_s_314, &fsel_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t>());
        auto res = ret_val_fsel_s_314;
        setArg(call_fsel_s_314, 0, fsel_s_314_arg0);
        setArg(call_fsel_s_314, 1, fsel_s_314_arg1);
        setArg(call_fsel_s_314, 2, 1);
        setRet(call_fsel_s_314, 0, ret_val_fsel_s_314);
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_operation(cc, bor, gen_ext(cc, 
                  res, 64, false), (uint64_t)((int64_t)- 1<<32))
              );
        InvokeNode* call_fget_flags_315;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_315 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_315, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_315;
        setRet(call_fget_flags_315, 0, ret_val_fget_flags_315);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 113);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 114: FCVT__W__S */
    continuation_e __fcvt__w__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt.w.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__W__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 114);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            InvokeNode* call_unbox_s_316;
            jh.cc.comment("//call_unbox_s");
            auto unbox_s_316_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_s_316 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_unbox_s_316, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
            auto frs1 = ret_val_unbox_s_316;
            setArg(call_unbox_s_316, 0, unbox_s_316_arg0);
            setRet(call_unbox_s_316, 0, ret_val_unbox_s_316);
            InvokeNode* call_fcvt_s_317;
            jh.cc.comment("//call_fcvt_s");
            auto fcvt_s_317_arg0 = frs1;
            auto fcvt_s_317_arg2 = get_rm(jh, rm);
            x86::Gp ret_val_fcvt_s_317 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fcvt_s_317, &fcvt_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint8_t>());
            auto res = gen_ext(cc, 
                gen_ext(cc, 
                    ret_val_fcvt_s_317, 32, false), 32, true);
            setArg(call_fcvt_s_317, 0, fcvt_s_317_arg0);
            setArg(call_fcvt_s_317, 1, 0);
            setArg(call_fcvt_s_317, 2, fcvt_s_317_arg2);
            setRet(call_fcvt_s_317, 0, ret_val_fcvt_s_317);
            if((rd)!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 32, true));
            }
            InvokeNode* call_fget_flags_318;
            jh.cc.comment("//call_fget_flags");
            x86::Gp ret_val_fget_flags_318 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fget_flags_318, &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_318;
            setRet(call_fget_flags_318, 0, ret_val_fget_flags_318);
            mov(cc, get_ptr_for(jh, traits::FCSR),
                  gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
                  ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
                  ))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 114);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 115: FCVT__WU__S */
    continuation_e __fcvt__wu__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt.wu.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__WU__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 115);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            InvokeNode* call_unbox_s_319;
            jh.cc.comment("//call_unbox_s");
            auto unbox_s_319_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_s_319 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_unbox_s_319, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
            auto frs1 = ret_val_unbox_s_319;
            setArg(call_unbox_s_319, 0, unbox_s_319_arg0);
            setRet(call_unbox_s_319, 0, ret_val_unbox_s_319);
            InvokeNode* call_fcvt_s_320;
            jh.cc.comment("//call_fcvt_s");
            auto fcvt_s_320_arg0 = frs1;
            auto fcvt_s_320_arg2 = get_rm(jh, (uint8_t)rm);
            x86::Gp ret_val_fcvt_s_320 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fcvt_s_320, &fcvt_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint8_t>());
            auto res = gen_ext(cc, 
                gen_ext(cc, 
                    ret_val_fcvt_s_320, 32, false), 32, true);
            setArg(call_fcvt_s_320, 0, fcvt_s_320_arg0);
            setArg(call_fcvt_s_320, 1, 1);
            setArg(call_fcvt_s_320, 2, fcvt_s_320_arg2);
            setRet(call_fcvt_s_320, 0, ret_val_fcvt_s_320);
            if((rd)!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 32, true));
            }
            InvokeNode* call_fget_flags_321;
            jh.cc.comment("//call_fget_flags");
            x86::Gp ret_val_fget_flags_321 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fget_flags_321, &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_321;
            setRet(call_fget_flags_321, 0, ret_val_fget_flags_321);
            mov(cc, get_ptr_for(jh, traits::FCSR),
                  gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
                  ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
                  ))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 115);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 116: FEQ__S */
    continuation_e __feq__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "feq.s"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FEQ__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 116);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            InvokeNode* call_unbox_s_322;
            jh.cc.comment("//call_unbox_s");
            auto unbox_s_322_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_s_322 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_unbox_s_322, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
            auto frs1 = ret_val_unbox_s_322;
            setArg(call_unbox_s_322, 0, unbox_s_322_arg0);
            setRet(call_unbox_s_322, 0, ret_val_unbox_s_322);
            InvokeNode* call_unbox_s_323;
            jh.cc.comment("//call_unbox_s");
            auto unbox_s_323_arg0 = load_reg_from_mem(jh, traits::F0 + rs2);
            x86::Gp ret_val_unbox_s_323 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_unbox_s_323, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
            auto frs2 = ret_val_unbox_s_323;
            setArg(call_unbox_s_323, 0, unbox_s_323_arg0);
            setRet(call_unbox_s_323, 0, ret_val_unbox_s_323);
            InvokeNode* call_fcmp_s_324;
            jh.cc.comment("//call_fcmp_s");
            auto fcmp_s_324_arg0 = frs1;
            auto fcmp_s_324_arg1 = frs2;
            x86::Gp ret_val_fcmp_s_324 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fcmp_s_324, &fcmp_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t>());
            auto res = ret_val_fcmp_s_324;
            setArg(call_fcmp_s_324, 0, fcmp_s_324_arg0);
            setArg(call_fcmp_s_324, 1, fcmp_s_324_arg1);
            setArg(call_fcmp_s_324, 2, 0);
            setRet(call_fcmp_s_324, 0, ret_val_fcmp_s_324);
            if((rd)!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      res);
            }
            InvokeNode* call_fget_flags_325;
            jh.cc.comment("//call_fget_flags");
            x86::Gp ret_val_fget_flags_325 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fget_flags_325, &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_325;
            setRet(call_fget_flags_325, 0, ret_val_fget_flags_325);
            mov(cc, get_ptr_for(jh, traits::FCSR),
                  gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
                  ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
                  ))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 116);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 117: FLT__S */
    continuation_e __flt__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "flt.s"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FLT__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 117);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            InvokeNode* call_unbox_s_326;
            jh.cc.comment("//call_unbox_s");
            auto unbox_s_326_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_s_326 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_unbox_s_326, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
            auto frs1 = ret_val_unbox_s_326;
            setArg(call_unbox_s_326, 0, unbox_s_326_arg0);
            setRet(call_unbox_s_326, 0, ret_val_unbox_s_326);
            InvokeNode* call_unbox_s_327;
            jh.cc.comment("//call_unbox_s");
            auto unbox_s_327_arg0 = load_reg_from_mem(jh, traits::F0 + rs2);
            x86::Gp ret_val_unbox_s_327 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_unbox_s_327, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
            auto frs2 = ret_val_unbox_s_327;
            setArg(call_unbox_s_327, 0, unbox_s_327_arg0);
            setRet(call_unbox_s_327, 0, ret_val_unbox_s_327);
            InvokeNode* call_fcmp_s_328;
            jh.cc.comment("//call_fcmp_s");
            auto fcmp_s_328_arg0 = frs1;
            auto fcmp_s_328_arg1 = frs2;
            x86::Gp ret_val_fcmp_s_328 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fcmp_s_328, &fcmp_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t>());
            auto res = ret_val_fcmp_s_328;
            setArg(call_fcmp_s_328, 0, fcmp_s_328_arg0);
            setArg(call_fcmp_s_328, 1, fcmp_s_328_arg1);
            setArg(call_fcmp_s_328, 2, 2);
            setRet(call_fcmp_s_328, 0, ret_val_fcmp_s_328);
            if((rd)!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      res);
            }
            InvokeNode* call_fget_flags_329;
            jh.cc.comment("//call_fget_flags");
            x86::Gp ret_val_fget_flags_329 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fget_flags_329, &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_329;
            setRet(call_fget_flags_329, 0, ret_val_fget_flags_329);
            mov(cc, get_ptr_for(jh, traits::FCSR),
                  gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
                  ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
                  ))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 117);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 118: FLE__S */
    continuation_e __fle__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fle.s"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FLE__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 118);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            InvokeNode* call_unbox_s_330;
            jh.cc.comment("//call_unbox_s");
            auto unbox_s_330_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_s_330 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_unbox_s_330, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
            auto frs1 = ret_val_unbox_s_330;
            setArg(call_unbox_s_330, 0, unbox_s_330_arg0);
            setRet(call_unbox_s_330, 0, ret_val_unbox_s_330);
            InvokeNode* call_unbox_s_331;
            jh.cc.comment("//call_unbox_s");
            auto unbox_s_331_arg0 = load_reg_from_mem(jh, traits::F0 + rs2);
            x86::Gp ret_val_unbox_s_331 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_unbox_s_331, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
            auto frs2 = ret_val_unbox_s_331;
            setArg(call_unbox_s_331, 0, unbox_s_331_arg0);
            setRet(call_unbox_s_331, 0, ret_val_unbox_s_331);
            InvokeNode* call_fcmp_s_332;
            jh.cc.comment("//call_fcmp_s");
            auto fcmp_s_332_arg0 = frs1;
            auto fcmp_s_332_arg1 = frs2;
            x86::Gp ret_val_fcmp_s_332 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fcmp_s_332, &fcmp_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t>());
            auto res = ret_val_fcmp_s_332;
            setArg(call_fcmp_s_332, 0, fcmp_s_332_arg0);
            setArg(call_fcmp_s_332, 1, fcmp_s_332_arg1);
            setArg(call_fcmp_s_332, 2, 1);
            setRet(call_fcmp_s_332, 0, ret_val_fcmp_s_332);
            if((rd)!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      res);
            }
            InvokeNode* call_fget_flags_333;
            jh.cc.comment("//call_fget_flags");
            x86::Gp ret_val_fget_flags_333 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fget_flags_333, &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_333;
            setRet(call_fget_flags_333, 0, ret_val_fget_flags_333);
            mov(cc, get_ptr_for(jh, traits::FCSR),
                  gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
                  ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
                  ))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 118);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 119: FCLASS__S */
    continuation_e __fclass__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "fclass.s"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCLASS__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 119);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            InvokeNode* call_unbox_s_335;
            jh.cc.comment("//call_unbox_s");
            auto unbox_s_335_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_s_335 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_unbox_s_335, &unbox_s, FuncSignature::build<uint32_t, uint64_t>());
            InvokeNode* call_fclass_s_334;
            jh.cc.comment("//call_fclass_s");
            auto fclass_s_334_arg0 = ret_val_unbox_s_335;
            x86::Gp ret_val_fclass_s_334 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fclass_s_334, &fclass_s, FuncSignature::build<uint32_t, uint32_t>());
            auto res = ret_val_fclass_s_334;
            setArg(call_unbox_s_335, 0, unbox_s_335_arg0);
            setRet(call_unbox_s_335, 0, ret_val_unbox_s_335);setArg(call_fclass_s_334, 0, fclass_s_334_arg0);
            setRet(call_fclass_s_334, 0, ret_val_fclass_s_334);
            if((rd)!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      res);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 119);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 120: FCVT__S__W */
    continuation_e __fcvt__s__w(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt.s.w"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__S__W_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 120);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            InvokeNode* call_fcvt_s_336;
            jh.cc.comment("//call_fcvt_s");
            auto fcvt_s_336_arg0 = gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs1), 32, false);
            auto fcvt_s_336_arg2 = get_rm(jh, (uint8_t)rm);
            x86::Gp ret_val_fcvt_s_336 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fcvt_s_336, &fcvt_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint8_t>());
            auto res = ret_val_fcvt_s_336;
            setArg(call_fcvt_s_336, 0, fcvt_s_336_arg0);
            setArg(call_fcvt_s_336, 1, 2);
            setArg(call_fcvt_s_336, 2, fcvt_s_336_arg2);
            setRet(call_fcvt_s_336, 0, ret_val_fcvt_s_336);
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  gen_operation(cc, bor, gen_ext(cc, 
                      res, 64, false), (uint64_t)((int64_t)- 1<<32))
                  );
            InvokeNode* call_fget_flags_337;
            jh.cc.comment("//call_fget_flags");
            x86::Gp ret_val_fget_flags_337 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fget_flags_337, &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_337;
            setRet(call_fget_flags_337, 0, ret_val_fget_flags_337);
            mov(cc, get_ptr_for(jh, traits::FCSR),
                  gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
                  ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
                  ))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 120);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 121: FCVT__S__WU */
    continuation_e __fcvt__s__wu(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt.s.wu"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__S__WU_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 121);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            InvokeNode* call_fcvt_s_338;
            jh.cc.comment("//call_fcvt_s");
            auto fcvt_s_338_arg0 = gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs1), 32, false);
            auto fcvt_s_338_arg2 = get_rm(jh, (uint8_t)rm);
            x86::Gp ret_val_fcvt_s_338 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fcvt_s_338, &fcvt_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint8_t>());
            auto res = ret_val_fcvt_s_338;
            setArg(call_fcvt_s_338, 0, fcvt_s_338_arg0);
            setArg(call_fcvt_s_338, 1, 3);
            setArg(call_fcvt_s_338, 2, fcvt_s_338_arg2);
            setRet(call_fcvt_s_338, 0, ret_val_fcvt_s_338);
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  gen_operation(cc, bor, gen_ext(cc, 
                      res, 64, false), (uint64_t)((int64_t)- 1<<32))
                  );
            InvokeNode* call_fget_flags_339;
            jh.cc.comment("//call_fget_flags");
            x86::Gp ret_val_fget_flags_339 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fget_flags_339, &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_339;
            setRet(call_fget_flags_339, 0, ret_val_fget_flags_339);
            mov(cc, get_ptr_for(jh, traits::FCSR),
                  gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
                  ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
                  ))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 121);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 122: FMV__X__W */
    continuation_e __fmv__x__w(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "fmv.x.w"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FMV__X__W_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 122);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if((rd)!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          gen_ext(cc, 
                              gen_ext(cc, 
                                  load_reg_from_mem(jh, traits::F0 + rs1), 32, false), 32, true), 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 122);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 123: FMV__W__X */
    continuation_e __fmv__w__x(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "fmv.w.x"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FMV__W__X_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 123);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  gen_operation(cc, bor, gen_ext(cc, 
                      load_reg_from_mem(jh, traits::X0 + rs1), 64, false), (uint64_t)((int64_t)- 1<<32))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 123);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 124: C__FLW */
    continuation_e __c__flw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} f(8+{rd}), {uimm}({rs1})", fmt::arg("mnemonic", "c.flw"),
                fmt::arg("rd", rd), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__FLW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 124);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1+8), uimm)
            ), 32, false);
        auto res = gen_ext(cc, 
            gen_read_mem(jh, traits::MEM, offs, 4), 32, false);
        if(static_cast<uint32_t>(traits::FLEN)==32){
            mov(cc, get_ptr_for(jh, traits::F0+ rd+8),
                  gen_ext(cc, res, 64, false)
            );
        }
        else{
            mov(cc, get_ptr_for(jh, traits::F0+ rd+8),
                  gen_operation(cc, bor, res, ((uint64_t)((int64_t)- 1<<32)))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 124);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 125: C__FSW */
    continuation_e __c__fsw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} f(8+{rs2}), {uimm}({rs1})", fmt::arg("mnemonic", "c.fsw"),
                fmt::arg("rs2", rs2), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__FSW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 125);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1+8), uimm)
            ), 32, false);
        gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
            load_reg_from_mem(jh, traits::F0 + rs2+8), 32, false), 4);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 125);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 126: C__FLWSP */
    continuation_e __c__flwsp(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t uimm = ((bit_sub<2,2>(instr) << 6) | (bit_sub<4,3>(instr) << 2) | (bit_sub<12,1>(instr) << 5));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} f {rd}, {uimm}(x2)", fmt::arg("mnemonic", "c.flwsp"),
                fmt::arg("rd", rd), fmt::arg("uimm", uimm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__FLWSP_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 126);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + 2), uimm)
            ), 32, false);
        auto res = gen_ext(cc, 
            gen_read_mem(jh, traits::MEM, offs, 4), 32, false);
        if(static_cast<uint32_t>(traits::FLEN)==32){
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  gen_ext(cc, res, 64, false)
            );
        }
        else{
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  gen_operation(cc, bor, res, ((uint64_t)((int64_t)- 1<<32)))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 126);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 127: C__FSWSP */
    continuation_e __c__fswsp(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t uimm = ((bit_sub<7,2>(instr) << 6) | (bit_sub<9,4>(instr) << 2));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} f {rs2}, {uimm}(x2), ", fmt::arg("mnemonic", "c.fswsp"),
                fmt::arg("rs2", rs2), fmt::arg("uimm", uimm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__FSWSP_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 127);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + 2), uimm)
            ), 32, false);
        gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
            load_reg_from_mem(jh, traits::F0 + rs2), 32, false), 4);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 127);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 128: FLD */
    continuation_e __fld(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "fld"),
                fmt::arg("rd", fname(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FLD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 128);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  gen_ext(cc, gen_read_mem(jh, traits::MEM, offs, 8), 64, false)
            );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 128);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 129: FSD */
    continuation_e __fsd(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "fsd"),
                fmt::arg("rs2", fname(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FSD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 129);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            gen_write_mem(jh, traits::MEM, offs, load_reg_from_mem(jh, traits::F0 + rs2), 8);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 129);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 130: FMADD_D */
    continuation_e __fmadd_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rs3 = ((bit_sub<27,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}, {rs3}", fmt::arg("mnemonic", "fmadd_d"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FMADD_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 130);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_fmadd_d_340;
        jh.cc.comment("//call_fmadd_d");
        auto fmadd_d_340_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        auto fmadd_d_340_arg1 = load_reg_from_mem(jh, traits::F0 + rs2);
        auto fmadd_d_340_arg2 = load_reg_from_mem(jh, traits::F0 + rs3);
        auto fmadd_d_340_arg4 = get_rm(jh, rm);
        x86::Gp ret_val_fmadd_d_340 = get_reg_Gp(jh.cc, 64, true);
        jh.cc.invoke(&call_fmadd_d_340, &fmadd_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint8_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_fmadd_d_340);
        setArg(call_fmadd_d_340, 0, fmadd_d_340_arg0);
        setArg(call_fmadd_d_340, 1, fmadd_d_340_arg1);
        setArg(call_fmadd_d_340, 2, fmadd_d_340_arg2);
        setArg(call_fmadd_d_340, 3, 0);
        setArg(call_fmadd_d_340, 4, fmadd_d_340_arg4);
        setRet(call_fmadd_d_340, 0, ret_val_fmadd_d_340);
        InvokeNode* call_fget_flags_341;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_341 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_341, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_341;
        setRet(call_fget_flags_341, 0, ret_val_fget_flags_341);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 130);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 131: FMSUB_D */
    continuation_e __fmsub_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rs3 = ((bit_sub<27,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}, {rs3}", fmt::arg("mnemonic", "fmsub_d"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FMSUB_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 131);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_fmadd_d_342;
        jh.cc.comment("//call_fmadd_d");
        auto fmadd_d_342_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        auto fmadd_d_342_arg1 = load_reg_from_mem(jh, traits::F0 + rs2);
        auto fmadd_d_342_arg2 = load_reg_from_mem(jh, traits::F0 + rs3);
        auto fmadd_d_342_arg4 = get_rm(jh, rm);
        x86::Gp ret_val_fmadd_d_342 = get_reg_Gp(jh.cc, 64, true);
        jh.cc.invoke(&call_fmadd_d_342, &fmadd_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint8_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_fmadd_d_342);
        setArg(call_fmadd_d_342, 0, fmadd_d_342_arg0);
        setArg(call_fmadd_d_342, 1, fmadd_d_342_arg1);
        setArg(call_fmadd_d_342, 2, fmadd_d_342_arg2);
        setArg(call_fmadd_d_342, 3, 1);
        setArg(call_fmadd_d_342, 4, fmadd_d_342_arg4);
        setRet(call_fmadd_d_342, 0, ret_val_fmadd_d_342);
        InvokeNode* call_fget_flags_343;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_343 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_343, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_343;
        setRet(call_fget_flags_343, 0, ret_val_fget_flags_343);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 131);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 132: FNMADD_D */
    continuation_e __fnmadd_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rs3 = ((bit_sub<27,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}, {rs3}", fmt::arg("mnemonic", "fnmadd_d"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FNMADD_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 132);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_fmadd_d_344;
        jh.cc.comment("//call_fmadd_d");
        auto fmadd_d_344_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        auto fmadd_d_344_arg1 = load_reg_from_mem(jh, traits::F0 + rs2);
        auto fmadd_d_344_arg2 = load_reg_from_mem(jh, traits::F0 + rs3);
        auto fmadd_d_344_arg4 = get_rm(jh, rm);
        x86::Gp ret_val_fmadd_d_344 = get_reg_Gp(jh.cc, 64, true);
        jh.cc.invoke(&call_fmadd_d_344, &fmadd_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint8_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_fmadd_d_344);
        setArg(call_fmadd_d_344, 0, fmadd_d_344_arg0);
        setArg(call_fmadd_d_344, 1, fmadd_d_344_arg1);
        setArg(call_fmadd_d_344, 2, fmadd_d_344_arg2);
        setArg(call_fmadd_d_344, 3, 2);
        setArg(call_fmadd_d_344, 4, fmadd_d_344_arg4);
        setRet(call_fmadd_d_344, 0, ret_val_fmadd_d_344);
        InvokeNode* call_fget_flags_345;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_345 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_345, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_345;
        setRet(call_fget_flags_345, 0, ret_val_fget_flags_345);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 132);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 133: FNMSUB_D */
    continuation_e __fnmsub_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rs3 = ((bit_sub<27,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}, {rs3}", fmt::arg("mnemonic", "fnmsub_d"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FNMSUB_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 133);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_fmadd_d_346;
        jh.cc.comment("//call_fmadd_d");
        auto fmadd_d_346_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        auto fmadd_d_346_arg1 = load_reg_from_mem(jh, traits::F0 + rs2);
        auto fmadd_d_346_arg2 = load_reg_from_mem(jh, traits::F0 + rs3);
        auto fmadd_d_346_arg4 = get_rm(jh, rm);
        x86::Gp ret_val_fmadd_d_346 = get_reg_Gp(jh.cc, 64, true);
        jh.cc.invoke(&call_fmadd_d_346, &fmadd_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint8_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_fmadd_d_346);
        setArg(call_fmadd_d_346, 0, fmadd_d_346_arg0);
        setArg(call_fmadd_d_346, 1, fmadd_d_346_arg1);
        setArg(call_fmadd_d_346, 2, fmadd_d_346_arg2);
        setArg(call_fmadd_d_346, 3, 3);
        setArg(call_fmadd_d_346, 4, fmadd_d_346_arg4);
        setRet(call_fmadd_d_346, 0, ret_val_fmadd_d_346);
        InvokeNode* call_fget_flags_347;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_347 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_347, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_347;
        setRet(call_fget_flags_347, 0, ret_val_fget_flags_347);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 133);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 134: FADD_D */
    continuation_e __fadd_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fadd_d"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FADD_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 134);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_fadd_d_348;
        jh.cc.comment("//call_fadd_d");
        auto fadd_d_348_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        auto fadd_d_348_arg1 = load_reg_from_mem(jh, traits::F0 + rs2);
        auto fadd_d_348_arg2 = get_rm(jh, rm);
        x86::Gp ret_val_fadd_d_348 = get_reg_Gp(jh.cc, 64, true);
        jh.cc.invoke(&call_fadd_d_348, &fadd_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint8_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_fadd_d_348);
        setArg(call_fadd_d_348, 0, fadd_d_348_arg0);
        setArg(call_fadd_d_348, 1, fadd_d_348_arg1);
        setArg(call_fadd_d_348, 2, fadd_d_348_arg2);
        setRet(call_fadd_d_348, 0, ret_val_fadd_d_348);
        InvokeNode* call_fget_flags_349;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_349 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_349, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_349;
        setRet(call_fget_flags_349, 0, ret_val_fget_flags_349);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 134);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 135: FSUB_D */
    continuation_e __fsub_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsub_d"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FSUB_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 135);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_fsub_d_350;
        jh.cc.comment("//call_fsub_d");
        auto fsub_d_350_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        auto fsub_d_350_arg1 = load_reg_from_mem(jh, traits::F0 + rs2);
        auto fsub_d_350_arg2 = get_rm(jh, rm);
        x86::Gp ret_val_fsub_d_350 = get_reg_Gp(jh.cc, 64, true);
        jh.cc.invoke(&call_fsub_d_350, &fsub_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint8_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_fsub_d_350);
        setArg(call_fsub_d_350, 0, fsub_d_350_arg0);
        setArg(call_fsub_d_350, 1, fsub_d_350_arg1);
        setArg(call_fsub_d_350, 2, fsub_d_350_arg2);
        setRet(call_fsub_d_350, 0, ret_val_fsub_d_350);
        InvokeNode* call_fget_flags_351;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_351 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_351, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_351;
        setRet(call_fget_flags_351, 0, ret_val_fget_flags_351);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 135);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 136: FMUL_D */
    continuation_e __fmul_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fmul_d"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FMUL_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 136);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_fmul_d_352;
        jh.cc.comment("//call_fmul_d");
        auto fmul_d_352_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        auto fmul_d_352_arg1 = load_reg_from_mem(jh, traits::F0 + rs2);
        auto fmul_d_352_arg2 = get_rm(jh, rm);
        x86::Gp ret_val_fmul_d_352 = get_reg_Gp(jh.cc, 64, true);
        jh.cc.invoke(&call_fmul_d_352, &fmul_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint8_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_fmul_d_352);
        setArg(call_fmul_d_352, 0, fmul_d_352_arg0);
        setArg(call_fmul_d_352, 1, fmul_d_352_arg1);
        setArg(call_fmul_d_352, 2, fmul_d_352_arg2);
        setRet(call_fmul_d_352, 0, ret_val_fmul_d_352);
        InvokeNode* call_fget_flags_353;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_353 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_353, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_353;
        setRet(call_fget_flags_353, 0, ret_val_fget_flags_353);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 136);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 137: FDIV_D */
    continuation_e __fdiv_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fdiv_d"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FDIV_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 137);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_fdiv_d_354;
        jh.cc.comment("//call_fdiv_d");
        auto fdiv_d_354_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        auto fdiv_d_354_arg1 = load_reg_from_mem(jh, traits::F0 + rs2);
        auto fdiv_d_354_arg2 = get_rm(jh, rm);
        x86::Gp ret_val_fdiv_d_354 = get_reg_Gp(jh.cc, 64, true);
        jh.cc.invoke(&call_fdiv_d_354, &fdiv_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint8_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_fdiv_d_354);
        setArg(call_fdiv_d_354, 0, fdiv_d_354_arg0);
        setArg(call_fdiv_d_354, 1, fdiv_d_354_arg1);
        setArg(call_fdiv_d_354, 2, fdiv_d_354_arg2);
        setRet(call_fdiv_d_354, 0, ret_val_fdiv_d_354);
        InvokeNode* call_fget_flags_355;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_355 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_355, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_355;
        setRet(call_fget_flags_355, 0, ret_val_fget_flags_355);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 137);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 138: FSQRT_D */
    continuation_e __fsqrt_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fsqrt_d"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FSQRT_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 138);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_fsqrt_d_356;
        jh.cc.comment("//call_fsqrt_d");
        auto fsqrt_d_356_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        auto fsqrt_d_356_arg1 = get_rm(jh, rm);
        x86::Gp ret_val_fsqrt_d_356 = get_reg_Gp(jh.cc, 64, true);
        jh.cc.invoke(&call_fsqrt_d_356, &fsqrt_d, FuncSignature::build<uint64_t, uint64_t, uint8_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_fsqrt_d_356);
        setArg(call_fsqrt_d_356, 0, fsqrt_d_356_arg0);
        setArg(call_fsqrt_d_356, 1, fsqrt_d_356_arg1);
        setRet(call_fsqrt_d_356, 0, ret_val_fsqrt_d_356);
        InvokeNode* call_fget_flags_357;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_357 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_357, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_357;
        setRet(call_fget_flags_357, 0, ret_val_fget_flags_357);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 138);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 139: FSGNJ_D */
    continuation_e __fsgnj_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnj_d"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FSGNJ_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 139);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_operation(jh.cc, bor, gen_ext(jh.cc, gen_operation(jh.cc, shl, gen_slice(jh.cc, load_reg_from_mem(jh, traits::F0 + rs2), 63, 63-63+1), 63), 64, false), gen_slice(jh.cc, load_reg_from_mem(jh, traits::F0 + rs1), 0, 62-0+1)));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 139);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 140: FSGNJN_D */
    continuation_e __fsgnjn_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnjn_d"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FSGNJN_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 140);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_operation(jh.cc, bor, gen_ext(jh.cc, gen_operation(jh.cc, shl, gen_operation(cc, bnot, gen_slice(jh.cc, load_reg_from_mem(jh, traits::F0 + rs2), 63, 63-63+1)), 63), 64, false), gen_slice(jh.cc, load_reg_from_mem(jh, traits::F0 + rs1), 0, 62-0+1)));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 140);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 141: FSGNJX_D */
    continuation_e __fsgnjx_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnjx_d"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FSGNJX_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 141);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_operation(cc, bxor, load_reg_from_mem(jh, traits::F0 + rs1), (gen_operation(cc, band, load_reg_from_mem(jh, traits::F0 + rs2), ((uint64_t)1<<63))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 141);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 142: FMIN_D */
    continuation_e __fmin_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fmin_d"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FMIN_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 142);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_fsel_d_358;
        jh.cc.comment("//call_fsel_d");
        auto fsel_d_358_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        auto fsel_d_358_arg1 = load_reg_from_mem(jh, traits::F0 + rs2);
        x86::Gp ret_val_fsel_d_358 = get_reg_Gp(jh.cc, 64, true);
        jh.cc.invoke(&call_fsel_d_358, &fsel_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint32_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_fsel_d_358);
        setArg(call_fsel_d_358, 0, fsel_d_358_arg0);
        setArg(call_fsel_d_358, 1, fsel_d_358_arg1);
        setArg(call_fsel_d_358, 2, 0);
        setRet(call_fsel_d_358, 0, ret_val_fsel_d_358);
        InvokeNode* call_fget_flags_359;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_359 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_359, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_359;
        setRet(call_fget_flags_359, 0, ret_val_fget_flags_359);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 142);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 143: FMAX_D */
    continuation_e __fmax_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fmax_d"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FMAX_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 143);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_fsel_d_360;
        jh.cc.comment("//call_fsel_d");
        auto fsel_d_360_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        auto fsel_d_360_arg1 = load_reg_from_mem(jh, traits::F0 + rs2);
        x86::Gp ret_val_fsel_d_360 = get_reg_Gp(jh.cc, 64, true);
        jh.cc.invoke(&call_fsel_d_360, &fsel_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint32_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_fsel_d_360);
        setArg(call_fsel_d_360, 0, fsel_d_360_arg0);
        setArg(call_fsel_d_360, 1, fsel_d_360_arg1);
        setArg(call_fsel_d_360, 2, 1);
        setRet(call_fsel_d_360, 0, ret_val_fsel_d_360);
        InvokeNode* call_fget_flags_361;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_361 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_361, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_361;
        setRet(call_fget_flags_361, 0, ret_val_fget_flags_361);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 143);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 144: FCVT_S_D */
    continuation_e __fcvt_s_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_s_d"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT_S_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 144);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_fconv_d2f_362;
        jh.cc.comment("//call_fconv_d2f");
        auto fconv_d2f_362_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
        auto fconv_d2f_362_arg1 = get_rm(jh, rm);
        x86::Gp ret_val_fconv_d2f_362 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fconv_d2f_362, &fconv_d2f, FuncSignature::build<uint32_t, uint64_t, uint8_t>());
        auto res = ret_val_fconv_d2f_362;
        setArg(call_fconv_d2f_362, 0, fconv_d2f_362_arg0);
        setArg(call_fconv_d2f_362, 1, fconv_d2f_362_arg1);
        setRet(call_fconv_d2f_362, 0, ret_val_fconv_d2f_362);
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              gen_ext(cc, 
                  (gen_operation(cc, add, res, ((int64_t)- 1<<32))
                  ), 64, true));
        InvokeNode* call_fget_flags_363;
        jh.cc.comment("//call_fget_flags");
        x86::Gp ret_val_fget_flags_363 = get_reg_Gp(jh.cc, 32, true);
        jh.cc.invoke(&call_fget_flags_363, &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_363;
        setRet(call_fget_flags_363, 0, ret_val_fget_flags_363);
        mov(cc, get_ptr_for(jh, traits::FCSR),
              gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
              ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
              ))
              );
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 144);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 145: FCVT_D_S */
    continuation_e __fcvt_d_s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_d_s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT_D_S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 145);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        InvokeNode* call_fconv_f2d_364;
        jh.cc.comment("//call_fconv_f2d");
        auto fconv_f2d_364_arg0 = gen_ext(cc, 
            load_reg_from_mem(jh, traits::F0 + rs1), 32, false);
        auto fconv_f2d_364_arg1 = get_rm(jh, rm);
        x86::Gp ret_val_fconv_f2d_364 = get_reg_Gp(jh.cc, 64, true);
        jh.cc.invoke(&call_fconv_f2d_364, &fconv_f2d, FuncSignature::build<uint64_t, uint32_t, uint8_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_fconv_f2d_364);
        setArg(call_fconv_f2d_364, 0, fconv_f2d_364_arg0);
        setArg(call_fconv_f2d_364, 1, fconv_f2d_364_arg1);
        setRet(call_fconv_f2d_364, 0, ret_val_fconv_f2d_364);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 145);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 146: FEQ_D */
    continuation_e __feq_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "feq_d"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FEQ_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 146);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            InvokeNode* call_fcmp_d_365;
            jh.cc.comment("//call_fcmp_d");
            auto fcmp_d_365_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
            auto fcmp_d_365_arg1 = load_reg_from_mem(jh, traits::F0 + rs2);
            x86::Gp ret_val_fcmp_d_365 = get_reg_Gp(jh.cc, 64, true);
            jh.cc.invoke(&call_fcmp_d_365, &fcmp_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint32_t>());
            auto res = ret_val_fcmp_d_365;
            setArg(call_fcmp_d_365, 0, fcmp_d_365_arg0);
            setArg(call_fcmp_d_365, 1, fcmp_d_365_arg1);
            setArg(call_fcmp_d_365, 2, 0);
            setRet(call_fcmp_d_365, 0, ret_val_fcmp_d_365);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 32, false));
            }
            InvokeNode* call_fget_flags_366;
            jh.cc.comment("//call_fget_flags");
            x86::Gp ret_val_fget_flags_366 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fget_flags_366, &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_366;
            setRet(call_fget_flags_366, 0, ret_val_fget_flags_366);
            mov(cc, get_ptr_for(jh, traits::FCSR),
                  gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
                  ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
                  ))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 146);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 147: FLT_D */
    continuation_e __flt_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "flt_d"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FLT_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 147);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            InvokeNode* call_fcmp_d_367;
            jh.cc.comment("//call_fcmp_d");
            auto fcmp_d_367_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
            auto fcmp_d_367_arg1 = load_reg_from_mem(jh, traits::F0 + rs2);
            x86::Gp ret_val_fcmp_d_367 = get_reg_Gp(jh.cc, 64, true);
            jh.cc.invoke(&call_fcmp_d_367, &fcmp_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint32_t>());
            auto res = ret_val_fcmp_d_367;
            setArg(call_fcmp_d_367, 0, fcmp_d_367_arg0);
            setArg(call_fcmp_d_367, 1, fcmp_d_367_arg1);
            setArg(call_fcmp_d_367, 2, 2);
            setRet(call_fcmp_d_367, 0, ret_val_fcmp_d_367);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 32, false));
            }
            InvokeNode* call_fget_flags_368;
            jh.cc.comment("//call_fget_flags");
            x86::Gp ret_val_fget_flags_368 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fget_flags_368, &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_368;
            setRet(call_fget_flags_368, 0, ret_val_fget_flags_368);
            mov(cc, get_ptr_for(jh, traits::FCSR),
                  gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
                  ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
                  ))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 147);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 148: FLE_D */
    continuation_e __fle_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fle_d"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FLE_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 148);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            InvokeNode* call_fcmp_d_369;
            jh.cc.comment("//call_fcmp_d");
            auto fcmp_d_369_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
            auto fcmp_d_369_arg1 = load_reg_from_mem(jh, traits::F0 + rs2);
            x86::Gp ret_val_fcmp_d_369 = get_reg_Gp(jh.cc, 64, true);
            jh.cc.invoke(&call_fcmp_d_369, &fcmp_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint32_t>());
            auto res = ret_val_fcmp_d_369;
            setArg(call_fcmp_d_369, 0, fcmp_d_369_arg0);
            setArg(call_fcmp_d_369, 1, fcmp_d_369_arg1);
            setArg(call_fcmp_d_369, 2, 1);
            setRet(call_fcmp_d_369, 0, ret_val_fcmp_d_369);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 32, false));
            }
            InvokeNode* call_fget_flags_370;
            jh.cc.comment("//call_fget_flags");
            x86::Gp ret_val_fget_flags_370 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fget_flags_370, &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_370;
            setRet(call_fget_flags_370, 0, ret_val_fget_flags_370);
            mov(cc, get_ptr_for(jh, traits::FCSR),
                  gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
                  ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
                  ))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 148);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 149: FCLASS_D */
    continuation_e __fclass_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "fclass_d"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCLASS_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 149);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){
                InvokeNode* call_fclass_d_371;
                jh.cc.comment("//call_fclass_d");
                auto fclass_d_371_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
                x86::Gp ret_val_fclass_d_371 = get_reg_Gp(jh.cc, 64, true);
                jh.cc.invoke(&call_fclass_d_371, &fclass_d, FuncSignature::build<uint64_t, uint64_t>());
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          ret_val_fclass_d_371, 32, false));
                setArg(call_fclass_d_371, 0, fclass_d_371_arg0);
                setRet(call_fclass_d_371, 0, ret_val_fclass_d_371);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 149);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 150: FCVT_W_D */
    continuation_e __fcvt_w_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_w_d"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT_W_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 150);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            InvokeNode* call_fcvt_64_32_372;
            jh.cc.comment("//call_fcvt_64_32");
            auto fcvt_64_32_372_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
            auto fcvt_64_32_372_arg2 = get_rm(jh, rm);
            x86::Gp ret_val_fcvt_64_32_372 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fcvt_64_32_372, &fcvt_64_32, FuncSignature::build<uint32_t, uint64_t, uint32_t, uint8_t>());
            auto res = gen_ext(cc, 
                gen_ext(cc, 
                    ret_val_fcvt_64_32_372, 32, false), 32, true);
            setArg(call_fcvt_64_32_372, 0, fcvt_64_32_372_arg0);
            setArg(call_fcvt_64_32_372, 1, 0);
            setArg(call_fcvt_64_32_372, 2, fcvt_64_32_372_arg2);
            setRet(call_fcvt_64_32_372, 0, ret_val_fcvt_64_32_372);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 32, true));
            }
            InvokeNode* call_fget_flags_373;
            jh.cc.comment("//call_fget_flags");
            x86::Gp ret_val_fget_flags_373 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fget_flags_373, &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_373;
            setRet(call_fget_flags_373, 0, ret_val_fget_flags_373);
            mov(cc, get_ptr_for(jh, traits::FCSR),
                  gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
                  ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
                  ))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 150);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 151: FCVT_WU_D */
    continuation_e __fcvt_wu_d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_wu_d"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT_WU_D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 151);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            InvokeNode* call_fcvt_64_32_374;
            jh.cc.comment("//call_fcvt_64_32");
            auto fcvt_64_32_374_arg0 = load_reg_from_mem(jh, traits::F0 + rs1);
            auto fcvt_64_32_374_arg2 = get_rm(jh, rm);
            x86::Gp ret_val_fcvt_64_32_374 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fcvt_64_32_374, &fcvt_64_32, FuncSignature::build<uint32_t, uint64_t, uint32_t, uint8_t>());
            auto res = gen_ext(cc, 
                gen_ext(cc, 
                    ret_val_fcvt_64_32_374, 32, false), 32, true);
            setArg(call_fcvt_64_32_374, 0, fcvt_64_32_374_arg0);
            setArg(call_fcvt_64_32_374, 1, 1);
            setArg(call_fcvt_64_32_374, 2, fcvt_64_32_374_arg2);
            setRet(call_fcvt_64_32_374, 0, ret_val_fcvt_64_32_374);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 32, true));
            }
            InvokeNode* call_fget_flags_375;
            jh.cc.comment("//call_fget_flags");
            x86::Gp ret_val_fget_flags_375 = get_reg_Gp(jh.cc, 32, true);
            jh.cc.invoke(&call_fget_flags_375, &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_375;
            setRet(call_fget_flags_375, 0, ret_val_fget_flags_375);
            mov(cc, get_ptr_for(jh, traits::FCSR),
                  gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))
                  ), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK))
                  ))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 151);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 152: FCVT_D_W */
    continuation_e __fcvt_d_w(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_d_w"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT_D_W_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 152);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            InvokeNode* call_fcvt_32_64_376;
            jh.cc.comment("//call_fcvt_32_64");
            auto fcvt_32_64_376_arg0 = gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs1), 32, false);
            auto fcvt_32_64_376_arg2 = get_rm(jh, rm);
            x86::Gp ret_val_fcvt_32_64_376 = get_reg_Gp(jh.cc, 64, true);
            jh.cc.invoke(&call_fcvt_32_64_376, &fcvt_32_64, FuncSignature::build<uint64_t, uint32_t, uint32_t, uint8_t>());
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  ret_val_fcvt_32_64_376);
            setArg(call_fcvt_32_64_376, 0, fcvt_32_64_376_arg0);
            setArg(call_fcvt_32_64_376, 1, 2);
            setArg(call_fcvt_32_64_376, 2, fcvt_32_64_376_arg2);
            setRet(call_fcvt_32_64_376, 0, ret_val_fcvt_32_64_376);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 152);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 153: FCVT_D_WU */
    continuation_e __fcvt_d_wu(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_d_wu"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT_D_WU_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 153);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            InvokeNode* call_fcvt_32_64_377;
            jh.cc.comment("//call_fcvt_32_64");
            auto fcvt_32_64_377_arg0 = gen_ext(cc, 
                load_reg_from_mem(jh, traits::X0 + rs1), 32, false);
            auto fcvt_32_64_377_arg2 = get_rm(jh, rm);
            x86::Gp ret_val_fcvt_32_64_377 = get_reg_Gp(jh.cc, 64, true);
            jh.cc.invoke(&call_fcvt_32_64_377, &fcvt_32_64, FuncSignature::build<uint64_t, uint32_t, uint32_t, uint8_t>());
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  ret_val_fcvt_32_64_377);
            setArg(call_fcvt_32_64_377, 0, fcvt_32_64_377_arg0);
            setArg(call_fcvt_32_64_377, 1, 3);
            setArg(call_fcvt_32_64_377, 2, fcvt_32_64_377_arg2);
            setRet(call_fcvt_32_64_377, 0, ret_val_fcvt_32_64_377);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 153);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 154: C__FLD */
    continuation_e __c__fld(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,2>(instr) << 6) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} f(8+{rd}), {uimm}({rs1})", fmt::arg("mnemonic", "c.fld"),
                fmt::arg("rd", rd), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__FLD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 154);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1+8), uimm)
            ), 32, false);
        auto res = gen_ext(cc, 
            gen_read_mem(jh, traits::MEM, offs, 8), 64, false);
        if(static_cast<uint32_t>(traits::FLEN)==64){
            mov(cc, get_ptr_for(jh, traits::F0+ rd+8),
                  res);
        }
        else{
            mov(cc, get_ptr_for(jh, traits::F0+ rd+8),
                  gen_operation(cc, bor, res, ((uint8_t)(- 1<<64)))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 154);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 155: C__FSD */
    continuation_e __c__fsd(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,2>(instr) << 6) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} f(8+{rs2}), {uimm}({rs1})", fmt::arg("mnemonic", "c.fsd"),
                fmt::arg("rs2", rs2), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__FSD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 155);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + rs1+8), uimm)
            ), 32, false);
        gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
            load_reg_from_mem(jh, traits::F0 + rs2+8), 64, false), 8);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 155);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 156: C__FLDSP */
    continuation_e __c__fldsp(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t uimm = ((bit_sub<2,3>(instr) << 6) | (bit_sub<5,2>(instr) << 3) | (bit_sub<12,1>(instr) << 5));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} f {rd}, {uimm}(x2)", fmt::arg("mnemonic", "c.fldsp"),
                fmt::arg("rd", rd), fmt::arg("uimm", uimm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__FLDSP_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 156);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + 2), uimm)
            ), 32, false);
        auto res = gen_ext(cc, 
            gen_read_mem(jh, traits::MEM, offs, 8), 64, false);
        if(static_cast<uint32_t>(traits::FLEN)==64){
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  res);
        }
        else{
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  gen_operation(cc, bor, res, ((uint8_t)(- 1<<64)))
                  );
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 156);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 157: C__FSDSP */
    continuation_e __c__fsdsp(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint16_t uimm = ((bit_sub<7,3>(instr) << 6) | (bit_sub<10,3>(instr) << 3));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} f {rs2}, {uimm}(x2), ", fmt::arg("mnemonic", "c.fsdsp"),
                fmt::arg("rs2", rs2), fmt::arg("uimm", uimm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__FSDSP_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 157);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem(jh, traits::X0 + 2), uimm)
            ), 32, false);
        gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
            load_reg_from_mem(jh, traits::F0 + rs2), 64, false), 8);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 157);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /****************************************************************************
     * end opcode definitions
     ****************************************************************************/
    continuation_e illegal_instruction(virt_addr_t &pc, code_word_t instr, jit_holder& jh ) {
        x86::Compiler& cc = jh.cc;
        if(this->disass_enabled){          
            auto mnemonic = std::string("illegal_instruction");
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);
        }
        cc.comment(fmt::format("illegal_instruction{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, instr_descr.size());
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc + ((instr & 3) == 3 ? 4 : 2);
        mov(cc, jh.next_pc, pc.val);
        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        gen_raise(jh, 0, 2);
        gen_sync(jh, POST_SYNC, instr_descr.size());
        gen_instr_epilogue(jh);
        return ILLEGAL_INSTR;
    }
};

template <typename ARCH> vm_impl<ARCH>::vm_impl() { this(new ARCH()); }

template <typename ARCH>
vm_impl<ARCH>::vm_impl(ARCH &core, unsigned core_id, unsigned cluster_id)
: vm_base<ARCH>(core, core_id, cluster_id)
, instr_decoder([this]() {
        std::vector<generic_instruction_descriptor> g_instr_descr;
        g_instr_descr.reserve(instr_descr.size());
        for (uint32_t i = 0; i < instr_descr.size(); ++i) {
            generic_instruction_descriptor new_instr_descr {instr_descr[i].value, instr_descr[i].mask, i};
            g_instr_descr.push_back(new_instr_descr);
        }
        return std::move(g_instr_descr);
    }()) {}

template <typename ARCH>
continuation_e vm_impl<ARCH>::gen_single_inst_behavior(virt_addr_t &pc, unsigned int &inst_cnt, jit_holder& jh) {
    enum {TRAP_ID=1<<16};
    code_word_t instr = 0;
    phys_addr_t paddr(pc);
    auto *const data = (uint8_t *)&instr;
    if(this->core.has_mmu())
        paddr = this->core.virt2phys(pc);
    auto res = this->core.read(paddr, 4, data);
    if (res != iss::Ok)
        return ILLEGAL_FETCH;
    if (instr == 0x0000006f || (instr&0xffff)==0xa001)
        return JUMP_TO_SELF;
    ++inst_cnt;
    uint32_t inst_index = instr_decoder.decode_instr(instr);
    compile_func f = nullptr;
    if(inst_index < instr_descr.size())
        f = instr_descr[inst_index].op;
    if (f == nullptr) 
        f = &this_class::illegal_instruction;
    return (this->*f)(pc, instr, jh);
}
template <typename ARCH>
void vm_impl<ARCH>::gen_instr_prologue(jit_holder& jh) {
    auto& cc = jh.cc;

    cc.comment("//gen_instr_prologue");

    x86_reg_t current_trap_state = get_reg_for(cc, traits::TRAP_STATE);
    mov(cc, current_trap_state, get_ptr_for(jh, traits::TRAP_STATE));
    mov(cc, get_ptr_for(jh, traits::PENDING_TRAP), current_trap_state);

}
template <typename ARCH>
void vm_impl<ARCH>::gen_instr_epilogue(jit_holder& jh) {
    auto& cc = jh.cc;

    cc.comment("//gen_instr_epilogue");
    x86_reg_t current_trap_state = get_reg_for(cc, traits::TRAP_STATE);
    mov(cc, current_trap_state, get_ptr_for(jh, traits::TRAP_STATE));
    cmp(cc, current_trap_state, 0);
    cc.jne(jh.trap_entry);
    cc.inc(get_ptr_for(jh, traits::ICOUNT));
}
template <typename ARCH>
void vm_impl<ARCH>::gen_block_prologue(jit_holder& jh){
    jh.pc = load_reg_from_mem_Gp(jh, traits::PC);
    jh.next_pc = load_reg_from_mem_Gp(jh, traits::NEXT_PC);
    jh.globals.resize(GLOBALS_SIZE);
    jh.globals[TVAL] = get_reg_Gp(jh.cc, 64, false);
}
template <typename ARCH>
void vm_impl<ARCH>::gen_block_epilogue(jit_holder& jh){
    x86::Compiler& cc = jh.cc;
    cc.comment("//gen_block_epilogue");
    cc.ret(jh.next_pc);

    cc.bind(jh.trap_entry);
    this->write_back(jh);

    x86::Gp current_trap_state = get_reg_for_Gp(cc, traits::TRAP_STATE);
    mov(cc, current_trap_state, get_ptr_for(jh, traits::TRAP_STATE));

    x86::Gp current_pc = get_reg_for_Gp(cc, traits::PC);
    mov(cc, current_pc, get_ptr_for(jh, traits::PC));

    cc.comment("//enter trap call;");
    InvokeNode* call_enter_trap;
    cc.invoke(&call_enter_trap, &enter_trap, FuncSignature::build<uint64_t, void*, uint64_t, uint64_t, uint64_t>());
    call_enter_trap->setArg(0, jh.arch_if_ptr);
    call_enter_trap->setArg(1, current_trap_state);
    call_enter_trap->setArg(2, current_pc);
    call_enter_trap->setArg(3, jh.globals[TVAL]);

    x86_reg_t current_next_pc = get_reg_for(cc, traits::NEXT_PC);
    mov(cc, current_next_pc, get_ptr_for(jh, traits::NEXT_PC));
    mov(cc, jh.next_pc, current_next_pc);

    mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(UNKNOWN_JUMP));
    cc.ret(jh.next_pc);
}
template <typename ARCH>
inline void vm_impl<ARCH>::gen_raise(jit_holder& jh, uint16_t trap_id, uint16_t cause) {
    auto& cc = jh.cc;
    cc.comment("//gen_raise");
    auto tmp1 = get_reg_for(cc, traits::TRAP_STATE);
    mov(cc, tmp1, 0x80ULL << 24 | (cause << 16) | trap_id);
    mov(cc, get_ptr_for(jh, traits::TRAP_STATE), tmp1);
    cc.jmp(jh.trap_entry);
}
template <typename ARCH>
template <typename T, typename>
void vm_impl<ARCH>::gen_set_tval(jit_holder& jh, T new_tval) {
        mov(jh.cc, jh.globals[TVAL], new_tval);
    }
template <typename ARCH>
void vm_impl<ARCH>::gen_set_tval(jit_holder& jh, x86_reg_t _new_tval) {
    if(nonstd::holds_alternative<x86::Gp>(_new_tval)) {
        x86::Gp new_tval = nonstd::get<x86::Gp>(_new_tval);
        if(new_tval.size() < 8)
            new_tval = gen_ext_Gp(jh.cc, new_tval, 64, false);
        mov(jh.cc, jh.globals[TVAL], new_tval);
    } else {
        throw std::runtime_error("Variant not supported in gen_set_tval");
    }
}

} // namespace tgc5c

template <>
std::unique_ptr<vm_if> create<arch::rv32gc>(arch::rv32gc *core, unsigned short port, bool dump) {
    auto ret = new rv32gc::vm_impl<arch::rv32gc>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret, port);
    return std::unique_ptr<vm_if>(ret);
}
} // namespace asmjit
} // namespace iss

#include <iss/arch/riscv_hart_m_p.h>
#include <iss/arch/riscv_hart_mu_p.h>
#include <iss/factory.h>
namespace iss {
namespace {
volatile std::array<bool, 2> dummy = {
        core_factory::instance().register_creator("rv32gc|m_p|asmjit", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_m_p<iss::arch::rv32gc>();
		    auto vm = new asmjit::rv32gc::vm_impl<arch::rv32gc>(*cpu, false);
		    if (port != 0) debugger::server<debugger::gdb_session>::run_server(vm, port);
            if(init_data){
                auto* cb = reinterpret_cast<semihosting_cb_t<arch::traits<arch::rv32gc>::reg_t>*>(init_data);
                cpu->set_semihosting_callback(*cb);
            }
            return {cpu_ptr{cpu}, vm_ptr{vm}};
        }),
        core_factory::instance().register_creator("rv32gc|mu_p|asmjit", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_mu_p<iss::arch::rv32gc>();
		    auto vm = new asmjit::rv32gc::vm_impl<arch::rv32gc>(*cpu, false);
		    if (port != 0) debugger::server<debugger::gdb_session>::run_server(vm, port);
            if(init_data){
                auto* cb = reinterpret_cast<semihosting_cb_t<arch::traits<arch::rv32gc>::reg_t>*>(init_data);
                cpu->set_semihosting_callback(*cb);
            }
            return {cpu_ptr{cpu}, vm_ptr{vm}};
        })
};
}
}
// clang-format on
