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
#include <iss/arch/rv32imacf.h>
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/server.h>
#include <iss/iss.h>
#include <iss/llvm/vm_base.h>
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
namespace llvm {
namespace fp_impl {
void add_fp_functions_2_module(::llvm::Module *, unsigned, unsigned);
}

namespace rv32imacf {
using namespace ::llvm;
using namespace iss::arch;
using namespace iss::debugger;

template <typename ARCH> class vm_impl : public iss::llvm::vm_base<ARCH> {
public:
    using traits = arch::traits<ARCH>;
    using super = typename iss::llvm::vm_base<ARCH>;
    using virt_addr_t = typename super::virt_addr_t;
    using phys_addr_t = typename super::phys_addr_t;
    using code_word_t = typename super::code_word_t;
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
    using vm_base<ARCH>::get_reg_ptr;

    inline const char *name(size_t index){return traits::reg_aliases.at(index);}

    inline const char *fname(size_t index){return index < 32?name(index+traits::F0):"illegal";}   

    template <typename T> inline ConstantInt *size(T type) {
        return ConstantInt::get(getContext(), APInt(32, type->getType()->getScalarSizeInBits()));
    }

    void setup_module(Module* m) override {
        super::setup_module(m);
        iss::llvm::fp_impl::add_fp_functions_2_module(m, traits::FP_REGS_SIZE, traits::XLEN);
    }

    inline Value *gen_choose(Value *cond, Value *trueVal, Value *falseVal, unsigned size) {
        return super::gen_cond_assign(cond, this->gen_ext(trueVal, size), this->gen_ext(falseVal, size));
    }

    std::tuple<continuation_e, BasicBlock *> gen_single_inst_behavior(virt_addr_t &, unsigned int &, BasicBlock *) override;

    void gen_leave_behavior(BasicBlock *leave_blk) override;
    void gen_raise_trap(uint16_t trap_id, uint16_t cause);
    void gen_leave_trap(unsigned lvl);
    void gen_wait(unsigned type);
    void set_tval(uint64_t new_tval);
    void set_tval(Value* new_tval);
    void gen_trap_behavior(BasicBlock *) override;
    void gen_instr_prologue();
    void gen_instr_epilogue(BasicBlock *bb);

    inline Value *gen_reg_load(unsigned i, unsigned level = 0) {
        return this->builder.CreateLoad(this->get_typeptr(i), get_reg_ptr(i), false);
    }

    inline void gen_set_pc(virt_addr_t pc, unsigned reg_num) {
        Value *next_pc_v = this->builder.CreateSExtOrTrunc(this->gen_const(traits::XLEN, pc.val),
                                                           this->get_type(traits::XLEN));
        this->builder.CreateStore(next_pc_v, get_reg_ptr(reg_num), true);
    }

    // some compile time constants

    using this_class = vm_impl<ARCH>;
    using compile_func = std::tuple<continuation_e, BasicBlock *> (this_class::*)(virt_addr_t &pc,
                                                                                  code_word_t instr,
                                                                                  BasicBlock *bb);
    template<unsigned W, typename U, typename S = typename std::make_signed<U>::type>
    inline S sext(U from) {
        auto mask = (1ULL<<W) - 1;
        auto sign_mask = 1ULL<<(W-1);
        return (from & mask) | ((from & sign_mask) ? ~mask : 0);
    }

    Value* get_rm(BasicBlock* bb, uint8_t get_rm_rm){
        Value* rm = this->gen_const(8, get_rm_rm);
        auto rm_eff =this->gen_choose(this->builder.CreateICmp(ICmpInst::ICMP_EQ,
           rm,
           this->gen_const(8,7))
        ,
        this->gen_slice(this->gen_reg_load(traits::FCSR), 5, 7-5+1),
        rm,
        8);
        {
        auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
        auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
        this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_UGT,
           rm_eff,
           this->gen_const(8,4))
        ), bb_then,  bb_merge);
        this->builder.SetInsertPoint(bb_then);
        {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        this->builder.CreateBr(bb_merge);
        this->builder.SetInsertPoint(bb_merge);
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

    const std::array<instruction_descriptor, 128> instr_descr = {{
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
    }};

    //needs to be declared after instr_descr
    decoder instr_decoder;

    /* instruction definitions */
    /* instruction 0: LUI */
    std::tuple<continuation_e, BasicBlock*> __lui(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint32_t imm = ((bit_sub<12,20>(instr) << 12));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "lui"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("LUI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,0);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_const(32,(uint32_t)((int32_t)imm)),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 0);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 1: AUIPC */
    std::tuple<continuation_e, BasicBlock*> __auipc(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint32_t imm = ((bit_sub<12,20>(instr) << 12));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#08x}", fmt::arg("mnemonic", "auipc"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("AUIPC_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,1);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_const(32,(uint32_t)(PC+(int32_t)imm)),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 1);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 2: JAL */
    std::tuple<continuation_e, BasicBlock*> __jal(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint32_t imm = ((bit_sub<12,8>(instr) << 12) | (bit_sub<20,1>(instr) << 11) | (bit_sub<21,10>(instr) << 1) | (bit_sub<31,1>(instr) << 20));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#0x}", fmt::arg("mnemonic", "jal"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("JAL_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,2);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto new_pc =(uint32_t)(PC+(int32_t)sext<21>(imm));
            if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){ this->set_tval(new_pc);
            this->gen_raise_trap(0, 0);
            }
            else{
                if(rd!=0) {
                    this->builder.CreateStore(
                    this->gen_const(32,(uint32_t)(PC+4)),
                    get_reg_ptr(rd + traits::X0), false);
                }
                auto PC_val_v = new_pc;
                this->builder.CreateStore(this->gen_const(32,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
                this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
            }
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 2);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 3: JALR */
    std::tuple<continuation_e, BasicBlock*> __jalr(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm:#0x}", fmt::arg("mnemonic", "jalr"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("JALR_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,3);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto addr_mask =(uint32_t)- 2;
            auto new_pc =this->gen_ext(
                (this->builder.CreateAnd(
                   (this->builder.CreateAdd(
                      this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
                      this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 64,true))
                   ),
                   this->gen_ext(this->gen_const(32,addr_mask), 64,false))
                ),
                32, true);
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            auto bb_else = BasicBlock::Create(this->mod->getContext(), "bb_else", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateURem(
               new_pc,
               this->gen_const(32,static_cast<uint32_t>(traits::INSTR_ALIGNMENT)))
            ), bb_then, bb_else);
            this->builder.SetInsertPoint(bb_then);
            {
                this->set_tval(new_pc);
                this->gen_raise_trap(0, 0);
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_else);
            {
                if(rd!=0) {
                    this->builder.CreateStore(
                    this->gen_const(32,(uint32_t)(PC+4)),
                    get_reg_ptr(rd + traits::X0), false);
                }
                auto PC_val_v = new_pc;
                this->builder.CreateStore(PC_val_v, get_reg_ptr(traits::NEXT_PC), false);                            
                this->builder.CreateStore(this->gen_const(32, static_cast<int>(UNKNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 3);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 4: BEQ */
    std::tuple<continuation_e, BasicBlock*> __beq(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "beq"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("BEQ_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,4);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_EQ,
               this->gen_reg_load(traits::X0+ rs1),
               this->gen_reg_load(traits::X0+ rs2))
            ), bb_then,  bb_merge);
            this->builder.SetInsertPoint(bb_then);
            {
                auto new_pc =(uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){ this->set_tval(new_pc);
                this->gen_raise_trap(0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    this->builder.CreateStore(this->gen_const(32,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
                    this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 4);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 5: BNE */
    std::tuple<continuation_e, BasicBlock*> __bne(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bne"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("BNE_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,5);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_NE,
               this->gen_reg_load(traits::X0+ rs1),
               this->gen_reg_load(traits::X0+ rs2))
            ), bb_then,  bb_merge);
            this->builder.SetInsertPoint(bb_then);
            {
                auto new_pc =(uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){ this->set_tval(new_pc);
                this->gen_raise_trap(0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    this->builder.CreateStore(this->gen_const(32,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
                    this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 5);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 6: BLT */
    std::tuple<continuation_e, BasicBlock*> __blt(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "blt"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("BLT_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,6);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_SLT,
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs1),
                   32, false),
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs2),
                   32, false))
            ), bb_then,  bb_merge);
            this->builder.SetInsertPoint(bb_then);
            {
                auto new_pc =(uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){ this->set_tval(new_pc);
                this->gen_raise_trap(0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    this->builder.CreateStore(this->gen_const(32,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
                    this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 6);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 7: BGE */
    std::tuple<continuation_e, BasicBlock*> __bge(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bge"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("BGE_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,7);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_SGE,
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs1),
                   32, false),
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs2),
                   32, false))
            ), bb_then,  bb_merge);
            this->builder.SetInsertPoint(bb_then);
            {
                auto new_pc =(uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){ this->set_tval(new_pc);
                this->gen_raise_trap(0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    this->builder.CreateStore(this->gen_const(32,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
                    this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 7);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 8: BLTU */
    std::tuple<continuation_e, BasicBlock*> __bltu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bltu"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("BLTU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,8);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_ULT,
               this->gen_reg_load(traits::X0+ rs1),
               this->gen_reg_load(traits::X0+ rs2))
            ), bb_then,  bb_merge);
            this->builder.SetInsertPoint(bb_then);
            {
                auto new_pc =(uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){ this->set_tval(new_pc);
                this->gen_raise_trap(0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    this->builder.CreateStore(this->gen_const(32,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
                    this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 8);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 9: BGEU */
    std::tuple<continuation_e, BasicBlock*> __bgeu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bgeu"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("BGEU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,9);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_UGE,
               this->gen_reg_load(traits::X0+ rs1),
               this->gen_reg_load(traits::X0+ rs2))
            ), bb_then,  bb_merge);
            this->builder.SetInsertPoint(bb_then);
            {
                auto new_pc =(uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){ this->set_tval(new_pc);
                this->gen_raise_trap(0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    this->builder.CreateStore(this->gen_const(32,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
                    this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 9);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 10: LB */
    std::tuple<continuation_e, BasicBlock*> __lb(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lb"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("LB_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,10);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto load_address =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 64,true))
                ),
                32, true);
            auto res =this->gen_ext(
                this->gen_read_mem(traits::MEM, load_address, 1),
                8, false);
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    res,
                    32, true),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 10);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 11: LH */
    std::tuple<continuation_e, BasicBlock*> __lh(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lh"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("LH_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,11);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto load_address =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 64,true))
                ),
                32, true);
            auto res =this->gen_ext(
                this->gen_read_mem(traits::MEM, load_address, 2),
                16, false);
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    res,
                    32, true),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 11);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 12: LW */
    std::tuple<continuation_e, BasicBlock*> __lw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lw"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("LW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,12);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto load_address =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 64,true))
                ),
                32, true);
            auto res =this->gen_ext(
                this->gen_read_mem(traits::MEM, load_address, 4),
                32, false);
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    res,
                    32, true),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 12);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 13: LBU */
    std::tuple<continuation_e, BasicBlock*> __lbu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lbu"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("LBU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,13);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto load_address =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 64,true))
                ),
                32, true);
            auto res =this->gen_read_mem(traits::MEM, load_address, 1);
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    res,
                    32, false),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 13);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 14: LHU */
    std::tuple<continuation_e, BasicBlock*> __lhu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lhu"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("LHU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,14);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto load_address =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 64,true))
                ),
                32, true);
            auto res =this->gen_read_mem(traits::MEM, load_address, 2);
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    res,
                    32, false),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 14);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 15: SB */
    std::tuple<continuation_e, BasicBlock*> __sb(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sb"),
                fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SB_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,15);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto store_address =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 64,true))
                ),
                32, true);
            this->gen_write_mem(traits::MEM,
            store_address,
            this->gen_ext(
                this->gen_reg_load(traits::X0+ rs2),
                8, false));
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 15);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 16: SH */
    std::tuple<continuation_e, BasicBlock*> __sh(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sh"),
                fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SH_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,16);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto store_address =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 64,true))
                ),
                32, true);
            this->gen_write_mem(traits::MEM,
            store_address,
            this->gen_ext(
                this->gen_reg_load(traits::X0+ rs2),
                16, false));
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 16);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 17: SW */
    std::tuple<continuation_e, BasicBlock*> __sw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sw"),
                fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,17);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto store_address =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 64,true))
                ),
                32, true);
            this->gen_write_mem(traits::MEM,
            store_address,
            this->gen_ext(
                this->gen_reg_load(traits::X0+ rs2),
                32, false));
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 17);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 18: ADDI */
    std::tuple<continuation_e, BasicBlock*> __addi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "addi"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("ADDI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,18);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    (this->builder.CreateAdd(
                       this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
                       this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 64,true))
                    ),
                    32, true),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 18);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 19: SLTI */
    std::tuple<continuation_e, BasicBlock*> __slti(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "slti"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SLTI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,19);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(this->gen_choose((this->builder.CreateICmp(ICmpInst::ICMP_SLT,
                   this->gen_ext(
                       this->gen_reg_load(traits::X0+ rs1), 32,true),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 32,true))
                ),
                this->gen_const(8,1),
                this->gen_const(8,0),
                8), 32),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 19);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 20: SLTIU */
    std::tuple<continuation_e, BasicBlock*> __sltiu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "sltiu"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SLTIU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,20);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(this->gen_choose((this->builder.CreateICmp(ICmpInst::ICMP_ULT,
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_const(32,(uint32_t)((int16_t)sext<12>(imm))))
                ),
                this->gen_const(8,1),
                this->gen_const(8,0),
                8), 32),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 20);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 21: XORI */
    std::tuple<continuation_e, BasicBlock*> __xori(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "xori"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("XORI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,21);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateXor(
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_const(32,(uint32_t)((int16_t)sext<12>(imm))))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 21);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 22: ORI */
    std::tuple<continuation_e, BasicBlock*> __ori(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "ori"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("ORI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,22);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateOr(
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_const(32,(uint32_t)((int16_t)sext<12>(imm))))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 22);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 23: ANDI */
    std::tuple<continuation_e, BasicBlock*> __andi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "andi"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("ANDI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,23);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateAnd(
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_const(32,(uint32_t)((int16_t)sext<12>(imm))))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 23);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 24: SLLI */
    std::tuple<continuation_e, BasicBlock*> __slli(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "slli"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SLLI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,24);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateShl(
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_ext(this->gen_const(8,shamt), 32,false))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 24);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 25: SRLI */
    std::tuple<continuation_e, BasicBlock*> __srli(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srli"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SRLI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,25);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateLShr(
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_ext(this->gen_const(8,shamt), 32,false))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 25);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 26: SRAI */
    std::tuple<continuation_e, BasicBlock*> __srai(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srai"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SRAI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,26);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    (this->builder.CreateAShr(
                       this->gen_ext(
                           this->gen_reg_load(traits::X0+ rs1), 32,true),
                       this->gen_ext(this->gen_const(8,shamt), 32,false))
                    ), 32,false),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 26);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 27: ADD */
    std::tuple<continuation_e, BasicBlock*> __add(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "add"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("ADD_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,27);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    (this->builder.CreateAdd(
                       this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
                       this->gen_ext(this->gen_reg_load(traits::X0+ rs2), 64,false))
                    ),
                    32, false),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 27);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 28: SUB */
    std::tuple<continuation_e, BasicBlock*> __sub(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sub"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SUB_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,28);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    (this->builder.CreateSub(
                       this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
                       this->gen_ext(this->gen_reg_load(traits::X0+ rs2), 64,false))
                    ),
                    32, true),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 28);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 29: SLL */
    std::tuple<continuation_e, BasicBlock*> __sll(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sll"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SLL_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,29);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(this->builder.CreateShl(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
                   (this->builder.CreateAnd(
                      this->gen_ext(this->gen_reg_load(traits::X0+ rs2), 64,false),
                      this->gen_const(64,(static_cast<uint32_t>(traits::XLEN)-1)))
                   ))
                , 32, false),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 29);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 30: SLT */
    std::tuple<continuation_e, BasicBlock*> __slt(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "slt"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SLT_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,30);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(this->gen_choose(this->builder.CreateICmp(ICmpInst::ICMP_SLT,
                   this->gen_ext(
                       this->gen_reg_load(traits::X0+ rs1), 32,true),
                   this->gen_ext(
                       this->gen_reg_load(traits::X0+ rs2), 32,true))
                ,
                this->gen_const(8,1),
                this->gen_const(8,0),
                8), 32),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 30);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 31: SLTU */
    std::tuple<continuation_e, BasicBlock*> __sltu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sltu"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SLTU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,31);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(this->gen_choose(this->builder.CreateICmp(ICmpInst::ICMP_ULT,
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_reg_load(traits::X0+ rs2))
                ,
                this->gen_const(8,1),
                this->gen_const(8,0),
                8), 32),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 31);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 32: XOR */
    std::tuple<continuation_e, BasicBlock*> __xor(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "xor"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("XOR_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,32);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateXor(
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_reg_load(traits::X0+ rs2))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 32);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 33: SRL */
    std::tuple<continuation_e, BasicBlock*> __srl(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "srl"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SRL_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,33);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(this->builder.CreateLShr(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
                   (this->builder.CreateAnd(
                      this->gen_ext(this->gen_reg_load(traits::X0+ rs2), 64,false),
                      this->gen_const(64,(static_cast<uint32_t>(traits::XLEN)-1)))
                   ))
                , 32, false),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 33);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 34: SRA */
    std::tuple<continuation_e, BasicBlock*> __sra(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sra"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SRA_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,34);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    (this->gen_ext(this->builder.CreateAShr(
                       this->gen_ext(this->gen_ext(
                           this->gen_reg_load(traits::X0+ rs1), 32,true), 64,true),
                       (this->builder.CreateAnd(
                          this->gen_ext(this->gen_reg_load(traits::X0+ rs2), 64,false),
                          this->gen_const(64,(static_cast<uint32_t>(traits::XLEN)-1)))
                       ))
                    , 32, true)),
                    32, true),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 34);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 35: OR */
    std::tuple<continuation_e, BasicBlock*> __or(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "or"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("OR_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,35);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateOr(
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_reg_load(traits::X0+ rs2))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 35);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 36: AND */
    std::tuple<continuation_e, BasicBlock*> __and(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "and"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("AND_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,36);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateAnd(
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_reg_load(traits::X0+ rs2))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 36);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 37: FENCE */
    std::tuple<continuation_e, BasicBlock*> __fence(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t succ = ((bit_sub<20,4>(instr)));
        uint8_t pred = ((bit_sub<24,4>(instr)));
        uint8_t fm = ((bit_sub<28,4>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {pred}, {succ} ({fm} , {rs1}, {rd})", fmt::arg("mnemonic", "fence"),
                fmt::arg("pred", pred), fmt::arg("succ", succ), fmt::arg("fm", fm), fmt::arg("rs1", name(rs1)), fmt::arg("rd", name(rd)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FENCE_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,37);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->gen_write_mem(traits::FENCE,
        static_cast<uint32_t>(traits::fence),
        this->gen_const(8,(uint8_t)pred<<4|succ));
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 37);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 38: ECALL */
    std::tuple<continuation_e, BasicBlock*> __ecall(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "ecall";
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("ECALL_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,38);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        this->gen_raise_trap(0, 11);
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(TRAP,nullptr);
        
        this->gen_sync(POST_SYNC, 38);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 39: EBREAK */
    std::tuple<continuation_e, BasicBlock*> __ebreak(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "ebreak";
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("EBREAK_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,39);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        this->gen_raise_trap(0, 3);
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(TRAP,nullptr);
        
        this->gen_sync(POST_SYNC, 39);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 40: MRET */
    std::tuple<continuation_e, BasicBlock*> __mret(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "mret";
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("MRET_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,40);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        this->gen_leave_trap(3);
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(TRAP,nullptr);
        
        this->gen_sync(POST_SYNC, 40);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 41: WFI */
    std::tuple<continuation_e, BasicBlock*> __wfi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "wfi";
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("WFI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,41);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        std::vector<Value*> wait_379_args{
            this->gen_ext(this->gen_const(8,1), 32)
        };
        this->builder.CreateCall(this->mod->getFunction("wait"), wait_379_args);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 41);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 42: CSRRW */
    std::tuple<continuation_e, BasicBlock*> __csrrw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {rs1}", fmt::arg("mnemonic", "csrrw"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("CSRRW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,42);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrs1 =this->gen_reg_load(traits::X0+ rs1);
            if(rd!=0){ auto xrd =this->gen_read_mem(traits::CSR, csr, 4);
            this->gen_write_mem(traits::CSR,
            csr,
            xrs1);
            this->builder.CreateStore(
            xrd,
            get_reg_ptr(rd + traits::X0), false);
            }
            else{
                this->gen_write_mem(traits::CSR,
                csr,
                xrs1);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 42);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 43: CSRRS */
    std::tuple<continuation_e, BasicBlock*> __csrrs(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {rs1}", fmt::arg("mnemonic", "csrrs"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("CSRRS_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,43);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrd =this->gen_read_mem(traits::CSR, csr, 4);
            auto xrs1 =this->gen_reg_load(traits::X0+ rs1);
            if(rs1!=0) {
                this->gen_write_mem(traits::CSR,
                csr,
                this->builder.CreateOr(
                   xrd,
                   xrs1)
                );
            }
            if(rd!=0) {
                this->builder.CreateStore(
                xrd,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 43);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 44: CSRRC */
    std::tuple<continuation_e, BasicBlock*> __csrrc(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {rs1}", fmt::arg("mnemonic", "csrrc"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("CSRRC_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,44);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrd =this->gen_read_mem(traits::CSR, csr, 4);
            auto xrs1 =this->gen_reg_load(traits::X0+ rs1);
            if(rs1!=0) {
                this->gen_write_mem(traits::CSR,
                csr,
                this->builder.CreateAnd(
                   xrd,
                   this->builder.CreateNot(xrs1))
                );
            }
            if(rd!=0) {
                this->builder.CreateStore(
                xrd,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 44);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 45: CSRRWI */
    std::tuple<continuation_e, BasicBlock*> __csrrwi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t zimm = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {zimm:#0x}", fmt::arg("mnemonic", "csrrwi"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("zimm", zimm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("CSRRWI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,45);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrd =this->gen_read_mem(traits::CSR, csr, 4);
            this->gen_write_mem(traits::CSR,
            csr,
            this->gen_const(32,(uint32_t)zimm));
            if(rd!=0) {
                this->builder.CreateStore(
                xrd,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 45);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 46: CSRRSI */
    std::tuple<continuation_e, BasicBlock*> __csrrsi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t zimm = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {zimm:#0x}", fmt::arg("mnemonic", "csrrsi"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("zimm", zimm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("CSRRSI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,46);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrd =this->gen_read_mem(traits::CSR, csr, 4);
            if(zimm!=0) {
                this->gen_write_mem(traits::CSR,
                csr,
                this->builder.CreateOr(
                   xrd,
                   this->gen_const(32,(uint32_t)zimm))
                );
            }
            if(rd!=0) {
                this->builder.CreateStore(
                xrd,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 46);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 47: CSRRCI */
    std::tuple<continuation_e, BasicBlock*> __csrrci(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t zimm = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {zimm:#0x}", fmt::arg("mnemonic", "csrrci"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("zimm", zimm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("CSRRCI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,47);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrd =this->gen_read_mem(traits::CSR, csr, 4);
            if(zimm!=0) {
                this->gen_write_mem(traits::CSR,
                csr,
                this->builder.CreateAnd(
                   xrd,
                   this->gen_const(32,~ ((uint32_t)zimm)))
                );
            }
            if(rd!=0) {
                this->builder.CreateStore(
                xrd,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 47);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 48: FENCE_I */
    std::tuple<continuation_e, BasicBlock*> __fence_i(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rd}, {imm}", fmt::arg("mnemonic", "fence_i"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FENCE_I_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,48);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->gen_write_mem(traits::FENCE,
        static_cast<uint32_t>(traits::fencei),
        this->gen_const(16,imm));
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(FLUSH,nullptr);
        
        this->gen_sync(POST_SYNC, 48);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 49: MUL */
    std::tuple<continuation_e, BasicBlock*> __mul(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mul"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("MUL_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,49);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto res =this->builder.CreateMul(
               this->gen_ext(this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs1), 32,true), 64,true),
               this->gen_ext(this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs2), 32,true), 64,true))
            ;
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    res,
                    32, true),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 49);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 50: MULH */
    std::tuple<continuation_e, BasicBlock*> __mulh(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mulh"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("MULH_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,50);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto res =this->builder.CreateMul(
               this->gen_ext(this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs1), 32,true), 64,true),
               this->gen_ext(this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs2), 32,true), 64,true))
            ;
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    (this->builder.CreateAShr(
                       res,
                       this->gen_ext(this->gen_const(32,static_cast<uint32_t>(traits::XLEN)), 64,false))
                    ),
                    32, true),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 50);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 51: MULHSU */
    std::tuple<continuation_e, BasicBlock*> __mulhsu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mulhsu"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("MULHSU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,51);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto res =this->builder.CreateMul(
               this->gen_ext(this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs1), 32,true), 64,true),
               this->gen_ext(this->gen_reg_load(traits::X0+ rs2), 64,false))
            ;
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    (this->builder.CreateAShr(
                       res,
                       this->gen_ext(this->gen_const(32,static_cast<uint32_t>(traits::XLEN)), 64,false))
                    ),
                    32, true),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 51);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 52: MULHU */
    std::tuple<continuation_e, BasicBlock*> __mulhu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mulhu"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("MULHU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,52);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto res =this->builder.CreateMul(
               this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
               this->gen_ext(this->gen_reg_load(traits::X0+ rs2), 64,false))
            ;
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    (this->builder.CreateLShr(
                       res,
                       this->gen_ext(this->gen_const(32,static_cast<uint32_t>(traits::XLEN)), 64,false))
                    ),
                    32, false),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 52);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 53: DIV */
    std::tuple<continuation_e, BasicBlock*> __div(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "div"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("DIV_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,53);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto dividend =this->gen_ext(
                this->gen_reg_load(traits::X0+ rs1), 32,true);
            auto divisor =this->gen_ext(
                this->gen_reg_load(traits::X0+ rs2), 32,true);
            if(rd!=0){ {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            auto bb_else = BasicBlock::Create(this->mod->getContext(), "bb_else", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_NE,
               divisor,
               this->gen_ext(this->gen_const(8,0), 32,false))
            ), bb_then, bb_else);
            this->builder.SetInsertPoint(bb_then);
            {
                auto MMIN =((uint32_t)1)<<(static_cast<uint32_t>(traits::XLEN)-1);
                {
                auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
                auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
                auto bb_else = BasicBlock::Create(this->mod->getContext(), "bb_else", this->func, bb_merge);
                this->builder.CreateCondBr(this->gen_bool(this->builder.CreateAnd(
                   this->builder.CreateICmp(ICmpInst::ICMP_EQ,
                      this->gen_reg_load(traits::X0+ rs1),
                      this->gen_const(32,MMIN))
                   ,
                   this->builder.CreateICmp(ICmpInst::ICMP_EQ,
                      divisor,
                      this->gen_ext(this->gen_const(8,- 1), 32,true))
                   )
                ), bb_then, bb_else);
                this->builder.SetInsertPoint(bb_then);
                {
                    this->builder.CreateStore(
                    this->gen_const(32,MMIN),
                    get_reg_ptr(rd + traits::X0), false);
                }
                this->builder.CreateBr(bb_merge);
                this->builder.SetInsertPoint(bb_else);
                {
                    this->builder.CreateStore(
                    this->gen_ext(
                        (this->builder.CreateSDiv(
                           this->gen_ext(dividend, 64,true),
                           this->gen_ext(divisor, 64,true))
                        ),
                        32, true),
                    get_reg_ptr(rd + traits::X0), false);
                }
                this->builder.CreateBr(bb_merge);
                this->builder.SetInsertPoint(bb_merge);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_else);
            {
                this->builder.CreateStore(
                this->gen_const(32,(uint32_t)- 1),
                get_reg_ptr(rd + traits::X0), false);
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 53);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 54: DIVU */
    std::tuple<continuation_e, BasicBlock*> __divu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "divu"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("DIVU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,54);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            auto bb_else = BasicBlock::Create(this->mod->getContext(), "bb_else", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_NE,
               this->gen_reg_load(traits::X0+ rs2),
               this->gen_ext(this->gen_const(8,0), 32,false))
            ), bb_then, bb_else);
            this->builder.SetInsertPoint(bb_then);
            {
                if(rd!=0) {
                    this->builder.CreateStore(
                    this->builder.CreateUDiv(
                       this->gen_reg_load(traits::X0+ rs1),
                       this->gen_reg_load(traits::X0+ rs2))
                    ,
                    get_reg_ptr(rd + traits::X0), false);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_else);
            {
                if(rd!=0) {
                    this->builder.CreateStore(
                    this->gen_const(32,(uint32_t)- 1),
                    get_reg_ptr(rd + traits::X0), false);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 54);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 55: REM */
    std::tuple<continuation_e, BasicBlock*> __rem(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "rem"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("REM_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,55);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            auto bb_else = BasicBlock::Create(this->mod->getContext(), "bb_else", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_NE,
               this->gen_reg_load(traits::X0+ rs2),
               this->gen_ext(this->gen_const(8,0), 32,false))
            ), bb_then, bb_else);
            this->builder.SetInsertPoint(bb_then);
            {
                auto MMIN =(uint32_t)1<<(static_cast<uint32_t>(traits::XLEN)-1);
                {
                auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
                auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
                auto bb_else = BasicBlock::Create(this->mod->getContext(), "bb_else", this->func, bb_merge);
                this->builder.CreateCondBr(this->gen_bool(this->builder.CreateAnd(
                   this->builder.CreateICmp(ICmpInst::ICMP_EQ,
                      this->gen_reg_load(traits::X0+ rs1),
                      this->gen_const(32,MMIN))
                   ,
                   this->builder.CreateICmp(ICmpInst::ICMP_EQ,
                      this->gen_ext(
                          this->gen_reg_load(traits::X0+ rs2),
                          32, false),
                      this->gen_ext(this->gen_const(8,- 1), 32,true))
                   )
                ), bb_then, bb_else);
                this->builder.SetInsertPoint(bb_then);
                {
                    if(rd!=0) {
                        this->builder.CreateStore(
                        this->gen_ext(this->gen_const(8,0), 32),
                        get_reg_ptr(rd + traits::X0), false);
                    }
                }
                this->builder.CreateBr(bb_merge);
                this->builder.SetInsertPoint(bb_else);
                {
                    if(rd!=0) {
                        this->builder.CreateStore(
                        this->gen_ext(
                            (this->builder.CreateSRem(
                               this->gen_ext(
                                   this->gen_reg_load(traits::X0+ rs1), 32,true),
                               this->gen_ext(
                                   this->gen_reg_load(traits::X0+ rs2), 32,true))
                            ), 32,false),
                        get_reg_ptr(rd + traits::X0), false);
                    }
                }
                this->builder.CreateBr(bb_merge);
                this->builder.SetInsertPoint(bb_merge);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_else);
            {
                if(rd!=0) {
                    this->builder.CreateStore(
                    this->gen_reg_load(traits::X0+ rs1),
                    get_reg_ptr(rd + traits::X0), false);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 55);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 56: REMU */
    std::tuple<continuation_e, BasicBlock*> __remu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "remu"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("REMU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,56);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            auto bb_else = BasicBlock::Create(this->mod->getContext(), "bb_else", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_NE,
               this->gen_reg_load(traits::X0+ rs2),
               this->gen_ext(this->gen_const(8,0), 32,false))
            ), bb_then, bb_else);
            this->builder.SetInsertPoint(bb_then);
            {
                if(rd!=0) {
                    this->builder.CreateStore(
                    this->builder.CreateURem(
                       this->gen_reg_load(traits::X0+ rs1),
                       this->gen_reg_load(traits::X0+ rs2))
                    ,
                    get_reg_ptr(rd + traits::X0), false);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_else);
            {
                if(rd!=0) {
                    this->builder.CreateStore(
                    this->gen_reg_load(traits::X0+ rs1),
                    get_reg_ptr(rd + traits::X0), false);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 56);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 57: LRW */
    std::tuple<continuation_e, BasicBlock*> __lrw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {aq}, {rl}", fmt::arg("mnemonic", "lrw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("aq", name(aq)), fmt::arg("rl", name(rl)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("LRW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,57);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){ auto offs =this->gen_reg_load(traits::X0+ rs1);
            this->builder.CreateStore(
            this->gen_ext(
                (this->gen_ext(
                    this->gen_read_mem(traits::MEM, offs, 4),
                    32, false)),
                32, true),
            get_reg_ptr(rd + traits::X0), false);
            this->gen_write_mem(traits::RES,
            offs,
            this->gen_const(8,(uint8_t)- 1));
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 57);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 58: SCW */
    std::tuple<continuation_e, BasicBlock*> __scw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {aq}, {rl}", fmt::arg("mnemonic", "scw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", name(aq)), fmt::arg("rl", name(rl)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SCW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,58);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs =this->gen_reg_load(traits::X0+ rs1);
            auto res1 =this->gen_read_mem(traits::RES, offs, 1);
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_NE,
               res1,
               this->gen_ext(this->gen_const(8,0), 32,false))
            ), bb_then,  bb_merge);
            this->builder.SetInsertPoint(bb_then);
            {
                this->gen_write_mem(traits::MEM,
                offs,
                this->gen_ext(
                    this->gen_reg_load(traits::X0+ rs2),
                    32, false));
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(this->gen_choose(res1,
                this->gen_const(8,0),
                this->gen_const(8,1),
                8), 32),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 58);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 59: AMOSWAPW */
    std::tuple<continuation_e, BasicBlock*> __amoswapw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoswapw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("AMOSWAPW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,59);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs =this->gen_reg_load(traits::X0+ rs1);
            auto res =this->gen_reg_load(traits::X0+ rs2);
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    (this->gen_ext(
                        this->gen_read_mem(traits::MEM, offs, 4),
                        32, false)),
                    32, true),
                get_reg_ptr(rd + traits::X0), false);
            }
            this->gen_write_mem(traits::MEM,
            offs,
            this->gen_ext(
                res,
                32, false));
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 59);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 60: AMOADDW */
    std::tuple<continuation_e, BasicBlock*> __amoaddw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoaddw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("AMOADDW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,60);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs =this->gen_reg_load(traits::X0+ rs1);
            auto res1 =this->gen_ext(
                this->gen_read_mem(traits::MEM, offs, 4),
                32, false);
            auto res2 =this->builder.CreateAdd(
               this->gen_ext(res1, 64,true),
               this->gen_ext(this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs2),
                   32, false), 64,true))
            ;
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    res1,
                    32, true),
                get_reg_ptr(rd + traits::X0), false);
            }
            this->gen_write_mem(traits::MEM,
            offs,
            this->gen_ext(
                res2,
                32, true));
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 60);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 61: AMOXORW */
    std::tuple<continuation_e, BasicBlock*> __amoxorw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoxorw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("AMOXORW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,61);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs =this->gen_reg_load(traits::X0+ rs1);
            auto res1 =this->gen_read_mem(traits::MEM, offs, 4);
            auto res2 =this->builder.CreateXor(
               res1,
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs2),
                   32, false))
            ;
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    this->gen_ext(
                        this->gen_ext(
                            res1, 32,true),
                        32, true), 32,false),
                get_reg_ptr(rd + traits::X0), false);
            }
            this->gen_write_mem(traits::MEM,
            offs,
            res2);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 61);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 62: AMOANDW */
    std::tuple<continuation_e, BasicBlock*> __amoandw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoandw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("AMOANDW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,62);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs =this->gen_reg_load(traits::X0+ rs1);
            auto res1 =this->gen_read_mem(traits::MEM, offs, 4);
            auto res2 =this->builder.CreateAnd(
               res1,
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs2),
                   32, false))
            ;
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    this->gen_ext(
                        this->gen_ext(
                            res1, 32,true),
                        32, true), 32,false),
                get_reg_ptr(rd + traits::X0), false);
            }
            this->gen_write_mem(traits::MEM,
            offs,
            res2);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 62);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 63: AMOORW */
    std::tuple<continuation_e, BasicBlock*> __amoorw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoorw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("AMOORW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,63);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs =this->gen_reg_load(traits::X0+ rs1);
            auto res1 =this->gen_read_mem(traits::MEM, offs, 4);
            auto res2 =this->builder.CreateOr(
               res1,
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs2),
                   32, false))
            ;
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    this->gen_ext(
                        this->gen_ext(
                            res1, 32,true),
                        32, true), 32,false),
                get_reg_ptr(rd + traits::X0), false);
            }
            this->gen_write_mem(traits::MEM,
            offs,
            res2);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 63);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 64: AMOMINW */
    std::tuple<continuation_e, BasicBlock*> __amominw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amominw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("AMOMINW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,64);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs =this->gen_reg_load(traits::X0+ rs1);
            auto res1 =this->gen_ext(
                this->gen_read_mem(traits::MEM, offs, 4),
                32, false);
            auto res2 =this->gen_choose(this->builder.CreateICmp(ICmpInst::ICMP_SGT,
               res1,
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs2),
                   32, false))
            ,
            this->gen_ext(
                this->gen_reg_load(traits::X0+ rs2),
                32, false),
            this->gen_ext(
                res1, 32,false),
            32);
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    res1,
                    32, true),
                get_reg_ptr(rd + traits::X0), false);
            }
            this->gen_write_mem(traits::MEM,
            offs,
            res2);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 64);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 65: AMOMAXW */
    std::tuple<continuation_e, BasicBlock*> __amomaxw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amomaxw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("AMOMAXW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,65);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs =this->gen_reg_load(traits::X0+ rs1);
            auto res1 =this->gen_ext(
                this->gen_read_mem(traits::MEM, offs, 4),
                32, false);
            auto res2 =this->gen_choose(this->builder.CreateICmp(ICmpInst::ICMP_SLT,
               res1,
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs2),
                   32, false))
            ,
            this->gen_ext(
                this->gen_reg_load(traits::X0+ rs2),
                32, false),
            this->gen_ext(
                res1, 32,false),
            32);
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    res1,
                    32, true),
                get_reg_ptr(rd + traits::X0), false);
            }
            this->gen_write_mem(traits::MEM,
            offs,
            res2);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 65);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 66: AMOMINUW */
    std::tuple<continuation_e, BasicBlock*> __amominuw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amominuw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("AMOMINUW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,66);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs =this->gen_reg_load(traits::X0+ rs1);
            auto res1 =this->gen_read_mem(traits::MEM, offs, 4);
            auto res2 =this->gen_choose(this->builder.CreateICmp(ICmpInst::ICMP_UGT,
               res1,
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs2),
                   32, false))
            ,
            this->gen_ext(
                this->gen_reg_load(traits::X0+ rs2),
                32, false),
            res1,
            32);
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    this->gen_ext(
                        this->gen_ext(
                            res1, 32,true),
                        32, true), 32,false),
                get_reg_ptr(rd + traits::X0), false);
            }
            this->gen_write_mem(traits::MEM,
            offs,
            res2);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 66);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 67: AMOMAXUW */
    std::tuple<continuation_e, BasicBlock*> __amomaxuw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amomaxuw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("AMOMAXUW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,67);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs =this->gen_reg_load(traits::X0+ rs1);
            auto res1 =this->gen_read_mem(traits::MEM, offs, 4);
            auto res2 =this->gen_choose(this->builder.CreateICmp(ICmpInst::ICMP_ULT,
               res1,
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs2),
                   32, false))
            ,
            this->gen_ext(
                this->gen_reg_load(traits::X0+ rs2),
                32, false),
            res1,
            32);
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    this->gen_ext(
                        this->gen_ext(
                            res1, 32,true),
                        32, true), 32,false),
                get_reg_ptr(rd + traits::X0), false);
            }
            this->gen_write_mem(traits::MEM,
            offs,
            res2);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 67);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 68: C__ADDI4SPN */
    std::tuple<continuation_e, BasicBlock*> __c__addi4spn(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<2,3>(instr)));
        uint16_t imm = ((bit_sub<5,1>(instr) << 3) | (bit_sub<6,1>(instr) << 2) | (bit_sub<7,4>(instr) << 6) | (bit_sub<11,2>(instr) << 4));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c.addi4spn"),
                fmt::arg("rd", name(8+rd)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__ADDI4SPN_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,68);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(imm) {
            this->builder.CreateStore(
            this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ 2), 64,false),
                   this->gen_ext(this->gen_const(16,imm), 64,false))
                ),
                32, false),
            get_reg_ptr(rd+8 + traits::X0), false);
        }
        else{
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 68);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 69: C__LW */
    std::tuple<continuation_e, BasicBlock*> __c__lw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {uimm:#05x}({rs1})", fmt::arg("mnemonic", "c.lw"),
                fmt::arg("rd", name(8+rd)), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__LW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,69);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        auto offs =this->gen_ext(
            (this->builder.CreateAdd(
               this->gen_ext(this->gen_reg_load(traits::X0+ rs1+8), 64,false),
               this->gen_ext(this->gen_const(8,uimm), 64,false))
            ),
            32, false);
        this->builder.CreateStore(
        this->gen_ext(
            this->gen_ext(
                this->gen_read_mem(traits::MEM, offs, 4),
                32, false),
            32, true),
        get_reg_ptr(rd+8 + traits::X0), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 69);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 70: C__SW */
    std::tuple<continuation_e, BasicBlock*> __c__sw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {uimm:#05x}({rs1})", fmt::arg("mnemonic", "c.sw"),
                fmt::arg("rs2", name(8+rs2)), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__SW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,70);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        auto offs =this->gen_ext(
            (this->builder.CreateAdd(
               this->gen_ext(this->gen_reg_load(traits::X0+ rs1+8), 64,false),
               this->gen_ext(this->gen_const(8,uimm), 64,false))
            ),
            32, false);
        this->gen_write_mem(traits::MEM,
        offs,
        this->gen_ext(
            this->gen_reg_load(traits::X0+ rs2+8),
            32, false));
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 70);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 71: C__ADDI */
    std::tuple<continuation_e, BasicBlock*> __c__addi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t imm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.addi"),
                fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__ADDI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,71);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rs1!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    (this->builder.CreateAdd(
                       this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
                       this->gen_ext(this->gen_const(8,(int8_t)sext<6>(imm)), 64,true))
                    ),
                    32, true),
                get_reg_ptr(rs1 + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 71);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 72: C__NOP */
    std::tuple<continuation_e, BasicBlock*> __c__nop(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t nzimm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "c.nop";
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__NOP_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,72);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 72);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 73: C__JAL */
    std::tuple<continuation_e, BasicBlock*> __c__jal(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,3>(instr) << 1) | (bit_sub<6,1>(instr) << 7) | (bit_sub<7,1>(instr) << 6) | (bit_sub<8,1>(instr) << 10) | (bit_sub<9,2>(instr) << 8) | (bit_sub<11,1>(instr) << 4) | (bit_sub<12,1>(instr) << 11));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {imm:#05x}", fmt::arg("mnemonic", "c.jal"),
                fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__JAL_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,73);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        this->builder.CreateStore(
        this->gen_const(32,(uint32_t)(PC+2)),
        get_reg_ptr(1 + traits::X0), false);
        auto PC_val_v = (uint32_t)(PC+(int16_t)sext<12>(imm));
        this->builder.CreateStore(this->gen_const(32,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
        this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 73);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 74: C__LI */
    std::tuple<continuation_e, BasicBlock*> __c__li(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t imm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c.li"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__LI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,74);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_const(32,(uint32_t)((int8_t)sext<6>(imm))),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 74);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 75: C__LUI */
    std::tuple<continuation_e, BasicBlock*> __c__lui(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint32_t imm = ((bit_sub<2,5>(instr) << 12) | (bit_sub<12,1>(instr) << 17));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c.lui"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__LUI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,75);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(imm==0||rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        if(rd!=0) {
            this->builder.CreateStore(
            this->gen_const(32,(uint32_t)((int32_t)sext<18>(imm))),
            get_reg_ptr(rd + traits::X0), false);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 75);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 76: C__ADDI16SP */
    std::tuple<continuation_e, BasicBlock*> __c__addi16sp(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t nzimm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 7) | (bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 4) | (bit_sub<12,1>(instr) << 9));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {nzimm:#05x}", fmt::arg("mnemonic", "c.addi16sp"),
                fmt::arg("nzimm", nzimm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__ADDI16SP_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,76);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(nzimm) {
            this->builder.CreateStore(
            this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ 2), 64,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<10>(nzimm)), 64,true))
                ),
                32, true),
            get_reg_ptr(2 + traits::X0), false);
        }
        else{
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 76);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 77: __reserved_clui */
    std::tuple<continuation_e, BasicBlock*> ____reserved_clui(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = ".reserved_clui";
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("__reserved_clui_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,77);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 77);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 78: C__SRLI */
    std::tuple<continuation_e, BasicBlock*> __c__srli(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t shamt = ((bit_sub<2,5>(instr)));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {shamt}", fmt::arg("mnemonic", "c.srli"),
                fmt::arg("rs1", name(8+rs1)), fmt::arg("shamt", shamt));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__SRLI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,78);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(
        this->builder.CreateLShr(
           this->gen_reg_load(traits::X0+ rs1+8),
           this->gen_ext(this->gen_const(8,shamt), 32,false))
        ,
        get_reg_ptr(rs1+8 + traits::X0), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 78);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 79: C__SRAI */
    std::tuple<continuation_e, BasicBlock*> __c__srai(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t shamt = ((bit_sub<2,5>(instr)));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {shamt}", fmt::arg("mnemonic", "c.srai"),
                fmt::arg("rs1", name(8+rs1)), fmt::arg("shamt", shamt));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__SRAI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,79);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(shamt){ this->builder.CreateStore(
        this->gen_ext(
            (this->builder.CreateAShr(
               (this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs1+8),
                   32, false)),
               this->gen_ext(this->gen_const(8,shamt), 32,false))
            ),
            32, true),
        get_reg_ptr(rs1+8 + traits::X0), false);
        }
        else{
            if(static_cast<uint32_t>(traits::XLEN)==128){ this->builder.CreateStore(
            this->gen_ext(
                (this->builder.CreateAShr(
                   (this->gen_ext(
                       this->gen_reg_load(traits::X0+ rs1+8),
                       32, false)),
                   this->gen_ext(this->gen_const(8,64), 32,false))
                ),
                32, true),
            get_reg_ptr(rs1+8 + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 79);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 80: C__ANDI */
    std::tuple<continuation_e, BasicBlock*> __c__andi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t imm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.andi"),
                fmt::arg("rs1", name(8+rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__ANDI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,80);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(
        this->gen_ext(
            (this->builder.CreateAnd(
               this->gen_reg_load(traits::X0+ rs1+8),
               this->gen_ext(this->gen_const(8,(int8_t)sext<6>(imm)), 32,true))
            ),
            32, true),
        get_reg_ptr(rs1+8 + traits::X0), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 80);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 81: C__SUB */
    std::tuple<continuation_e, BasicBlock*> __c__sub(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.sub"),
                fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__SUB_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,81);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(
        this->gen_ext(
            (this->builder.CreateSub(
               this->gen_ext(this->gen_reg_load(traits::X0+ rd+8), 64,false),
               this->gen_ext(this->gen_reg_load(traits::X0+ rs2+8), 64,false))
            ),
            32, true),
        get_reg_ptr(rd+8 + traits::X0), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 81);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 82: C__XOR */
    std::tuple<continuation_e, BasicBlock*> __c__xor(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.xor"),
                fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__XOR_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,82);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(
        this->builder.CreateXor(
           this->gen_reg_load(traits::X0+ rd+8),
           this->gen_reg_load(traits::X0+ rs2+8))
        ,
        get_reg_ptr(rd+8 + traits::X0), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 82);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 83: C__OR */
    std::tuple<continuation_e, BasicBlock*> __c__or(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.or"),
                fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__OR_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,83);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(
        this->builder.CreateOr(
           this->gen_reg_load(traits::X0+ rd+8),
           this->gen_reg_load(traits::X0+ rs2+8))
        ,
        get_reg_ptr(rd+8 + traits::X0), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 83);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 84: C__AND */
    std::tuple<continuation_e, BasicBlock*> __c__and(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.and"),
                fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__AND_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,84);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(
        this->builder.CreateAnd(
           this->gen_reg_load(traits::X0+ rd+8),
           this->gen_reg_load(traits::X0+ rs2+8))
        ,
        get_reg_ptr(rd+8 + traits::X0), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 84);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 85: C__J */
    std::tuple<continuation_e, BasicBlock*> __c__j(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,3>(instr) << 1) | (bit_sub<6,1>(instr) << 7) | (bit_sub<7,1>(instr) << 6) | (bit_sub<8,1>(instr) << 10) | (bit_sub<9,2>(instr) << 8) | (bit_sub<11,1>(instr) << 4) | (bit_sub<12,1>(instr) << 11));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {imm:#05x}", fmt::arg("mnemonic", "c.j"),
                fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__J_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,85);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        auto PC_val_v = (uint32_t)(PC+(int16_t)sext<12>(imm));
        this->builder.CreateStore(this->gen_const(32,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
        this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 85);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 86: C__BEQZ */
    std::tuple<continuation_e, BasicBlock*> __c__beqz(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 1) | (bit_sub<5,2>(instr) << 6) | (bit_sub<10,2>(instr) << 3) | (bit_sub<12,1>(instr) << 8));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.beqz"),
                fmt::arg("rs1", name(8+rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__BEQZ_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,86);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        {
        auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
        auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
        this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_EQ,
           this->gen_reg_load(traits::X0+ rs1+8),
           this->gen_ext(this->gen_const(8,0), 32,false))
        ), bb_then,  bb_merge);
        this->builder.SetInsertPoint(bb_then);
        {
            auto PC_val_v = (uint32_t)(PC+(int16_t)sext<9>(imm));
            this->builder.CreateStore(this->gen_const(32,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
            this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        }
        this->builder.CreateBr(bb_merge);
        this->builder.SetInsertPoint(bb_merge);
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 86);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 87: C__BNEZ */
    std::tuple<continuation_e, BasicBlock*> __c__bnez(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 1) | (bit_sub<5,2>(instr) << 6) | (bit_sub<10,2>(instr) << 3) | (bit_sub<12,1>(instr) << 8));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.bnez"),
                fmt::arg("rs1", name(8+rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__BNEZ_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,87);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        {
        auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
        auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
        this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_NE,
           this->gen_reg_load(traits::X0+ rs1+8),
           this->gen_ext(this->gen_const(8,0), 32,false))
        ), bb_then,  bb_merge);
        this->builder.SetInsertPoint(bb_then);
        {
            auto PC_val_v = (uint32_t)(PC+(int16_t)sext<9>(imm));
            this->builder.CreateStore(this->gen_const(32,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
            this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        }
        this->builder.CreateBr(bb_merge);
        this->builder.SetInsertPoint(bb_merge);
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 87);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 88: C__SLLI */
    std::tuple<continuation_e, BasicBlock*> __c__slli(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t nzuimm = ((bit_sub<2,5>(instr)));
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {nzuimm}", fmt::arg("mnemonic", "c.slli"),
                fmt::arg("rs1", name(rs1)), fmt::arg("nzuimm", nzuimm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__SLLI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,88);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rs1!=0) {
                this->builder.CreateStore(
                this->builder.CreateShl(
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_ext(this->gen_const(8,nzuimm), 32,false))
                ,
                get_reg_ptr(rs1 + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 88);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 89: C__LWSP */
    std::tuple<continuation_e, BasicBlock*> __c__lwsp(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t uimm = ((bit_sub<2,2>(instr) << 6) | (bit_sub<4,3>(instr) << 2) | (bit_sub<12,1>(instr) << 5));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, sp, {uimm:#05x}", fmt::arg("mnemonic", "c.lwsp"),
                fmt::arg("rd", name(rd)), fmt::arg("uimm", uimm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__LWSP_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,89);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rd==0) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ 2), 64,false),
                   this->gen_ext(this->gen_const(8,uimm), 64,false))
                ),
                32, false);
            this->builder.CreateStore(
            this->gen_ext(
                this->gen_ext(
                    this->gen_read_mem(traits::MEM, offs, 4),
                    32, false),
                32, true),
            get_reg_ptr(rd + traits::X0), false);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 89);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 90: C__MV */
    std::tuple<continuation_e, BasicBlock*> __c__mv(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.mv"),
                fmt::arg("rd", name(rd)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__MV_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,90);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_reg_load(traits::X0+ rs2),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 90);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 91: C__JR */
    std::tuple<continuation_e, BasicBlock*> __c__jr(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}", fmt::arg("mnemonic", "c.jr"),
                fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__JR_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,91);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rs1&&rs1<static_cast<uint32_t>(traits::RFS)){ auto addr_mask =(uint32_t)- 2;
        auto PC_val_v = this->builder.CreateAnd(
           this->gen_reg_load(traits::X0+ rs1%static_cast<uint32_t>(traits::RFS)),
           this->gen_const(32,addr_mask))
        ;
        this->builder.CreateStore(PC_val_v, get_reg_ptr(traits::NEXT_PC), false);                            
        this->builder.CreateStore(this->gen_const(32, static_cast<int>(UNKNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        }
        else{
            this->gen_raise_trap(0, 2);
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 91);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 92: __reserved_cmv */
    std::tuple<continuation_e, BasicBlock*> ____reserved_cmv(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = ".reserved_cmv";
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("__reserved_cmv_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,92);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->gen_raise_trap(0, 2);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 92);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 93: C__ADD */
    std::tuple<continuation_e, BasicBlock*> __c__add(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.add"),
                fmt::arg("rd", name(rd)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__ADD_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,93);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    (this->builder.CreateAdd(
                       this->gen_ext(this->gen_reg_load(traits::X0+ rd), 64,false),
                       this->gen_ext(this->gen_reg_load(traits::X0+ rs2), 64,false))
                    ),
                    32, false),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 93);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 94: C__JALR */
    std::tuple<continuation_e, BasicBlock*> __c__jalr(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}", fmt::arg("mnemonic", "c.jalr"),
                fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__JALR_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,94);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto addr_mask =(uint32_t)- 2;
            auto new_pc =this->gen_reg_load(traits::X0+ rs1);
            this->builder.CreateStore(
            this->gen_const(32,(uint32_t)(PC+2)),
            get_reg_ptr(1 + traits::X0), false);
            auto PC_val_v = this->builder.CreateAnd(
               new_pc,
               this->gen_const(32,addr_mask))
            ;
            this->builder.CreateStore(PC_val_v, get_reg_ptr(traits::NEXT_PC), false);                            
            this->builder.CreateStore(this->gen_const(32, static_cast<int>(UNKNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 94);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 95: C__EBREAK */
    std::tuple<continuation_e, BasicBlock*> __c__ebreak(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "c.ebreak";
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__EBREAK_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,95);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->gen_raise_trap(0, 3);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 95);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 96: C__SWSP */
    std::tuple<continuation_e, BasicBlock*> __c__swsp(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t uimm = ((bit_sub<7,2>(instr) << 6) | (bit_sub<9,4>(instr) << 2));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {uimm:#05x}(sp)", fmt::arg("mnemonic", "c.swsp"),
                fmt::arg("rs2", name(rs2)), fmt::arg("uimm", uimm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__SWSP_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,96);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ 2), 64,false),
                   this->gen_ext(this->gen_const(8,uimm), 64,false))
                ),
                32, false);
            this->gen_write_mem(traits::MEM,
            offs,
            this->gen_ext(
                this->gen_reg_load(traits::X0+ rs2),
                32, false));
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 96);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 97: DII */
    std::tuple<continuation_e, BasicBlock*> __dii(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "dii";
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("DII_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,97);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 97);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 98: FLW */
    std::tuple<continuation_e, BasicBlock*> __flw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "flw"),
                fmt::arg("rd", fname(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FLW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,98);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 64,true))
                ),
                32, true);
            this->builder.CreateStore(
            this->gen_ext(
                this->gen_read_mem(traits::MEM, offs, 4),
                32, false),
            get_reg_ptr(rd + traits::F0), false);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 98);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 99: FSW */
    std::tuple<continuation_e, BasicBlock*> __fsw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "fsw"),
                fmt::arg("rs2", fname(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FSW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,99);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 64,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 64,true))
                ),
                32, true);
            this->gen_write_mem(traits::MEM,
            offs,
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs2),
                32, false));
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 99);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 100: FMADD__S */
    std::tuple<continuation_e, BasicBlock*> __fmadd__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rs3 = ((bit_sub<27,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}, {rs3}", fmt::arg("mnemonic", "fmadd.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FMADD__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,100);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        std::vector<Value*> fmadd_s_423_args{
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs1),
                32, false),
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs2),
                32, false),
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs3),
                32, false),
            this->gen_ext(this->gen_const(8,0), 32),
            this->gen_ext(get_rm(bb, (uint8_t)rm), 8)
        };
        this->builder.CreateStore(
        this->gen_ext(
            this->builder.CreateCall(this->mod->getFunction("fmadd_s"), fmadd_s_423_args),
            32, false),
        get_reg_ptr(rd + traits::F0), false);
        std::vector<Value*> fget_flags_425_args{
        };
        auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_425_args);
        this->builder.CreateStore(
        this->builder.CreateOr(
           (this->builder.CreateAnd(
              this->gen_reg_load(traits::FCSR),
              this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
           ),
           (this->builder.CreateAnd(
              flags,
              this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
           ))
        ,
        get_reg_ptr(traits::FCSR), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 100);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 101: FMSUB__S */
    std::tuple<continuation_e, BasicBlock*> __fmsub__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rs3 = ((bit_sub<27,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}, {rs3}", fmt::arg("mnemonic", "fmsub.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FMSUB__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,101);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        std::vector<Value*> fmadd_s_426_args{
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs1),
                32, false),
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs2),
                32, false),
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs3),
                32, false),
            this->gen_ext(this->gen_const(8,1), 32),
            this->gen_ext(get_rm(bb, (uint8_t)rm), 8)
        };
        this->builder.CreateStore(
        this->builder.CreateCall(this->mod->getFunction("fmadd_s"), fmadd_s_426_args),
        get_reg_ptr(rd + traits::F0), false);
        std::vector<Value*> fget_flags_428_args{
        };
        auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_428_args);
        this->builder.CreateStore(
        this->builder.CreateOr(
           (this->builder.CreateAnd(
              this->gen_reg_load(traits::FCSR),
              this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
           ),
           (this->builder.CreateAnd(
              flags,
              this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
           ))
        ,
        get_reg_ptr(traits::FCSR), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 101);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 102: FNMADD__S */
    std::tuple<continuation_e, BasicBlock*> __fnmadd__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rs3 = ((bit_sub<27,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, name(rd), {rs1}, {rs2}, {rs3}", fmt::arg("mnemonic", "fnmadd.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FNMADD__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,102);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        std::vector<Value*> fmadd_s_429_args{
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs1),
                32, false),
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs2),
                32, false),
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs3),
                32, false),
            this->gen_ext(this->gen_const(8,2), 32),
            this->gen_ext(get_rm(bb, (uint8_t)rm), 8)
        };
        this->builder.CreateStore(
        this->builder.CreateCall(this->mod->getFunction("fmadd_s"), fmadd_s_429_args),
        get_reg_ptr(rd + traits::F0), false);
        std::vector<Value*> fget_flags_431_args{
        };
        auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_431_args);
        this->builder.CreateStore(
        this->builder.CreateOr(
           (this->builder.CreateAnd(
              this->gen_reg_load(traits::FCSR),
              this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
           ),
           (this->builder.CreateAnd(
              flags,
              this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
           ))
        ,
        get_reg_ptr(traits::FCSR), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 102);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 103: FNMSUB__S */
    std::tuple<continuation_e, BasicBlock*> __fnmsub__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rs3 = ((bit_sub<27,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}, {rs3}", fmt::arg("mnemonic", "fnmsub.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FNMSUB__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,103);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        std::vector<Value*> fmadd_s_432_args{
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs1),
                32, false),
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs2),
                32, false),
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs3),
                32, false),
            this->gen_ext(this->gen_const(8,3), 32),
            this->gen_ext(get_rm(bb, (uint8_t)rm), 8)
        };
        this->builder.CreateStore(
        this->builder.CreateCall(this->mod->getFunction("fmadd_s"), fmadd_s_432_args),
        get_reg_ptr(rd + traits::F0), false);
        std::vector<Value*> fget_flags_434_args{
        };
        auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_434_args);
        this->builder.CreateStore(
        this->builder.CreateOr(
           (this->builder.CreateAnd(
              this->gen_reg_load(traits::FCSR),
              this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
           ),
           (this->builder.CreateAnd(
              flags,
              this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
           ))
        ,
        get_reg_ptr(traits::FCSR), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 103);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 104: FADD__S */
    std::tuple<continuation_e, BasicBlock*> __fadd__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fadd.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FADD__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,104);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        std::vector<Value*> fadd_s_435_args{
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs1),
                32, false),
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs2),
                32, false),
            this->gen_ext(get_rm(bb, (uint8_t)rm), 8)
        };
        this->builder.CreateStore(
        this->builder.CreateCall(this->mod->getFunction("fadd_s"), fadd_s_435_args),
        get_reg_ptr(rd + traits::F0), false);
        std::vector<Value*> fget_flags_437_args{
        };
        auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_437_args);
        this->builder.CreateStore(
        this->builder.CreateOr(
           (this->builder.CreateAnd(
              this->gen_reg_load(traits::FCSR),
              this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
           ),
           (this->builder.CreateAnd(
              flags,
              this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
           ))
        ,
        get_reg_ptr(traits::FCSR), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 104);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 105: FSUB__S */
    std::tuple<continuation_e, BasicBlock*> __fsub__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsub.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FSUB__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,105);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        std::vector<Value*> fsub_s_438_args{
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs1),
                32, false),
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs2),
                32, false),
            this->gen_ext(get_rm(bb, (uint8_t)rm), 8)
        };
        this->builder.CreateStore(
        this->builder.CreateCall(this->mod->getFunction("fsub_s"), fsub_s_438_args),
        get_reg_ptr(rd + traits::F0), false);
        std::vector<Value*> fget_flags_440_args{
        };
        auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_440_args);
        this->builder.CreateStore(
        this->builder.CreateOr(
           (this->builder.CreateAnd(
              this->gen_reg_load(traits::FCSR),
              this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
           ),
           (this->builder.CreateAnd(
              flags,
              this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
           ))
        ,
        get_reg_ptr(traits::FCSR), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 105);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 106: FMUL__S */
    std::tuple<continuation_e, BasicBlock*> __fmul__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fmul.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FMUL__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,106);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        std::vector<Value*> fmul_s_441_args{
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs1),
                32, false),
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs2),
                32, false),
            this->gen_ext(get_rm(bb, (uint8_t)rm), 8)
        };
        this->builder.CreateStore(
        this->builder.CreateCall(this->mod->getFunction("fmul_s"), fmul_s_441_args),
        get_reg_ptr(rd + traits::F0), false);
        std::vector<Value*> fget_flags_443_args{
        };
        auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_443_args);
        this->builder.CreateStore(
        this->builder.CreateOr(
           (this->builder.CreateAnd(
              this->gen_reg_load(traits::FCSR),
              this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
           ),
           (this->builder.CreateAnd(
              flags,
              this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
           ))
        ,
        get_reg_ptr(traits::FCSR), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 106);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 107: FDIV__S */
    std::tuple<continuation_e, BasicBlock*> __fdiv__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fdiv.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FDIV__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,107);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        std::vector<Value*> fdiv_s_444_args{
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs1),
                32, false),
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs2),
                32, false),
            this->gen_ext(get_rm(bb, (uint8_t)rm), 8)
        };
        this->builder.CreateStore(
        this->builder.CreateCall(this->mod->getFunction("fdiv_s"), fdiv_s_444_args),
        get_reg_ptr(rd + traits::F0), false);
        std::vector<Value*> fget_flags_446_args{
        };
        auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_446_args);
        this->builder.CreateStore(
        this->builder.CreateOr(
           (this->builder.CreateAnd(
              this->gen_reg_load(traits::FCSR),
              this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
           ),
           (this->builder.CreateAnd(
              flags,
              this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
           ))
        ,
        get_reg_ptr(traits::FCSR), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 107);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 108: FSQRT__S */
    std::tuple<continuation_e, BasicBlock*> __fsqrt__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fsqrt.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FSQRT__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,108);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        std::vector<Value*> fsqrt_s_447_args{
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs1),
                32, false),
            this->gen_ext(get_rm(bb, (uint8_t)rm), 8)
        };
        this->builder.CreateStore(
        this->builder.CreateCall(this->mod->getFunction("fsqrt_s"), fsqrt_s_447_args),
        get_reg_ptr(rd + traits::F0), false);
        std::vector<Value*> fget_flags_449_args{
        };
        auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_449_args);
        this->builder.CreateStore(
        this->builder.CreateOr(
           (this->builder.CreateAnd(
              this->gen_reg_load(traits::FCSR),
              this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
           ),
           (this->builder.CreateAnd(
              flags,
              this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
           ))
        ,
        get_reg_ptr(traits::FCSR), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 108);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 109: FSGNJ__S */
    std::tuple<continuation_e, BasicBlock*> __fsgnj__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnj.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FSGNJ__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,109);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(
        this->builder.CreateOr(this->builder.CreateShl(this->gen_ext(this->gen_slice(this->gen_reg_load(traits::F0+ rs2), 31, 31-31+1), 32), 31), this->gen_slice(this->gen_reg_load(traits::F0+ rs1), 0, 30-0+1)),
        get_reg_ptr(rd + traits::F0), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 109);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 110: FSGNJN__S */
    std::tuple<continuation_e, BasicBlock*> __fsgnjn__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnjn.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FSGNJN__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,110);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(
        this->builder.CreateOr(this->builder.CreateShl(this->gen_ext(this->builder.CreateNot(this->gen_slice(this->gen_reg_load(traits::F0+ rs2), 31, 31-31+1)), 32), 31), this->gen_slice(this->gen_reg_load(traits::F0+ rs1), 0, 30-0+1)),
        get_reg_ptr(rd + traits::F0), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 110);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 111: FSGNJX__S */
    std::tuple<continuation_e, BasicBlock*> __fsgnjx__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnjx.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FSGNJX__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,111);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(
        this->builder.CreateXor(
           this->gen_reg_load(traits::F0+ rs1),
           (this->builder.CreateAnd(
              this->gen_reg_load(traits::F0+ rs2),
              this->gen_const(32,(uint32_t)2147483648))
           ))
        ,
        get_reg_ptr(rd + traits::F0), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 111);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 112: FMIN__S */
    std::tuple<continuation_e, BasicBlock*> __fmin__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fmin.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FMIN__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,112);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        std::vector<Value*> fsel_s_450_args{
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs1),
                32, false),
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs2),
                32, false),
            this->gen_ext(this->gen_const(8,0), 32)
        };
        this->builder.CreateStore(
        this->builder.CreateCall(this->mod->getFunction("fsel_s"), fsel_s_450_args),
        get_reg_ptr(rd + traits::F0), false);
        std::vector<Value*> fget_flags_451_args{
        };
        auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_451_args);
        this->builder.CreateStore(
        this->builder.CreateOr(
           (this->builder.CreateAnd(
              this->gen_reg_load(traits::FCSR),
              this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
           ),
           (this->builder.CreateAnd(
              flags,
              this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
           ))
        ,
        get_reg_ptr(traits::FCSR), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 112);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 113: FMAX__S */
    std::tuple<continuation_e, BasicBlock*> __fmax__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fmax.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FMAX__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,113);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        std::vector<Value*> fsel_s_452_args{
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs1),
                32, false),
            this->gen_ext(
                this->gen_reg_load(traits::F0+ rs2),
                32, false),
            this->gen_ext(this->gen_const(8,1), 32)
        };
        this->builder.CreateStore(
        this->builder.CreateCall(this->mod->getFunction("fsel_s"), fsel_s_452_args),
        get_reg_ptr(rd + traits::F0), false);
        std::vector<Value*> fget_flags_453_args{
        };
        auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_453_args);
        this->builder.CreateStore(
        this->builder.CreateOr(
           (this->builder.CreateAnd(
              this->gen_reg_load(traits::FCSR),
              this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
           ),
           (this->builder.CreateAnd(
              flags,
              this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
           ))
        ,
        get_reg_ptr(traits::FCSR), false);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 113);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 114: FCVT__W__S */
    std::tuple<continuation_e, BasicBlock*> __fcvt__w__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt.w.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FCVT__W__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,114);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            std::vector<Value*> fcvt_s_455_args{
                this->gen_ext(
                    this->gen_reg_load(traits::F0+ rs1),
                    32, false),
                this->gen_ext(this->gen_const(8,0), 32),
                this->gen_ext(get_rm(bb, (uint8_t)rm), 8)
            };
            auto res =this->builder.CreateCall(this->mod->getFunction("fcvt_s"), fcvt_s_455_args);
            if((rd)!=0) {
                this->builder.CreateStore(
                res,
                get_reg_ptr(rd + traits::X0), false);
            }
            std::vector<Value*> fget_flags_457_args{
            };
            auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_457_args);
            this->builder.CreateStore(
            this->builder.CreateOr(
               (this->builder.CreateAnd(
                  this->gen_reg_load(traits::FCSR),
                  this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
               ),
               (this->builder.CreateAnd(
                  flags,
                  this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
               ))
            ,
            get_reg_ptr(traits::FCSR), false);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 114);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 115: FCVT__WU__S */
    std::tuple<continuation_e, BasicBlock*> __fcvt__wu__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt.wu.s"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FCVT__WU__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,115);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            std::vector<Value*> fcvt_s_459_args{
                this->gen_ext(
                    this->gen_reg_load(traits::F0+ rs1),
                    32, false),
                this->gen_ext(this->gen_const(8,1), 32),
                this->gen_ext(get_rm(bb, (uint8_t)rm), 8)
            };
            auto res =this->builder.CreateCall(this->mod->getFunction("fcvt_s"), fcvt_s_459_args);
            if((rd)!=0) {
                this->builder.CreateStore(
                res,
                get_reg_ptr(rd + traits::X0), false);
            }
            std::vector<Value*> fget_flags_461_args{
            };
            auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_461_args);
            this->builder.CreateStore(
            this->builder.CreateOr(
               (this->builder.CreateAnd(
                  this->gen_reg_load(traits::FCSR),
                  this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
               ),
               (this->builder.CreateAnd(
                  flags,
                  this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
               ))
            ,
            get_reg_ptr(traits::FCSR), false);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 115);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 116: FEQ__S */
    std::tuple<continuation_e, BasicBlock*> __feq__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "feq.s"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FEQ__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,116);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            std::vector<Value*> fcmp_s_463_args{
                this->gen_ext(
                    this->gen_reg_load(traits::F0+ rs1),
                    32, false),
                this->gen_ext(
                    this->gen_reg_load(traits::F0+ rs2),
                    32, false),
                this->gen_ext(this->gen_const(8,0), 32)
            };
            auto res =this->builder.CreateCall(this->mod->getFunction("fcmp_s"), fcmp_s_463_args);
            if((rd)!=0) {
                this->builder.CreateStore(
                res,
                get_reg_ptr(rd + traits::X0), false);
            }
            std::vector<Value*> fget_flags_464_args{
            };
            auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_464_args);
            this->builder.CreateStore(
            this->builder.CreateOr(
               (this->builder.CreateAnd(
                  this->gen_reg_load(traits::FCSR),
                  this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
               ),
               (this->builder.CreateAnd(
                  flags,
                  this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
               ))
            ,
            get_reg_ptr(traits::FCSR), false);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 116);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 117: FLT__S */
    std::tuple<continuation_e, BasicBlock*> __flt__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "flt.s"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FLT__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,117);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            std::vector<Value*> fcmp_s_466_args{
                this->gen_ext(
                    this->gen_reg_load(traits::F0+ rs1),
                    32, false),
                this->gen_ext(
                    this->gen_reg_load(traits::F0+ rs2),
                    32, false),
                this->gen_ext(this->gen_const(8,2), 32)
            };
            auto res =this->builder.CreateCall(this->mod->getFunction("fcmp_s"), fcmp_s_466_args);
            if((rd)!=0) {
                this->builder.CreateStore(
                res,
                get_reg_ptr(rd + traits::X0), false);
            }
            std::vector<Value*> fget_flags_467_args{
            };
            auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_467_args);
            this->builder.CreateStore(
            this->builder.CreateOr(
               (this->builder.CreateAnd(
                  this->gen_reg_load(traits::FCSR),
                  this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
               ),
               (this->builder.CreateAnd(
                  flags,
                  this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
               ))
            ,
            get_reg_ptr(traits::FCSR), false);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 117);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 118: FLE__S */
    std::tuple<continuation_e, BasicBlock*> __fle__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fle.s"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FLE__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,118);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            std::vector<Value*> fcmp_s_469_args{
                this->gen_ext(
                    this->gen_reg_load(traits::F0+ rs1),
                    32, false),
                this->gen_ext(
                    this->gen_reg_load(traits::F0+ rs2),
                    32, false),
                this->gen_ext(this->gen_const(8,1), 32)
            };
            auto res =this->builder.CreateCall(this->mod->getFunction("fcmp_s"), fcmp_s_469_args);
            if((rd)!=0) {
                this->builder.CreateStore(
                res,
                get_reg_ptr(rd + traits::X0), false);
            }
            std::vector<Value*> fget_flags_470_args{
            };
            auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_470_args);
            this->builder.CreateStore(
            this->builder.CreateOr(
               (this->builder.CreateAnd(
                  this->gen_reg_load(traits::FCSR),
                  this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
               ),
               (this->builder.CreateAnd(
                  flags,
                  this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
               ))
            ,
            get_reg_ptr(traits::FCSR), false);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 118);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 119: FCLASS__S */
    std::tuple<continuation_e, BasicBlock*> __fclass__s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "fclass.s"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FCLASS__S_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,119);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            std::vector<Value*> fclass_s_472_args{
                this->gen_ext(
                    this->gen_reg_load(traits::F0+ rs1),
                    32, false)
            };
            auto res =this->builder.CreateCall(this->mod->getFunction("fclass_s"), fclass_s_472_args);
            if((rd)!=0) {
                this->builder.CreateStore(
                res,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 119);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 120: FCVT__S__W */
    std::tuple<continuation_e, BasicBlock*> __fcvt__s__w(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt.s.w"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FCVT__S__W_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,120);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            std::vector<Value*> fcvt_s_474_args{
                this->gen_ext(
                    this->gen_reg_load(traits::X0+ rs1),
                    32, false),
                this->gen_ext(this->gen_const(8,2), 32),
                this->gen_ext(get_rm(bb, (uint8_t)rm), 8)
            };
            this->builder.CreateStore(
            this->builder.CreateCall(this->mod->getFunction("fcvt_s"), fcvt_s_474_args),
            get_reg_ptr(rd + traits::F0), false);
            std::vector<Value*> fget_flags_476_args{
            };
            auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_476_args);
            this->builder.CreateStore(
            this->builder.CreateOr(
               (this->builder.CreateAnd(
                  this->gen_reg_load(traits::FCSR),
                  this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
               ),
               (this->builder.CreateAnd(
                  flags,
                  this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
               ))
            ,
            get_reg_ptr(traits::FCSR), false);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 120);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 121: FCVT__S__WU */
    std::tuple<continuation_e, BasicBlock*> __fcvt__s__wu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt.s.wu"),
                fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FCVT__S__WU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,121);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            std::vector<Value*> fcvt_s_478_args{
                this->gen_ext(
                    this->gen_reg_load(traits::X0+ rs1),
                    32, false),
                this->gen_ext(this->gen_const(8,3), 32),
                this->gen_ext(get_rm(bb, (uint8_t)rm), 8)
            };
            this->builder.CreateStore(
            this->builder.CreateCall(this->mod->getFunction("fcvt_s"), fcvt_s_478_args),
            get_reg_ptr(rd + traits::F0), false);
            std::vector<Value*> fget_flags_480_args{
            };
            auto flags =this->builder.CreateCall(this->mod->getFunction("fget_flags"), fget_flags_480_args);
            this->builder.CreateStore(
            this->builder.CreateOr(
               (this->builder.CreateAnd(
                  this->gen_reg_load(traits::FCSR),
                  this->gen_const(32,~ static_cast<uint32_t>(traits::FFLAG_MASK)))
               ),
               (this->builder.CreateAnd(
                  flags,
                  this->gen_const(32,static_cast<uint32_t>(traits::FFLAG_MASK)))
               ))
            ,
            get_reg_ptr(traits::FCSR), false);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 121);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 122: FMV__X__W */
    std::tuple<continuation_e, BasicBlock*> __fmv__x__w(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "fmv.x.w"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FMV__X__W_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,122);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if((rd)!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    this->gen_reg_load(traits::F0+ rs1),
                    32, false),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 122);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 123: FMV__W__X */
    std::tuple<continuation_e, BasicBlock*> __fmv__w__x(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "fmv.w.x"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FMV__W__X_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,123);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<int32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            this->builder.CreateStore(
            this->builder.CreateOr(
               this->gen_const(32,(uint32_t)((int64_t)- 1<<32)),
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs1),
                   32, false))
            ,
            get_reg_ptr(rd + traits::F0), false);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 123);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 124: C__FLW */
    std::tuple<continuation_e, BasicBlock*> __c__flw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f(8+{rd}), {uimm}({rs1})", fmt::arg("mnemonic", "c.flw"),
                fmt::arg("rd", rd), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__FLW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,124);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        auto offs =this->gen_ext(
            (this->builder.CreateAdd(
               this->gen_ext(this->gen_reg_load(traits::X0+ rs1+8), 64,false),
               this->gen_ext(this->gen_const(8,uimm), 64,false))
            ),
            32, false);
        auto res =this->gen_ext(
            this->gen_read_mem(traits::MEM, offs, 4),
            32, false);
        if(static_cast<uint32_t>(traits::FLEN)==32) {
            this->builder.CreateStore(
            res,
            get_reg_ptr(rd+8 + traits::F0), false);
        }
        else{
            this->builder.CreateStore(
            this->builder.CreateOr(
               this->gen_const(32,((uint32_t)((int32_t)- 1<<32))),
               res)
            ,
            get_reg_ptr(rd+8 + traits::F0), false);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 124);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 125: C__FSW */
    std::tuple<continuation_e, BasicBlock*> __c__fsw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f(8+{rs2}), {uimm}({rs1})", fmt::arg("mnemonic", "c.fsw"),
                fmt::arg("rs2", rs2), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__FSW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,125);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        auto offs =this->gen_ext(
            (this->builder.CreateAdd(
               this->gen_ext(this->gen_reg_load(traits::X0+ rs1+8), 64,false),
               this->gen_ext(this->gen_const(8,uimm), 64,false))
            ),
            32, false);
        this->gen_write_mem(traits::MEM,
        offs,
        this->gen_ext(
            this->gen_reg_load(traits::F0+ rs2+8),
            32, false));
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 125);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 126: C__FLWSP */
    std::tuple<continuation_e, BasicBlock*> __c__flwsp(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t uimm = ((bit_sub<2,2>(instr) << 6) | (bit_sub<4,3>(instr) << 2) | (bit_sub<12,1>(instr) << 5));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f {rd}, {uimm}(x2)", fmt::arg("mnemonic", "c.flwsp"),
                fmt::arg("rd", rd), fmt::arg("uimm", uimm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__FLWSP_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,126);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        auto offs =this->gen_ext(
            (this->builder.CreateAdd(
               this->gen_ext(this->gen_reg_load(traits::X0+ 2), 64,false),
               this->gen_ext(this->gen_const(8,uimm), 64,false))
            ),
            32, false);
        auto res =this->gen_ext(
            this->gen_read_mem(traits::MEM, offs, 4),
            32, false);
        if(static_cast<uint32_t>(traits::FLEN)==32) {
            this->builder.CreateStore(
            res,
            get_reg_ptr(rd + traits::F0), false);
        }
        else{
            this->builder.CreateStore(
            this->builder.CreateOr(
               this->gen_const(32,((uint32_t)((int32_t)- 1<<32))),
               res)
            ,
            get_reg_ptr(rd + traits::F0), false);
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 126);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 127: C__FSWSP */
    std::tuple<continuation_e, BasicBlock*> __c__fswsp(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t uimm = ((bit_sub<7,2>(instr) << 6) | (bit_sub<9,4>(instr) << 2));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f {rs2}, {uimm}(x2), ", fmt::arg("mnemonic", "c.fswsp"),
                fmt::arg("rs2", rs2), fmt::arg("uimm", uimm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("C__FSWSP_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,127);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 2;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        auto offs =this->gen_ext(
            (this->builder.CreateAdd(
               this->gen_ext(this->gen_reg_load(traits::X0+ 2), 64,false),
               this->gen_ext(this->gen_const(8,uimm), 64,false))
            ),
            32, false);
        this->gen_write_mem(traits::MEM,
        offs,
        this->gen_ext(
            this->gen_reg_load(traits::F0+ rs2),
            32, false));
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 127);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /****************************************************************************
     * end opcode definitions
     ****************************************************************************/
    std::tuple<continuation_e, BasicBlock *> illegal_instruction(virt_addr_t &pc, code_word_t instr, BasicBlock *bb) {
        if(this->disass_enabled){
            auto mnemonic = std::string("illegal_instruction");
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        this->gen_sync(iss::PRE_SYNC, instr_descr.size());
        this->builder.CreateStore(this->builder.CreateLoad(this->get_typeptr(traits::NEXT_PC), get_reg_ptr(traits::NEXT_PC), true),
                                   get_reg_ptr(traits::PC), true);
        this->builder.CreateStore(
            this->builder.CreateAdd(this->builder.CreateLoad(this->get_typeptr(traits::ICOUNT), get_reg_ptr(traits::ICOUNT), true),
                                     this->gen_const(64U, 1)),
            get_reg_ptr(traits::ICOUNT), true);
        pc = pc + ((instr & 3) == 3 ? 4 : 2);
        this->set_tval(instr);
        this->gen_raise_trap(0, 2);     // illegal instruction trap
		this->gen_sync(iss::POST_SYNC, instr_descr.size());
        bb = this->leave_blk;
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
        return std::make_tuple(ILLEGAL_INSTR, nullptr);
    }    
};

template <typename CODE_WORD> void debug_fn(CODE_WORD instr) {
    volatile CODE_WORD x = instr;
    instr = 2 * x;
}

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
std::tuple<continuation_e, BasicBlock *>
vm_impl<ARCH>::gen_single_inst_behavior(virt_addr_t &pc, unsigned int &inst_cnt, BasicBlock *this_block) {
    // we fetch at max 4 byte, alignment is 2
    enum {TRAP_ID=1<<16};
    code_word_t instr = 0;
    // const typename traits::addr_t upper_bits = ~traits::PGMASK;
    phys_addr_t paddr(pc);
    auto *const data = (uint8_t *)&instr;
    if(this->core.has_mmu())
        paddr = this->core.virt2phys(pc);
    auto res = this->core.read(paddr, 4, data);
    if (res != iss::Ok) 
        return std::make_tuple(ILLEGAL_FETCH, nullptr);
    if (instr == 0x0000006f || (instr&0xffff)==0xa001)
        return std::make_tuple(JUMP_TO_SELF, nullptr);
    ++inst_cnt;
    uint32_t inst_index = instr_decoder.decode_instr(instr);
    compile_func f = nullptr;
    if(inst_index < instr_descr.size())
        f = instr_descr[inst_index].op;
    if (f == nullptr) {
        f = &this_class::illegal_instruction;
    }
    return (this->*f)(pc, instr, this_block);
}

template <typename ARCH>
void vm_impl<ARCH>::gen_leave_behavior(BasicBlock *leave_blk) {
    this->builder.SetInsertPoint(leave_blk);
    this->builder.CreateRet(this->builder.CreateLoad(this->get_typeptr(traits::NEXT_PC),get_reg_ptr(traits::NEXT_PC), false));
}

template <typename ARCH>
void vm_impl<ARCH>::gen_raise_trap(uint16_t trap_id, uint16_t cause) {
    auto *TRAP_val = this->gen_const(32, 0x80 << 24 | (cause << 16) | trap_id);
    this->builder.CreateStore(TRAP_val, get_reg_ptr(traits::TRAP_STATE), true);
    this->builder.CreateBr(this->trap_blk);
}

template <typename ARCH>
void vm_impl<ARCH>::gen_leave_trap(unsigned lvl) {
    std::vector<Value *> args{ this->core_ptr, ConstantInt::get(getContext(), APInt(64, lvl)) };
    this->builder.CreateCall(this->mod->getFunction("leave_trap"), args);
    this->builder.CreateStore(this->gen_const(32U, static_cast<int>(UNKNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
}

template <typename ARCH>
void vm_impl<ARCH>::gen_wait(unsigned type) {
    std::vector<Value *> args{ this->core_ptr, ConstantInt::get(getContext(), APInt(64, type)) };
    this->builder.CreateCall(this->mod->getFunction("wait"), args);
}

template <typename ARCH>
inline void vm_impl<ARCH>::set_tval(uint64_t tval) {
    auto tmp_tval = this->gen_const(64, tval);
    this->set_tval(tmp_tval);
}
template <typename ARCH>
inline void vm_impl<ARCH>::set_tval(Value* new_tval) {
    this->builder.CreateStore(this->gen_ext(new_tval, 64, false), this->tval);
}
template <typename ARCH> 
void vm_impl<ARCH>::gen_trap_behavior(BasicBlock *trap_blk) {
    this->builder.SetInsertPoint(trap_blk);
    auto *trap_state_val = this->builder.CreateLoad(this->get_typeptr(traits::TRAP_STATE), get_reg_ptr(traits::TRAP_STATE), true);
    auto *cur_pc_val = this->builder.CreateLoad(this->get_typeptr(traits::PC), get_reg_ptr(traits::PC), true);
    std::vector<Value *> args{this->core_ptr,
                                this->adj_to64(trap_state_val),
                                this->adj_to64(cur_pc_val),
                              this->adj_to64(this->builder.CreateLoad(this->get_type(64),this->tval))};
    this->builder.CreateCall(this->mod->getFunction("enter_trap"), args);
    this->builder.CreateStore(this->gen_const(32U, static_cast<int>(UNKNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);

    auto *trap_addr_val = this->builder.CreateLoad(this->get_typeptr(traits::NEXT_PC), get_reg_ptr(traits::NEXT_PC), false);
    this->builder.CreateRet(trap_addr_val);
}
template <typename ARCH>
void vm_impl<ARCH>::gen_instr_prologue() {
    auto* trap_val =
        this->builder.CreateLoad(this->get_typeptr(arch::traits<ARCH>::PENDING_TRAP), get_reg_ptr(arch::traits<ARCH>::PENDING_TRAP));
    this->builder.CreateStore(trap_val, get_reg_ptr(arch::traits<ARCH>::TRAP_STATE), false);
}
            

template <typename ARCH>
void vm_impl<ARCH>::gen_instr_epilogue(BasicBlock *bb) {
    auto* target_bb = BasicBlock::Create(this->mod->getContext(), "", this->func, bb);
    auto *v = this->builder.CreateLoad(this->get_typeptr(traits::TRAP_STATE), get_reg_ptr(traits::TRAP_STATE), true);
    this->gen_cond_branch(this->builder.CreateICmp(
                              ICmpInst::ICMP_EQ, v,
                              ConstantInt::get(getContext(), APInt(v->getType()->getIntegerBitWidth(), 0))),
                          target_bb, this->trap_blk, 1);
    this->builder.SetInsertPoint(target_bb);
    // update icount
    auto* icount_val = this->builder.CreateAdd(
        this->builder.CreateLoad(this->get_typeptr(arch::traits<ARCH>::ICOUNT), get_reg_ptr(arch::traits<ARCH>::ICOUNT)), this->gen_const(64U, 1));
    this->builder.CreateStore(icount_val, get_reg_ptr(arch::traits<ARCH>::ICOUNT), false);
}

} // namespace rv32imacf

template <>
std::unique_ptr<vm_if> create<arch::rv32imacf>(arch::rv32imacf *core, unsigned short port, bool dump) {
    auto ret = new rv32imacf::vm_impl<arch::rv32imacf>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret, port);
    return std::unique_ptr<vm_if>(ret);
}
} // namespace llvm
} // namespace iss

#include <iss/arch/riscv_hart_m_p.h>
#include <iss/arch/riscv_hart_mu_p.h>
#include <iss/factory.h>
namespace iss {
namespace {
volatile std::array<bool, 2> dummy = {
        core_factory::instance().register_creator("rv32imacf|m_p|llvm", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_m_p<iss::arch::rv32imacf>();
		    auto vm = new llvm::rv32imacf::vm_impl<arch::rv32imacf>(*cpu, false);
		    if (port != 0) debugger::server<debugger::gdb_session>::run_server(vm, port);
            if(init_data){
                auto* cb = reinterpret_cast<std::function<void(arch_if*, arch::traits<arch::rv32imacf>::reg_t*, arch::traits<arch::rv32imacf>::reg_t*)>*>(init_data);
                cpu->set_semihosting_callback(*cb);
            }
            return {cpu_ptr{cpu}, vm_ptr{vm}};
        }),
        core_factory::instance().register_creator("rv32imacf|mu_p|llvm", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_mu_p<iss::arch::rv32imacf>();
		    auto vm = new llvm::rv32imacf::vm_impl<arch::rv32imacf>(*cpu, false);
		    if (port != 0) debugger::server<debugger::gdb_session>::run_server(vm, port);
            if(init_data){
                auto* cb = reinterpret_cast<std::function<void(arch_if*, arch::traits<arch::rv32imacf>::reg_t*, arch::traits<arch::rv32imacf>::reg_t*)>*>(init_data);
                cpu->set_semihosting_callback(*cb);
            }
            return {cpu_ptr{cpu}, vm_ptr{vm}};
        })
};
}
}
// clang-format on
