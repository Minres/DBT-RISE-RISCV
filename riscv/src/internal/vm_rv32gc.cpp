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

#include <iss/arch/rv32gc.h>
#include <iss/arch/riscv_hart_msu_vp.h>
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/server.h>
#include <iss/iss.h>
#include <iss/llvm/vm_base.h>
#include <util/logging.h>

#include <fmt/format.h>

#include <array>
#include <iss/debugger/riscv_target_adapter.h>

namespace iss {
namespace vm {
namespace fp_impl {
void add_fp_functions_2_module(llvm::Module *, unsigned);
}
}

namespace rv32gc {
using namespace iss::arch;
using namespace llvm;
using namespace iss::debugger;
using namespace iss::vm::llvm;

template <typename ARCH> class vm_impl : public vm_base<ARCH> {
public:
    using super = typename iss::vm::llvm::vm_base<ARCH>;
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

    inline const char *name(size_t index){return traits<ARCH>::reg_aliases.at(index);}

    template <typename T> inline ConstantInt *size(T type) {
        return ConstantInt::get(getContext(), APInt(32, type->getType()->getScalarSizeInBits()));
    }

    void setup_module(Module* m) override {
        super::setup_module(m);
        iss::vm::fp_impl::add_fp_functions_2_module(m, traits<ARCH>::FP_REGS_SIZE);
    }

    inline Value *gen_choose(Value *cond, Value *trueVal, Value *falseVal, unsigned size) {
        return super::gen_cond_assign(cond, this->gen_ext(trueVal, size), this->gen_ext(falseVal, size));
    }

    std::tuple<continuation_e, BasicBlock *> gen_single_inst_behavior(virt_addr_t &, unsigned int &, BasicBlock *) override;

    void gen_leave_behavior(BasicBlock *leave_blk) override;

    void gen_raise_trap(uint16_t trap_id, uint16_t cause);

    void gen_leave_trap(unsigned lvl);

    void gen_wait(unsigned type);

    void gen_trap_behavior(BasicBlock *) override;

    void gen_trap_check(BasicBlock *bb);

    inline Value *gen_reg_load(unsigned i, unsigned level = 0) {
        return this->builder.CreateLoad(get_reg_ptr(i), false);
    }

    inline void gen_set_pc(virt_addr_t pc, unsigned reg_num) {
        Value *next_pc_v = this->builder.CreateSExtOrTrunc(this->gen_const(traits<ARCH>::XLEN, pc.val),
                                                           this->get_type(traits<ARCH>::XLEN));
        this->builder.CreateStore(next_pc_v, get_reg_ptr(reg_num), true);
    }

    // some compile time constants
    // enum { MASK16 = 0b1111110001100011, MASK32 = 0b11111111111100000111000001111111 };
    enum { MASK16 = 0b1111111111111111, MASK32 = 0b11111111111100000111000001111111 };
    enum { EXTR_MASK16 = MASK16 >> 2, EXTR_MASK32 = MASK32 >> 2 };
    enum { LUT_SIZE = 1 << util::bit_count(EXTR_MASK32), LUT_SIZE_C = 1 << util::bit_count(EXTR_MASK16) };

    using this_class = vm_impl<ARCH>;
    using compile_func = std::tuple<continuation_e, BasicBlock *> (this_class::*)(virt_addr_t &pc,
                                                                                  code_word_t instr,
                                                                                  BasicBlock *bb);
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

    const std::array<InstructionDesriptor, 159> instr_descr = {{
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
        /* instruction MUL */
        {32, 0b00000010000000000000000000110011, 0b11111110000000000111000001111111, &this_class::__mul},
        /* instruction MULH */
        {32, 0b00000010000000000001000000110011, 0b11111110000000000111000001111111, &this_class::__mulh},
        /* instruction MULHSU */
        {32, 0b00000010000000000010000000110011, 0b11111110000000000111000001111111, &this_class::__mulhsu},
        /* instruction MULHU */
        {32, 0b00000010000000000011000000110011, 0b11111110000000000111000001111111, &this_class::__mulhu},
        /* instruction DIV */
        {32, 0b00000010000000000100000000110011, 0b11111110000000000111000001111111, &this_class::__div},
        /* instruction DIVU */
        {32, 0b00000010000000000101000000110011, 0b11111110000000000111000001111111, &this_class::__divu},
        /* instruction REM */
        {32, 0b00000010000000000110000000110011, 0b11111110000000000111000001111111, &this_class::__rem},
        /* instruction REMU */
        {32, 0b00000010000000000111000000110011, 0b11111110000000000111000001111111, &this_class::__remu},
        /* instruction LR.W */
        {32, 0b00010000000000000010000000101111, 0b11111001111100000111000001111111, &this_class::__lr_w},
        /* instruction SC.W */
        {32, 0b00011000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__sc_w},
        /* instruction AMOSWAP.W */
        {32, 0b00001000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoswap_w},
        /* instruction AMOADD.W */
        {32, 0b00000000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoadd_w},
        /* instruction AMOXOR.W */
        {32, 0b00100000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoxor_w},
        /* instruction AMOAND.W */
        {32, 0b01100000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoand_w},
        /* instruction AMOOR.W */
        {32, 0b01000000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoor_w},
        /* instruction AMOMIN.W */
        {32, 0b10000000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amomin_w},
        /* instruction AMOMAX.W */
        {32, 0b10100000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amomax_w},
        /* instruction AMOMINU.W */
        {32, 0b11000000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amominu_w},
        /* instruction AMOMAXU.W */
        {32, 0b11100000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amomaxu_w},
        /* instruction C.ADDI4SPN */
        {16, 0b0000000000000000, 0b1110000000000011, &this_class::__c_addi4spn},
        /* instruction C.LW */
        {16, 0b0100000000000000, 0b1110000000000011, &this_class::__c_lw},
        /* instruction C.SW */
        {16, 0b1100000000000000, 0b1110000000000011, &this_class::__c_sw},
        /* instruction C.ADDI */
        {16, 0b0000000000000001, 0b1110000000000011, &this_class::__c_addi},
        /* instruction C.NOP */
        {16, 0b0000000000000001, 0b1111111111111111, &this_class::__c_nop},
        /* instruction C.JAL */
        {16, 0b0010000000000001, 0b1110000000000011, &this_class::__c_jal},
        /* instruction C.LI */
        {16, 0b0100000000000001, 0b1110000000000011, &this_class::__c_li},
        /* instruction C.LUI */
        {16, 0b0110000000000001, 0b1110000000000011, &this_class::__c_lui},
        /* instruction C.ADDI16SP */
        {16, 0b0110000100000001, 0b1110111110000011, &this_class::__c_addi16sp},
        /* instruction C.SRLI */
        {16, 0b1000000000000001, 0b1111110000000011, &this_class::__c_srli},
        /* instruction C.SRAI */
        {16, 0b1000010000000001, 0b1111110000000011, &this_class::__c_srai},
        /* instruction C.ANDI */
        {16, 0b1000100000000001, 0b1110110000000011, &this_class::__c_andi},
        /* instruction C.SUB */
        {16, 0b1000110000000001, 0b1111110001100011, &this_class::__c_sub},
        /* instruction C.XOR */
        {16, 0b1000110000100001, 0b1111110001100011, &this_class::__c_xor},
        /* instruction C.OR */
        {16, 0b1000110001000001, 0b1111110001100011, &this_class::__c_or},
        /* instruction C.AND */
        {16, 0b1000110001100001, 0b1111110001100011, &this_class::__c_and},
        /* instruction C.J */
        {16, 0b1010000000000001, 0b1110000000000011, &this_class::__c_j},
        /* instruction C.BEQZ */
        {16, 0b1100000000000001, 0b1110000000000011, &this_class::__c_beqz},
        /* instruction C.BNEZ */
        {16, 0b1110000000000001, 0b1110000000000011, &this_class::__c_bnez},
        /* instruction C.SLLI */
        {16, 0b0000000000000010, 0b1111000000000011, &this_class::__c_slli},
        /* instruction C.LWSP */
        {16, 0b0100000000000010, 0b1110000000000011, &this_class::__c_lwsp},
        /* instruction C.MV */
        {16, 0b1000000000000010, 0b1111000000000011, &this_class::__c_mv},
        /* instruction C.JR */
        {16, 0b1000000000000010, 0b1111000001111111, &this_class::__c_jr},
        /* instruction C.ADD */
        {16, 0b1001000000000010, 0b1111000000000011, &this_class::__c_add},
        /* instruction C.JALR */
        {16, 0b1001000000000010, 0b1111000001111111, &this_class::__c_jalr},
        /* instruction C.EBREAK */
        {16, 0b1001000000000010, 0b1111111111111111, &this_class::__c_ebreak},
        /* instruction C.SWSP */
        {16, 0b1100000000000010, 0b1110000000000011, &this_class::__c_swsp},
        /* instruction DII */
        {16, 0b0000000000000000, 0b1111111111111111, &this_class::__dii},
        /* instruction FLW */
        {32, 0b00000000000000000010000000000111, 0b00000000000000000111000001111111, &this_class::__flw},
        /* instruction FSW */
        {32, 0b00000000000000000010000000100111, 0b00000000000000000111000001111111, &this_class::__fsw},
        /* instruction FMADD.S */
        {32, 0b00000000000000000000000001000011, 0b00000110000000000000000001111111, &this_class::__fmadd_s},
        /* instruction FMSUB.S */
        {32, 0b00000000000000000000000001000111, 0b00000110000000000000000001111111, &this_class::__fmsub_s},
        /* instruction FNMADD.S */
        {32, 0b00000000000000000000000001001111, 0b00000110000000000000000001111111, &this_class::__fnmadd_s},
        /* instruction FNMSUB.S */
        {32, 0b00000000000000000000000001001011, 0b00000110000000000000000001111111, &this_class::__fnmsub_s},
        /* instruction FADD.S */
        {32, 0b00000000000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fadd_s},
        /* instruction FSUB.S */
        {32, 0b00001000000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fsub_s},
        /* instruction FMUL.S */
        {32, 0b00010000000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fmul_s},
        /* instruction FDIV.S */
        {32, 0b00011000000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fdiv_s},
        /* instruction FSQRT.S */
        {32, 0b01011000000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fsqrt_s},
        /* instruction FSGNJ.S */
        {32, 0b00100000000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnj_s},
        /* instruction FSGNJN.S */
        {32, 0b00100000000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnjn_s},
        /* instruction FSGNJX.S */
        {32, 0b00100000000000000010000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnjx_s},
        /* instruction FMIN.S */
        {32, 0b00101000000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fmin_s},
        /* instruction FMAX.S */
        {32, 0b00101000000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__fmax_s},
        /* instruction FCVT.W.S */
        {32, 0b11000000000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_w_s},
        /* instruction FCVT.WU.S */
        {32, 0b11000000000100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_wu_s},
        /* instruction FEQ.S */
        {32, 0b10100000000000000010000001010011, 0b11111110000000000111000001111111, &this_class::__feq_s},
        /* instruction FLT.S */
        {32, 0b10100000000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__flt_s},
        /* instruction FLE.S */
        {32, 0b10100000000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fle_s},
        /* instruction FCLASS.S */
        {32, 0b11100000000000000001000001010011, 0b11111111111100000111000001111111, &this_class::__fclass_s},
        /* instruction FCVT.S.W */
        {32, 0b11010000000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_s_w},
        /* instruction FCVT.S.WU */
        {32, 0b11010000000100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_s_wu},
        /* instruction FMV.X.W */
        {32, 0b11100000000000000000000001010011, 0b11111111111100000111000001111111, &this_class::__fmv_x_w},
        /* instruction FMV.W.X */
        {32, 0b11110000000000000000000001010011, 0b11111111111100000111000001111111, &this_class::__fmv_w_x},
        /* instruction C.FLW */
        {16, 0b0110000000000000, 0b1110000000000011, &this_class::__c_flw},
        /* instruction C.FSW */
        {16, 0b1110000000000000, 0b1110000000000011, &this_class::__c_fsw},
        /* instruction C.FLWSP */
        {16, 0b0110000000000010, 0b1110000000000011, &this_class::__c_flwsp},
        /* instruction C.FSWSP */
        {16, 0b1110000000000010, 0b1110000000000011, &this_class::__c_fswsp},
        /* instruction FLD */
        {32, 0b00000000000000000011000000000111, 0b00000000000000000111000001111111, &this_class::__fld},
        /* instruction FSD */
        {32, 0b00000000000000000011000000100111, 0b00000000000000000111000001111111, &this_class::__fsd},
        /* instruction FMADD.D */
        {32, 0b00000010000000000000000001000011, 0b00000110000000000000000001111111, &this_class::__fmadd_d},
        /* instruction FMSUB.D */
        {32, 0b00000010000000000000000001000111, 0b00000110000000000000000001111111, &this_class::__fmsub_d},
        /* instruction FNMADD.D */
        {32, 0b00000010000000000000000001001111, 0b00000110000000000000000001111111, &this_class::__fnmadd_d},
        /* instruction FNMSUB.D */
        {32, 0b00000010000000000000000001001011, 0b00000110000000000000000001111111, &this_class::__fnmsub_d},
        /* instruction FADD.D */
        {32, 0b00000010000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fadd_d},
        /* instruction FSUB.D */
        {32, 0b00001010000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fsub_d},
        /* instruction FMUL.D */
        {32, 0b00010010000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fmul_d},
        /* instruction FDIV.D */
        {32, 0b00011010000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fdiv_d},
        /* instruction FSQRT.D */
        {32, 0b01011010000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fsqrt_d},
        /* instruction FSGNJ.D */
        {32, 0b00100010000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnj_d},
        /* instruction FSGNJN.D */
        {32, 0b00100010000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnjn_d},
        /* instruction FSGNJX.D */
        {32, 0b00100010000000000010000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnjx_d},
        /* instruction FMIN.D */
        {32, 0b00101010000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fmin_d},
        /* instruction FMAX.D */
        {32, 0b00101010000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__fmax_d},
        /* instruction FCVT.S.D */
        {32, 0b01000000000100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_s_d},
        /* instruction FCVT.D.S */
        {32, 0b01000010000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_d_s},
        /* instruction FEQ.D */
        {32, 0b10100010000000000010000001010011, 0b11111110000000000111000001111111, &this_class::__feq_d},
        /* instruction FLT.D */
        {32, 0b10100010000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__flt_d},
        /* instruction FLE.D */
        {32, 0b10100010000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fle_d},
        /* instruction FCLASS.D */
        {32, 0b11100010000000000001000001010011, 0b11111111111100000111000001111111, &this_class::__fclass_d},
        /* instruction FCVT.W.D */
        {32, 0b11000010000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_w_d},
        /* instruction FCVT.WU.D */
        {32, 0b11000010000100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_wu_d},
        /* instruction FCVT.D.W */
        {32, 0b11010010000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_d_w},
        /* instruction FCVT.D.WU */
        {32, 0b11010010000100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_d_wu},
        /* instruction C.FLD */
        {16, 0b0010000000000000, 0b1110000000000011, &this_class::__c_fld},
        /* instruction C.FSD */
        {16, 0b1010000000000000, 0b1110000000000011, &this_class::__c_fsd},
        /* instruction C.FLDSP */
        {16, 0b0010000000000010, 0b1110000000000011, &this_class::__c_fldsp},
        /* instruction C.FSDSP */
        {16, 0b1010000000000010, 0b1110000000000011, &this_class::__c_fsdsp},
    }};
 
    /* instruction definitions */
    /* instruction 0: LUI */
    std::tuple<continuation_e, BasicBlock*> __lui(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("LUI");
    	
    	this->gen_sync(PRE_SYNC, 0);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	int32_t imm = signextend<int32_t,32>((bit_sub<12,20>(instr) << 12));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_const(32U, imm);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 0);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 1: AUIPC */
    std::tuple<continuation_e, BasicBlock*> __auipc(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("AUIPC");
    	
    	this->gen_sync(PRE_SYNC, 1);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	int32_t imm = signextend<int32_t,32>((bit_sub<12,20>(instr) << 12));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, imm));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 1);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 2: JAL */
    std::tuple<continuation_e, BasicBlock*> __jal(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("JAL");
    	
    	this->gen_sync(PRE_SYNC, 2);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	int32_t imm = signextend<int32_t,21>((bit_sub<12,8>(instr) << 12) | (bit_sub<20,1>(instr) << 11) | (bit_sub<21,10>(instr) << 1) | (bit_sub<31,1>(instr) << 20));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 4));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	Value* PC_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        cur_pc_val,
    	        32, true),
    	    this->gen_const(32U, imm));
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 2);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 3: JALR */
    std::tuple<continuation_e, BasicBlock*> __jalr(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("JALR");
    	
    	this->gen_sync(PRE_SYNC, 3);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* new_pc_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, imm));
    	Value* align_val = this->builder.CreateAnd(
    	    new_pc_val,
    	    this->gen_const(32U, 0x1));
    	{
    	    BasicBlock* bbnext = BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	    BasicBlock* bb_then = BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	    BasicBlock* bb_else = BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	    // this->builder.SetInsertPoint(bb);
    	    this->gen_cond_branch(this->builder.CreateICmp(
    	        ICmpInst::ICMP_NE,
    	        align_val,
    	        this->gen_const(32U, 0)),
    	        bb_then,
    	        bb_else);
    	    this->builder.SetInsertPoint(bb_then);
    	    {
    	        this->gen_raise_trap(0, 0);
    	    }
    	    this->builder.CreateBr(bbnext);
    	    this->builder.SetInsertPoint(bb_else);
    	    {
    	        if(rd != 0){
    	            Value* Xtmp0_val = this->builder.CreateAdd(
    	                cur_pc_val,
    	                this->gen_const(32U, 4));
    	            this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	        }
    	        Value* PC_val = this->builder.CreateAnd(
    	            new_pc_val,
    	            this->builder.CreateNot(this->gen_const(32U, 0x1)));
    	        this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	        this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()), get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	    }
    	    this->builder.CreateBr(bbnext);
    	    bb=bbnext;
    	}
    	this->builder.SetInsertPoint(bb);
    	this->gen_sync(POST_SYNC, 3);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 4: BEQ */
    std::tuple<continuation_e, BasicBlock*> __beq(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("BEQ");
    	
    	this->gen_sync(PRE_SYNC, 4);
    	
    	int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_EQ,
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, imm)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 4)),
    	    32);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 4);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 5: BNE */
    std::tuple<continuation_e, BasicBlock*> __bne(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("BNE");
    	
    	this->gen_sync(PRE_SYNC, 5);
    	
    	int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_NE,
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, imm)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 4)),
    	    32);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 5);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 6: BLT */
    std::tuple<continuation_e, BasicBlock*> __blt(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("BLT");
    	
    	this->gen_sync(PRE_SYNC, 6);
    	
    	int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_SLT,
    	        this->gen_ext(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            32, true),
    	        this->gen_ext(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            32, true)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, imm)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 4)),
    	    32);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 6);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 7: BGE */
    std::tuple<continuation_e, BasicBlock*> __bge(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("BGE");
    	
    	this->gen_sync(PRE_SYNC, 7);
    	
    	int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_SGE,
    	        this->gen_ext(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            32, true),
    	        this->gen_ext(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            32, true)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, imm)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 4)),
    	    32);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 7);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 8: BLTU */
    std::tuple<continuation_e, BasicBlock*> __bltu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("BLTU");
    	
    	this->gen_sync(PRE_SYNC, 8);
    	
    	int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_ULT,
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, imm)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 4)),
    	    32);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 8);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 9: BGEU */
    std::tuple<continuation_e, BasicBlock*> __bgeu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("BGEU");
    	
    	this->gen_sync(PRE_SYNC, 9);
    	
    	int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_UGE,
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, imm)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 4)),
    	    32);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 9);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 10: LB */
    std::tuple<continuation_e, BasicBlock*> __lb(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("LB");
    	
    	this->gen_sync(PRE_SYNC, 10);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, imm));
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 8/8),
    	        32,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 10);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 11: LH */
    std::tuple<continuation_e, BasicBlock*> __lh(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("LH");
    	
    	this->gen_sync(PRE_SYNC, 11);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, imm));
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 16/8),
    	        32,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 11);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 12: LW */
    std::tuple<continuation_e, BasicBlock*> __lw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("LW");
    	
    	this->gen_sync(PRE_SYNC, 12);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, imm));
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	        32,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 12);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 13: LBU */
    std::tuple<continuation_e, BasicBlock*> __lbu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("LBU");
    	
    	this->gen_sync(PRE_SYNC, 13);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, imm));
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 8/8),
    	        32,
    	        false);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 13);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 14: LHU */
    std::tuple<continuation_e, BasicBlock*> __lhu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("LHU");
    	
    	this->gen_sync(PRE_SYNC, 14);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, imm));
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 16/8),
    	        32,
    	        false);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 14);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 15: SB */
    std::tuple<continuation_e, BasicBlock*> __sb(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SB");
    	
    	this->gen_sync(PRE_SYNC, 15);
    	
    	int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, imm));
    	Value* MEMtmp0_val = this->gen_reg_load(rs2 + traits<ARCH>::X0, 0);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(8)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 15);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 16: SH */
    std::tuple<continuation_e, BasicBlock*> __sh(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SH");
    	
    	this->gen_sync(PRE_SYNC, 16);
    	
    	int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, imm));
    	Value* MEMtmp0_val = this->gen_reg_load(rs2 + traits<ARCH>::X0, 0);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(16)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 16);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 17: SW */
    std::tuple<continuation_e, BasicBlock*> __sw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SW");
    	
    	this->gen_sync(PRE_SYNC, 17);
    	
    	int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, imm));
    	Value* MEMtmp0_val = this->gen_reg_load(rs2 + traits<ARCH>::X0, 0);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 17);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 18: ADDI */
    std::tuple<continuation_e, BasicBlock*> __addi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("ADDI");
    	
    	this->gen_sync(PRE_SYNC, 18);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateAdd(
    	        this->gen_ext(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            32, true),
    	        this->gen_const(32U, imm));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 18);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 19: SLTI */
    std::tuple<continuation_e, BasicBlock*> __slti(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SLTI");
    	
    	this->gen_sync(PRE_SYNC, 19);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_SLT,
    	            this->gen_ext(
    	                this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	                32, true),
    	            this->gen_const(32U, imm)),
    	        this->gen_const(32U, 1),
    	        this->gen_const(32U, 0),
    	        32);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 19);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 20: SLTIU */
    std::tuple<continuation_e, BasicBlock*> __sltiu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SLTIU");
    	
    	this->gen_sync(PRE_SYNC, 20);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	int32_t full_imm_val = imm;
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, full_imm_val)),
    	        this->gen_const(32U, 1),
    	        this->gen_const(32U, 0),
    	        32);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 20);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 21: XORI */
    std::tuple<continuation_e, BasicBlock*> __xori(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("XORI");
    	
    	this->gen_sync(PRE_SYNC, 21);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateXor(
    	        this->gen_ext(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            32, true),
    	        this->gen_const(32U, imm));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 21);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 22: ORI */
    std::tuple<continuation_e, BasicBlock*> __ori(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("ORI");
    	
    	this->gen_sync(PRE_SYNC, 22);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateOr(
    	        this->gen_ext(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            32, true),
    	        this->gen_const(32U, imm));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 22);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 23: ANDI */
    std::tuple<continuation_e, BasicBlock*> __andi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("ANDI");
    	
    	this->gen_sync(PRE_SYNC, 23);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateAnd(
    	        this->gen_ext(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            32, true),
    	        this->gen_const(32U, imm));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 23);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 24: SLLI */
    std::tuple<continuation_e, BasicBlock*> __slli(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SLLI");
    	
    	this->gen_sync(PRE_SYNC, 24);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(shamt > 31){
    	    this->gen_raise_trap(0, 0);
    	} else {
    	    if(rd != 0){
    	        Value* Xtmp0_val = this->builder.CreateShl(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, shamt));
    	        this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	    }
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 24);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 25: SRLI */
    std::tuple<continuation_e, BasicBlock*> __srli(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SRLI");
    	
    	this->gen_sync(PRE_SYNC, 25);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(shamt > 31){
    	    this->gen_raise_trap(0, 0);
    	} else {
    	    if(rd != 0){
    	        Value* Xtmp0_val = this->builder.CreateLShr(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, shamt));
    	        this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	    }
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 25);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 26: SRAI */
    std::tuple<continuation_e, BasicBlock*> __srai(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SRAI");
    	
    	this->gen_sync(PRE_SYNC, 26);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(shamt > 31){
    	    this->gen_raise_trap(0, 0);
    	} else {
    	    if(rd != 0){
    	        Value* Xtmp0_val = this->builder.CreateAShr(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, shamt));
    	        this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	    }
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 26);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 27: ADD */
    std::tuple<continuation_e, BasicBlock*> __add(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("ADD");
    	
    	this->gen_sync(PRE_SYNC, 27);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateAdd(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 27);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 28: SUB */
    std::tuple<continuation_e, BasicBlock*> __sub(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SUB");
    	
    	this->gen_sync(PRE_SYNC, 28);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateSub(
    	         this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	         this->gen_reg_load(rs2 + traits<ARCH>::X0, 0));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 28);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 29: SLL */
    std::tuple<continuation_e, BasicBlock*> __sll(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SLL");
    	
    	this->gen_sync(PRE_SYNC, 29);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateShl(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->builder.CreateAnd(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            this->builder.CreateSub(
    	                 this->gen_const(32U, 32),
    	                 this->gen_const(32U, 1))));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 29);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 30: SLT */
    std::tuple<continuation_e, BasicBlock*> __slt(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SLT");
    	
    	this->gen_sync(PRE_SYNC, 30);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_SLT,
    	            this->gen_ext(
    	                this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	                32, true),
    	            this->gen_ext(
    	                this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	                32, true)),
    	        this->gen_const(32U, 1),
    	        this->gen_const(32U, 0),
    	        32);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 30);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 31: SLTU */
    std::tuple<continuation_e, BasicBlock*> __sltu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SLTU");
    	
    	this->gen_sync(PRE_SYNC, 31);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_ext(
    	                this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	                32,
    	                false),
    	            this->gen_ext(
    	                this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	                32,
    	                false)),
    	        this->gen_const(32U, 1),
    	        this->gen_const(32U, 0),
    	        32);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 31);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 32: XOR */
    std::tuple<continuation_e, BasicBlock*> __xor(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("XOR");
    	
    	this->gen_sync(PRE_SYNC, 32);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateXor(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 32);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 33: SRL */
    std::tuple<continuation_e, BasicBlock*> __srl(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SRL");
    	
    	this->gen_sync(PRE_SYNC, 33);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateLShr(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->builder.CreateAnd(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            this->builder.CreateSub(
    	                 this->gen_const(32U, 32),
    	                 this->gen_const(32U, 1))));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 33);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 34: SRA */
    std::tuple<continuation_e, BasicBlock*> __sra(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SRA");
    	
    	this->gen_sync(PRE_SYNC, 34);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateAShr(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->builder.CreateAnd(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            this->builder.CreateSub(
    	                 this->gen_const(32U, 32),
    	                 this->gen_const(32U, 1))));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 34);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 35: OR */
    std::tuple<continuation_e, BasicBlock*> __or(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("OR");
    	
    	this->gen_sync(PRE_SYNC, 35);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateOr(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 35);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 36: AND */
    std::tuple<continuation_e, BasicBlock*> __and(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("AND");
    	
    	this->gen_sync(PRE_SYNC, 36);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateAnd(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 36);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 37: FENCE */
    std::tuple<continuation_e, BasicBlock*> __fence(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FENCE");
    	
    	this->gen_sync(PRE_SYNC, 37);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t succ = ((bit_sub<20,4>(instr)));
    	uint8_t pred = ((bit_sub<24,4>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("fence"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* FENCEtmp0_val = this->builder.CreateOr(
    	    this->builder.CreateShl(
    	        this->gen_const(32U, pred),
    	        this->gen_const(32U, 4)),
    	    this->gen_const(32U, succ));
    	this->gen_write_mem(
    	    traits<ARCH>::FENCE,
    	    this->gen_const(64U, 0),
    	    this->builder.CreateZExtOrTrunc(FENCEtmp0_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 37);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 38: FENCE_I */
    std::tuple<continuation_e, BasicBlock*> __fence_i(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FENCE_I");
    	
    	this->gen_sync(PRE_SYNC, 38);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint16_t imm = ((bit_sub<20,12>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("fence_i"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* FENCEtmp0_val = this->gen_const(32U, imm);
    	this->gen_write_mem(
    	    traits<ARCH>::FENCE,
    	    this->gen_const(64U, 1),
    	    this->builder.CreateZExtOrTrunc(FENCEtmp0_val,this->get_type(32)));
    	this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()), get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 38);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(FLUSH, nullptr);
    }
    
    /* instruction 39: ECALL */
    std::tuple<continuation_e, BasicBlock*> __ecall(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("ECALL");
    	
    	this->gen_sync(PRE_SYNC, 39);
    	
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("ecall"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	this->gen_raise_trap(0, 11);
    	this->gen_sync(POST_SYNC, 39);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 40: EBREAK */
    std::tuple<continuation_e, BasicBlock*> __ebreak(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("EBREAK");
    	
    	this->gen_sync(PRE_SYNC, 40);
    	
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("ebreak"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	this->gen_raise_trap(0, 3);
    	this->gen_sync(POST_SYNC, 40);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 41: URET */
    std::tuple<continuation_e, BasicBlock*> __uret(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("URET");
    	
    	this->gen_sync(PRE_SYNC, 41);
    	
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("uret"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	this->gen_leave_trap(0);
    	this->gen_sync(POST_SYNC, 41);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 42: SRET */
    std::tuple<continuation_e, BasicBlock*> __sret(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SRET");
    	
    	this->gen_sync(PRE_SYNC, 42);
    	
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("sret"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	this->gen_leave_trap(1);
    	this->gen_sync(POST_SYNC, 42);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 43: MRET */
    std::tuple<continuation_e, BasicBlock*> __mret(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("MRET");
    	
    	this->gen_sync(PRE_SYNC, 43);
    	
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("mret"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	this->gen_leave_trap(3);
    	this->gen_sync(POST_SYNC, 43);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 44: WFI */
    std::tuple<continuation_e, BasicBlock*> __wfi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("WFI");
    	
    	this->gen_sync(PRE_SYNC, 44);
    	
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("wfi"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	this->gen_wait(1);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 44);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 45: SFENCE.VMA */
    std::tuple<continuation_e, BasicBlock*> __sfence_vma(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SFENCE.VMA");
    	
    	this->gen_sync(PRE_SYNC, 45);
    	
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("sfence.vma"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* FENCEtmp0_val = this->gen_const(32U, rs1);
    	this->gen_write_mem(
    	    traits<ARCH>::FENCE,
    	    this->gen_const(64U, 2),
    	    this->builder.CreateZExtOrTrunc(FENCEtmp0_val,this->get_type(32)));
    	Value* FENCEtmp1_val = this->gen_const(32U, rs2);
    	this->gen_write_mem(
    	    traits<ARCH>::FENCE,
    	    this->gen_const(64U, 3),
    	    this->builder.CreateZExtOrTrunc(FENCEtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 45);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 46: CSRRW */
    std::tuple<continuation_e, BasicBlock*> __csrrw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("CSRRW");
    	
    	this->gen_sync(PRE_SYNC, 46);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* rs_val_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	if(rd != 0){
    	    Value* csr_val_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, csr), 32/8);
    	    Value* CSRtmp0_val = rs_val_val;
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, csr),
    	        this->builder.CreateZExtOrTrunc(CSRtmp0_val,this->get_type(32)));
    	    Value* Xtmp1_val = csr_val_val;
    	    this->builder.CreateStore(Xtmp1_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	} else {
    	    Value* CSRtmp2_val = rs_val_val;
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, csr),
    	        this->builder.CreateZExtOrTrunc(CSRtmp2_val,this->get_type(32)));
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 46);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 47: CSRRS */
    std::tuple<continuation_e, BasicBlock*> __csrrs(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("CSRRS");
    	
    	this->gen_sync(PRE_SYNC, 47);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* xrd_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, csr), 32/8);
    	Value* xrs1_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	if(rd != 0){
    	    Value* Xtmp0_val = xrd_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	if(rs1 != 0){
    	    Value* CSRtmp1_val = this->builder.CreateOr(
    	        xrd_val,
    	        xrs1_val);
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, csr),
    	        this->builder.CreateZExtOrTrunc(CSRtmp1_val,this->get_type(32)));
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 47);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 48: CSRRC */
    std::tuple<continuation_e, BasicBlock*> __csrrc(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("CSRRC");
    	
    	this->gen_sync(PRE_SYNC, 48);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* xrd_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, csr), 32/8);
    	Value* xrs1_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	if(rd != 0){
    	    Value* Xtmp0_val = xrd_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	if(rs1 != 0){
    	    Value* CSRtmp1_val = this->builder.CreateAnd(
    	        xrd_val,
    	        this->builder.CreateNot(xrs1_val));
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, csr),
    	        this->builder.CreateZExtOrTrunc(CSRtmp1_val,this->get_type(32)));
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 48);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 49: CSRRWI */
    std::tuple<continuation_e, BasicBlock*> __csrrwi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("CSRRWI");
    	
    	this->gen_sync(PRE_SYNC, 49);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, csr), 32/8);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	Value* CSRtmp1_val = this->gen_ext(
    	    this->gen_const(32U, zimm),
    	    32,
    	    false);
    	this->gen_write_mem(
    	    traits<ARCH>::CSR,
    	    this->gen_const(16U, csr),
    	    this->builder.CreateZExtOrTrunc(CSRtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 49);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 50: CSRRSI */
    std::tuple<continuation_e, BasicBlock*> __csrrsi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("CSRRSI");
    	
    	this->gen_sync(PRE_SYNC, 50);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, csr), 32/8);
    	if(zimm != 0){
    	    Value* CSRtmp0_val = this->builder.CreateOr(
    	        res_val,
    	        this->gen_ext(
    	            this->gen_const(32U, zimm),
    	            32,
    	            false));
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, csr),
    	        this->builder.CreateZExtOrTrunc(CSRtmp0_val,this->get_type(32)));
    	}
    	if(rd != 0){
    	    Value* Xtmp1_val = res_val;
    	    this->builder.CreateStore(Xtmp1_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 50);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 51: CSRRCI */
    std::tuple<continuation_e, BasicBlock*> __csrrci(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("CSRRCI");
    	
    	this->gen_sync(PRE_SYNC, 51);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, csr), 32/8);
    	if(rd != 0){
    	    Value* Xtmp0_val = res_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	if(zimm != 0){
    	    Value* CSRtmp1_val = this->builder.CreateAnd(
    	        res_val,
    	        this->builder.CreateNot(this->gen_ext(
    	            this->gen_const(32U, zimm),
    	            32,
    	            false)));
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, csr),
    	        this->builder.CreateZExtOrTrunc(CSRtmp1_val,this->get_type(32)));
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 51);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 52: MUL */
    std::tuple<continuation_e, BasicBlock*> __mul(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("MUL");
    	
    	this->gen_sync(PRE_SYNC, 52);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* res_val = this->builder.CreateMul(
    	        this->gen_ext(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            128,
    	            false),
    	        this->gen_ext(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            128,
    	            false));
    	    Value* Xtmp0_val = this->gen_ext(
    	        res_val,
    	        32,
    	        false);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 52);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 53: MULH */
    std::tuple<continuation_e, BasicBlock*> __mulh(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("MULH");
    	
    	this->gen_sync(PRE_SYNC, 53);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* res_val = this->builder.CreateMul(
    	        this->gen_ext(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            128,
    	            true),
    	        this->gen_ext(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            128,
    	            true));
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->builder.CreateLShr(
    	            res_val,
    	            this->gen_const(32U, 32)),
    	        32,
    	        false);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 53);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 54: MULHSU */
    std::tuple<continuation_e, BasicBlock*> __mulhsu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("MULHSU");
    	
    	this->gen_sync(PRE_SYNC, 54);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* res_val = this->builder.CreateMul(
    	        this->gen_ext(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            128,
    	            true),
    	        this->gen_ext(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            128,
    	            false));
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->builder.CreateLShr(
    	            res_val,
    	            this->gen_const(32U, 32)),
    	        32,
    	        false);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 54);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 55: MULHU */
    std::tuple<continuation_e, BasicBlock*> __mulhu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("MULHU");
    	
    	this->gen_sync(PRE_SYNC, 55);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* res_val = this->builder.CreateMul(
    	        this->gen_ext(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            128,
    	            false),
    	        this->gen_ext(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            128,
    	            false));
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->builder.CreateLShr(
    	            res_val,
    	            this->gen_const(32U, 32)),
    	        32,
    	        false);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 55);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 56: DIV */
    std::tuple<continuation_e, BasicBlock*> __div(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("DIV");
    	
    	this->gen_sync(PRE_SYNC, 56);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    {
    	        BasicBlock* bbnext = BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	        BasicBlock* bb_then = BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	        BasicBlock* bb_else = BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	        // this->builder.SetInsertPoint(bb);
    	        this->gen_cond_branch(this->builder.CreateICmp(
    	            ICmpInst::ICMP_NE,
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, 0)),
    	            bb_then,
    	            bb_else);
    	        this->builder.SetInsertPoint(bb_then);
    	        {
    	            uint32_t M1_val = - 1;
    	            uint32_t MMIN_val = - 1 << 32 - 1;
    	            {
    	                BasicBlock* bbnext = BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	                BasicBlock* bb_then = BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	                BasicBlock* bb_else = BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	                // this->builder.SetInsertPoint(bb);
    	                this->gen_cond_branch(this->builder.CreateICmp(
    	                    ICmpInst::ICMP_EQ,
    	                    this->gen_ext(
    	                        this->gen_reg_load(rs1 + traits<ARCH>::X0, 1),
    	                        32, true),
    	                    this->gen_ext(
    	                        this->gen_const(32U, MMIN_val),
    	                        32, true)),
    	                    bb_then,
    	                    bb_else);
    	                this->builder.SetInsertPoint(bb_then);
    	                {
    	                    {
    	                        BasicBlock* bbnext = BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	                        BasicBlock* bb_then = BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	                        BasicBlock* bb_else = BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	                        // this->builder.SetInsertPoint(bb);
    	                        this->gen_cond_branch(this->builder.CreateICmp(
    	                            ICmpInst::ICMP_EQ,
    	                            this->gen_ext(
    	                                this->gen_reg_load(rs2 + traits<ARCH>::X0, 2),
    	                                32, true),
    	                            this->gen_ext(
    	                                this->gen_const(32U, M1_val),
    	                                32, true)),
    	                            bb_then,
    	                            bb_else);
    	                        this->builder.SetInsertPoint(bb_then);
    	                        {
    	                            Value* Xtmp0_val = this->gen_const(32U, MMIN_val);
    	                            this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	                        }
    	                        this->builder.CreateBr(bbnext);
    	                        this->builder.SetInsertPoint(bb_else);
    	                        {
    	                            Value* Xtmp1_val = this->builder.CreateSDiv(
    	                                this->gen_ext(
    	                                    this->gen_reg_load(rs1 + traits<ARCH>::X0, 3),
    	                                    32, true),
    	                                this->gen_ext(
    	                                    this->gen_reg_load(rs2 + traits<ARCH>::X0, 3),
    	                                    32, true));
    	                            this->builder.CreateStore(Xtmp1_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	                        }
    	                        this->builder.CreateBr(bbnext);
    	                        bb=bbnext;
    	                    }
    	                    this->builder.SetInsertPoint(bb);
    	                }
    	                this->builder.CreateBr(bbnext);
    	                this->builder.SetInsertPoint(bb_else);
    	                {
    	                    Value* Xtmp2_val = this->builder.CreateSDiv(
    	                        this->gen_ext(
    	                            this->gen_reg_load(rs1 + traits<ARCH>::X0, 2),
    	                            32, true),
    	                        this->gen_ext(
    	                            this->gen_reg_load(rs2 + traits<ARCH>::X0, 2),
    	                            32, true));
    	                    this->builder.CreateStore(Xtmp2_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	                }
    	                this->builder.CreateBr(bbnext);
    	                bb=bbnext;
    	            }
    	            this->builder.SetInsertPoint(bb);
    	        }
    	        this->builder.CreateBr(bbnext);
    	        this->builder.SetInsertPoint(bb_else);
    	        {
    	            Value* Xtmp3_val = this->builder.CreateNeg(this->gen_const(32U, 1));
    	            this->builder.CreateStore(Xtmp3_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	        }
    	        this->builder.CreateBr(bbnext);
    	        bb=bbnext;
    	    }
    	    this->builder.SetInsertPoint(bb);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 56);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 57: DIVU */
    std::tuple<continuation_e, BasicBlock*> __divu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("DIVU");
    	
    	this->gen_sync(PRE_SYNC, 57);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    {
    	        BasicBlock* bbnext = BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	        BasicBlock* bb_then = BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	        BasicBlock* bb_else = BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	        // this->builder.SetInsertPoint(bb);
    	        this->gen_cond_branch(this->builder.CreateICmp(
    	            ICmpInst::ICMP_NE,
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, 0)),
    	            bb_then,
    	            bb_else);
    	        this->builder.SetInsertPoint(bb_then);
    	        {
    	            Value* Xtmp0_val = this->builder.CreateUDiv(
    	                this->gen_ext(
    	                    this->gen_reg_load(rs1 + traits<ARCH>::X0, 1),
    	                    32,
    	                    false),
    	                this->gen_ext(
    	                    this->gen_reg_load(rs2 + traits<ARCH>::X0, 1),
    	                    32,
    	                    false));
    	            this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	        }
    	        this->builder.CreateBr(bbnext);
    	        this->builder.SetInsertPoint(bb_else);
    	        {
    	            Value* Xtmp1_val = this->builder.CreateNeg(this->gen_const(32U, 1));
    	            this->builder.CreateStore(Xtmp1_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	        }
    	        this->builder.CreateBr(bbnext);
    	        bb=bbnext;
    	    }
    	    this->builder.SetInsertPoint(bb);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 57);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 58: REM */
    std::tuple<continuation_e, BasicBlock*> __rem(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("REM");
    	
    	this->gen_sync(PRE_SYNC, 58);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    {
    	        BasicBlock* bbnext = BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	        BasicBlock* bb_then = BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	        BasicBlock* bb_else = BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	        // this->builder.SetInsertPoint(bb);
    	        this->gen_cond_branch(this->builder.CreateICmp(
    	            ICmpInst::ICMP_NE,
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, 0)),
    	            bb_then,
    	            bb_else);
    	        this->builder.SetInsertPoint(bb_then);
    	        {
    	            uint32_t M1_val = - 1;
    	            uint32_t MMIN_val = - 1 << 32 - 1;
    	            {
    	                BasicBlock* bbnext = BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	                BasicBlock* bb_then = BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	                BasicBlock* bb_else = BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	                // this->builder.SetInsertPoint(bb);
    	                this->gen_cond_branch(this->builder.CreateICmp(
    	                    ICmpInst::ICMP_EQ,
    	                    this->gen_ext(
    	                        this->gen_reg_load(rs1 + traits<ARCH>::X0, 1),
    	                        32, true),
    	                    this->gen_ext(
    	                        this->gen_const(32U, MMIN_val),
    	                        32, true)),
    	                    bb_then,
    	                    bb_else);
    	                this->builder.SetInsertPoint(bb_then);
    	                {
    	                    {
    	                        BasicBlock* bbnext = BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	                        BasicBlock* bb_then = BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	                        BasicBlock* bb_else = BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	                        // this->builder.SetInsertPoint(bb);
    	                        this->gen_cond_branch(this->builder.CreateICmp(
    	                            ICmpInst::ICMP_EQ,
    	                            this->gen_ext(
    	                                this->gen_reg_load(rs2 + traits<ARCH>::X0, 2),
    	                                32, true),
    	                            this->gen_ext(
    	                                this->gen_const(32U, M1_val),
    	                                32, true)),
    	                            bb_then,
    	                            bb_else);
    	                        this->builder.SetInsertPoint(bb_then);
    	                        {
    	                            Value* Xtmp0_val = this->gen_const(32U, 0);
    	                            this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	                        }
    	                        this->builder.CreateBr(bbnext);
    	                        this->builder.SetInsertPoint(bb_else);
    	                        {
    	                            Value* Xtmp1_val = this->builder.CreateURem(
    	                                this->gen_ext(
    	                                    this->gen_reg_load(rs1 + traits<ARCH>::X0, 3),
    	                                    32,
    	                                    true),
    	                                this->gen_ext(
    	                                    this->gen_reg_load(rs2 + traits<ARCH>::X0, 3),
    	                                    32,
    	                                    true));
    	                            this->builder.CreateStore(Xtmp1_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	                        }
    	                        this->builder.CreateBr(bbnext);
    	                        bb=bbnext;
    	                    }
    	                    this->builder.SetInsertPoint(bb);
    	                }
    	                this->builder.CreateBr(bbnext);
    	                this->builder.SetInsertPoint(bb_else);
    	                {
    	                    Value* Xtmp2_val = this->builder.CreateURem(
    	                        this->gen_ext(
    	                            this->gen_reg_load(rs1 + traits<ARCH>::X0, 2),
    	                            32,
    	                            true),
    	                        this->gen_ext(
    	                            this->gen_reg_load(rs2 + traits<ARCH>::X0, 2),
    	                            32,
    	                            true));
    	                    this->builder.CreateStore(Xtmp2_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	                }
    	                this->builder.CreateBr(bbnext);
    	                bb=bbnext;
    	            }
    	            this->builder.SetInsertPoint(bb);
    	        }
    	        this->builder.CreateBr(bbnext);
    	        this->builder.SetInsertPoint(bb_else);
    	        {
    	            Value* Xtmp3_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 1);
    	            this->builder.CreateStore(Xtmp3_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	        }
    	        this->builder.CreateBr(bbnext);
    	        bb=bbnext;
    	    }
    	    this->builder.SetInsertPoint(bb);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 58);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 59: REMU */
    std::tuple<continuation_e, BasicBlock*> __remu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("REMU");
    	
    	this->gen_sync(PRE_SYNC, 59);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    {
    	        BasicBlock* bbnext = BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	        BasicBlock* bb_then = BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	        BasicBlock* bb_else = BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	        // this->builder.SetInsertPoint(bb);
    	        this->gen_cond_branch(this->builder.CreateICmp(
    	            ICmpInst::ICMP_NE,
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, 0)),
    	            bb_then,
    	            bb_else);
    	        this->builder.SetInsertPoint(bb_then);
    	        {
    	            Value* Xtmp0_val = this->builder.CreateURem(
    	                this->gen_ext(
    	                    this->gen_reg_load(rs1 + traits<ARCH>::X0, 1),
    	                    32,
    	                    false),
    	                this->gen_ext(
    	                    this->gen_reg_load(rs2 + traits<ARCH>::X0, 1),
    	                    32,
    	                    false));
    	            this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	        }
    	        this->builder.CreateBr(bbnext);
    	        this->builder.SetInsertPoint(bb_else);
    	        {
    	            Value* Xtmp1_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 1);
    	            this->builder.CreateStore(Xtmp1_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	        }
    	        this->builder.CreateBr(bbnext);
    	        bb=bbnext;
    	    }
    	    this->builder.SetInsertPoint(bb);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 59);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 60: LR.W */
    std::tuple<continuation_e, BasicBlock*> __lr_w(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("LR.W");
    	
    	this->gen_sync(PRE_SYNC, 60);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rl = ((bit_sub<25,1>(instr)));
    	uint8_t aq = ((bit_sub<26,1>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "lr.w"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* offs_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	        32,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	    Value* REStmp1_val = this->gen_ext(
    	        this->builder.CreateNeg(this->gen_const(8U, 1)),
    	        32,
    	        true);
    	    this->gen_write_mem(
    	        traits<ARCH>::RES,
    	        offs_val,
    	        this->builder.CreateZExtOrTrunc(REStmp1_val,this->get_type(32)));
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 60);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 61: SC.W */
    std::tuple<continuation_e, BasicBlock*> __sc_w(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SC.W");
    	
    	this->gen_sync(PRE_SYNC, 61);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rl = ((bit_sub<25,1>(instr)));
    	uint8_t aq = ((bit_sub<26,1>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sc.w"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_read_mem(traits<ARCH>::RES, offs_val, 32/8);
    	{
    	    BasicBlock* bbnext = BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	    BasicBlock* bb_then = BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	    // this->builder.SetInsertPoint(bb);
    	    this->gen_cond_branch(this->builder.CreateICmp(
    	        ICmpInst::ICMP_NE,
    	        res1_val,
    	        this->gen_const(32U, 0)),
    	        bb_then,
    	        bbnext);
    	    this->builder.SetInsertPoint(bb_then);
    	    {
    	        Value* MEMtmp0_val = this->gen_reg_load(rs2 + traits<ARCH>::X0, 1);
    	        this->gen_write_mem(
    	            traits<ARCH>::MEM,
    	            offs_val,
    	            this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(32)));
    	    }
    	    this->builder.CreateBr(bbnext);
    	    bb=bbnext;
    	}
    	this->builder.SetInsertPoint(bb);
    	if(rd != 0){
    	    Value* Xtmp1_val = this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_NE,
    	            res1_val,
    	            this->gen_const(32U, 0)),
    	        this->gen_const(32U, 0),
    	        this->gen_const(32U, 1),
    	        32);
    	    this->builder.CreateStore(Xtmp1_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 61);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 62: AMOSWAP.W */
    std::tuple<continuation_e, BasicBlock*> __amoswap_w(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("AMOSWAP.W");
    	
    	this->gen_sync(PRE_SYNC, 62);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rl = ((bit_sub<25,1>(instr)));
    	uint8_t aq = ((bit_sub<26,1>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amoswap.w"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	        32,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	Value* MEMtmp1_val = this->gen_reg_load(rs2 + traits<ARCH>::X0, 0);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 62);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 63: AMOADD.W */
    std::tuple<continuation_e, BasicBlock*> __amoadd_w(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("AMOADD.W");
    	
    	this->gen_sync(PRE_SYNC, 63);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rl = ((bit_sub<25,1>(instr)));
    	uint8_t aq = ((bit_sub<26,1>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amoadd.w"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_ext(
    	    this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	    32,
    	    true);
    	if(rd != 0){
    	    Value* Xtmp0_val = res1_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	Value* res2_val = this->builder.CreateAdd(
    	    res1_val,
    	    this->gen_reg_load(rs2 + traits<ARCH>::X0, 0));
    	Value* MEMtmp1_val = res2_val;
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 63);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 64: AMOXOR.W */
    std::tuple<continuation_e, BasicBlock*> __amoxor_w(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("AMOXOR.W");
    	
    	this->gen_sync(PRE_SYNC, 64);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rl = ((bit_sub<25,1>(instr)));
    	uint8_t aq = ((bit_sub<26,1>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amoxor.w"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_ext(
    	    this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	    32,
    	    true);
    	if(rd != 0){
    	    Value* Xtmp0_val = res1_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	Value* res2_val = this->builder.CreateXor(
    	    res1_val,
    	    this->gen_reg_load(rs2 + traits<ARCH>::X0, 0));
    	Value* MEMtmp1_val = res2_val;
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 64);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 65: AMOAND.W */
    std::tuple<continuation_e, BasicBlock*> __amoand_w(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("AMOAND.W");
    	
    	this->gen_sync(PRE_SYNC, 65);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rl = ((bit_sub<25,1>(instr)));
    	uint8_t aq = ((bit_sub<26,1>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amoand.w"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_ext(
    	    this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	    32,
    	    true);
    	if(rd != 0){
    	    Value* Xtmp0_val = res1_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	Value* res2_val = this->builder.CreateAnd(
    	    res1_val,
    	    this->gen_reg_load(rs2 + traits<ARCH>::X0, 0));
    	Value* MEMtmp1_val = res2_val;
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 65);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 66: AMOOR.W */
    std::tuple<continuation_e, BasicBlock*> __amoor_w(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("AMOOR.W");
    	
    	this->gen_sync(PRE_SYNC, 66);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rl = ((bit_sub<25,1>(instr)));
    	uint8_t aq = ((bit_sub<26,1>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amoor.w"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_ext(
    	    this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	    32,
    	    true);
    	if(rd != 0){
    	    Value* Xtmp0_val = res1_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	Value* res2_val = this->builder.CreateOr(
    	    res1_val,
    	    this->gen_reg_load(rs2 + traits<ARCH>::X0, 0));
    	Value* MEMtmp1_val = res2_val;
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 66);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 67: AMOMIN.W */
    std::tuple<continuation_e, BasicBlock*> __amomin_w(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("AMOMIN.W");
    	
    	this->gen_sync(PRE_SYNC, 67);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rl = ((bit_sub<25,1>(instr)));
    	uint8_t aq = ((bit_sub<26,1>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amomin.w"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_ext(
    	    this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	    32,
    	    true);
    	if(rd != 0){
    	    Value* Xtmp0_val = res1_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	Value* res2_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_SGT,
    	        this->gen_ext(
    	            res1_val,
    	            32, true),
    	        this->gen_ext(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            32, true)),
    	    this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	    res1_val,
    	    32);
    	Value* MEMtmp1_val = res2_val;
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 67);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 68: AMOMAX.W */
    std::tuple<continuation_e, BasicBlock*> __amomax_w(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("AMOMAX.W");
    	
    	this->gen_sync(PRE_SYNC, 68);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rl = ((bit_sub<25,1>(instr)));
    	uint8_t aq = ((bit_sub<26,1>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amomax.w"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_ext(
    	    this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	    32,
    	    true);
    	if(rd != 0){
    	    Value* Xtmp0_val = res1_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	Value* res2_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_SLT,
    	        this->gen_ext(
    	            res1_val,
    	            32, true),
    	        this->gen_ext(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            32, true)),
    	    this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	    res1_val,
    	    32);
    	Value* MEMtmp1_val = res2_val;
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 68);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 69: AMOMINU.W */
    std::tuple<continuation_e, BasicBlock*> __amominu_w(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("AMOMINU.W");
    	
    	this->gen_sync(PRE_SYNC, 69);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rl = ((bit_sub<25,1>(instr)));
    	uint8_t aq = ((bit_sub<26,1>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amominu.w"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_ext(
    	    this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	    32,
    	    false);
    	if(rd != 0){
    	    Value* Xtmp0_val = res1_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	Value* res2_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_UGT,
    	        res1_val,
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0)),
    	    this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	    res1_val,
    	    32);
    	Value* MEMtmp1_val = res2_val;
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 69);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 70: AMOMAXU.W */
    std::tuple<continuation_e, BasicBlock*> __amomaxu_w(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("AMOMAXU.W");
    	
    	this->gen_sync(PRE_SYNC, 70);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rl = ((bit_sub<25,1>(instr)));
    	uint8_t aq = ((bit_sub<26,1>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amomaxu.w"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_ext(
    	    this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	    32,
    	    false);
    	if(rd != 0){
    	    Value* Xtmp0_val = res1_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	Value* res2_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_ULT,
    	        res1_val,
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0)),
    	    this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	    res1_val,
    	    32);
    	Value* MEMtmp1_val = res2_val;
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 70);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 71: C.ADDI4SPN */
    std::tuple<continuation_e, BasicBlock*> __c_addi4spn(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.ADDI4SPN");
    	
    	this->gen_sync(PRE_SYNC, 71);
    	
    	uint8_t rd = ((bit_sub<2,3>(instr)));
    	uint16_t imm = ((bit_sub<5,1>(instr) << 3) | (bit_sub<6,1>(instr) << 2) | (bit_sub<7,4>(instr) << 6) | (bit_sub<11,2>(instr) << 4));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c.addi4spn"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	if(imm == 0){
    	    this->gen_raise_trap(0, 2);
    	}
    	Value* Xtmp0_val = this->builder.CreateAdd(
    	    this->gen_reg_load(2 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, imm));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + 8 + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 71);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 72: C.LW */
    std::tuple<continuation_e, BasicBlock*> __c_lw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.LW");
    	
    	this->gen_sync(PRE_SYNC, 72);
    	
    	uint8_t rd = ((bit_sub<2,3>(instr)));
    	uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
    	uint8_t rs1 = ((bit_sub<7,3>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, {rs1}, {uimm:#05x}", fmt::arg("mnemonic", "c.lw"),
    	    	fmt::arg("rd", name(8+rd)), fmt::arg("rs1", name(8+rs1)), fmt::arg("uimm", uimm));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_reg_load(rs1 + 8 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, uimm));
    	Value* Xtmp0_val = this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8);
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + 8 + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 72);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 73: C.SW */
    std::tuple<continuation_e, BasicBlock*> __c_sw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.SW");
    	
    	this->gen_sync(PRE_SYNC, 73);
    	
    	uint8_t rs2 = ((bit_sub<2,3>(instr)));
    	uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
    	uint8_t rs1 = ((bit_sub<7,3>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rs1}, {rs2}, {uimm:#05x}", fmt::arg("mnemonic", "c.sw"),
    	    	fmt::arg("rs1", name(8+rs1)), fmt::arg("rs2", name(8+rs2)), fmt::arg("uimm", uimm));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_reg_load(rs1 + 8 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, uimm));
    	Value* MEMtmp0_val = this->gen_reg_load(rs2 + 8 + traits<ARCH>::X0, 0);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 73);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 74: C.ADDI */
    std::tuple<continuation_e, BasicBlock*> __c_addi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.ADDI");
    	
    	this->gen_sync(PRE_SYNC, 74);
    	
    	int8_t imm = signextend<int8_t,6>((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* Xtmp0_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, imm));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rs1 + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 74);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 75: C.NOP */
    std::tuple<continuation_e, BasicBlock*> __c_nop(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.NOP");
    	
    	this->gen_sync(PRE_SYNC, 75);
    	
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("c.nop"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	/* TODO: describe operations for C.NOP ! */
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 75);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 76: C.JAL */
    std::tuple<continuation_e, BasicBlock*> __c_jal(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.JAL");
    	
    	this->gen_sync(PRE_SYNC, 76);
    	
    	int16_t imm = signextend<int16_t,12>((bit_sub<2,1>(instr) << 5) | (bit_sub<3,3>(instr) << 1) | (bit_sub<6,1>(instr) << 7) | (bit_sub<7,1>(instr) << 6) | (bit_sub<8,1>(instr) << 10) | (bit_sub<9,2>(instr) << 8) | (bit_sub<11,1>(instr) << 4) | (bit_sub<12,1>(instr) << 11));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* Xtmp0_val = this->builder.CreateAdd(
    	    cur_pc_val,
    	    this->gen_const(32U, 2));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(1 + traits<ARCH>::X0), false);
    	Value* PC_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        cur_pc_val,
    	        32, true),
    	    this->gen_const(32U, imm));
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 76);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 77: C.LI */
    std::tuple<continuation_e, BasicBlock*> __c_li(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.LI");
    	
    	this->gen_sync(PRE_SYNC, 77);
    	
    	int8_t imm = signextend<int8_t,6>((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	if(rd == 0){
    	    this->gen_raise_trap(0, 2);
    	}
    	Value* Xtmp0_val = this->gen_const(32U, imm);
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 77);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 78: C.LUI */
    std::tuple<continuation_e, BasicBlock*> __c_lui(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.LUI");
    	
    	this->gen_sync(PRE_SYNC, 78);
    	
    	int32_t imm = signextend<int32_t,18>((bit_sub<2,5>(instr) << 12) | (bit_sub<12,1>(instr) << 17));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	if(rd == 0){
    	    this->gen_raise_trap(0, 2);
    	}
    	if(imm == 0){
    	    this->gen_raise_trap(0, 2);
    	}
    	Value* Xtmp0_val = this->gen_const(32U, imm);
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 78);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 79: C.ADDI16SP */
    std::tuple<continuation_e, BasicBlock*> __c_addi16sp(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.ADDI16SP");
    	
    	this->gen_sync(PRE_SYNC, 79);
    	
    	int16_t imm = signextend<int16_t,10>((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 7) | (bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 4) | (bit_sub<12,1>(instr) << 9));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {imm:#05x}", fmt::arg("mnemonic", "c.addi16sp"),
    	    	fmt::arg("imm", imm));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* Xtmp0_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(2 + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, imm));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(2 + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 79);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 80: C.SRLI */
    std::tuple<continuation_e, BasicBlock*> __c_srli(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.SRLI");
    	
    	this->gen_sync(PRE_SYNC, 80);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	uint8_t rs1_idx_val = rs1 + 8;
    	Value* Xtmp0_val = this->builder.CreateLShr(
    	    this->gen_reg_load(rs1_idx_val + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, shamt));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rs1_idx_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 80);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 81: C.SRAI */
    std::tuple<continuation_e, BasicBlock*> __c_srai(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.SRAI");
    	
    	this->gen_sync(PRE_SYNC, 81);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	uint8_t rs1_idx_val = rs1 + 8;
    	Value* Xtmp0_val = this->builder.CreateAShr(
    	    this->gen_reg_load(rs1_idx_val + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, shamt));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rs1_idx_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 81);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 82: C.ANDI */
    std::tuple<continuation_e, BasicBlock*> __c_andi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.ANDI");
    	
    	this->gen_sync(PRE_SYNC, 82);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	uint8_t rs1_idx_val = rs1 + 8;
    	Value* Xtmp0_val = this->builder.CreateAnd(
    	    this->gen_reg_load(rs1_idx_val + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, imm));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rs1_idx_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 82);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 83: C.SUB */
    std::tuple<continuation_e, BasicBlock*> __c_sub(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.SUB");
    	
    	this->gen_sync(PRE_SYNC, 83);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	uint8_t rd_idx_val = rd + 8;
    	Value* Xtmp0_val = this->builder.CreateSub(
    	     this->gen_reg_load(rd_idx_val + traits<ARCH>::X0, 0),
    	     this->gen_reg_load(rs2 + 8 + traits<ARCH>::X0, 0));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd_idx_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 83);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 84: C.XOR */
    std::tuple<continuation_e, BasicBlock*> __c_xor(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.XOR");
    	
    	this->gen_sync(PRE_SYNC, 84);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	uint8_t rd_idx_val = rd + 8;
    	Value* Xtmp0_val = this->builder.CreateXor(
    	    this->gen_reg_load(rd_idx_val + traits<ARCH>::X0, 0),
    	    this->gen_reg_load(rs2 + 8 + traits<ARCH>::X0, 0));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd_idx_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 84);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 85: C.OR */
    std::tuple<continuation_e, BasicBlock*> __c_or(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.OR");
    	
    	this->gen_sync(PRE_SYNC, 85);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	uint8_t rd_idx_val = rd + 8;
    	Value* Xtmp0_val = this->builder.CreateOr(
    	    this->gen_reg_load(rd_idx_val + traits<ARCH>::X0, 0),
    	    this->gen_reg_load(rs2 + 8 + traits<ARCH>::X0, 0));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd_idx_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 85);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 86: C.AND */
    std::tuple<continuation_e, BasicBlock*> __c_and(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.AND");
    	
    	this->gen_sync(PRE_SYNC, 86);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	uint8_t rd_idx_val = rd + 8;
    	Value* Xtmp0_val = this->builder.CreateAnd(
    	    this->gen_reg_load(rd_idx_val + traits<ARCH>::X0, 0),
    	    this->gen_reg_load(rs2 + 8 + traits<ARCH>::X0, 0));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd_idx_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 86);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 87: C.J */
    std::tuple<continuation_e, BasicBlock*> __c_j(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.J");
    	
    	this->gen_sync(PRE_SYNC, 87);
    	
    	int16_t imm = signextend<int16_t,12>((bit_sub<2,1>(instr) << 5) | (bit_sub<3,3>(instr) << 1) | (bit_sub<6,1>(instr) << 7) | (bit_sub<7,1>(instr) << 6) | (bit_sub<8,1>(instr) << 10) | (bit_sub<9,2>(instr) << 8) | (bit_sub<11,1>(instr) << 4) | (bit_sub<12,1>(instr) << 11));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* PC_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        cur_pc_val,
    	        32, true),
    	    this->gen_const(32U, imm));
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 87);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 88: C.BEQZ */
    std::tuple<continuation_e, BasicBlock*> __c_beqz(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.BEQZ");
    	
    	this->gen_sync(PRE_SYNC, 88);
    	
    	int16_t imm = signextend<int16_t,9>((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 1) | (bit_sub<5,2>(instr) << 6) | (bit_sub<10,2>(instr) << 3) | (bit_sub<12,1>(instr) << 8));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_EQ,
    	        this->gen_reg_load(rs1 + 8 + traits<ARCH>::X0, 0),
    	        this->gen_const(32U, 0)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, imm)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 2)),
    	    32);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 88);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 89: C.BNEZ */
    std::tuple<continuation_e, BasicBlock*> __c_bnez(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.BNEZ");
    	
    	this->gen_sync(PRE_SYNC, 89);
    	
    	int16_t imm = signextend<int16_t,9>((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 1) | (bit_sub<5,2>(instr) << 6) | (bit_sub<10,2>(instr) << 3) | (bit_sub<12,1>(instr) << 8));
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_NE,
    	        this->gen_reg_load(rs1 + 8 + traits<ARCH>::X0, 0),
    	        this->gen_const(32U, 0)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, imm)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 2)),
    	    32);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 89);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 90: C.SLLI */
    std::tuple<continuation_e, BasicBlock*> __c_slli(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.SLLI");
    	
    	this->gen_sync(PRE_SYNC, 90);
    	
    	uint8_t shamt = ((bit_sub<2,5>(instr)));
    	uint8_t rs1 = ((bit_sub<7,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rs1}, {shamt}", fmt::arg("mnemonic", "c.slli"),
    	    	fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	if(rs1 == 0){
    	    this->gen_raise_trap(0, 2);
    	}
    	Value* Xtmp0_val = this->builder.CreateShl(
    	    this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, shamt));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rs1 + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 90);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 91: C.LWSP */
    std::tuple<continuation_e, BasicBlock*> __c_lwsp(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.LWSP");
    	
    	this->gen_sync(PRE_SYNC, 91);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_reg_load(2 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, uimm));
    	Value* Xtmp0_val = this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8);
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 91);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 92: C.MV */
    std::tuple<continuation_e, BasicBlock*> __c_mv(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.MV");
    	
    	this->gen_sync(PRE_SYNC, 92);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* Xtmp0_val = this->gen_reg_load(rs2 + traits<ARCH>::X0, 0);
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 92);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 93: C.JR */
    std::tuple<continuation_e, BasicBlock*> __c_jr(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.JR");
    	
    	this->gen_sync(PRE_SYNC, 93);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* PC_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()), get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 93);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 94: C.ADD */
    std::tuple<continuation_e, BasicBlock*> __c_add(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.ADD");
    	
    	this->gen_sync(PRE_SYNC, 94);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* Xtmp0_val = this->builder.CreateAdd(
    	    this->gen_reg_load(rd + traits<ARCH>::X0, 0),
    	    this->gen_reg_load(rs2 + traits<ARCH>::X0, 0));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 94);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 95: C.JALR */
    std::tuple<continuation_e, BasicBlock*> __c_jalr(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.JALR");
    	
    	this->gen_sync(PRE_SYNC, 95);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* Xtmp0_val = this->builder.CreateAdd(
    	    cur_pc_val,
    	    this->gen_const(32U, 2));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(1 + traits<ARCH>::X0), false);
    	Value* PC_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()), get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 95);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 96: C.EBREAK */
    std::tuple<continuation_e, BasicBlock*> __c_ebreak(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.EBREAK");
    	
    	this->gen_sync(PRE_SYNC, 96);
    	
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("c.ebreak"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	this->gen_raise_trap(0, 3);
    	this->gen_sync(POST_SYNC, 96);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 97: C.SWSP */
    std::tuple<continuation_e, BasicBlock*> __c_swsp(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.SWSP");
    	
    	this->gen_sync(PRE_SYNC, 97);
    	
    	uint8_t rs2 = ((bit_sub<2,5>(instr)));
    	uint8_t uimm = ((bit_sub<7,2>(instr) << 6) | (bit_sub<9,4>(instr) << 2));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} x2+{uimm:#05x}, {rs2}", fmt::arg("mnemonic", "c.swsp"),
    	    	fmt::arg("uimm", uimm), fmt::arg("rs2", name(rs2)));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_reg_load(2 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, uimm));
    	Value* MEMtmp0_val = this->gen_reg_load(rs2 + traits<ARCH>::X0, 0);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 97);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 98: DII */
    std::tuple<continuation_e, BasicBlock*> __dii(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("DII");
    	
    	this->gen_sync(PRE_SYNC, 98);
    	
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("dii"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	this->gen_raise_trap(0, 2);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 98);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 99: FLW */
    std::tuple<continuation_e, BasicBlock*> __flw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FLW");
    	
    	this->gen_sync(PRE_SYNC, 99);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, {imm}(x{rs1})", fmt::arg("mnemonic", "flw"),
    	    	fmt::arg("rd", rd), fmt::arg("imm", imm), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, imm));
    	Value* res_val = this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8);
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 99);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 100: FSW */
    std::tuple<continuation_e, BasicBlock*> __fsw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FSW");
    	
    	this->gen_sync(PRE_SYNC, 100);
    	
    	int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rs2}, {imm}(x{rs1})", fmt::arg("mnemonic", "fsw"),
    	    	fmt::arg("rs2", rs2), fmt::arg("imm", imm), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, imm));
    	Value* MEMtmp0_val = this->builder.CreateTrunc(
    	    this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	    this-> get_type(32) 
    	);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 100);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 101: FMADD.S */
    std::tuple<continuation_e, BasicBlock*> __fmadd_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FMADD.S");
    	
    	this->gen_sync(PRE_SYNC, 101);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rs3 = ((bit_sub<27,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} x{rd}, f{rs1}, f{rs2}, f{rs3}", fmt::arg("mnemonic", "fmadd.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2), fmt::arg("rs3", rs3));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fmadd_s"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs3 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 0LL),
    	        32,
    	        false), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 101);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 102: FMSUB.S */
    std::tuple<continuation_e, BasicBlock*> __fmsub_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FMSUB.S");
    	
    	this->gen_sync(PRE_SYNC, 102);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rs3 = ((bit_sub<27,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} x{rd}, f{rs1}, f{rs2}, f{rs3}", fmt::arg("mnemonic", "fmsub.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2), fmt::arg("rs3", rs3));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fmadd_s"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs3 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 1LL),
    	        32,
    	        false), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 102);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 103: FNMADD.S */
    std::tuple<continuation_e, BasicBlock*> __fnmadd_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FNMADD.S");
    	
    	this->gen_sync(PRE_SYNC, 103);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rs3 = ((bit_sub<27,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} x{rd}, f{rs1}, f{rs2}, f{rs3}", fmt::arg("mnemonic", "fnmadd.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2), fmt::arg("rs3", rs3));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fmadd_s"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs3 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 2LL),
    	        32,
    	        false), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 103);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 104: FNMSUB.S */
    std::tuple<continuation_e, BasicBlock*> __fnmsub_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FNMSUB.S");
    	
    	this->gen_sync(PRE_SYNC, 104);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rs3 = ((bit_sub<27,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} x{rd}, f{rs1}, f{rs2}, f{rs3}", fmt::arg("mnemonic", "fnmsub.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2), fmt::arg("rs3", rs3));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fmadd_s"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs3 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 3LL),
    	        32,
    	        false), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 104);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 105: FADD.S */
    std::tuple<continuation_e, BasicBlock*> __fadd_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FADD.S");
    	
    	this->gen_sync(PRE_SYNC, 105);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fadd.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fadd_s"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 105);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 106: FSUB.S */
    std::tuple<continuation_e, BasicBlock*> __fsub_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FSUB.S");
    	
    	this->gen_sync(PRE_SYNC, 106);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fsub.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fsub_s"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 106);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 107: FMUL.S */
    std::tuple<continuation_e, BasicBlock*> __fmul_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FMUL.S");
    	
    	this->gen_sync(PRE_SYNC, 107);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fmul.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fmul_s"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 107);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 108: FDIV.S */
    std::tuple<continuation_e, BasicBlock*> __fdiv_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FDIV.S");
    	
    	this->gen_sync(PRE_SYNC, 108);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fdiv.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fdiv_s"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 108);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 109: FSQRT.S */
    std::tuple<continuation_e, BasicBlock*> __fsqrt_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FSQRT.S");
    	
    	this->gen_sync(PRE_SYNC, 109);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, f{rs1}", fmt::arg("mnemonic", "fsqrt.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fsqrt_s"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 109);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 110: FSGNJ.S */
    std::tuple<continuation_e, BasicBlock*> __fsgnj_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FSGNJ.S");
    	
    	this->gen_sync(PRE_SYNC, 110);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fsgnj.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateOr(
    	    this->builder.CreateAnd(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	            this-> get_type(32) 
    	        ),
    	        this->gen_const(32U, 0x7fffffff)),
    	    this->builder.CreateAnd(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	            this-> get_type(32) 
    	        ),
    	        this->gen_const(32U, 0x80000000)));
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 110);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 111: FSGNJN.S */
    std::tuple<continuation_e, BasicBlock*> __fsgnjn_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FSGNJN.S");
    	
    	this->gen_sync(PRE_SYNC, 111);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fsgnjn.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateOr(
    	    this->builder.CreateAnd(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	            this-> get_type(32) 
    	        ),
    	        this->gen_const(32U, 0x7fffffff)),
    	    this->builder.CreateAnd(
    	        this->builder.CreateNot(this->builder.CreateTrunc(
    	            this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	            this-> get_type(32) 
    	        )),
    	        this->gen_const(32U, 0x80000000)));
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 111);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 112: FSGNJX.S */
    std::tuple<continuation_e, BasicBlock*> __fsgnjx_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FSGNJX.S");
    	
    	this->gen_sync(PRE_SYNC, 112);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fsgnjx.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateXor(
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ),
    	    this->builder.CreateAnd(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	            this-> get_type(32) 
    	        ),
    	        this->gen_const(32U, 0x80000000)));
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 112);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 113: FMIN.S */
    std::tuple<continuation_e, BasicBlock*> __fmin_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FMIN.S");
    	
    	this->gen_sync(PRE_SYNC, 113);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fmin.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fsel_s"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 0LL),
    	        32,
    	        false)
    	});
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 113);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 114: FMAX.S */
    std::tuple<continuation_e, BasicBlock*> __fmax_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FMAX.S");
    	
    	this->gen_sync(PRE_SYNC, 114);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fmax.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fsel_s"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 1LL),
    	        32,
    	        false)
    	});
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 114);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 115: FCVT.W.S */
    std::tuple<continuation_e, BasicBlock*> __fcvt_w_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FCVT.W.S");
    	
    	this->gen_sync(PRE_SYNC, 115);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} x{rd}, f{rs1}", fmt::arg("mnemonic", "fcvt.w.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* Xtmp0_val = this->gen_ext(
    	    this->builder.CreateCall(this->mod->getFunction("fcvt_s"), std::vector<Value*>{
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	            this-> get_type(32) 
    	        ), 
    	        this->gen_ext(
    	            this->gen_const(64U, 0LL),
    	            32,
    	            false), 
    	        this->gen_const(8U, rm)
    	    }),
    	    32,
    	    true);
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 115);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 116: FCVT.WU.S */
    std::tuple<continuation_e, BasicBlock*> __fcvt_wu_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FCVT.WU.S");
    	
    	this->gen_sync(PRE_SYNC, 116);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} x{rd}, f{rs1}", fmt::arg("mnemonic", "fcvt.wu.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* Xtmp0_val = this->gen_ext(
    	    this->builder.CreateCall(this->mod->getFunction("fcvt_s"), std::vector<Value*>{
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	            this-> get_type(32) 
    	        ), 
    	        this->gen_ext(
    	            this->gen_const(64U, 1LL),
    	            32,
    	            false), 
    	        this->gen_const(8U, rm)
    	    }),
    	    32,
    	    false);
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 116);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 117: FEQ.S */
    std::tuple<continuation_e, BasicBlock*> __feq_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FEQ.S");
    	
    	this->gen_sync(PRE_SYNC, 117);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} x{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "feq.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* Xtmp0_val = this->builder.CreateCall(this->mod->getFunction("fcmp_s"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 0LL),
    	        32,
    	        false)
    	});
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 117);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 118: FLT.S */
    std::tuple<continuation_e, BasicBlock*> __flt_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FLT.S");
    	
    	this->gen_sync(PRE_SYNC, 118);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} x{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "flt.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* Xtmp0_val = this->builder.CreateCall(this->mod->getFunction("fcmp_s"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 2LL),
    	        32,
    	        false)
    	});
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 118);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 119: FLE.S */
    std::tuple<continuation_e, BasicBlock*> __fle_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FLE.S");
    	
    	this->gen_sync(PRE_SYNC, 119);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} x{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fle.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* Xtmp0_val = this->builder.CreateCall(this->mod->getFunction("fcmp_s"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 1LL),
    	        32,
    	        false)
    	});
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 119);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 120: FCLASS.S */
    std::tuple<continuation_e, BasicBlock*> __fclass_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FCLASS.S");
    	
    	this->gen_sync(PRE_SYNC, 120);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} x{rd}, f{rs1}", fmt::arg("mnemonic", "fclass.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* Xtmp0_val = this->builder.CreateCall(this->mod->getFunction("fclass_s"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    )
    	});
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 120);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 121: FCVT.S.W */
    std::tuple<continuation_e, BasicBlock*> __fcvt_s_w(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FCVT.S.W");
    	
    	this->gen_sync(PRE_SYNC, 121);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, x{rs1}", fmt::arg("mnemonic", "fcvt.s.w"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fcvt_s"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 2LL),
    	        32,
    	        false), 
    	    this->gen_const(8U, rm)
    	});
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 121);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 122: FCVT.S.WU */
    std::tuple<continuation_e, BasicBlock*> __fcvt_s_wu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FCVT.S.WU");
    	
    	this->gen_sync(PRE_SYNC, 122);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, x{rs1}", fmt::arg("mnemonic", "fcvt.s.wu"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fcvt_s"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 3LL),
    	        32,
    	        false), 
    	    this->gen_const(8U, rm)
    	});
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 122);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 123: FMV.X.W */
    std::tuple<continuation_e, BasicBlock*> __fmv_x_w(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FMV.X.W");
    	
    	this->gen_sync(PRE_SYNC, 123);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} x{rd}, f{rs1}", fmt::arg("mnemonic", "fmv.x.w"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* Xtmp0_val = this->gen_ext(
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ),
    	    32,
    	    true);
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 123);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 124: FMV.W.X */
    std::tuple<continuation_e, BasicBlock*> __fmv_w_x(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FMV.W.X");
    	
    	this->gen_sync(PRE_SYNC, 124);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, x{rs1}", fmt::arg("mnemonic", "fmv.w.x"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(64 == 32){
    	    Value* Ftmp0_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 124);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 125: C.FLW */
    std::tuple<continuation_e, BasicBlock*> __c_flw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.FLW");
    	
    	this->gen_sync(PRE_SYNC, 125);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_reg_load(rs1 + 8 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, uimm));
    	Value* res_val = this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8);
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + 8 + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + 8 + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 125);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 126: C.FSW */
    std::tuple<continuation_e, BasicBlock*> __c_fsw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.FSW");
    	
    	this->gen_sync(PRE_SYNC, 126);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_reg_load(rs1 + 8 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, uimm));
    	Value* MEMtmp0_val = this->builder.CreateTrunc(
    	    this->gen_reg_load(rs2 + 8 + traits<ARCH>::F0, 0),
    	    this-> get_type(32) 
    	);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 126);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 127: C.FLWSP */
    std::tuple<continuation_e, BasicBlock*> __c_flwsp(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.FLWSP");
    	
    	this->gen_sync(PRE_SYNC, 127);
    	
    	uint8_t uimm = ((bit_sub<2,2>(instr) << 6) | (bit_sub<4,3>(instr) << 2) | (bit_sub<12,1>(instr) << 5));
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, {uimm}(x2)", fmt::arg("mnemonic", "c.flwsp"),
    	    	fmt::arg("rd", rd), fmt::arg("uimm", uimm));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_reg_load(2 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, uimm));
    	Value* res_val = this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8);
    	if(64 == 32){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 32)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 127);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 128: C.FSWSP */
    std::tuple<continuation_e, BasicBlock*> __c_fswsp(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.FSWSP");
    	
    	this->gen_sync(PRE_SYNC, 128);
    	
    	uint8_t rs2 = ((bit_sub<2,5>(instr)));
    	uint8_t uimm = ((bit_sub<7,2>(instr) << 6) | (bit_sub<9,4>(instr) << 2));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rs2}, {uimm}(x2), ", fmt::arg("mnemonic", "c.fswsp"),
    	    	fmt::arg("rs2", rs2), fmt::arg("uimm", uimm));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_reg_load(2 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, uimm));
    	Value* MEMtmp0_val = this->builder.CreateTrunc(
    	    this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	    this-> get_type(32) 
    	);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 128);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 129: FLD */
    std::tuple<continuation_e, BasicBlock*> __fld(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FLD");
    	
    	this->gen_sync(PRE_SYNC, 129);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, {imm}({rs1})", fmt::arg("mnemonic", "fld"),
    	    	fmt::arg("rd", rd), fmt::arg("imm", imm), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, imm));
    	Value* res_val = this->gen_read_mem(traits<ARCH>::MEM, offs_val, 64/8);
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 129);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 130: FSD */
    std::tuple<continuation_e, BasicBlock*> __fsd(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FSD");
    	
    	this->gen_sync(PRE_SYNC, 130);
    	
    	int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rs2}, {imm}({rs1})", fmt::arg("mnemonic", "fsd"),
    	    	fmt::arg("rs2", rs2), fmt::arg("imm", imm), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, imm));
    	Value* MEMtmp0_val = this->builder.CreateTrunc(
    	    this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	    this-> get_type(64) 
    	);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(64)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 130);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 131: FMADD.D */
    std::tuple<continuation_e, BasicBlock*> __fmadd_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FMADD.D");
    	
    	this->gen_sync(PRE_SYNC, 131);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rs3 = ((bit_sub<27,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, f{rs1}, f{rs2}, f{rs3}", fmt::arg("mnemonic", "fmadd.d"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2), fmt::arg("rs3", rs3));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fmadd_d"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs3 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 0LL),
    	        64,
    	        false), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 131);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 132: FMSUB.D */
    std::tuple<continuation_e, BasicBlock*> __fmsub_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FMSUB.D");
    	
    	this->gen_sync(PRE_SYNC, 132);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rs3 = ((bit_sub<27,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, f{rs1}, f{rs2}, f{rs3}", fmt::arg("mnemonic", "fmsub.d"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2), fmt::arg("rs3", rs3));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fmadd_d"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs3 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 1LL),
    	        32,
    	        false), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 132);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 133: FNMADD.D */
    std::tuple<continuation_e, BasicBlock*> __fnmadd_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FNMADD.D");
    	
    	this->gen_sync(PRE_SYNC, 133);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rs3 = ((bit_sub<27,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, f{rs1}, f{rs2}, f{rs3}", fmt::arg("mnemonic", "fnmadd.d"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2), fmt::arg("rs3", rs3));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fmadd_d"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs3 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 2LL),
    	        32,
    	        false), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 133);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 134: FNMSUB.D */
    std::tuple<continuation_e, BasicBlock*> __fnmsub_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FNMSUB.D");
    	
    	this->gen_sync(PRE_SYNC, 134);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	uint8_t rs3 = ((bit_sub<27,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, f{rs1}, f{rs2}, f{rs3}", fmt::arg("mnemonic", "fnmsub.d"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2), fmt::arg("rs3", rs3));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fmadd_d"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs3 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 3LL),
    	        32,
    	        false), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 134);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 135: FADD.D */
    std::tuple<continuation_e, BasicBlock*> __fadd_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FADD.D");
    	
    	this->gen_sync(PRE_SYNC, 135);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fadd.d"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fadd_d"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 135);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 136: FSUB.D */
    std::tuple<continuation_e, BasicBlock*> __fsub_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FSUB.D");
    	
    	this->gen_sync(PRE_SYNC, 136);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fsub.d"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fsub_d"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 136);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 137: FMUL.D */
    std::tuple<continuation_e, BasicBlock*> __fmul_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FMUL.D");
    	
    	this->gen_sync(PRE_SYNC, 137);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fmul.d"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fmul_d"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 137);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 138: FDIV.D */
    std::tuple<continuation_e, BasicBlock*> __fdiv_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FDIV.D");
    	
    	this->gen_sync(PRE_SYNC, 138);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fdiv.d"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fdiv_d"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 138);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 139: FSQRT.D */
    std::tuple<continuation_e, BasicBlock*> __fsqrt_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FSQRT.D");
    	
    	this->gen_sync(PRE_SYNC, 139);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, f{rs1}", fmt::arg("mnemonic", "fsqrt.d"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fsqrt_d"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_const(8U, rm),
    	            this->gen_const(8U, 7)),
    	        this->gen_const(8U, rm),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	            this-> get_type(8) 
    	        ),
    	        8)
    	});
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 139);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 140: FSGNJ.D */
    std::tuple<continuation_e, BasicBlock*> __fsgnj_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FSGNJ.D");
    	
    	this->gen_sync(PRE_SYNC, 140);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fsgnj.d"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateOr(
    	    this->builder.CreateAnd(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	            this-> get_type(64) 
    	        ),
    	        this->gen_const(64U, 0x7fffffff)),
    	    this->builder.CreateAnd(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	            this-> get_type(64) 
    	        ),
    	        this->gen_const(64U, 0x80000000)));
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 140);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 141: FSGNJN.D */
    std::tuple<continuation_e, BasicBlock*> __fsgnjn_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FSGNJN.D");
    	
    	this->gen_sync(PRE_SYNC, 141);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fsgnjn.d"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateOr(
    	    this->builder.CreateAnd(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	            this-> get_type(64) 
    	        ),
    	        this->gen_const(64U, 0x7fffffff)),
    	    this->builder.CreateAnd(
    	        this->builder.CreateNot(this->builder.CreateTrunc(
    	            this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	            this-> get_type(64) 
    	        )),
    	        this->gen_const(64U, 0x80000000)));
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 141);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 142: FSGNJX.D */
    std::tuple<continuation_e, BasicBlock*> __fsgnjx_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FSGNJX.D");
    	
    	this->gen_sync(PRE_SYNC, 142);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fsgnjx.d"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateXor(
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ),
    	    this->builder.CreateAnd(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	            this-> get_type(64) 
    	        ),
    	        this->gen_const(64U, 0x80000000)));
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 142);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 143: FMIN.D */
    std::tuple<continuation_e, BasicBlock*> __fmin_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FMIN.D");
    	
    	this->gen_sync(PRE_SYNC, 143);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fmin.d"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fsel_d"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 0LL),
    	        32,
    	        false)
    	});
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 143);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 144: FMAX.D */
    std::tuple<continuation_e, BasicBlock*> __fmax_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FMAX.D");
    	
    	this->gen_sync(PRE_SYNC, 144);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fmax.d"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fsel_d"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 1LL),
    	        32,
    	        false)
    	});
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 144);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 145: FCVT.S.D */
    std::tuple<continuation_e, BasicBlock*> __fcvt_s_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FCVT.S.D");
    	
    	this->gen_sync(PRE_SYNC, 145);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, f{rs1}", fmt::arg("mnemonic", "fcvt.s.d"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fconv_d2f"), std::vector<Value*>{
    	    this->gen_reg_load(rs1 + traits<ARCH>::F0, 0), 
    	    this->gen_const(8U, rm)
    	});
    	uint64_t upper_val = - 1;
    	Value* Ftmp0_val = this->builder.CreateOr(
    	    this->builder.CreateShl(
    	        this->gen_const(64U, upper_val),
    	        this->gen_const(64U, 32)),
    	    this->gen_ext(
    	        res_val,
    	        64,
    	        false));
    	this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 145);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 146: FCVT.D.S */
    std::tuple<continuation_e, BasicBlock*> __fcvt_d_s(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FCVT.D.S");
    	
    	this->gen_sync(PRE_SYNC, 146);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, f{rs1}", fmt::arg("mnemonic", "fcvt.d.s"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fconv_f2d"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(32) 
    	    ), 
    	    this->gen_const(8U, rm)
    	});
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 146);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 147: FEQ.D */
    std::tuple<continuation_e, BasicBlock*> __feq_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FEQ.D");
    	
    	this->gen_sync(PRE_SYNC, 147);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "feq.d"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* Xtmp0_val = this->builder.CreateCall(this->mod->getFunction("fcmp_d"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 0LL),
    	        32,
    	        false)
    	});
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 147);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 148: FLT.D */
    std::tuple<continuation_e, BasicBlock*> __flt_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FLT.D");
    	
    	this->gen_sync(PRE_SYNC, 148);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "flt.d"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* Xtmp0_val = this->builder.CreateCall(this->mod->getFunction("fcmp_d"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 2LL),
    	        32,
    	        false)
    	});
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 148);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 149: FLE.D */
    std::tuple<continuation_e, BasicBlock*> __fle_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FLE.D");
    	
    	this->gen_sync(PRE_SYNC, 149);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fle.d"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* Xtmp0_val = this->builder.CreateCall(this->mod->getFunction("fcmp_d"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    ), 
    	    this->gen_ext(
    	        this->gen_const(64U, 1LL),
    	        32,
    	        false)
    	});
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 149);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 150: FCLASS.D */
    std::tuple<continuation_e, BasicBlock*> __fclass_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FCLASS.D");
    	
    	this->gen_sync(PRE_SYNC, 150);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, f{rs1}", fmt::arg("mnemonic", "fclass.d"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* Xtmp0_val = this->builder.CreateCall(this->mod->getFunction("fclass_d"), std::vector<Value*>{
    	    this->builder.CreateTrunc(
    	        this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	        this-> get_type(64) 
    	    )
    	});
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 150);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 151: FCVT.W.D */
    std::tuple<continuation_e, BasicBlock*> __fcvt_w_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FCVT.W.D");
    	
    	this->gen_sync(PRE_SYNC, 151);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, f{rs1}", fmt::arg("mnemonic", "fcvt.w.d"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* Xtmp0_val = this->gen_ext(
    	    this->builder.CreateCall(this->mod->getFunction("fcvt_d"), std::vector<Value*>{
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	            this-> get_type(64) 
    	        ), 
    	        this->gen_ext(
    	            this->gen_const(64U, 0LL),
    	            32,
    	            false), 
    	        this->gen_const(8U, rm)
    	    }),
    	    32,
    	    true);
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 151);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 152: FCVT.WU.D */
    std::tuple<continuation_e, BasicBlock*> __fcvt_wu_d(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FCVT.WU.D");
    	
    	this->gen_sync(PRE_SYNC, 152);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} {rd}, f{rs1}", fmt::arg("mnemonic", "fcvt.wu.d"),
    	    	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* Xtmp0_val = this->gen_ext(
    	    this->builder.CreateCall(this->mod->getFunction("fcvt_d"), std::vector<Value*>{
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs1 + traits<ARCH>::F0, 0),
    	            this-> get_type(64) 
    	        ), 
    	        this->gen_ext(
    	            this->gen_const(64U, 1LL),
    	            32,
    	            false), 
    	        this->gen_const(8U, rm)
    	    }),
    	    32,
    	    false);
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	Value* flags_val = this->builder.CreateCall(this->mod->getFunction("fget_flags"), std::vector<Value*>{
    	});
    	Value* FCSR_val = this->builder.CreateAdd(
    	    this->builder.CreateAnd(
    	        this->gen_reg_load(traits<ARCH>::FCSR, 0),
    	        this->builder.CreateNot(this->gen_const(32U, 0x1f))),
    	    flags_val);
    	this->builder.CreateStore(FCSR_val, get_reg_ptr(traits<ARCH>::FCSR), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 152);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 153: FCVT.D.W */
    std::tuple<continuation_e, BasicBlock*> __fcvt_d_w(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FCVT.D.W");
    	
    	this->gen_sync(PRE_SYNC, 153);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, {rs1}", fmt::arg("mnemonic", "fcvt.d.w"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fcvt_d"), std::vector<Value*>{
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        64,
    	        true), 
    	    this->gen_ext(
    	        this->gen_const(64U, 2LL),
    	        32,
    	        false), 
    	    this->gen_const(8U, rm)
    	});
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 153);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 154: FCVT.D.WU */
    std::tuple<continuation_e, BasicBlock*> __fcvt_d_wu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FCVT.D.WU");
    	
    	this->gen_sync(PRE_SYNC, 154);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rm = ((bit_sub<12,3>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, {rs1}", fmt::arg("mnemonic", "fcvt.d.wu"),
    	    	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->builder.CreateCall(this->mod->getFunction("fcvt_d"), std::vector<Value*>{
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        64,
    	        false), 
    	    this->gen_ext(
    	        this->gen_const(64U, 3LL),
    	        32,
    	        false), 
    	    this->gen_const(8U, rm)
    	});
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 154);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 155: C.FLD */
    std::tuple<continuation_e, BasicBlock*> __c_fld(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.FLD");
    	
    	this->gen_sync(PRE_SYNC, 155);
    	
    	uint8_t rd = ((bit_sub<2,3>(instr)));
    	uint8_t uimm = ((bit_sub<5,2>(instr) << 6) | (bit_sub<10,3>(instr) << 3));
    	uint8_t rs1 = ((bit_sub<7,3>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f(8+{rd}), {uimm}({rs1})", fmt::arg("mnemonic", "c.fld"),
    	    	fmt::arg("rd", rd), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_reg_load(rs1 + 8 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, uimm));
    	Value* res_val = this->gen_read_mem(traits<ARCH>::MEM, offs_val, 64/8);
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + 8 + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        res_val);
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + 8 + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 155);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 156: C.FSD */
    std::tuple<continuation_e, BasicBlock*> __c_fsd(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.FSD");
    	
    	this->gen_sync(PRE_SYNC, 156);
    	
    	uint8_t rs2 = ((bit_sub<2,3>(instr)));
    	uint8_t uimm = ((bit_sub<5,2>(instr) << 6) | (bit_sub<10,3>(instr) << 3));
    	uint8_t rs1 = ((bit_sub<7,3>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f(8+{rs2}), {uimm}({rs1})", fmt::arg("mnemonic", "c.fsd"),
    	    	fmt::arg("rs2", rs2), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_reg_load(rs1 + 8 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, uimm));
    	Value* MEMtmp0_val = this->builder.CreateTrunc(
    	    this->gen_reg_load(rs2 + 8 + traits<ARCH>::F0, 0),
    	    this-> get_type(64) 
    	);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(64)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 156);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 157: C.FLDSP */
    std::tuple<continuation_e, BasicBlock*> __c_fldsp(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.FLDSP");
    	
    	this->gen_sync(PRE_SYNC, 157);
    	
    	uint16_t uimm = ((bit_sub<2,3>(instr) << 6) | (bit_sub<5,2>(instr) << 3) | (bit_sub<12,1>(instr) << 5));
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rd}, {uimm}(x2)", fmt::arg("mnemonic", "c.fldsp"),
    	    	fmt::arg("rd", rd), fmt::arg("uimm", uimm));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_reg_load(2 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, uimm));
    	Value* res_val = this->gen_read_mem(traits<ARCH>::MEM, offs_val, 64/8);
    	if(64 == 64){
    	    Value* Ftmp0_val = res_val;
    	    this->builder.CreateStore(Ftmp0_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	} else {
    	    uint64_t upper_val = - 1;
    	    Value* Ftmp1_val = this->builder.CreateOr(
    	        this->builder.CreateShl(
    	            this->gen_const(64U, upper_val),
    	            this->gen_const(64U, 64)),
    	        this->gen_ext(
    	            res_val,
    	            64,
    	            false));
    	    this->builder.CreateStore(Ftmp1_val, get_reg_ptr(rd + traits<ARCH>::F0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 157);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 158: C.FSDSP */
    std::tuple<continuation_e, BasicBlock*> __c_fsdsp(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("C.FSDSP");
    	
    	this->gen_sync(PRE_SYNC, 158);
    	
    	uint8_t rs2 = ((bit_sub<2,5>(instr)));
    	uint16_t uimm = ((bit_sub<7,3>(instr) << 6) | (bit_sub<10,3>(instr) << 3));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    auto mnemonic = fmt::format(
    	        "{mnemonic:10} f{rs2}, {uimm}(x2), ", fmt::arg("mnemonic", "c.fsdsp"),
    	    	fmt::arg("rs2", rs2), fmt::arg("uimm", uimm));
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(mnemonic),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_reg_load(2 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, uimm));
    	Value* MEMtmp0_val = this->builder.CreateTrunc(
    	    this->gen_reg_load(rs2 + traits<ARCH>::F0, 0),
    	    this-> get_type(64) 
    	);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(64)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 158);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /****************************************************************************
     * end opcode definitions
     ****************************************************************************/
    std::tuple<continuation_e, BasicBlock *> illegal_intruction(virt_addr_t &pc, code_word_t instr, BasicBlock *bb) {
		this->gen_sync(iss::PRE_SYNC, instr_descr.size());
        this->builder.CreateStore(this->builder.CreateLoad(get_reg_ptr(traits<ARCH>::NEXT_PC), true),
                                   get_reg_ptr(traits<ARCH>::PC), true);
        this->builder.CreateStore(
            this->builder.CreateAdd(this->builder.CreateLoad(get_reg_ptr(traits<ARCH>::ICOUNT), true),
                                     this->gen_const(64U, 1)),
            get_reg_ptr(traits<ARCH>::ICOUNT), true);
        pc = pc + ((instr & 3) == 3 ? 4 : 2);
        this->gen_raise_trap(0, 2);     // illegal instruction trap
		this->gen_sync(iss::POST_SYNC, instr_descr.size());
        this->gen_trap_check(this->leave_blk);
        return std::make_tuple(BRANCH, nullptr);
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
std::tuple<continuation_e, BasicBlock *>
vm_impl<ARCH>::gen_single_inst_behavior(virt_addr_t &pc, unsigned int &inst_cnt, BasicBlock *this_block) {
    // we fetch at max 4 byte, alignment is 2
    code_word_t insn = 0;
    const typename traits<ARCH>::addr_t upper_bits = ~traits<ARCH>::PGMASK;
    phys_addr_t paddr(pc);
    try {
        auto *const data = (uint8_t *)&insn;
        paddr = this->core.v2p(pc);
        if ((pc.val & upper_bits) != ((pc.val + 2) & upper_bits)) { // we may cross a page boundary
            auto res = this->core.read(paddr, 2, data);
            if (res != iss::Ok) throw trap_access(1, pc.val);
            if ((insn & 0x3) == 0x3) { // this is a 32bit instruction
                res = this->core.read(this->core.v2p(pc + 2), 2, data + 2);
            }
        } else {
            auto res = this->core.read(paddr, 4, data);
            if (res != iss::Ok) throw trap_access(1, pc.val);
        }
    } catch (trap_access &ta) {
        throw trap_access(ta.id, pc.val);
    }
    if (insn == 0x0000006f || (insn&0xffff)==0xa001) throw simulation_stopped(0); // 'J 0' or 'C.J 0'
    // curr pc on stack
    ++inst_cnt;
    auto lut_val = extract_fields(insn);
    auto f = qlut[insn & 0x3][lut_val];
    if (f == nullptr) {
        f = &this_class::illegal_intruction;
    }
    return (this->*f)(pc, insn, this_block);
}

template <typename ARCH> void vm_impl<ARCH>::gen_leave_behavior(BasicBlock *leave_blk) {
    this->builder.SetInsertPoint(leave_blk);
    this->builder.CreateRet(this->builder.CreateLoad(get_reg_ptr(arch::traits<ARCH>::NEXT_PC), false));
}

template <typename ARCH> void vm_impl<ARCH>::gen_raise_trap(uint16_t trap_id, uint16_t cause) {
    auto *TRAP_val = this->gen_const(32, 0x80 << 24 | (cause << 16) | trap_id);
    this->builder.CreateStore(TRAP_val, get_reg_ptr(traits<ARCH>::TRAP_STATE), true);
    this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()), get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
}

template <typename ARCH> void vm_impl<ARCH>::gen_leave_trap(unsigned lvl) {
    std::vector<Value *> args{ this->core_ptr, ConstantInt::get(getContext(), APInt(64, lvl)) };
    this->builder.CreateCall(this->mod->getFunction("leave_trap"), args);
    auto *PC_val = this->gen_read_mem(traits<ARCH>::CSR, (lvl << 8) + 0x41, traits<ARCH>::XLEN / 8);
    this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()), get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
}

template <typename ARCH> void vm_impl<ARCH>::gen_wait(unsigned type) {
    std::vector<Value *> args{ this->core_ptr, ConstantInt::get(getContext(), APInt(64, type)) };
    this->builder.CreateCall(this->mod->getFunction("wait"), args);
}

template <typename ARCH> void vm_impl<ARCH>::gen_trap_behavior(BasicBlock *trap_blk) {
    this->builder.SetInsertPoint(trap_blk);
    auto *trap_state_val = this->builder.CreateLoad(get_reg_ptr(traits<ARCH>::TRAP_STATE), true);
    this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()),
                              get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    std::vector<Value *> args{this->core_ptr, this->adj_to64(trap_state_val),
                              this->adj_to64(this->builder.CreateLoad(get_reg_ptr(traits<ARCH>::PC), false))};
    this->builder.CreateCall(this->mod->getFunction("enter_trap"), args);
    auto *trap_addr_val = this->builder.CreateLoad(get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    this->builder.CreateRet(trap_addr_val);
}

template <typename ARCH> inline void vm_impl<ARCH>::gen_trap_check(BasicBlock *bb) {
    auto *v = this->builder.CreateLoad(get_reg_ptr(arch::traits<ARCH>::TRAP_STATE), true);
    this->gen_cond_branch(this->builder.CreateICmp(
                              ICmpInst::ICMP_EQ, v,
                              ConstantInt::get(getContext(), APInt(v->getType()->getIntegerBitWidth(), 0))),
                          bb, this->trap_blk, 1);
}

} // namespace rv32gc

template <>
std::unique_ptr<vm_if> create<arch::rv32gc>(arch::rv32gc *core, unsigned short port, bool dump) {
    auto ret = new rv32gc::vm_impl<arch::rv32gc>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret, port);
    return std::unique_ptr<vm_if>(ret);
}

} // namespace iss
