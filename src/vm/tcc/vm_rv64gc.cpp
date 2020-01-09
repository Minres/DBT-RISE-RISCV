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

#include <iss/arch/rv64gc.h>
#include <iss/arch/riscv_hart_msu_vp.h>
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/server.h>
#include <iss/iss.h>
#include <iss/llvm/vm_base.h>
#include <util/logging.h>

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#include <array>
#include <iss/debugger/riscv_target_adapter.h>

namespace iss {
namespace vm {
namespace fp_impl {
void add_fp_functions_2_module(llvm::Module *, unsigned, unsigned);
}
}

namespace tcc {
namespace rv64gc {
using namespace iss::arch;
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

    using this_class = vm_impl<ARCH>;
    using compile_ret_t = std::tuple<continuation_e>;
    using compile_func = compile_ret_t (this_class::*)(virt_addr_t &pc, code_word_t instr, std::ostringstream&);

    inline const char *name(size_t index){return traits<ARCH>::reg_aliases.at(index);}

    template <typename T> inline ConstantInt *size(T type) {
        return ConstantInt::get(getContext(), APInt(32, type->getType()->getScalarSizeInBits()));
    }

    void setup_module(Module* m) override {
        super::setup_module(m);
        iss::vm::fp_impl::add_fp_functions_2_module(m, traits<ARCH>::FP_REGS_SIZE, traits<ARCH>::XLEN);
    }

    inline Value *gen_choose(Value *cond, Value *trueVal, Value *falseVal, unsigned size) {
        return super::gen_cond_assign(cond, this->gen_ext(trueVal, size), this->gen_ext(falseVal, size));
    }

    compile_ret_t gen_single_inst_behavior(virt_addr_t &, unsigned int &, std::ostringstream&) override;

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

    const std::array<InstructionDesriptor, 204> instr_descr = {{
         /* entries are: size, valid value, valid mask, function ptr */
        /* instruction JALR */
        {32, 0b00000000000000000000000001100111, 0b00000000000000000111000001111111, &this_class::__jalr},
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
        {16, 0b1000000000000001, 0b1110110000000011, &this_class::__c_srli},
        /* instruction C.SRAI */
        {16, 0b1000010000000001, 0b1110110000000011, &this_class::__c_srai},
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
        {16, 0b0000000000000010, 0b1110000000000011, &this_class::__c_slli},
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
        /* instruction C.FLD */
        {16, 0b0010000000000000, 0b1110000000000011, &this_class::__c_fld},
        /* instruction C.FSD */
        {16, 0b1010000000000000, 0b1110000000000011, &this_class::__c_fsd},
        /* instruction C.FLDSP */
        {16, 0b0010000000000010, 0b1110000000000011, &this_class::__c_fldsp},
        /* instruction C.FSDSP */
        {16, 0b1010000000000010, 0b1110000000000011, &this_class::__c_fsdsp},
        /* instruction C.FLW */
        {16, 0b0110000000000000, 0b1110000000000011, &this_class::__c_flw},
        /* instruction C.FSW */
        {16, 0b1110000000000000, 0b1110000000000011, &this_class::__c_fsw},
        /* instruction C.FLWSP */
        {16, 0b0110000000000010, 0b1110000000000011, &this_class::__c_flwsp},
        /* instruction C.FSWSP */
        {16, 0b1110000000000010, 0b1110000000000011, &this_class::__c_fswsp},
        /* instruction C.LD */
        {16, 0b0110000000000000, 0b1110000000000011, &this_class::__c_ld},
        /* instruction C.SD */
        {16, 0b1110000000000000, 0b1110000000000011, &this_class::__c_sd},
        /* instruction C.SUBW */
        {16, 0b1001110000000001, 0b1111110001100011, &this_class::__c_subw},
        /* instruction C.ADDW */
        {16, 0b1001110000100001, 0b1111110001100011, &this_class::__c_addw},
        /* instruction C.ADDIW */
        {16, 0b0010000000000001, 0b1110000000000011, &this_class::__c_addiw},
        /* instruction C.LDSP */
        {16, 0b0110000000000010, 0b1110000000000011, &this_class::__c_ldsp},
        /* instruction C.SDSP */
        {16, 0b1110000000000010, 0b1110000000000011, &this_class::__c_sdsp},
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
        /* instruction FCVT.L.D */
        {32, 0b11000010001000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_l_d},
        /* instruction FCVT.LU.D */
        {32, 0b11000010001100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_lu_d},
        /* instruction FCVT.D.L */
        {32, 0b11010010001000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_d_l},
        /* instruction FCVT.D.LU */
        {32, 0b11010010001100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_d_lu},
        /* instruction FMV.X.D */
        {32, 0b11100010000000000000000001010011, 0b11111111111100000111000001111111, &this_class::__fmv_x_d},
        /* instruction FMV.D.X */
        {32, 0b11110010000000000000000001010011, 0b11111111111100000111000001111111, &this_class::__fmv_d_x},
        /* instruction LUI */
        {32, 0b00000000000000000000000000110111, 0b00000000000000000000000001111111, &this_class::__lui},
        /* instruction AUIPC */
        {32, 0b00000000000000000000000000010111, 0b00000000000000000000000001111111, &this_class::__auipc},
        /* instruction JAL */
        {32, 0b00000000000000000000000001101111, 0b00000000000000000000000001111111, &this_class::__jal},
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
        /* instruction FCVT.L.S */
        {32, 0b11000000001000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_l_s},
        /* instruction FCVT.LU.S */
        {32, 0b11000000001100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_lu_s},
        /* instruction FCVT.S.L */
        {32, 0b11010000001000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_s_l},
        /* instruction FCVT.S.LU */
        {32, 0b11010000001100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt_s_lu},
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
        /* instruction LR.D */
        {32, 0b00010000000000000011000000101111, 0b11111001111100000111000001111111, &this_class::__lr_d},
        /* instruction SC.D */
        {32, 0b00011000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__sc_d},
        /* instruction AMOSWAP.D */
        {32, 0b00001000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amoswap_d},
        /* instruction AMOADD.D */
        {32, 0b00000000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amoadd_d},
        /* instruction AMOXOR.D */
        {32, 0b00100000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amoxor_d},
        /* instruction AMOAND.D */
        {32, 0b01100000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amoand_d},
        /* instruction AMOOR.D */
        {32, 0b01000000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amoor_d},
        /* instruction AMOMIN.D */
        {32, 0b10000000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amomin_d},
        /* instruction AMOMAX.D */
        {32, 0b10100000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amomax_d},
        /* instruction AMOMINU.D */
        {32, 0b11000000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amominu_d},
        /* instruction AMOMAXU.D */
        {32, 0b11100000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amomaxu_d},
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
        /* instruction MULW */
        {32, 0b00000010000000000000000000111011, 0b11111110000000000111000001111111, &this_class::__mulw},
        /* instruction DIVW */
        {32, 0b00000010000000000100000000111011, 0b11111110000000000111000001111111, &this_class::__divw},
        /* instruction DIVUW */
        {32, 0b00000010000000000101000000111011, 0b11111110000000000111000001111111, &this_class::__divuw},
        /* instruction REMW */
        {32, 0b00000010000000000110000000111011, 0b11111110000000000111000001111111, &this_class::__remw},
        /* instruction REMUW */
        {32, 0b00000010000000000111000000111011, 0b11111110000000000111000001111111, &this_class::__remuw},
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
    /* instruction 0: JALR */
    compile_ret_t __jalr(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 1: C.ADDI4SPN */
    compile_ret_t __c_addi4spn(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 2: C.LW */
    compile_ret_t __c_lw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 3: C.SW */
    compile_ret_t __c_sw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 4: C.ADDI */
    compile_ret_t __c_addi(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 5: C.NOP */
    compile_ret_t __c_nop(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 6: C.JAL */
    compile_ret_t __c_jal(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 7: C.LI */
    compile_ret_t __c_li(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 8: C.LUI */
    compile_ret_t __c_lui(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 9: C.ADDI16SP */
    compile_ret_t __c_addi16sp(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 10: C.SRLI */
    compile_ret_t __c_srli(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 11: C.SRAI */
    compile_ret_t __c_srai(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 12: C.ANDI */
    compile_ret_t __c_andi(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 13: C.SUB */
    compile_ret_t __c_sub(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 14: C.XOR */
    compile_ret_t __c_xor(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 15: C.OR */
    compile_ret_t __c_or(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 16: C.AND */
    compile_ret_t __c_and(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 17: C.J */
    compile_ret_t __c_j(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 18: C.BEQZ */
    compile_ret_t __c_beqz(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 19: C.BNEZ */
    compile_ret_t __c_bnez(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 20: C.SLLI */
    compile_ret_t __c_slli(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 21: C.LWSP */
    compile_ret_t __c_lwsp(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 22: C.MV */
    compile_ret_t __c_mv(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 23: C.JR */
    compile_ret_t __c_jr(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 24: C.ADD */
    compile_ret_t __c_add(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 25: C.JALR */
    compile_ret_t __c_jalr(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 26: C.EBREAK */
    compile_ret_t __c_ebreak(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 27: C.SWSP */
    compile_ret_t __c_swsp(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 28: DII */
    compile_ret_t __dii(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 29: C.FLD */
    compile_ret_t __c_fld(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 30: C.FSD */
    compile_ret_t __c_fsd(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 31: C.FLDSP */
    compile_ret_t __c_fldsp(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 32: C.FSDSP */
    compile_ret_t __c_fsdsp(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 33: C.FLW */
    compile_ret_t __c_flw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 34: C.FSW */
    compile_ret_t __c_fsw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 35: C.FLWSP */
    compile_ret_t __c_flwsp(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 36: C.FSWSP */
    compile_ret_t __c_fswsp(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 37: C.LD */
    compile_ret_t __c_ld(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 38: C.SD */
    compile_ret_t __c_sd(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 39: C.SUBW */
    compile_ret_t __c_subw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 40: C.ADDW */
    compile_ret_t __c_addw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 41: C.ADDIW */
    compile_ret_t __c_addiw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 42: C.LDSP */
    compile_ret_t __c_ldsp(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 43: C.SDSP */
    compile_ret_t __c_sdsp(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 44: FLD */
    compile_ret_t __fld(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 45: FSD */
    compile_ret_t __fsd(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 46: FMADD.D */
    compile_ret_t __fmadd_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 47: FMSUB.D */
    compile_ret_t __fmsub_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 48: FNMADD.D */
    compile_ret_t __fnmadd_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 49: FNMSUB.D */
    compile_ret_t __fnmsub_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 50: FADD.D */
    compile_ret_t __fadd_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 51: FSUB.D */
    compile_ret_t __fsub_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 52: FMUL.D */
    compile_ret_t __fmul_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 53: FDIV.D */
    compile_ret_t __fdiv_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 54: FSQRT.D */
    compile_ret_t __fsqrt_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 55: FSGNJ.D */
    compile_ret_t __fsgnj_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 56: FSGNJN.D */
    compile_ret_t __fsgnjn_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 57: FSGNJX.D */
    compile_ret_t __fsgnjx_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 58: FMIN.D */
    compile_ret_t __fmin_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 59: FMAX.D */
    compile_ret_t __fmax_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 60: FCVT.S.D */
    compile_ret_t __fcvt_s_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 61: FCVT.D.S */
    compile_ret_t __fcvt_d_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 62: FEQ.D */
    compile_ret_t __feq_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 63: FLT.D */
    compile_ret_t __flt_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 64: FLE.D */
    compile_ret_t __fle_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 65: FCLASS.D */
    compile_ret_t __fclass_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 66: FCVT.W.D */
    compile_ret_t __fcvt_w_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 67: FCVT.WU.D */
    compile_ret_t __fcvt_wu_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 68: FCVT.D.W */
    compile_ret_t __fcvt_d_w(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 69: FCVT.D.WU */
    compile_ret_t __fcvt_d_wu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 70: FCVT.L.D */
    compile_ret_t __fcvt_l_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 71: FCVT.LU.D */
    compile_ret_t __fcvt_lu_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 72: FCVT.D.L */
    compile_ret_t __fcvt_d_l(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 73: FCVT.D.LU */
    compile_ret_t __fcvt_d_lu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 74: FMV.X.D */
    compile_ret_t __fmv_x_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 75: FMV.D.X */
    compile_ret_t __fmv_d_x(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 76: LUI */
    compile_ret_t __lui(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 77: AUIPC */
    compile_ret_t __auipc(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 78: JAL */
    compile_ret_t __jal(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 79: BEQ */
    compile_ret_t __beq(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 80: BNE */
    compile_ret_t __bne(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 81: BLT */
    compile_ret_t __blt(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 82: BGE */
    compile_ret_t __bge(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 83: BLTU */
    compile_ret_t __bltu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 84: BGEU */
    compile_ret_t __bgeu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 85: LB */
    compile_ret_t __lb(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 86: LH */
    compile_ret_t __lh(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 87: LW */
    compile_ret_t __lw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 88: LBU */
    compile_ret_t __lbu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 89: LHU */
    compile_ret_t __lhu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 90: SB */
    compile_ret_t __sb(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 91: SH */
    compile_ret_t __sh(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 92: SW */
    compile_ret_t __sw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 93: ADDI */
    compile_ret_t __addi(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 94: SLTI */
    compile_ret_t __slti(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 95: SLTIU */
    compile_ret_t __sltiu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 96: XORI */
    compile_ret_t __xori(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 97: ORI */
    compile_ret_t __ori(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 98: ANDI */
    compile_ret_t __andi(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 99: SLLI */
    compile_ret_t __slli(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 100: SRLI */
    compile_ret_t __srli(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 101: SRAI */
    compile_ret_t __srai(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 102: ADD */
    compile_ret_t __add(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 103: SUB */
    compile_ret_t __sub(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 104: SLL */
    compile_ret_t __sll(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 105: SLT */
    compile_ret_t __slt(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 106: SLTU */
    compile_ret_t __sltu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 107: XOR */
    compile_ret_t __xor(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 108: SRL */
    compile_ret_t __srl(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 109: SRA */
    compile_ret_t __sra(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 110: OR */
    compile_ret_t __or(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 111: AND */
    compile_ret_t __and(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 112: FENCE */
    compile_ret_t __fence(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 113: FENCE_I */
    compile_ret_t __fence_i(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 114: ECALL */
    compile_ret_t __ecall(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 115: EBREAK */
    compile_ret_t __ebreak(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 116: URET */
    compile_ret_t __uret(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 117: SRET */
    compile_ret_t __sret(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 118: MRET */
    compile_ret_t __mret(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 119: WFI */
    compile_ret_t __wfi(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 120: SFENCE.VMA */
    compile_ret_t __sfence_vma(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 121: CSRRW */
    compile_ret_t __csrrw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 122: CSRRS */
    compile_ret_t __csrrs(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 123: CSRRC */
    compile_ret_t __csrrc(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 124: CSRRWI */
    compile_ret_t __csrrwi(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 125: CSRRSI */
    compile_ret_t __csrrsi(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 126: CSRRCI */
    compile_ret_t __csrrci(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 127: FLW */
    compile_ret_t __flw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 128: FSW */
    compile_ret_t __fsw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 129: FMADD.S */
    compile_ret_t __fmadd_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 130: FMSUB.S */
    compile_ret_t __fmsub_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 131: FNMADD.S */
    compile_ret_t __fnmadd_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 132: FNMSUB.S */
    compile_ret_t __fnmsub_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 133: FADD.S */
    compile_ret_t __fadd_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 134: FSUB.S */
    compile_ret_t __fsub_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 135: FMUL.S */
    compile_ret_t __fmul_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 136: FDIV.S */
    compile_ret_t __fdiv_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 137: FSQRT.S */
    compile_ret_t __fsqrt_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 138: FSGNJ.S */
    compile_ret_t __fsgnj_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 139: FSGNJN.S */
    compile_ret_t __fsgnjn_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 140: FSGNJX.S */
    compile_ret_t __fsgnjx_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 141: FMIN.S */
    compile_ret_t __fmin_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 142: FMAX.S */
    compile_ret_t __fmax_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 143: FCVT.W.S */
    compile_ret_t __fcvt_w_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 144: FCVT.WU.S */
    compile_ret_t __fcvt_wu_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 145: FEQ.S */
    compile_ret_t __feq_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 146: FLT.S */
    compile_ret_t __flt_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 147: FLE.S */
    compile_ret_t __fle_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 148: FCLASS.S */
    compile_ret_t __fclass_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 149: FCVT.S.W */
    compile_ret_t __fcvt_s_w(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 150: FCVT.S.WU */
    compile_ret_t __fcvt_s_wu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 151: FMV.X.W */
    compile_ret_t __fmv_x_w(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 152: FMV.W.X */
    compile_ret_t __fmv_w_x(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 153: FCVT.L.S */
    compile_ret_t __fcvt_l_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 154: FCVT.LU.S */
    compile_ret_t __fcvt_lu_s(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 155: FCVT.S.L */
    compile_ret_t __fcvt_s_l(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 156: FCVT.S.LU */
    compile_ret_t __fcvt_s_lu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 157: LR.W */
    compile_ret_t __lr_w(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 158: SC.W */
    compile_ret_t __sc_w(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 159: AMOSWAP.W */
    compile_ret_t __amoswap_w(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 160: AMOADD.W */
    compile_ret_t __amoadd_w(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 161: AMOXOR.W */
    compile_ret_t __amoxor_w(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 162: AMOAND.W */
    compile_ret_t __amoand_w(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 163: AMOOR.W */
    compile_ret_t __amoor_w(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 164: AMOMIN.W */
    compile_ret_t __amomin_w(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 165: AMOMAX.W */
    compile_ret_t __amomax_w(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 166: AMOMINU.W */
    compile_ret_t __amominu_w(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 167: AMOMAXU.W */
    compile_ret_t __amomaxu_w(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 168: LR.D */
    compile_ret_t __lr_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 169: SC.D */
    compile_ret_t __sc_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 170: AMOSWAP.D */
    compile_ret_t __amoswap_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 171: AMOADD.D */
    compile_ret_t __amoadd_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 172: AMOXOR.D */
    compile_ret_t __amoxor_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 173: AMOAND.D */
    compile_ret_t __amoand_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 174: AMOOR.D */
    compile_ret_t __amoor_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 175: AMOMIN.D */
    compile_ret_t __amomin_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 176: AMOMAX.D */
    compile_ret_t __amomax_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 177: AMOMINU.D */
    compile_ret_t __amominu_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 178: AMOMAXU.D */
    compile_ret_t __amomaxu_d(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 179: MUL */
    compile_ret_t __mul(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 180: MULH */
    compile_ret_t __mulh(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 181: MULHSU */
    compile_ret_t __mulhsu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 182: MULHU */
    compile_ret_t __mulhu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 183: DIV */
    compile_ret_t __div(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 184: DIVU */
    compile_ret_t __divu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 185: REM */
    compile_ret_t __rem(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 186: REMU */
    compile_ret_t __remu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 187: MULW */
    compile_ret_t __mulw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 188: DIVW */
    compile_ret_t __divw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 189: DIVUW */
    compile_ret_t __divuw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 190: REMW */
    compile_ret_t __remw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 191: REMUW */
    compile_ret_t __remuw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 192: LWU */
    compile_ret_t __lwu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 193: LD */
    compile_ret_t __ld(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 194: SD */
    compile_ret_t __sd(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 195: ADDIW */
    compile_ret_t __addiw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 196: SLLIW */
    compile_ret_t __slliw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 197: SRLIW */
    compile_ret_t __srliw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 198: SRAIW */
    compile_ret_t __sraiw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 199: ADDW */
    compile_ret_t __addw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 200: SUBW */
    compile_ret_t __subw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 201: SLLW */
    compile_ret_t __sllw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 202: SRLW */
    compile_ret_t __srlw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 203: SRAW */
    compile_ret_t __sraw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /****************************************************************************
     * end opcode definitions
     ****************************************************************************/
    compile_ret_t illegal_intruction(virt_addr_t &pc, code_word_t instr, std::stringstream& os) {
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
        return BRANCH;
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
std::tuple<continuation_e>
vm_impl<ARCH>::gen_single_inst_behavior(virt_addr_t &pc, unsigned int &inst_cnt, std::ostringstrem& os) {
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
} // namespace rv64gc

template <>
std::unique_ptr<vm_if> create<arch::rv64gc>(arch::rv64gc *core, unsigned short port, bool dump) {
    auto ret = new rv64gc::vm_impl<arch::rv64gc>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret, port);
    return std::unique_ptr<vm_if>(ret);
}
}
} // namespace iss
