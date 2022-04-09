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
#include <iss/arch/rv64gc.h>
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
namespace rv64gc {
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

    virt_addr_t execute_inst(finish_cond_e cond, virt_addr_t start, uint64_t icount_limit) override;

    // some compile time constants
    // enum { MASK16 = 0b1111110001100011, MASK32 = 0b11111111111100000111000001111111 };
    enum { MASK16 = 0b1111111111111111, MASK32 = 0b11111111111100000111000001111111 };
    enum { EXTR_MASK16 = MASK16 >> 2, EXTR_MASK32 = MASK32 >> 2 };
    enum { LUT_SIZE = 1 << util::bit_count(static_cast<uint32_t>(EXTR_MASK32)), LUT_SIZE_C = 1 << util::bit_count(static_cast<uint32_t>(EXTR_MASK16)) };

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

    const std::array<InstructionDesriptor, 204> instr_descr = {{
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
        /* instruction C.FLW */
        {16, 0b0110000000000000, 0b1110000000000011, &this_class::__c_flw},
        /* instruction C.FSW */
        {16, 0b1110000000000000, 0b1110000000000011, &this_class::__c_fsw},
        /* instruction C.FLWSP */
        {16, 0b0110000000000010, 0b1110000000000011, &this_class::__c_flwsp},
        /* instruction C.FSWSP */
        {16, 0b1110000000000010, 0b1110000000000011, &this_class::__c_fswsp},
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
        if(rd != 0){
            auto Xtmp0_val = (cur_pc_val + 4);
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto PC_val = (new_pc_val & ~(0x1));
        super::template get_reg(traits<ARCH>::NEXT_PC) = PC_val;
        super::template get_reg(traits<ARCH>::LAST_BRANCH) = std::numeric_limits<uint32_t>::max();
        this->do_sync(POST_SYNC, 3);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 24: SLLI */
    compile_ret_t __slli(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 24);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "slli"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(shamt > 31){
            raise_trap(0, 0);
        } else {
            if(rd != 0){
                auto Xtmp0_val = (super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)<<(shamt));
                super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
            }
        }
        this->do_sync(POST_SYNC, 24);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 25: SRLI */
    compile_ret_t __srli(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 25);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srli"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(shamt > 31){
            raise_trap(0, 0);
        } else {
            if(rd != 0){
                auto Xtmp0_val = (static_cast<uint64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0))>>(shamt));
                super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
            }
        }
        this->do_sync(POST_SYNC, 25);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 26: SRAI */
    compile_ret_t __srai(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 26);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srai"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(shamt > 31){
            raise_trap(0, 0);
        } else {
            if(rd != 0){
                auto Xtmp0_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0))>>(shamt));
                super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
            }
        }
        this->do_sync(POST_SYNC, 26);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 64: MUL */
    compile_ret_t __mul(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 64);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mul"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto res_val = (super::template zext<uint128_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) * super::template zext<uint128_t>(super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)));
            auto Xtmp0_val = super::template zext<uint64_t>(res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 64);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 65: MULH */
    compile_ret_t __mulh(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 65);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mulh"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto res_val = (super::template sext<int128_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) * super::template sext<int128_t>(super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)));
            auto Xtmp0_val = super::template zext<uint64_t>((res_val >> (64)));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 65);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 66: MULHSU */
    compile_ret_t __mulhsu(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 66);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mulhsu"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto res_val = (super::template sext<int128_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) * super::template zext<uint128_t>(super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)));
            auto Xtmp0_val = super::template zext<uint64_t>((res_val >> (64)));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 66);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 67: MULHU */
    compile_ret_t __mulhu(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 67);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mulhu"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto res_val = (super::template zext<uint128_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) * super::template zext<uint128_t>(super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)));
            auto Xtmp0_val = super::template zext<uint64_t>((res_val >> (64)));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 67);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 68: DIV */
    compile_ret_t __div(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 68);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "div"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            {
                if((super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0) != 0)) {
                    uint64_t M1_val = - 1;
                    uint8_t XLM1_val = 64 - 1;
                    uint64_t ONE_val = 1;
                    uint64_t MMIN_val = ONE_val << XLM1_val;
                    {
                        if(((super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0) == MMIN_val) && (super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0) == M1_val))) {
                            auto Xtmp0_val = MMIN_val;
                            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
                        }
                        else {
                            auto Xtmp1_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) / static_cast<int64_t>(super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)));
                            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp1_val;
                        }
                    }
                }
                else {
                    auto Xtmp2_val = -(1);
                    super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp2_val;
                }
            }
        }
        this->do_sync(POST_SYNC, 68);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 69: DIVU */
    compile_ret_t __divu(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 69);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "divu"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            {
                if((super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0) != 0)) {
                    auto Xtmp0_val = (super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0) / super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0));
                    super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
                }
                else {
                    auto Xtmp1_val = -(1);
                    super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp1_val;
                }
            }
        }
        this->do_sync(POST_SYNC, 69);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 70: REM */
    compile_ret_t __rem(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 70);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "rem"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            {
                if((super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0) != 0)) {
                    uint64_t M1_val = - 1;
                    uint32_t XLM1_val = 64 - 1;
                    uint64_t ONE_val = 1;
                    uint64_t MMIN_val = ONE_val << XLM1_val;
                    {
                        if(((super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0) == MMIN_val) && (super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0) == M1_val))) {
                            auto Xtmp0_val = 0;
                            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
                        }
                        else {
                            auto Xtmp1_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) % static_cast<int64_t>(super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)));
                            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp1_val;
                        }
                    }
                }
                else {
                    auto Xtmp2_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
                    super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp2_val;
                }
            }
        }
        this->do_sync(POST_SYNC, 70);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 71: REMU */
    compile_ret_t __remu(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 71);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "remu"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            {
                if((super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0) != 0)) {
                    auto Xtmp0_val = (super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0) % super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0));
                    super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
                }
                else {
                    auto Xtmp1_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
                    super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp1_val;
                }
            }
        }
        this->do_sync(POST_SYNC, 71);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 72: MULW */
    compile_ret_t __mulw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 72);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mulw"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto Xtmp0_val = super::template sext<int64_t>((static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
            ) * static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)
            )));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 72);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 73: DIVW */
    compile_ret_t __divw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 73);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "divw"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            {
                if((super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0) != 0)) {
                    uint32_t M1_val = - 1;
                    uint32_t ONE_val = 1;
                    uint32_t MMIN_val = ONE_val << 31;
                    {
                        if(((static_cast<uint32_t>(
                            super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
                        ) == MMIN_val) && (static_cast<uint32_t>(
                            super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)
                        ) == M1_val))) {
                            auto Xtmp0_val = (-(1) << 31);
                            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
                        }
                        else {
                            auto Xtmp1_val = super::template sext<int64_t>((static_cast<int64_t>(static_cast<uint32_t>(
                                super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
                            )) / static_cast<int64_t>(static_cast<uint32_t>(
                                super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)
                            ))));
                            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp1_val;
                        }
                    }
                }
                else {
                    auto Xtmp2_val = -(1);
                    super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp2_val;
                }
            }
        }
        this->do_sync(POST_SYNC, 73);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 74: DIVUW */
    compile_ret_t __divuw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 74);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "divuw"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            {
                if((static_cast<uint32_t>(
                    super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)
                ) != 0)) {
                    auto Xtmp0_val = super::template sext<int64_t>((static_cast<uint32_t>(
                        super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
                    ) / static_cast<uint32_t>(
                        super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)
                    )));
                    super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
                }
                else {
                    auto Xtmp1_val = -(1);
                    super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp1_val;
                }
            }
        }
        this->do_sync(POST_SYNC, 74);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 75: REMW */
    compile_ret_t __remw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 75);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "remw"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            {
                if((super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0) != 0)) {
                    uint32_t M1_val = - 1;
                    uint32_t ONE_val = 1;
                    uint32_t MMIN_val = ONE_val << 31;
                    {
                        if(((static_cast<uint32_t>(
                            super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
                        ) == MMIN_val) && (super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0) == M1_val))) {
                            auto Xtmp0_val = 0;
                            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
                        }
                        else {
                            auto Xtmp1_val = super::template sext<int64_t>((static_cast<int64_t>(static_cast<uint32_t>(
                                super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
                            )) % static_cast<int64_t>(static_cast<uint32_t>(
                                super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)
                            ))));
                            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp1_val;
                        }
                    }
                }
                else {
                    auto Xtmp2_val = super::template sext<int64_t>(static_cast<uint32_t>(
                        super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
                    ));
                    super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp2_val;
                }
            }
        }
        this->do_sync(POST_SYNC, 75);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 76: REMUW */
    compile_ret_t __remuw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 76);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "remuw"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            {
                if((static_cast<uint32_t>(
                    super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)
                ) != 0)) {
                    auto Xtmp0_val = super::template sext<int64_t>((static_cast<uint32_t>(
                        super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
                    ) % static_cast<uint32_t>(
                        super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)
                    )));
                    super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
                }
                else {
                    auto Xtmp1_val = super::template sext<int64_t>(static_cast<uint32_t>(
                        super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
                    ));
                    super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp1_val;
                }
            }
        }
        this->do_sync(POST_SYNC, 76);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 77: LR.W */
    compile_ret_t __lr_w(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 77);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "lr.w"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
            auto Xtmp0_val = super::template sext<int64_t>(super::template read_mem<uint32_t>(traits<ARCH>::MEM, offs_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
            auto REStmp1_val = super::template sext<int8_t>(-(1));
            super::write_mem(traits<ARCH>::RES, offs_val, static_cast<uint32_t>(REStmp1_val));
        }
        this->do_sync(POST_SYNC, 77);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 78: SC.W */
    compile_ret_t __sc_w(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 78);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res1_val = super::template read_mem<uint32_t>(traits<ARCH>::RES, offs_val);
        {
            if((res1_val != 0)) {
                auto MEMtmp0_val = super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0);
                super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint32_t>(MEMtmp0_val));
            }
        }
        if(rd != 0){
            auto Xtmp1_val = (res1_val != super::template zext<uint64_t>(0))?
                0:
                1;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp1_val;
        }
        this->do_sync(POST_SYNC, 78);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 79: AMOSWAP.W */
    compile_ret_t __amoswap_w(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 79);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        if(rd != 0){
            auto Xtmp0_val = super::template sext<int64_t>(super::template read_mem<uint32_t>(traits<ARCH>::MEM, offs_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto MEMtmp1_val = super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0);
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint32_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 79);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 80: AMOADD.W */
    compile_ret_t __amoadd_w(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 80);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res1_val = super::template sext<int64_t>(super::template read_mem<uint32_t>(traits<ARCH>::MEM, offs_val));
        if(rd != 0){
            auto Xtmp0_val = res1_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto res2_val = (res1_val + super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0));
        auto MEMtmp1_val = res2_val;
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint32_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 80);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 81: AMOXOR.W */
    compile_ret_t __amoxor_w(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 81);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res1_val = super::template sext<int64_t>(super::template read_mem<uint32_t>(traits<ARCH>::MEM, offs_val));
        if(rd != 0){
            auto Xtmp0_val = res1_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto res2_val = (res1_val ^ super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0));
        auto MEMtmp1_val = res2_val;
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint32_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 81);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 82: AMOAND.W */
    compile_ret_t __amoand_w(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 82);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res1_val = super::template sext<int64_t>(super::template read_mem<uint32_t>(traits<ARCH>::MEM, offs_val));
        if(rd != 0){
            auto Xtmp0_val = res1_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto res2_val = (res1_val & super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0));
        auto MEMtmp1_val = res2_val;
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint32_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 82);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 83: AMOOR.W */
    compile_ret_t __amoor_w(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 83);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res1_val = super::template sext<int64_t>(super::template read_mem<uint32_t>(traits<ARCH>::MEM, offs_val));
        if(rd != 0){
            auto Xtmp0_val = res1_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto res2_val = (res1_val | super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0));
        auto MEMtmp1_val = res2_val;
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint32_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 83);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 84: AMOMIN.W */
    compile_ret_t __amomin_w(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 84);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res1_val = super::template sext<int64_t>(super::template read_mem<uint32_t>(traits<ARCH>::MEM, offs_val));
        if(rd != 0){
            auto Xtmp0_val = res1_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto res2_val = (static_cast<int64_t>(res1_val) > static_cast<int64_t>(super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)))?
            super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0):
            res1_val;
        auto MEMtmp1_val = res2_val;
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint32_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 84);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 85: AMOMAX.W */
    compile_ret_t __amomax_w(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 85);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res1_val = super::template sext<int64_t>(super::template read_mem<uint32_t>(traits<ARCH>::MEM, offs_val));
        if(rd != 0){
            auto Xtmp0_val = res1_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto res2_val = (static_cast<int64_t>(res1_val) < static_cast<int64_t>(super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)))?
            super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0):
            res1_val;
        auto MEMtmp1_val = res2_val;
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint32_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 85);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 86: AMOMINU.W */
    compile_ret_t __amominu_w(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 86);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res1_val = super::template sext<int64_t>(super::template read_mem<uint32_t>(traits<ARCH>::MEM, offs_val));
        if(rd != 0){
            auto Xtmp0_val = res1_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto res2_val = (res1_val > super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0))?
            super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0):
            res1_val;
        auto MEMtmp1_val = res2_val;
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint32_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 86);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 87: AMOMAXU.W */
    compile_ret_t __amomaxu_w(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 87);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res1_val = super::template sext<int64_t>(super::template read_mem<uint32_t>(traits<ARCH>::MEM, offs_val));
        if(rd != 0){
            auto Xtmp0_val = res1_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto res2_val = (res1_val < super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0))?
            super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0):
            res1_val;
        auto MEMtmp1_val = res2_val;
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint32_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 87);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 88: LR.D */
    compile_ret_t __lr_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 88);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "lr.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(rd != 0){
            auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
            auto Xtmp0_val = super::template sext<int64_t>(super::template read_mem<uint64_t>(traits<ARCH>::MEM, offs_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
            auto REStmp1_val = super::template sext<int8_t>(-(1));
            super::write_mem(traits<ARCH>::RES, offs_val, static_cast<uint64_t>(REStmp1_val));
        }
        this->do_sync(POST_SYNC, 88);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 89: SC.D */
    compile_ret_t __sc_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 89);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sc.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res_val = super::template read_mem<uint8_t>(traits<ARCH>::RES, offs_val);
        {
            if((res_val != 0)) {
                auto MEMtmp0_val = super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0);
                super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint64_t>(MEMtmp0_val));if(rd != 0){
                    auto Xtmp1_val = 0;
                    super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp1_val;
                }
            }
            else {
                if(rd != 0){
                    auto Xtmp2_val = 1;
                    super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp2_val;
                }
            }
        }
        this->do_sync(POST_SYNC, 89);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 90: AMOSWAP.D */
    compile_ret_t __amoswap_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 90);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amoswap.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        if(rd != 0){
            auto Xtmp0_val = super::template sext<int64_t>(super::template read_mem<uint64_t>(traits<ARCH>::MEM, offs_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto MEMtmp1_val = super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0);
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint64_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 90);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 91: AMOADD.D */
    compile_ret_t __amoadd_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 91);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amoadd.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res_val = super::template sext<int64_t>(super::template read_mem<uint64_t>(traits<ARCH>::MEM, offs_val));
        if(rd != 0){
            auto Xtmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto res2_val = (res_val + super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0));
        auto MEMtmp1_val = res2_val;
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint64_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 91);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 92: AMOXOR.D */
    compile_ret_t __amoxor_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 92);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amoxor.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res_val = super::template sext<int64_t>(super::template read_mem<uint64_t>(traits<ARCH>::MEM, offs_val));
        if(rd != 0){
            auto Xtmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto res2_val = (res_val ^ super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0));
        auto MEMtmp1_val = res2_val;
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint64_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 92);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 93: AMOAND.D */
    compile_ret_t __amoand_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 93);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amoand.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res_val = super::template sext<int64_t>(super::template read_mem<uint64_t>(traits<ARCH>::MEM, offs_val));
        if(rd != 0){
            auto Xtmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto res2_val = (res_val & super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0));
        auto MEMtmp1_val = res2_val;
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint64_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 93);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 94: AMOOR.D */
    compile_ret_t __amoor_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 94);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amoor.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res_val = super::template sext<int64_t>(super::template read_mem<uint64_t>(traits<ARCH>::MEM, offs_val));
        if(rd != 0){
            auto Xtmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto res2_val = (res_val | super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0));
        auto MEMtmp1_val = res2_val;
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint64_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 94);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 95: AMOMIN.D */
    compile_ret_t __amomin_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 95);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amomin.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res1_val = super::template sext<int64_t>(super::template read_mem<uint64_t>(traits<ARCH>::MEM, offs_val));
        if(rd != 0){
            auto Xtmp0_val = res1_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto res2_val = (static_cast<int64_t>(res1_val) > static_cast<int64_t>(super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)))?
            super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0):
            res1_val;
        auto MEMtmp1_val = res2_val;
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint64_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 95);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 96: AMOMAX.D */
    compile_ret_t __amomax_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 96);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amomax.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res_val = super::template sext<int64_t>(super::template read_mem<uint64_t>(traits<ARCH>::MEM, offs_val));
        if(rd != 0){
            auto Xtmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto res2_val = (static_cast<int64_t>(res_val) < static_cast<int64_t>(super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0)))?
            super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0):
            res_val;
        auto MEMtmp1_val = res2_val;
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint64_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 96);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 97: AMOMINU.D */
    compile_ret_t __amominu_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 97);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amominu.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res_val = super::template sext<int64_t>(super::template read_mem<uint64_t>(traits<ARCH>::MEM, offs_val));
        if(rd != 0){
            auto Xtmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto res2_val = (res_val > super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0))?
            super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0):
            res_val;
        auto MEMtmp1_val = res2_val;
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint64_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 97);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 98: AMOMAXU.D */
    compile_ret_t __amomaxu_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 98);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu={aq},rel={rl})", fmt::arg("mnemonic", "amomaxu.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        auto res1_val = super::template sext<int64_t>(super::template read_mem<uint64_t>(traits<ARCH>::MEM, offs_val));
        if(rd != 0){
            auto Xtmp0_val = res1_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        auto res2_val = (res1_val < super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0))?
            super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0):
            res1_val;
        auto MEMtmp1_val = res2_val;
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint64_t>(MEMtmp1_val));
        this->do_sync(POST_SYNC, 98);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 99: FLW */
    compile_ret_t __flw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 99);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, {imm}(x{rs1})", fmt::arg("mnemonic", "flw"),
            	fmt::arg("rd", rd), fmt::arg("imm", imm), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
        auto res_val = super::template read_mem<uint32_t>(traits<ARCH>::MEM, offs_val);
        if(64 == 32){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 99);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 100: FSW */
    compile_ret_t __fsw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 100);
        
        int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rs2}, {imm}(x{rs1})", fmt::arg("mnemonic", "fsw"),
            	fmt::arg("rs2", rs2), fmt::arg("imm", imm), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
        auto MEMtmp0_val = static_cast<uint32_t>(
            super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
        );
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint32_t>(MEMtmp0_val));
        this->do_sync(POST_SYNC, 100);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 101: FMADD.S */
    compile_ret_t __fmadd_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 101);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Ftmp0_val = fmadd_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0), 
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0), 
                super::template get_reg<reg_t>(rs3 + traits<ARCH>::F0), 
                super::template zext<uint32_t>(0), 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto frs2_val = unbox_s(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            );
            auto frs3_val = unbox_s(
                super::template get_reg<reg_t>(rs3 + traits<ARCH>::F0)
            );
            auto res_val = fmadd_s(
                frs1_val, 
                frs2_val, 
                frs3_val, 
                super::template zext<uint32_t>(0), 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 101);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 102: FMSUB.S */
    compile_ret_t __fmsub_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 102);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Ftmp0_val = fmadd_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0), 
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0), 
                super::template get_reg<reg_t>(rs3 + traits<ARCH>::F0), 
                super::template zext<uint32_t>(1), 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto frs2_val = unbox_s(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            );
            auto frs3_val = unbox_s(
                super::template get_reg<reg_t>(rs3 + traits<ARCH>::F0)
            );
            auto res_val = fmadd_s(
                frs1_val, 
                frs2_val, 
                frs3_val, 
                super::template zext<uint32_t>(1), 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 102);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 103: FNMADD.S */
    compile_ret_t __fnmadd_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 103);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Ftmp0_val = fmadd_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0), 
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0), 
                super::template get_reg<reg_t>(rs3 + traits<ARCH>::F0), 
                super::template zext<uint32_t>(2), 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto frs2_val = unbox_s(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            );
            auto frs3_val = unbox_s(
                super::template get_reg<reg_t>(rs3 + traits<ARCH>::F0)
            );
            auto res_val = fmadd_s(
                frs1_val, 
                frs2_val, 
                frs3_val, 
                super::template zext<uint32_t>(2), 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 103);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 104: FNMSUB.S */
    compile_ret_t __fnmsub_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 104);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Ftmp0_val = fmadd_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0), 
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0), 
                super::template get_reg<reg_t>(rs3 + traits<ARCH>::F0), 
                super::template zext<uint32_t>(3), 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto frs2_val = unbox_s(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            );
            auto frs3_val = unbox_s(
                super::template get_reg<reg_t>(rs3 + traits<ARCH>::F0)
            );
            auto res_val = fmadd_s(
                frs1_val, 
                frs2_val, 
                frs3_val, 
                super::template zext<uint32_t>(3), 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 104);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 105: FADD.S */
    compile_ret_t __fadd_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 105);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fadd.s"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Ftmp0_val = fadd_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0), 
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0), 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto frs2_val = unbox_s(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            );
            auto res_val = fadd_s(
                frs1_val, 
                frs2_val, 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 105);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 106: FSUB.S */
    compile_ret_t __fsub_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 106);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fsub.s"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Ftmp0_val = fsub_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0), 
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0), 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto frs2_val = unbox_s(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            );
            auto res_val = fsub_s(
                frs1_val, 
                frs2_val, 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 106);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 107: FMUL.S */
    compile_ret_t __fmul_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 107);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fmul.s"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Ftmp0_val = fmul_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0), 
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0), 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto frs2_val = unbox_s(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            );
            auto res_val = fmul_s(
                frs1_val, 
                frs2_val, 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 107);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 108: FDIV.S */
    compile_ret_t __fdiv_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 108);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fdiv.s"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Ftmp0_val = fdiv_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0), 
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0), 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto frs2_val = unbox_s(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            );
            auto res_val = fdiv_s(
                frs1_val, 
                frs2_val, 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 108);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 109: FSQRT.S */
    compile_ret_t __fsqrt_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 109);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, f{rs1}", fmt::arg("mnemonic", "fsqrt.s"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Ftmp0_val = fsqrt_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0), 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto res_val = fsqrt_s(
                frs1_val, 
                ((rm) < 7)?
                    (rm):
                    static_cast<uint8_t>(
                        super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                    )
            );
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 109);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 110: FSGNJ.S */
    compile_ret_t __fsgnj_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 110);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fsgnj.s"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Ftmp0_val = ((super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0) & 0x7fffffff) | (super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0) & 0x80000000));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto frs2_val = unbox_s(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            );
            auto res_val = ((frs1_val & 0x7fffffff) | (frs2_val & 0x80000000));
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 110);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 111: FSGNJN.S */
    compile_ret_t __fsgnjn_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 111);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fsgnjn.s"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Ftmp0_val = ((super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0) & 0x7fffffff) | (~(super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)) & 0x80000000));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto frs2_val = unbox_s(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            );
            auto res_val = ((frs1_val & 0x7fffffff) | (~(frs2_val) & 0x80000000));
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 111);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 112: FSGNJX.S */
    compile_ret_t __fsgnjx_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 112);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fsgnjx.s"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Ftmp0_val = (super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0) ^ (super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0) & 0x80000000));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto frs2_val = unbox_s(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            );
            auto res_val = (frs1_val ^ (frs2_val & 0x80000000));
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 112);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 113: FMIN.S */
    compile_ret_t __fmin_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 113);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fmin.s"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Ftmp0_val = fsel_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0), 
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0), 
                super::template zext<uint32_t>(0)
            );
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto frs2_val = unbox_s(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            );
            auto res_val = fsel_s(
                frs1_val, 
                frs2_val, 
                super::template zext<uint32_t>(0)
            );
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 113);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 114: FMAX.S */
    compile_ret_t __fmax_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 114);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fmax.s"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Ftmp0_val = fsel_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0), 
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0), 
                super::template zext<uint32_t>(1)
            );
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto frs2_val = unbox_s(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            );
            auto res_val = fsel_s(
                frs1_val, 
                frs2_val, 
                super::template zext<uint32_t>(1)
            );
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 114);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 115: FCVT.W.S */
    compile_ret_t __fcvt_w_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 115);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}", fmt::arg("mnemonic", "fcvt.w.s"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Xtmp0_val = super::template sext<int64_t>(fcvt_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0), 
                super::template zext<uint32_t>(0), 
                (rm)
            ));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto Xtmp1_val = super::template sext<int64_t>(fcvt_s(
                frs1_val, 
                super::template zext<uint32_t>(0), 
                (rm)
            ));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 115);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 116: FCVT.WU.S */
    compile_ret_t __fcvt_wu_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 116);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}", fmt::arg("mnemonic", "fcvt.wu.s"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Xtmp0_val = super::template sext<int64_t>(fcvt_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0), 
                super::template zext<uint32_t>(1), 
                (rm)
            ));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto Xtmp1_val = super::template sext<int64_t>(fcvt_s(
                frs1_val, 
                super::template zext<uint32_t>(1), 
                (rm)
            ));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 116);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 117: FEQ.S */
    compile_ret_t __feq_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 117);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "feq.s"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Xtmp0_val = super::template zext<uint64_t>(fcmp_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0), 
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0), 
                super::template zext<uint32_t>(0)
            ));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto frs2_val = unbox_s(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            );
            auto Xtmp1_val = super::template zext<uint64_t>(fcmp_s(
                frs1_val, 
                frs2_val, 
                super::template zext<uint32_t>(0)
            ));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 117);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 118: FLT.S */
    compile_ret_t __flt_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 118);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "flt.s"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Xtmp0_val = super::template zext<uint64_t>(fcmp_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0), 
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0), 
                super::template zext<uint32_t>(2)
            ));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto frs2_val = unbox_s(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            );
            auto Xtmp1_val = super::template zext<uint64_t>(fcmp_s(
                frs1_val, 
                frs2_val, 
                super::template zext<uint32_t>(2)
            ));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp1_val;
        }
        auto Xtmp2_val = fcmp_s(
            static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            ), 
            super::template zext<uint32_t>(2)
        );
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp2_val;
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 118);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 119: FLE.S */
    compile_ret_t __fle_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 119);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fle.s"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Xtmp0_val = super::template zext<uint64_t>(fcmp_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0), 
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0), 
                super::template zext<uint32_t>(1)
            ));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        } else {
            auto frs1_val = unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            );
            auto frs2_val = unbox_s(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            );
            auto Xtmp1_val = super::template zext<uint64_t>(fcmp_s(
                frs1_val, 
                frs2_val, 
                super::template zext<uint32_t>(1)
            ));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 119);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 120: FCLASS.S */
    compile_ret_t __fclass_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 120);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}", fmt::arg("mnemonic", "fclass.s"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto Xtmp0_val = fclass_s(
            unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            )
        );
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 120);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 121: FCVT.S.W */
    compile_ret_t __fcvt_s_w(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 121);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, {rs1}", fmt::arg("mnemonic", "fcvt.s.w"),
            	fmt::arg("rd", rd), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Ftmp0_val = fcvt_s(
                static_cast<uint32_t>(
                    super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
                ), 
                super::template zext<uint32_t>(2), 
                (rm)
            );
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            auto res_val = fcvt_s(
                static_cast<uint32_t>(
                    super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
                ), 
                super::template zext<uint32_t>(2), 
                (rm)
            );
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 121);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 122: FCVT.S.WU */
    compile_ret_t __fcvt_s_wu(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 122);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, {rs1}", fmt::arg("mnemonic", "fcvt.s.wu"),
            	fmt::arg("rd", rd), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Ftmp0_val = fcvt_s(
                static_cast<uint32_t>(
                    super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
                ), 
                super::template zext<uint32_t>(3), 
                (rm)
            );
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            auto res_val = fcvt_s(
                static_cast<uint32_t>(
                    super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
                ), 
                super::template zext<uint32_t>(3), 
                (rm)
            );
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 122);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 123: FMV.X.W */
    compile_ret_t __fmv_x_w(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 123);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}", fmt::arg("mnemonic", "fmv.x.w"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto Xtmp0_val = super::template sext<int64_t>(static_cast<uint32_t>(
            super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
        ));
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 123);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 124: FMV.W.X */
    compile_ret_t __fmv_w_x(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 124);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, {rs1}", fmt::arg("mnemonic", "fmv.w.x"),
            	fmt::arg("rd", rd), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        if(64 == 32){
            auto Ftmp0_val = static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
            );
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
            )));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 124);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 125: FCVT.L.S */
    compile_ret_t __fcvt_l_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 125);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} x{rd}, f{rs1}", fmt::arg("mnemonic", "fcvt.l.s"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fcvt_32_64(
            unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            super::template zext<uint32_t>(0), 
            (rm)
        );
        auto Xtmp0_val = super::template sext<int64_t>(res_val);
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 125);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 126: FCVT.LU.S */
    compile_ret_t __fcvt_lu_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 126);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} x{rd}, f{rs1}", fmt::arg("mnemonic", "fcvt.lu.s"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fcvt_32_64(
            unbox_s(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            super::template zext<uint32_t>(1), 
            (rm)
        );
        auto Xtmp0_val = super::template zext<uint64_t>(res_val);
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 126);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 127: FCVT.S.L */
    compile_ret_t __fcvt_s_l(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 127);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, x{rs1}", fmt::arg("mnemonic", "fcvt.s.l"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fcvt_64_32(
            super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0), 
            super::template zext<uint32_t>(2), 
            (rm)
        );
        if(64 == 32){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 127);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 128: FCVT.S.LU */
    compile_ret_t __fcvt_s_lu(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 128);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, x{rs1}", fmt::arg("mnemonic", "fcvt.s.lu"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fcvt_64_32(
            super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0), 
            super::template zext<uint32_t>(3), 
            (rm)
        );
        if(64 == 32){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 128);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 129: FLD */
    compile_ret_t __fld(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 129);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, {imm}({rs1})", fmt::arg("mnemonic", "fld"),
            	fmt::arg("rd", rd), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
        auto res_val = super::template read_mem<uint64_t>(traits<ARCH>::MEM, offs_val);
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 129);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 130: FSD */
    compile_ret_t __fsd(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 130);
        
        int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rs2}, {imm}({rs1})", fmt::arg("mnemonic", "fsd"),
            	fmt::arg("rs2", rs2), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto offs_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
        auto MEMtmp0_val = static_cast<uint64_t>(
            super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
        );
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint64_t>(MEMtmp0_val));
        this->do_sync(POST_SYNC, 130);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 131: FMADD.D */
    compile_ret_t __fmadd_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 131);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fmadd_d(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            ), 
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs3 + traits<ARCH>::F0)
            ), 
            super::template zext<uint32_t>(0), 
            ((rm) < 7)?
                (rm):
                static_cast<uint8_t>(
                    super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                )
        );
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 131);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 132: FMSUB.D */
    compile_ret_t __fmsub_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 132);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fmadd_d(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            ), 
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs3 + traits<ARCH>::F0)
            ), 
            super::template zext<uint32_t>(1), 
            ((rm) < 7)?
                (rm):
                static_cast<uint8_t>(
                    super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                )
        );
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 132);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 133: FNMADD.D */
    compile_ret_t __fnmadd_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 133);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fmadd_d(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            ), 
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs3 + traits<ARCH>::F0)
            ), 
            super::template zext<uint32_t>(2), 
            ((rm) < 7)?
                (rm):
                static_cast<uint8_t>(
                    super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                )
        );
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 133);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 134: FNMSUB.D */
    compile_ret_t __fnmsub_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 134);
        
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
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fmadd_d(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            ), 
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs3 + traits<ARCH>::F0)
            ), 
            super::template zext<uint32_t>(3), 
            ((rm) < 7)?
                (rm):
                static_cast<uint8_t>(
                    super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                )
        );
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 134);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 135: FADD.D */
    compile_ret_t __fadd_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 135);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fadd.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fadd_d(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            ), 
            ((rm) < 7)?
                (rm):
                static_cast<uint8_t>(
                    super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                )
        );
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 135);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 136: FSUB.D */
    compile_ret_t __fsub_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 136);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fsub.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fsub_d(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            ), 
            ((rm) < 7)?
                (rm):
                static_cast<uint8_t>(
                    super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                )
        );
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 136);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 137: FMUL.D */
    compile_ret_t __fmul_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 137);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fmul.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fmul_d(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            ), 
            ((rm) < 7)?
                (rm):
                static_cast<uint8_t>(
                    super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                )
        );
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 137);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 138: FDIV.D */
    compile_ret_t __fdiv_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 138);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fdiv.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fdiv_d(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            ), 
            ((rm) < 7)?
                (rm):
                static_cast<uint8_t>(
                    super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                )
        );
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 138);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 139: FSQRT.D */
    compile_ret_t __fsqrt_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 139);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}", fmt::arg("mnemonic", "fsqrt.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fsqrt_d(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            ((rm) < 7)?
                (rm):
                static_cast<uint8_t>(
                    super::template get_reg<reg_t>(traits<ARCH>::FCSR)
                )
        );
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 139);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 140: FSGNJ.D */
    compile_ret_t __fsgnj_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 140);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fsgnj.d"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        uint64_t ONE_val = 1;
        uint64_t MSK1_val = ONE_val << 63;
        uint64_t MSK2_val = MSK1_val - 1;
        auto res_val = ((static_cast<uint64_t>(
            super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
        ) & MSK2_val) | (static_cast<uint64_t>(
            super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
        ) & MSK1_val));
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 140);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 141: FSGNJN.D */
    compile_ret_t __fsgnjn_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 141);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fsgnjn.d"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        uint64_t ONE_val = 1;
        uint64_t MSK1_val = ONE_val << 63;
        uint64_t MSK2_val = MSK1_val - 1;
        auto res_val = ((static_cast<uint64_t>(
            super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
        ) & MSK2_val) | (~(static_cast<uint64_t>(
            super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
        )) & MSK1_val));
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 141);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 142: FSGNJX.D */
    compile_ret_t __fsgnjx_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 142);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fsgnjx.d"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        uint64_t ONE_val = 1;
        uint64_t MSK1_val = ONE_val << 63;
        auto res_val = (static_cast<uint64_t>(
            super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
        ) ^ (static_cast<uint64_t>(
            super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
        ) & MSK1_val));
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 142);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 143: FMIN.D */
    compile_ret_t __fmin_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 143);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fmin.d"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fsel_d(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            ), 
            super::template zext<uint32_t>(0)
        );
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 143);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 144: FMAX.D */
    compile_ret_t __fmax_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 144);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fmax.d"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fsel_d(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            ), 
            super::template zext<uint32_t>(1)
        );
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 144);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 145: FCVT.S.D */
    compile_ret_t __fcvt_s_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 145);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, f{rs1}", fmt::arg("mnemonic", "fcvt.s.d"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fconv_d2f(
            super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0), 
            (rm)
        );
        uint64_t upper_val = - 1;
        auto Ftmp0_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
        super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        this->do_sync(POST_SYNC, 145);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 146: FCVT.D.S */
    compile_ret_t __fcvt_d_s(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 146);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, f{rs1}", fmt::arg("mnemonic", "fcvt.d.s"),
            	fmt::arg("rd", rd), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fconv_f2d(
            static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            (rm)
        );
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 146);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 147: FEQ.D */
    compile_ret_t __feq_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 147);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "feq.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto Xtmp0_val = super::template zext<uint64_t>(fcmp_d(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            ), 
            super::template zext<uint32_t>(0)
        ));
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 147);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 148: FLT.D */
    compile_ret_t __flt_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 148);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "flt.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto Xtmp0_val = super::template zext<uint64_t>(fcmp_d(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            ), 
            super::template zext<uint32_t>(2)
        ));
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 148);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 149: FLE.D */
    compile_ret_t __fle_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 149);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}, f{rs2}", fmt::arg("mnemonic", "fle.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1), fmt::arg("rs2", rs2));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto Xtmp0_val = super::template zext<uint64_t>(fcmp_d(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
            ), 
            super::template zext<uint32_t>(1)
        ));
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 149);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 150: FCLASS.D */
    compile_ret_t __fclass_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 150);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}", fmt::arg("mnemonic", "fclass.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto Xtmp0_val = fclass_d(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            )
        );
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 150);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 151: FCVT.W.D */
    compile_ret_t __fcvt_w_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 151);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}", fmt::arg("mnemonic", "fcvt.w.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto Xtmp0_val = super::template sext<int64_t>(fcvt_64_32(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            super::template zext<uint32_t>(0), 
            (rm)
        ));
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 151);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 152: FCVT.WU.D */
    compile_ret_t __fcvt_wu_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 152);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}", fmt::arg("mnemonic", "fcvt.wu.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto Xtmp0_val = super::template sext<int64_t>(fcvt_64_32(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            super::template zext<uint32_t>(1), 
            (rm)
        ));
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 152);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 153: FCVT.D.W */
    compile_ret_t __fcvt_d_w(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 153);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, {rs1}", fmt::arg("mnemonic", "fcvt.d.w"),
            	fmt::arg("rd", rd), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fcvt_32_64(
            super::template sext<int32_t>(static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
            )), 
            super::template zext<uint32_t>(2), 
            (rm)
        );
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 153);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 154: FCVT.D.WU */
    compile_ret_t __fcvt_d_wu(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 154);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, {rs1}", fmt::arg("mnemonic", "fcvt.d.wu"),
            	fmt::arg("rd", rd), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fcvt_32_64(
            super::template zext<uint32_t>(static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
            )), 
            super::template zext<uint32_t>(3), 
            (rm)
        );
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 154);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 155: FCVT.L.D */
    compile_ret_t __fcvt_l_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 155);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}", fmt::arg("mnemonic", "fcvt.l.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto Xtmp0_val = super::template sext<int64_t>(fcvt_d(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            super::template zext<uint32_t>(0), 
            (rm)
        ));
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 155);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 156: FCVT.LU.D */
    compile_ret_t __fcvt_lu_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 156);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}", fmt::arg("mnemonic", "fcvt.lu.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto Xtmp0_val = super::template sext<int64_t>(fcvt_d(
            static_cast<uint64_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0)
            ), 
            super::template zext<uint32_t>(1), 
            (rm)
        ));
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        auto flags_val = fget_flags(
        );
        auto FCSR_val = ((super::template get_reg<reg_t>(traits<ARCH>::FCSR) & ~((0x1f))) + flags_val);
        super::template get_reg(traits<ARCH>::FCSR) = FCSR_val;
        this->do_sync(POST_SYNC, 156);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 157: FCVT.D.L */
    compile_ret_t __fcvt_d_l(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 157);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, {rs1}", fmt::arg("mnemonic", "fcvt.d.l"),
            	fmt::arg("rd", rd), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fcvt_d(
            super::template sext<int32_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)), 
            super::template zext<uint32_t>(2), 
            (rm)
        );
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 157);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 158: FCVT.D.LU */
    compile_ret_t __fcvt_d_lu(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 158);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, {rs1}", fmt::arg("mnemonic", "fcvt.d.lu"),
            	fmt::arg("rd", rd), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto res_val = fcvt_d(
            super::template zext<uint32_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)), 
            super::template zext<uint32_t>(3), 
            (rm)
        );
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 158);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 159: FMV.X.D */
    compile_ret_t __fmv_x_d(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 159);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, f{rs1}", fmt::arg("mnemonic", "fmv.x.d"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", rs1));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto Xtmp0_val = super::template sext<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::F0));
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 159);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 160: FMV.D.X */
    compile_ret_t __fmv_d_x(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 160);
        
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, {rs1}", fmt::arg("mnemonic", "fmv.d.x"),
            	fmt::arg("rd", rd), fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 4;
        auto Ftmp0_val = super::template zext<uint64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0));
        super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        this->do_sync(POST_SYNC, 160);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 161: C.ADDI4SPN */
    compile_ret_t __c_addi4spn(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 161);
        
        uint8_t rd = ((bit_sub<2,3>(instr)));
        uint16_t imm = ((bit_sub<5,1>(instr) << 3) | (bit_sub<6,1>(instr) << 2) | (bit_sub<7,4>(instr) << 6) | (bit_sub<11,2>(instr) << 4));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c.addi4spn"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        if(imm == 0){
            raise_trap(0, 2);
        }
        auto Xtmp0_val = (super::template get_reg<reg_t>(2 + traits<ARCH>::X0) + (imm));
        super::template get_reg<reg_t>(rd + 8 + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 161);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 162: C.LW */
    compile_ret_t __c_lw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 162);
        
        uint8_t rd = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {uimm:#05x}({rs1})", fmt::arg("mnemonic", "c.lw"),
            	fmt::arg("rd", name(8+rd)), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto offs_val = (super::template get_reg<reg_t>(rs1 + 8 + traits<ARCH>::X0) + (uimm));
        auto Xtmp0_val = super::template sext<int64_t>(super::template read_mem<uint32_t>(traits<ARCH>::MEM, offs_val));
        super::template get_reg<reg_t>(rd + 8 + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 162);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 163: C.SW */
    compile_ret_t __c_sw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 163);
        
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {uimm:#05x}({rs1})", fmt::arg("mnemonic", "c.sw"),
            	fmt::arg("rs2", name(8+rs2)), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto offs_val = (super::template get_reg<reg_t>(rs1 + 8 + traits<ARCH>::X0) + (uimm));
        auto MEMtmp0_val = super::template get_reg<reg_t>(rs2 + 8 + traits<ARCH>::X0);
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint32_t>(MEMtmp0_val));
        this->do_sync(POST_SYNC, 163);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 164: C.ADDI */
    compile_ret_t __c_addi(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 164);
        
        int8_t imm = signextend<int8_t,6>((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.addi"),
            	fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto Xtmp0_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)) + (imm));
        super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 164);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 165: C.NOP */
    compile_ret_t __c_nop(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 165);
        
        if(this->disass_enabled){
            /* generate console output when executing the command */
            this->core.disass_output(pc.val, "c.nop");
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        /* TODO: describe operations for C.NOP ! */
        this->do_sync(POST_SYNC, 165);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 166: C.JAL */
    compile_ret_t __c_jal(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 166);
        
        int16_t imm = signextend<int16_t,12>((bit_sub<2,1>(instr) << 5) | (bit_sub<3,3>(instr) << 1) | (bit_sub<6,1>(instr) << 7) | (bit_sub<7,1>(instr) << 6) | (bit_sub<8,1>(instr) << 10) | (bit_sub<9,2>(instr) << 8) | (bit_sub<11,1>(instr) << 4) | (bit_sub<12,1>(instr) << 11));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {imm:#05x}", fmt::arg("mnemonic", "c.jal"),
            	fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto Xtmp0_val = (cur_pc_val + 2);
        super::template get_reg<reg_t>(1 + traits<ARCH>::X0)=Xtmp0_val;
        auto PC_val = (static_cast<int64_t>(cur_pc_val) + (imm));
        super::template get_reg(traits<ARCH>::NEXT_PC) = PC_val;
        auto is_cont_v = PC_val !=pc.val;
        super::template get_reg(traits<ARCH>::LAST_BRANCH) = is_cont_v?1:0;
        this->do_sync(POST_SYNC, 166);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 167: C.LI */
    compile_ret_t __c_li(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 167);
        
        int8_t imm = signextend<int8_t,6>((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c.li"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        if(rd == 0){
            raise_trap(0, 2);
        }
        auto Xtmp0_val = (imm);
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 167);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 168: C.LUI */
    compile_ret_t __c_lui(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 168);
        
        int32_t imm = signextend<int32_t,18>((bit_sub<2,5>(instr) << 12) | (bit_sub<12,1>(instr) << 17));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c.lui"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        if(rd == 0){
            raise_trap(0, 2);
        }
        if(imm == 0){
            raise_trap(0, 2);
        }
        auto Xtmp0_val = (imm);
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 168);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 169: C.ADDI16SP */
    compile_ret_t __c_addi16sp(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 169);
        
        int16_t imm = signextend<int16_t,10>((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 7) | (bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 4) | (bit_sub<12,1>(instr) << 9));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {imm:#05x}", fmt::arg("mnemonic", "c.addi16sp"),
            	fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto Xtmp0_val = (static_cast<int64_t>(super::template get_reg<reg_t>(2 + traits<ARCH>::X0)) + (imm));
        super::template get_reg<reg_t>(2 + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 169);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 170: C.SRLI */
    compile_ret_t __c_srli(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 170);
        
        uint8_t shamt = ((bit_sub<2,5>(instr)));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {shamt}", fmt::arg("mnemonic", "c.srli"),
            	fmt::arg("rs1", name(8+rs1)), fmt::arg("shamt", shamt));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        uint8_t rs1_idx_val = rs1 + 8;
        auto Xtmp0_val = (static_cast<uint64_t>(super::template get_reg<reg_t>(rs1_idx_val + traits<ARCH>::X0))>>(shamt));
        super::template get_reg<reg_t>(rs1_idx_val + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 170);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 171: C.SRAI */
    compile_ret_t __c_srai(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 171);
        
        uint8_t shamt = ((bit_sub<2,5>(instr)));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {shamt}", fmt::arg("mnemonic", "c.srai"),
            	fmt::arg("rs1", name(8+rs1)), fmt::arg("shamt", shamt));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        uint8_t rs1_idx_val = rs1 + 8;
        auto Xtmp0_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1_idx_val + traits<ARCH>::X0))>>(shamt));
        super::template get_reg<reg_t>(rs1_idx_val + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 171);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 172: C.ANDI */
    compile_ret_t __c_andi(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 172);
        
        int8_t imm = signextend<int8_t,6>((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.andi"),
            	fmt::arg("rs1", name(8+rs1)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        uint8_t rs1_idx_val = rs1 + 8;
        auto Xtmp0_val = (static_cast<int64_t>(super::template get_reg<reg_t>(rs1_idx_val + traits<ARCH>::X0)) & (imm));
        super::template get_reg<reg_t>(rs1_idx_val + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 172);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 173: C.SUB */
    compile_ret_t __c_sub(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 173);
        
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.sub"),
            	fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        uint8_t rd_idx_val = rd + 8;
        auto Xtmp0_val = (super::template get_reg<reg_t>(rd_idx_val + traits<ARCH>::X0) - super::template get_reg<reg_t>(rs2 + 8 + traits<ARCH>::X0));
        super::template get_reg<reg_t>(rd_idx_val + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 173);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 174: C.XOR */
    compile_ret_t __c_xor(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 174);
        
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.xor"),
            	fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        uint8_t rd_idx_val = rd + 8;
        auto Xtmp0_val = (super::template get_reg<reg_t>(rd_idx_val + traits<ARCH>::X0) ^ super::template get_reg<reg_t>(rs2 + 8 + traits<ARCH>::X0));
        super::template get_reg<reg_t>(rd_idx_val + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 174);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 175: C.OR */
    compile_ret_t __c_or(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 175);
        
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.or"),
            	fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        uint8_t rd_idx_val = rd + 8;
        auto Xtmp0_val = (super::template get_reg<reg_t>(rd_idx_val + traits<ARCH>::X0) | super::template get_reg<reg_t>(rs2 + 8 + traits<ARCH>::X0));
        super::template get_reg<reg_t>(rd_idx_val + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 175);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 176: C.AND */
    compile_ret_t __c_and(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 176);
        
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.and"),
            	fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        uint8_t rd_idx_val = rd + 8;
        auto Xtmp0_val = (super::template get_reg<reg_t>(rd_idx_val + traits<ARCH>::X0) & super::template get_reg<reg_t>(rs2 + 8 + traits<ARCH>::X0));
        super::template get_reg<reg_t>(rd_idx_val + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 176);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 177: C.J */
    compile_ret_t __c_j(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 177);
        
        int16_t imm = signextend<int16_t,12>((bit_sub<2,1>(instr) << 5) | (bit_sub<3,3>(instr) << 1) | (bit_sub<6,1>(instr) << 7) | (bit_sub<7,1>(instr) << 6) | (bit_sub<8,1>(instr) << 10) | (bit_sub<9,2>(instr) << 8) | (bit_sub<11,1>(instr) << 4) | (bit_sub<12,1>(instr) << 11));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {imm:#05x}", fmt::arg("mnemonic", "c.j"),
            	fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto PC_val = (static_cast<int64_t>(cur_pc_val) + (imm));
        super::template get_reg(traits<ARCH>::NEXT_PC) = PC_val;
        auto is_cont_v = PC_val !=pc.val;
        super::template get_reg(traits<ARCH>::LAST_BRANCH) = is_cont_v?1:0;
        this->do_sync(POST_SYNC, 177);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 178: C.BEQZ */
    compile_ret_t __c_beqz(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 178);
        
        int16_t imm = signextend<int16_t,9>((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 1) | (bit_sub<5,2>(instr) << 6) | (bit_sub<10,2>(instr) << 3) | (bit_sub<12,1>(instr) << 8));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.beqz"),
            	fmt::arg("rs1", name(8+rs1)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto PC_val = (super::template get_reg<reg_t>(rs1 + 8 + traits<ARCH>::X0) == 0)?
            (static_cast<int64_t>(cur_pc_val) + (imm)):
            (cur_pc_val + 2);
        super::template get_reg(traits<ARCH>::NEXT_PC) = PC_val;
        auto is_cont_v = PC_val !=pc.val;
        super::template get_reg(traits<ARCH>::LAST_BRANCH) = is_cont_v?1:0;
        this->do_sync(POST_SYNC, 178);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 179: C.BNEZ */
    compile_ret_t __c_bnez(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 179);
        
        int16_t imm = signextend<int16_t,9>((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 1) | (bit_sub<5,2>(instr) << 6) | (bit_sub<10,2>(instr) << 3) | (bit_sub<12,1>(instr) << 8));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.bnez"),
            	fmt::arg("rs1", name(8+rs1)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto PC_val = (super::template get_reg<reg_t>(rs1 + 8 + traits<ARCH>::X0) != 0)?
            (static_cast<int64_t>(cur_pc_val) + (imm)):
            (cur_pc_val + 2);
        super::template get_reg(traits<ARCH>::NEXT_PC) = PC_val;
        auto is_cont_v = PC_val !=pc.val;
        super::template get_reg(traits<ARCH>::LAST_BRANCH) = is_cont_v?1:0;
        this->do_sync(POST_SYNC, 179);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 180: C.SLLI */
    compile_ret_t __c_slli(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 180);
        
        uint8_t shamt = ((bit_sub<2,5>(instr)));
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {shamt}", fmt::arg("mnemonic", "c.slli"),
            	fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        if(rs1 == 0){
            raise_trap(0, 2);
        }
        auto Xtmp0_val = (super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)<<(shamt));
        super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 180);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 181: C.LWSP */
    compile_ret_t __c_lwsp(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 181);
        
        uint8_t uimm = ((bit_sub<2,2>(instr) << 6) | (bit_sub<4,3>(instr) << 2) | (bit_sub<12,1>(instr) << 5));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, sp, {uimm:#05x}", fmt::arg("mnemonic", "c.lwsp"),
            	fmt::arg("rd", name(rd)), fmt::arg("uimm", uimm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto offs_val = (super::template get_reg<reg_t>(2 + traits<ARCH>::X0) + (uimm));
        auto Xtmp0_val = super::template sext<int64_t>(super::template read_mem<uint32_t>(traits<ARCH>::MEM, offs_val));
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 181);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 182: C.MV */
    compile_ret_t __c_mv(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 182);
        
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.mv"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto Xtmp0_val = super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0);
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 182);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 183: C.JR */
    compile_ret_t __c_jr(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 183);
        
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}", fmt::arg("mnemonic", "c.jr"),
            	fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto PC_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        super::template get_reg(traits<ARCH>::NEXT_PC) = PC_val;
        super::template get_reg(traits<ARCH>::LAST_BRANCH) = std::numeric_limits<uint32_t>::max();
        this->do_sync(POST_SYNC, 183);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 184: C.ADD */
    compile_ret_t __c_add(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 184);
        
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.add"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs2", name(rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto Xtmp0_val = (super::template get_reg<reg_t>(rd + traits<ARCH>::X0) + super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0));
        super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 184);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 185: C.JALR */
    compile_ret_t __c_jalr(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 185);
        
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}", fmt::arg("mnemonic", "c.jalr"),
            	fmt::arg("rs1", name(rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto Xtmp0_val = (cur_pc_val + 2);
        super::template get_reg<reg_t>(1 + traits<ARCH>::X0)=Xtmp0_val;
        auto PC_val = super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0);
        super::template get_reg(traits<ARCH>::NEXT_PC) = PC_val;
        super::template get_reg(traits<ARCH>::LAST_BRANCH) = std::numeric_limits<uint32_t>::max();
        this->do_sync(POST_SYNC, 185);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 186: C.EBREAK */
    compile_ret_t __c_ebreak(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 186);
        
        if(this->disass_enabled){
            /* generate console output when executing the command */
            this->core.disass_output(pc.val, "c.ebreak");
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        raise_trap(0, 3);
        this->do_sync(POST_SYNC, 186);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 187: C.SWSP */
    compile_ret_t __c_swsp(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 187);
        
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t uimm = ((bit_sub<7,2>(instr) << 6) | (bit_sub<9,4>(instr) << 2));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {uimm:#05x}(sp)", fmt::arg("mnemonic", "c.swsp"),
            	fmt::arg("rs2", name(rs2)), fmt::arg("uimm", uimm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto offs_val = (super::template get_reg<reg_t>(2 + traits<ARCH>::X0) + (uimm));
        auto MEMtmp0_val = super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0);
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint32_t>(MEMtmp0_val));
        this->do_sync(POST_SYNC, 187);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 188: DII */
    compile_ret_t __dii(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 188);
        
        if(this->disass_enabled){
            /* generate console output when executing the command */
            this->core.disass_output(pc.val, "dii");
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        raise_trap(0, 2);
        this->do_sync(POST_SYNC, 188);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 189: C.LD */
    compile_ret_t __c_ld(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 189);
        
        uint8_t rd = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,2>(instr) << 6) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {uimm},({rs1})", fmt::arg("mnemonic", "c.ld"),
            	fmt::arg("rd", name(8+rd)), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto offs_val = (super::template get_reg<reg_t>(rs1 + 8 + traits<ARCH>::X0) + (uimm));
        auto Xtmp0_val = super::template sext<int64_t>(super::template read_mem<uint64_t>(traits<ARCH>::MEM, offs_val));
        super::template get_reg<reg_t>(rd + 8 + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 189);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 190: C.SD */
    compile_ret_t __c_sd(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 190);
        
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,2>(instr) << 6) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {uimm},({rs1})", fmt::arg("mnemonic", "c.sd"),
            	fmt::arg("rs2", name(8+rs2)), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto offs_val = (super::template get_reg<reg_t>(rs1 + 8 + traits<ARCH>::X0) + (uimm));
        auto MEMtmp0_val = super::template get_reg<reg_t>(rs2 + 8 + traits<ARCH>::X0);
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint64_t>(MEMtmp0_val));
        this->do_sync(POST_SYNC, 190);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 191: C.SUBW */
    compile_ret_t __c_subw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 191);
        
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rd}, {rs2}", fmt::arg("mnemonic", "c.subw"),
            	fmt::arg("rd", name(8+rd)), fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto res_val = (static_cast<uint32_t>(
            super::template get_reg<reg_t>(rd + 8 + traits<ARCH>::X0)
        ) - static_cast<uint32_t>(
            super::template get_reg<reg_t>(rs2 + 8 + traits<ARCH>::X0)
        ));
        auto Xtmp0_val = super::template sext<int64_t>(res_val);
        super::template get_reg<reg_t>(rd + 8 + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 191);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 192: C.ADDW */
    compile_ret_t __c_addw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 192);
        
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rd}, {rs2}", fmt::arg("mnemonic", "c.addw"),
            	fmt::arg("rd", name(8+rd)), fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto res_val = (static_cast<uint32_t>(
            super::template get_reg<reg_t>(rd + 8 + traits<ARCH>::X0)
        ) + static_cast<uint32_t>(
            super::template get_reg<reg_t>(rs2 + 8 + traits<ARCH>::X0)
        ));
        auto Xtmp0_val = super::template sext<int64_t>(res_val);
        super::template get_reg<reg_t>(rd + 8 + traits<ARCH>::X0)=Xtmp0_val;
        this->do_sync(POST_SYNC, 192);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 193: C.ADDIW */
    compile_ret_t __c_addiw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 193);
        
        int8_t imm = signextend<int8_t,6>((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.addiw"),
            	fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        if(rs1 != 0){
            auto res_val = (static_cast<int32_t>(static_cast<uint32_t>(
                super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)
            )) + (imm));
            auto Xtmp0_val = super::template sext<int64_t>(res_val);
            super::template get_reg<reg_t>(rs1 + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 193);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 194: C.LDSP */
    compile_ret_t __c_ldsp(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 194);
        
        uint16_t uimm = ((bit_sub<2,3>(instr) << 6) | (bit_sub<5,2>(instr) << 3) | (bit_sub<12,1>(instr) << 5));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {uimm}(sp)", fmt::arg("mnemonic", "c.ldsp"),
            	fmt::arg("rd", name(rd)), fmt::arg("uimm", uimm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto offs_val = (super::template get_reg<reg_t>(2 + traits<ARCH>::X0) + (uimm));
        if(rd != 0){
            auto Xtmp0_val = super::template sext<int64_t>(super::template read_mem<uint64_t>(traits<ARCH>::MEM, offs_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::X0)=Xtmp0_val;
        }
        this->do_sync(POST_SYNC, 194);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 195: C.SDSP */
    compile_ret_t __c_sdsp(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 195);
        
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint16_t uimm = ((bit_sub<7,3>(instr) << 6) | (bit_sub<10,3>(instr) << 3));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {uimm}(sp)", fmt::arg("mnemonic", "c.sdsp"),
            	fmt::arg("rs2", name(rs2)), fmt::arg("uimm", uimm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto offs_val = (super::template get_reg<reg_t>(2 + traits<ARCH>::X0) + (uimm));
        auto MEMtmp0_val = super::template get_reg<reg_t>(rs2 + traits<ARCH>::X0);
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint64_t>(MEMtmp0_val));
        this->do_sync(POST_SYNC, 195);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 196: C.FLW */
    compile_ret_t __c_flw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 196);
        
        uint8_t rd = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f(8+{rd}), {uimm}({rs1})", fmt::arg("mnemonic", "c.flw"),
            	fmt::arg("rd", rd), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto offs_val = (super::template get_reg<reg_t>(rs1 + 8 + traits<ARCH>::X0) + (uimm));
        auto res_val = super::template read_mem<uint32_t>(traits<ARCH>::MEM, offs_val);
        if(64 == 32){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + 8 + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + 8 + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 196);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 197: C.FSW */
    compile_ret_t __c_fsw(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 197);
        
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f(8+{rs2}), {uimm}({rs1})", fmt::arg("mnemonic", "c.fsw"),
            	fmt::arg("rs2", rs2), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto offs_val = (super::template get_reg<reg_t>(rs1 + 8 + traits<ARCH>::X0) + (uimm));
        auto MEMtmp0_val = static_cast<uint32_t>(
            super::template get_reg<reg_t>(rs2 + 8 + traits<ARCH>::F0)
        );
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint32_t>(MEMtmp0_val));
        this->do_sync(POST_SYNC, 197);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 198: C.FLWSP */
    compile_ret_t __c_flwsp(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 198);
        
        uint8_t uimm = ((bit_sub<2,2>(instr) << 6) | (bit_sub<4,3>(instr) << 2) | (bit_sub<12,1>(instr) << 5));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, {uimm}(x2)", fmt::arg("mnemonic", "c.flwsp"),
            	fmt::arg("rd", rd), fmt::arg("uimm", uimm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto offs_val = (super::template get_reg<reg_t>(2 + traits<ARCH>::X0) + (uimm));
        auto res_val = super::template read_mem<uint32_t>(traits<ARCH>::MEM, offs_val);
        if(64 == 32){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 32) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 198);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 199: C.FSWSP */
    compile_ret_t __c_fswsp(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 199);
        
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t uimm = ((bit_sub<7,2>(instr) << 6) | (bit_sub<9,4>(instr) << 2));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rs2}, {uimm}(x2), ", fmt::arg("mnemonic", "c.fswsp"),
            	fmt::arg("rs2", rs2), fmt::arg("uimm", uimm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto offs_val = (super::template get_reg<reg_t>(2 + traits<ARCH>::X0) + (uimm));
        auto MEMtmp0_val = static_cast<uint32_t>(
            super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
        );
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint32_t>(MEMtmp0_val));
        this->do_sync(POST_SYNC, 199);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 200: C.FLD */
    compile_ret_t __c_fld(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 200);
        
        uint8_t rd = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,2>(instr) << 6) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f(8+{rd}), {uimm}({rs1})", fmt::arg("mnemonic", "c.fld"),
            	fmt::arg("rd", rd), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto offs_val = (super::template get_reg<reg_t>(rs1 + 8 + traits<ARCH>::X0) + (uimm));
        auto res_val = super::template read_mem<uint64_t>(traits<ARCH>::MEM, offs_val);
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + 8 + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | res_val);
            super::template get_reg<reg_t>(rd + 8 + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 200);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 201: C.FSD */
    compile_ret_t __c_fsd(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 201);
        
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,2>(instr) << 6) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f(8+{rs2}), {uimm}({rs1})", fmt::arg("mnemonic", "c.fsd"),
            	fmt::arg("rs2", rs2), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto offs_val = (super::template get_reg<reg_t>(rs1 + 8 + traits<ARCH>::X0) + (uimm));
        auto MEMtmp0_val = static_cast<uint64_t>(
            super::template get_reg<reg_t>(rs2 + 8 + traits<ARCH>::F0)
        );
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint64_t>(MEMtmp0_val));
        this->do_sync(POST_SYNC, 201);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 202: C.FLDSP */
    compile_ret_t __c_fldsp(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 202);
        
        uint16_t uimm = ((bit_sub<2,3>(instr) << 6) | (bit_sub<5,2>(instr) << 3) | (bit_sub<12,1>(instr) << 5));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rd}, {uimm}(x2)", fmt::arg("mnemonic", "c.fldsp"),
            	fmt::arg("rd", rd), fmt::arg("uimm", uimm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto offs_val = (super::template get_reg<reg_t>(2 + traits<ARCH>::X0) + (uimm));
        auto res_val = super::template read_mem<uint64_t>(traits<ARCH>::MEM, offs_val);
        if(64 == 64){
            auto Ftmp0_val = res_val;
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp0_val;
        } else {
            uint64_t upper_val = - 1;
            auto Ftmp1_val = ((upper_val << 64) | super::template zext<uint64_t>(res_val));
            super::template get_reg<reg_t>(rd + traits<ARCH>::F0)=Ftmp1_val;
        }
        this->do_sync(POST_SYNC, 202);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
        }
        pc.val=super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC);
        return pc;
    }
    
    /* instruction 203: C.FSDSP */
    compile_ret_t __c_fsdsp(virt_addr_t& pc, code_word_t instr){
        this->do_sync(PRE_SYNC, 203);
        
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint16_t uimm = ((bit_sub<7,3>(instr) << 6) | (bit_sub<10,3>(instr) << 3));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} f{rs2}, {uimm}(x2), ", fmt::arg("mnemonic", "c.fsdsp"),
            	fmt::arg("rs2", rs2), fmt::arg("uimm", uimm));
            this->core.disass_output(pc.val, mnemonic);
        }
        
        auto cur_pc_val = pc.val;
        super::template get_reg<reg_t>(arch::traits<ARCH>::NEXT_PC) = cur_pc_val + 2;
        auto offs_val = (super::template get_reg<reg_t>(2 + traits<ARCH>::X0) + (uimm));
        auto MEMtmp0_val = static_cast<uint64_t>(
            super::template get_reg<reg_t>(rs2 + traits<ARCH>::F0)
        );
        super::write_mem(traits<ARCH>::MEM, offs_val, static_cast<uint64_t>(MEMtmp0_val));
        this->do_sync(POST_SYNC, 203);
        auto& trap_state = super::template get_reg<uint32_t>(arch::traits<ARCH>::TRAP_STATE);
        // trap check
        if(trap_state!=0){
            auto& last_br = super::template get_reg<uint32_t>(arch::traits<ARCH>::LAST_BRANCH);
            last_br = std::numeric_limits<uint32_t>::max();
            super::core.enter_trap(trap_state, cur_pc_val, 0);
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

inline bool is_count_limit_enabled(finish_cond_e cond){
    return (cond & finish_cond_e::COUNT_LIMIT) == finish_cond_e::COUNT_LIMIT;
}

inline bool is_jump_to_self_enabled(finish_cond_e cond){
    return (cond & finish_cond_e::JUMP_TO_SELF) == finish_cond_e::JUMP_TO_SELF;
}

template <typename ARCH>
typename vm_base<ARCH>::virt_addr_t vm_impl<ARCH>::execute_inst(finish_cond_e cond, virt_addr_t start, uint64_t icount_limit) {
    // we fetch at max 4 byte, alignment is 2
    enum {TRAP_ID=1<<16};
    const typename traits<ARCH>::addr_t upper_bits = ~traits<ARCH>::PGMASK;
    code_word_t insn = 0;
    auto *const data = (uint8_t *)&insn;
    auto pc=start;
    while(!this->core.should_stop() && !(is_count_limit_enabled(cond) && this->core.get_icount() >= icount_limit)){
        auto paddr = this->core.v2p(pc);
        if ((pc.val & upper_bits) != ((pc.val + 2) & upper_bits)) { // we may cross a page boundary
            if (this->core.read(paddr, 2, data) != iss::Ok) throw trap_access(TRAP_ID, pc.val);
            if ((insn & 0x3) == 0x3) // this is a 32bit instruction
                if (this->core.read(this->core.v2p(pc + 2), 2, data + 2) != iss::Ok) throw trap_access(TRAP_ID, pc.val);
        } else {
            if (this->core.read(paddr, 4, data) != iss::Ok) throw trap_access(TRAP_ID, pc.val);
        }
        if (is_jump_to_self_enabled(cond) &&(insn == 0x0000006f || (insn&0xffff)==0xa001))
            throw simulation_stopped(0); // 'J 0' or 'C.J 0'
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
std::unique_ptr<vm_if> create<arch::rv64gc>(arch::rv64gc *core, unsigned short port, bool dump) {
    auto ret = new rv64gc::vm_impl<arch::rv64gc>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret, port);
    return std::unique_ptr<vm_if>(ret);
}
} // namespace interp
} // namespace iss
