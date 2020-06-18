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

#include <iss/arch/rv32imac.h>
#include <iss/arch/riscv_hart_msu_vp.h>
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/server.h>
#include <iss/iss.h>
#include <iss/tcc/vm_base.h>
#include <util/logging.h>
#include <sstream>

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#include <array>
#include <iss/debugger/riscv_target_adapter.h>

namespace iss {
namespace tcc {
namespace rv32imac {
using namespace iss::arch;
using namespace iss::debugger;

template <typename ARCH> class vm_impl : public iss::tcc::vm_base<ARCH> {
public:
    using super       = typename iss::tcc::vm_base<ARCH>;
    using virt_addr_t = typename super::virt_addr_t;
    using phys_addr_t = typename super::phys_addr_t;
    using code_word_t = typename super::code_word_t;
    using addr_t      = typename super::addr_t;
    using tu_builder  = typename super::tu_builder;

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
    using compile_func = compile_ret_t (this_class::*)(virt_addr_t &pc, code_word_t instr, tu_builder&);

    inline const char *name(size_t index){return traits<ARCH>::reg_aliases.at(index);}

    void setup_module(std::string m) override {
        super::setup_module(m);
    }

    compile_ret_t gen_single_inst_behavior(virt_addr_t &, unsigned int &, tu_builder&) override;

    void gen_trap_behavior(tu_builder& tu) override;

    void gen_raise_trap(tu_builder& tu, uint16_t trap_id, uint16_t cause);

    void gen_leave_trap(tu_builder& tu, unsigned lvl);

    void gen_wait(tu_builder& tu, unsigned type);

    inline void gen_trap_check(tu_builder& tu) {
        tu("if(*trap_state!=0) goto trap_entry;");
    }

    inline void gen_set_pc(tu_builder& tu, virt_addr_t pc, unsigned reg_num) {
        switch(reg_num){
        case traits<ARCH>::NEXT_PC:
            tu("*next_pc = {:#x};", pc.val);
            break;
        case traits<ARCH>::PC:
            tu("*pc = {:#x};", pc.val);
            break;
        default:
            if(!tu.defined_regs[reg_num]){
                tu("reg_t* reg{:02d} = (reg_t*){:#x};", reg_num, reinterpret_cast<uintptr_t>(get_reg_ptr(reg_num)));
            tu.defined_regs[reg_num]=true;
            }
            tu("*reg{:02d} = {:#x};", reg_num, pc.val);
        }
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

    const std::array<InstructionDesriptor, 99> instr_descr = {{
         /* entries are: size, valid value, valid mask, function ptr */
        /* instruction LUI, encoding '.........................0110111' */
        {32, 0b00000000000000000000000000110111, 0b00000000000000000000000001111111, &this_class::__lui},
        /* instruction AUIPC, encoding '.........................0010111' */
        {32, 0b00000000000000000000000000010111, 0b00000000000000000000000001111111, &this_class::__auipc},
        /* instruction JAL, encoding '.........................1101111' */
        {32, 0b00000000000000000000000001101111, 0b00000000000000000000000001111111, &this_class::__jal},
        /* instruction BEQ, encoding '.................000.....1100011' */
        {32, 0b00000000000000000000000001100011, 0b00000000000000000111000001111111, &this_class::__beq},
        /* instruction BNE, encoding '.................001.....1100011' */
        {32, 0b00000000000000000001000001100011, 0b00000000000000000111000001111111, &this_class::__bne},
        /* instruction BLT, encoding '.................100.....1100011' */
        {32, 0b00000000000000000100000001100011, 0b00000000000000000111000001111111, &this_class::__blt},
        /* instruction BGE, encoding '.................101.....1100011' */
        {32, 0b00000000000000000101000001100011, 0b00000000000000000111000001111111, &this_class::__bge},
        /* instruction BLTU, encoding '.................110.....1100011' */
        {32, 0b00000000000000000110000001100011, 0b00000000000000000111000001111111, &this_class::__bltu},
        /* instruction BGEU, encoding '.................111.....1100011' */
        {32, 0b00000000000000000111000001100011, 0b00000000000000000111000001111111, &this_class::__bgeu},
        /* instruction LB, encoding '.................000.....0000011' */
        {32, 0b00000000000000000000000000000011, 0b00000000000000000111000001111111, &this_class::__lb},
        /* instruction LH, encoding '.................001.....0000011' */
        {32, 0b00000000000000000001000000000011, 0b00000000000000000111000001111111, &this_class::__lh},
        /* instruction LW, encoding '.................010.....0000011' */
        {32, 0b00000000000000000010000000000011, 0b00000000000000000111000001111111, &this_class::__lw},
        /* instruction LBU, encoding '.................100.....0000011' */
        {32, 0b00000000000000000100000000000011, 0b00000000000000000111000001111111, &this_class::__lbu},
        /* instruction LHU, encoding '.................101.....0000011' */
        {32, 0b00000000000000000101000000000011, 0b00000000000000000111000001111111, &this_class::__lhu},
        /* instruction SB, encoding '.................000.....0100011' */
        {32, 0b00000000000000000000000000100011, 0b00000000000000000111000001111111, &this_class::__sb},
        /* instruction SH, encoding '.................001.....0100011' */
        {32, 0b00000000000000000001000000100011, 0b00000000000000000111000001111111, &this_class::__sh},
        /* instruction SW, encoding '.................010.....0100011' */
        {32, 0b00000000000000000010000000100011, 0b00000000000000000111000001111111, &this_class::__sw},
        /* instruction ADDI, encoding '.................000.....0010011' */
        {32, 0b00000000000000000000000000010011, 0b00000000000000000111000001111111, &this_class::__addi},
        /* instruction SLTI, encoding '.................010.....0010011' */
        {32, 0b00000000000000000010000000010011, 0b00000000000000000111000001111111, &this_class::__slti},
        /* instruction SLTIU, encoding '.................011.....0010011' */
        {32, 0b00000000000000000011000000010011, 0b00000000000000000111000001111111, &this_class::__sltiu},
        /* instruction XORI, encoding '.................100.....0010011' */
        {32, 0b00000000000000000100000000010011, 0b00000000000000000111000001111111, &this_class::__xori},
        /* instruction ORI, encoding '.................110.....0010011' */
        {32, 0b00000000000000000110000000010011, 0b00000000000000000111000001111111, &this_class::__ori},
        /* instruction ANDI, encoding '.................111.....0010011' */
        {32, 0b00000000000000000111000000010011, 0b00000000000000000111000001111111, &this_class::__andi},
        /* instruction SLLI, encoding '0000000..........001.....0010011' */
        {32, 0b00000000000000000001000000010011, 0b11111110000000000111000001111111, &this_class::__slli},
        /* instruction SRLI, encoding '0000000..........101.....0010011' */
        {32, 0b00000000000000000101000000010011, 0b11111110000000000111000001111111, &this_class::__srli},
        /* instruction SRAI, encoding '0100000..........101.....0010011' */
        {32, 0b01000000000000000101000000010011, 0b11111110000000000111000001111111, &this_class::__srai},
        /* instruction ADD, encoding '0000000..........000.....0110011' */
        {32, 0b00000000000000000000000000110011, 0b11111110000000000111000001111111, &this_class::__add},
        /* instruction SUB, encoding '0100000..........000.....0110011' */
        {32, 0b01000000000000000000000000110011, 0b11111110000000000111000001111111, &this_class::__sub},
        /* instruction SLL, encoding '0000000..........001.....0110011' */
        {32, 0b00000000000000000001000000110011, 0b11111110000000000111000001111111, &this_class::__sll},
        /* instruction SLT, encoding '0000000..........010.....0110011' */
        {32, 0b00000000000000000010000000110011, 0b11111110000000000111000001111111, &this_class::__slt},
        /* instruction SLTU, encoding '0000000..........011.....0110011' */
        {32, 0b00000000000000000011000000110011, 0b11111110000000000111000001111111, &this_class::__sltu},
        /* instruction XOR, encoding '0000000..........100.....0110011' */
        {32, 0b00000000000000000100000000110011, 0b11111110000000000111000001111111, &this_class::__xor},
        /* instruction SRL, encoding '0000000..........101.....0110011' */
        {32, 0b00000000000000000101000000110011, 0b11111110000000000111000001111111, &this_class::__srl},
        /* instruction SRA, encoding '0100000..........101.....0110011' */
        {32, 0b01000000000000000101000000110011, 0b11111110000000000111000001111111, &this_class::__sra},
        /* instruction OR, encoding '0000000..........110.....0110011' */
        {32, 0b00000000000000000110000000110011, 0b11111110000000000111000001111111, &this_class::__or},
        /* instruction AND, encoding '0000000..........111.....0110011' */
        {32, 0b00000000000000000111000000110011, 0b11111110000000000111000001111111, &this_class::__and},
        /* instruction FENCE, encoding '0000.............000.....0001111' */
        {32, 0b00000000000000000000000000001111, 0b11110000000000000111000001111111, &this_class::__fence},
        /* instruction FENCE_I, encoding '.................001.....0001111' */
        {32, 0b00000000000000000001000000001111, 0b00000000000000000111000001111111, &this_class::__fence_i},
        /* instruction ECALL, encoding '00000000000000000000000001110011' */
        {32, 0b00000000000000000000000001110011, 0b11111111111111111111111111111111, &this_class::__ecall},
        /* instruction EBREAK, encoding '00000000000100000000000001110011' */
        {32, 0b00000000000100000000000001110011, 0b11111111111111111111111111111111, &this_class::__ebreak},
        /* instruction URET, encoding '00000000001000000000000001110011' */
        {32, 0b00000000001000000000000001110011, 0b11111111111111111111111111111111, &this_class::__uret},
        /* instruction SRET, encoding '00010000001000000000000001110011' */
        {32, 0b00010000001000000000000001110011, 0b11111111111111111111111111111111, &this_class::__sret},
        /* instruction MRET, encoding '00110000001000000000000001110011' */
        {32, 0b00110000001000000000000001110011, 0b11111111111111111111111111111111, &this_class::__mret},
        /* instruction WFI, encoding '00010000010100000000000001110011' */
        {32, 0b00010000010100000000000001110011, 0b11111111111111111111111111111111, &this_class::__wfi},
        /* instruction SFENCE.VMA, encoding '0001001..........000000001110011' */
        {32, 0b00010010000000000000000001110011, 0b11111110000000000111111111111111, &this_class::__sfence_vma},
        /* instruction CSRRW, encoding '.................001.....1110011' */
        {32, 0b00000000000000000001000001110011, 0b00000000000000000111000001111111, &this_class::__csrrw},
        /* instruction CSRRS, encoding '.................010.....1110011' */
        {32, 0b00000000000000000010000001110011, 0b00000000000000000111000001111111, &this_class::__csrrs},
        /* instruction CSRRC, encoding '.................011.....1110011' */
        {32, 0b00000000000000000011000001110011, 0b00000000000000000111000001111111, &this_class::__csrrc},
        /* instruction CSRRWI, encoding '.................101.....1110011' */
        {32, 0b00000000000000000101000001110011, 0b00000000000000000111000001111111, &this_class::__csrrwi},
        /* instruction CSRRSI, encoding '.................110.....1110011' */
        {32, 0b00000000000000000110000001110011, 0b00000000000000000111000001111111, &this_class::__csrrsi},
        /* instruction CSRRCI, encoding '.................111.....1110011' */
        {32, 0b00000000000000000111000001110011, 0b00000000000000000111000001111111, &this_class::__csrrci},
        /* instruction MUL, encoding '0000001..........000.....0110011' */
        {32, 0b00000010000000000000000000110011, 0b11111110000000000111000001111111, &this_class::__mul},
        /* instruction MULH, encoding '0000001..........001.....0110011' */
        {32, 0b00000010000000000001000000110011, 0b11111110000000000111000001111111, &this_class::__mulh},
        /* instruction MULHSU, encoding '0000001..........010.....0110011' */
        {32, 0b00000010000000000010000000110011, 0b11111110000000000111000001111111, &this_class::__mulhsu},
        /* instruction MULHU, encoding '0000001..........011.....0110011' */
        {32, 0b00000010000000000011000000110011, 0b11111110000000000111000001111111, &this_class::__mulhu},
        /* instruction DIV, encoding '0000001..........100.....0110011' */
        {32, 0b00000010000000000100000000110011, 0b11111110000000000111000001111111, &this_class::__div},
        /* instruction DIVU, encoding '0000001..........101.....0110011' */
        {32, 0b00000010000000000101000000110011, 0b11111110000000000111000001111111, &this_class::__divu},
        /* instruction REM, encoding '0000001..........110.....0110011' */
        {32, 0b00000010000000000110000000110011, 0b11111110000000000111000001111111, &this_class::__rem},
        /* instruction REMU, encoding '0000001..........111.....0110011' */
        {32, 0b00000010000000000111000000110011, 0b11111110000000000111000001111111, &this_class::__remu},
        /* instruction LR.W, encoding '00010..00000.....010.....0101111' */
        {32, 0b00010000000000000010000000101111, 0b11111001111100000111000001111111, &this_class::__lr_w},
        /* instruction SC.W, encoding '00011............010.....0101111' */
        {32, 0b00011000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__sc_w},
        /* instruction AMOSWAP.W, encoding '00001............010.....0101111' */
        {32, 0b00001000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoswap_w},
        /* instruction AMOADD.W, encoding '00000............010.....0101111' */
        {32, 0b00000000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoadd_w},
        /* instruction AMOXOR.W, encoding '00100............010.....0101111' */
        {32, 0b00100000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoxor_w},
        /* instruction AMOAND.W, encoding '01100............010.....0101111' */
        {32, 0b01100000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoand_w},
        /* instruction AMOOR.W, encoding '01000............010.....0101111' */
        {32, 0b01000000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoor_w},
        /* instruction AMOMIN.W, encoding '10000............010.....0101111' */
        {32, 0b10000000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amomin_w},
        /* instruction AMOMAX.W, encoding '10100............010.....0101111' */
        {32, 0b10100000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amomax_w},
        /* instruction AMOMINU.W, encoding '11000............010.....0101111' */
        {32, 0b11000000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amominu_w},
        /* instruction AMOMAXU.W, encoding '11100............010.....0101111' */
        {32, 0b11100000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amomaxu_w},
        /* instruction JALR, encoding '.................000.....1100111' */
        {32, 0b00000000000000000000000001100111, 0b00000000000000000111000001111111, &this_class::__jalr},
        /* instruction C.ADDI4SPN, encoding '000...........00' */
        {16, 0b0000000000000000, 0b1110000000000011, &this_class::__c_addi4spn},
        /* instruction C.LW, encoding '010...........00' */
        {16, 0b0100000000000000, 0b1110000000000011, &this_class::__c_lw},
        /* instruction C.SW, encoding '110...........00' */
        {16, 0b1100000000000000, 0b1110000000000011, &this_class::__c_sw},
        /* instruction C.ADDI, encoding '000...........01' */
        {16, 0b0000000000000001, 0b1110000000000011, &this_class::__c_addi},
        /* instruction C.NOP, encoding '0000000000000001' */
        {16, 0b0000000000000001, 0b1111111111111111, &this_class::__c_nop},
        /* instruction C.JAL, encoding '001...........01' */
        {16, 0b0010000000000001, 0b1110000000000011, &this_class::__c_jal},
        /* instruction C.LI, encoding '010...........01' */
        {16, 0b0100000000000001, 0b1110000000000011, &this_class::__c_li},
        /* instruction C.LUI, encoding '011...........01' */
        {16, 0b0110000000000001, 0b1110000000000011, &this_class::__c_lui},
        /* instruction C.ADDI16SP, encoding '011.00010.....01' */
        {16, 0b0110000100000001, 0b1110111110000011, &this_class::__c_addi16sp},
        /* instruction C.SRLI, encoding '100000........01' */
        {16, 0b1000000000000001, 0b1111110000000011, &this_class::__c_srli},
        /* instruction C.SRAI, encoding '100001........01' */
        {16, 0b1000010000000001, 0b1111110000000011, &this_class::__c_srai},
        /* instruction C.ANDI, encoding '100.10........01' */
        {16, 0b1000100000000001, 0b1110110000000011, &this_class::__c_andi},
        /* instruction C.SUB, encoding '100011...00...01' */
        {16, 0b1000110000000001, 0b1111110001100011, &this_class::__c_sub},
        /* instruction C.XOR, encoding '100011...01...01' */
        {16, 0b1000110000100001, 0b1111110001100011, &this_class::__c_xor},
        /* instruction C.OR, encoding '100011...10...01' */
        {16, 0b1000110001000001, 0b1111110001100011, &this_class::__c_or},
        /* instruction C.AND, encoding '100011...11...01' */
        {16, 0b1000110001100001, 0b1111110001100011, &this_class::__c_and},
        /* instruction C.J, encoding '101...........01' */
        {16, 0b1010000000000001, 0b1110000000000011, &this_class::__c_j},
        /* instruction C.BEQZ, encoding '110...........01' */
        {16, 0b1100000000000001, 0b1110000000000011, &this_class::__c_beqz},
        /* instruction C.BNEZ, encoding '111...........01' */
        {16, 0b1110000000000001, 0b1110000000000011, &this_class::__c_bnez},
        /* instruction C.SLLI, encoding '0000..........10' */
        {16, 0b0000000000000010, 0b1111000000000011, &this_class::__c_slli},
        /* instruction C.LWSP, encoding '010...........10' */
        {16, 0b0100000000000010, 0b1110000000000011, &this_class::__c_lwsp},
        /* instruction C.MV, encoding '1000..........10' */
        {16, 0b1000000000000010, 0b1111000000000011, &this_class::__c_mv},
        /* instruction C.JR, encoding '1000.....0000010' */
        {16, 0b1000000000000010, 0b1111000001111111, &this_class::__c_jr},
        /* instruction C.ADD, encoding '1001..........10' */
        {16, 0b1001000000000010, 0b1111000000000011, &this_class::__c_add},
        /* instruction C.JALR, encoding '1001.....0000010' */
        {16, 0b1001000000000010, 0b1111000001111111, &this_class::__c_jalr},
        /* instruction C.EBREAK, encoding '1001000000000010' */
        {16, 0b1001000000000010, 0b1111111111111111, &this_class::__c_ebreak},
        /* instruction C.SWSP, encoding '110...........10' */
        {16, 0b1100000000000010, 0b1110000000000011, &this_class::__c_swsp},
        /* instruction DII, encoding '0000000000000000' */
        {16, 0b0000000000000000, 0b1111111111111111, &this_class::__dii},
    }};
 
    /* instruction definitions */
    /* instruction 0: LUI */
    compile_ret_t __lui(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LUI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 0);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        int32_t imm = signextend<int32_t,32>((bit_sub<12,20>(instr) << 12));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "lui"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.constant(imm, 32U), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 0);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 1: AUIPC */
    compile_ret_t __auipc(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("AUIPC_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 1);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        int32_t imm = signextend<int32_t,32>((bit_sub<12,20>(instr) << 12));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#08x}", fmt::arg("mnemonic", "auipc"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.add(
                tu.ext(
                    cur_pc_val,
                    32, false),
                tu.constant(imm, 32U)), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 1);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 2: JAL */
    compile_ret_t __jal(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("JAL_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 2);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        int32_t imm = signextend<int32_t,21>((bit_sub<12,8>(instr) << 12) | (bit_sub<20,1>(instr) << 11) | (bit_sub<21,10>(instr) << 1) | (bit_sub<31,1>(instr) << 20));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#0x}", fmt::arg("mnemonic", "jal"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.add(
                cur_pc_val,
                tu.constant(4, 32U)), rd + traits<ARCH>::X0);
        }
        auto PC_val_v = tu.assignment("PC_val", tu.add(
            tu.ext(
                cur_pc_val,
                32, false),
            tu.constant(imm, 32U)), 32);
        tu.store(PC_val_v, traits<ARCH>::NEXT_PC);
        auto is_cont_v = tu.choose(
            tu.icmp(ICmpInst::ICMP_NE, tu.ext(PC_val_v, 32U, true), tu.constant(pc.val, 32U)),
            tu.constant(0U, 32), tu.constant(1U, 32));
        tu.store(is_cont_v, traits<ARCH>::LAST_BRANCH);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 2);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 3: BEQ */
    compile_ret_t __beq(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BEQ_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 3);
        int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "beq"),
            	fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto PC_val_v = tu.assignment("PC_val", tu.choose(
            tu.icmp(
                ICmpInst::ICMP_EQ,
                tu.load(rs1 + traits<ARCH>::X0, 0),
                tu.load(rs2 + traits<ARCH>::X0, 0)),
            tu.add(
                tu.ext(
                    cur_pc_val,
                    32, false),
                tu.constant(imm, 32U)),
            tu.add(
                cur_pc_val,
                tu.constant(4, 32U))), 32);
        tu.store(PC_val_v, traits<ARCH>::NEXT_PC);
        auto is_cont_v = tu.choose(
            tu.icmp(ICmpInst::ICMP_NE, tu.ext(PC_val_v, 32U, true), tu.constant(pc.val, 32U)),
            tu.constant(0U, 32), tu.constant(1U, 32));
        tu.store(is_cont_v, traits<ARCH>::LAST_BRANCH);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 3);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 4: BNE */
    compile_ret_t __bne(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BNE_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 4);
        int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bne"),
            	fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto PC_val_v = tu.assignment("PC_val", tu.choose(
            tu.icmp(
                ICmpInst::ICMP_NE,
                tu.load(rs1 + traits<ARCH>::X0, 0),
                tu.load(rs2 + traits<ARCH>::X0, 0)),
            tu.add(
                tu.ext(
                    cur_pc_val,
                    32, false),
                tu.constant(imm, 32U)),
            tu.add(
                cur_pc_val,
                tu.constant(4, 32U))), 32);
        tu.store(PC_val_v, traits<ARCH>::NEXT_PC);
        auto is_cont_v = tu.choose(
            tu.icmp(ICmpInst::ICMP_NE, tu.ext(PC_val_v, 32U, true), tu.constant(pc.val, 32U)),
            tu.constant(0U, 32), tu.constant(1U, 32));
        tu.store(is_cont_v, traits<ARCH>::LAST_BRANCH);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 4);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 5: BLT */
    compile_ret_t __blt(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BLT_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 5);
        int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "blt"),
            	fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto PC_val_v = tu.assignment("PC_val", tu.choose(
            tu.icmp(
                ICmpInst::ICMP_SLT,
                tu.ext(
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    32, false),
                tu.ext(
                    tu.load(rs2 + traits<ARCH>::X0, 0),
                    32, false)),
            tu.add(
                tu.ext(
                    cur_pc_val,
                    32, false),
                tu.constant(imm, 32U)),
            tu.add(
                cur_pc_val,
                tu.constant(4, 32U))), 32);
        tu.store(PC_val_v, traits<ARCH>::NEXT_PC);
        auto is_cont_v = tu.choose(
            tu.icmp(ICmpInst::ICMP_NE, tu.ext(PC_val_v, 32U, true), tu.constant(pc.val, 32U)),
            tu.constant(0U, 32), tu.constant(1U, 32));
        tu.store(is_cont_v, traits<ARCH>::LAST_BRANCH);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 5);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 6: BGE */
    compile_ret_t __bge(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BGE_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 6);
        int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bge"),
            	fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto PC_val_v = tu.assignment("PC_val", tu.choose(
            tu.icmp(
                ICmpInst::ICMP_SGE,
                tu.ext(
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    32, false),
                tu.ext(
                    tu.load(rs2 + traits<ARCH>::X0, 0),
                    32, false)),
            tu.add(
                tu.ext(
                    cur_pc_val,
                    32, false),
                tu.constant(imm, 32U)),
            tu.add(
                cur_pc_val,
                tu.constant(4, 32U))), 32);
        tu.store(PC_val_v, traits<ARCH>::NEXT_PC);
        auto is_cont_v = tu.choose(
            tu.icmp(ICmpInst::ICMP_NE, tu.ext(PC_val_v, 32U, true), tu.constant(pc.val, 32U)),
            tu.constant(0U, 32), tu.constant(1U, 32));
        tu.store(is_cont_v, traits<ARCH>::LAST_BRANCH);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 6);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 7: BLTU */
    compile_ret_t __bltu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BLTU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 7);
        int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bltu"),
            	fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto PC_val_v = tu.assignment("PC_val", tu.choose(
            tu.icmp(
                ICmpInst::ICMP_ULT,
                tu.load(rs1 + traits<ARCH>::X0, 0),
                tu.load(rs2 + traits<ARCH>::X0, 0)),
            tu.add(
                tu.ext(
                    cur_pc_val,
                    32, false),
                tu.constant(imm, 32U)),
            tu.add(
                cur_pc_val,
                tu.constant(4, 32U))), 32);
        tu.store(PC_val_v, traits<ARCH>::NEXT_PC);
        auto is_cont_v = tu.choose(
            tu.icmp(ICmpInst::ICMP_NE, tu.ext(PC_val_v, 32U, true), tu.constant(pc.val, 32U)),
            tu.constant(0U, 32), tu.constant(1U, 32));
        tu.store(is_cont_v, traits<ARCH>::LAST_BRANCH);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 7);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 8: BGEU */
    compile_ret_t __bgeu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BGEU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 8);
        int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bgeu"),
            	fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto PC_val_v = tu.assignment("PC_val", tu.choose(
            tu.icmp(
                ICmpInst::ICMP_UGE,
                tu.load(rs1 + traits<ARCH>::X0, 0),
                tu.load(rs2 + traits<ARCH>::X0, 0)),
            tu.add(
                tu.ext(
                    cur_pc_val,
                    32, false),
                tu.constant(imm, 32U)),
            tu.add(
                cur_pc_val,
                tu.constant(4, 32U))), 32);
        tu.store(PC_val_v, traits<ARCH>::NEXT_PC);
        auto is_cont_v = tu.choose(
            tu.icmp(ICmpInst::ICMP_NE, tu.ext(PC_val_v, 32U, true), tu.constant(pc.val, 32U)),
            tu.constant(0U, 32), tu.constant(1U, 32));
        tu.store(is_cont_v, traits<ARCH>::LAST_BRANCH);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 8);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 9: LB */
    compile_ret_t __lb(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LB_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 9);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lb"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.add(
            tu.ext(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                32, false),
            tu.constant(imm, 32U)), 32);
        if(rd != 0){
            tu.store(tu.ext(
                tu.read_mem(traits<ARCH>::MEM, offs_val, 8),
                32,
                false), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 9);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 10: LH */
    compile_ret_t __lh(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LH_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 10);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lh"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.add(
            tu.ext(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                32, false),
            tu.constant(imm, 32U)), 32);
        if(rd != 0){
            tu.store(tu.ext(
                tu.read_mem(traits<ARCH>::MEM, offs_val, 16),
                32,
                false), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 10);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 11: LW */
    compile_ret_t __lw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 11);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lw"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.add(
            tu.ext(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                32, false),
            tu.constant(imm, 32U)), 32);
        if(rd != 0){
            tu.store(tu.ext(
                tu.read_mem(traits<ARCH>::MEM, offs_val, 32),
                32,
                false), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 11);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 12: LBU */
    compile_ret_t __lbu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LBU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 12);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lbu"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.add(
            tu.ext(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                32, false),
            tu.constant(imm, 32U)), 32);
        if(rd != 0){
            tu.store(tu.ext(
                tu.read_mem(traits<ARCH>::MEM, offs_val, 8),
                32,
                true), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 12);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 13: LHU */
    compile_ret_t __lhu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LHU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 13);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lhu"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.add(
            tu.ext(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                32, false),
            tu.constant(imm, 32U)), 32);
        if(rd != 0){
            tu.store(tu.ext(
                tu.read_mem(traits<ARCH>::MEM, offs_val, 16),
                32,
                true), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 13);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 14: SB */
    compile_ret_t __sb(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SB_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 14);
        int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sb"),
            	fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.add(
            tu.ext(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                32, false),
            tu.constant(imm, 32U)), 32);
        tu.write_mem(
            traits<ARCH>::MEM,
            offs_val,
            tu.trunc(tu.load(rs2 + traits<ARCH>::X0, 0), 8));
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 14);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 15: SH */
    compile_ret_t __sh(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SH_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 15);
        int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sh"),
            	fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.add(
            tu.ext(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                32, false),
            tu.constant(imm, 32U)), 32);
        tu.write_mem(
            traits<ARCH>::MEM,
            offs_val,
            tu.trunc(tu.load(rs2 + traits<ARCH>::X0, 0), 16));
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 15);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 16: SW */
    compile_ret_t __sw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 16);
        int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sw"),
            	fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.add(
            tu.ext(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                32, false),
            tu.constant(imm, 32U)), 32);
        tu.write_mem(
            traits<ARCH>::MEM,
            offs_val,
            tu.trunc(tu.load(rs2 + traits<ARCH>::X0, 0), 32));
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 16);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 17: ADDI */
    compile_ret_t __addi(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("ADDI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 17);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "addi"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.add(
                tu.ext(
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    32, false),
                tu.constant(imm, 32U)), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 17);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 18: SLTI */
    compile_ret_t __slti(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLTI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 18);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "slti"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.choose(
                tu.icmp(
                    ICmpInst::ICMP_SLT,
                    tu.ext(
                        tu.load(rs1 + traits<ARCH>::X0, 0),
                        32, false),
                    tu.constant(imm, 32U)),
                tu.constant(1, 32U),
                tu.constant(0, 32U)), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 18);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 19: SLTIU */
    compile_ret_t __sltiu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLTIU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 19);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "sltiu"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        int32_t full_imm_val = imm;
        if(rd != 0){
            tu.store(tu.choose(
                tu.icmp(
                    ICmpInst::ICMP_ULT,
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    tu.constant(full_imm_val, 32U)),
                tu.constant(1, 32U),
                tu.constant(0, 32U)), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 19);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 20: XORI */
    compile_ret_t __xori(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("XORI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 20);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "xori"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.l_xor(
                tu.ext(
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    32, false),
                tu.constant(imm, 32U)), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 20);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 21: ORI */
    compile_ret_t __ori(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("ORI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 21);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "ori"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.l_or(
                tu.ext(
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    32, false),
                tu.constant(imm, 32U)), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 21);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 22: ANDI */
    compile_ret_t __andi(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("ANDI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 22);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "andi"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.l_and(
                tu.ext(
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    32, false),
                tu.constant(imm, 32U)), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 22);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 23: SLLI */
    compile_ret_t __slli(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLLI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 23);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "slli"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(shamt > 31){
            this->gen_raise_trap(tu, 0, 0);
        } else {
            if(rd != 0){
                tu.store(tu.shl(
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    tu.constant(shamt, 32U)), rd + traits<ARCH>::X0);
            }
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 23);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 24: SRLI */
    compile_ret_t __srli(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRLI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 24);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srli"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(shamt > 31){
            this->gen_raise_trap(tu, 0, 0);
        } else {
            if(rd != 0){
                tu.store(tu.lshr(
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    tu.constant(shamt, 32U)), rd + traits<ARCH>::X0);
            }
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 24);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 25: SRAI */
    compile_ret_t __srai(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRAI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 25);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srai"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(shamt > 31){
            this->gen_raise_trap(tu, 0, 0);
        } else {
            if(rd != 0){
                tu.store(tu.ashr(
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    tu.constant(shamt, 32U)), rd + traits<ARCH>::X0);
            }
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 25);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 26: ADD */
    compile_ret_t __add(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("ADD_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 26);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "add"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.add(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                tu.load(rs2 + traits<ARCH>::X0, 0)), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 26);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 27: SUB */
    compile_ret_t __sub(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SUB_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 27);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sub"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.sub(
                 tu.load(rs1 + traits<ARCH>::X0, 0),
                 tu.load(rs2 + traits<ARCH>::X0, 0)), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 27);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 28: SLL */
    compile_ret_t __sll(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLL_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 28);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sll"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.shl(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                tu.l_and(
                    tu.load(rs2 + traits<ARCH>::X0, 0),
                    tu.sub(
                         tu.constant(32, 32U),
                         tu.constant(1, 32U)))), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 28);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 29: SLT */
    compile_ret_t __slt(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLT_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 29);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "slt"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.choose(
                tu.icmp(
                    ICmpInst::ICMP_SLT,
                    tu.ext(
                        tu.load(rs1 + traits<ARCH>::X0, 0),
                        32, false),
                    tu.ext(
                        tu.load(rs2 + traits<ARCH>::X0, 0),
                        32, false)),
                tu.constant(1, 32U),
                tu.constant(0, 32U)), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 29);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 30: SLTU */
    compile_ret_t __sltu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLTU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 30);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sltu"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.choose(
                tu.icmp(
                    ICmpInst::ICMP_ULT,
                    tu.ext(
                        tu.load(rs1 + traits<ARCH>::X0, 0),
                        32,
                        true),
                    tu.ext(
                        tu.load(rs2 + traits<ARCH>::X0, 0),
                        32,
                        true)),
                tu.constant(1, 32U),
                tu.constant(0, 32U)), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 30);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 31: XOR */
    compile_ret_t __xor(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("XOR_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 31);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "xor"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.l_xor(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                tu.load(rs2 + traits<ARCH>::X0, 0)), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 31);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 32: SRL */
    compile_ret_t __srl(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRL_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 32);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "srl"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.lshr(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                tu.l_and(
                    tu.load(rs2 + traits<ARCH>::X0, 0),
                    tu.sub(
                         tu.constant(32, 32U),
                         tu.constant(1, 32U)))), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 32);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 33: SRA */
    compile_ret_t __sra(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRA_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 33);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sra"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.ashr(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                tu.l_and(
                    tu.load(rs2 + traits<ARCH>::X0, 0),
                    tu.sub(
                         tu.constant(32, 32U),
                         tu.constant(1, 32U)))), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 33);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 34: OR */
    compile_ret_t __or(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("OR_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 34);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "or"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.l_or(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                tu.load(rs2 + traits<ARCH>::X0, 0)), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 34);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 35: AND */
    compile_ret_t __and(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("AND_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 35);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "and"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.l_and(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                tu.load(rs2 + traits<ARCH>::X0, 0)), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 35);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 36: FENCE */
    compile_ret_t __fence(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("FENCE_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 36);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t succ = ((bit_sub<20,4>(instr)));
        uint8_t pred = ((bit_sub<24,4>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, "fence");
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        tu.write_mem(
            traits<ARCH>::FENCE,
            tu.constant(0, 64U),
            tu.trunc(tu.l_or(
                tu.shl(
                    tu.constant(pred, 32U),
                    tu.constant(4, 32U)),
                tu.constant(succ, 32U)), 32));
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 36);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 37: FENCE_I */
    compile_ret_t __fence_i(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("FENCE_I_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 37);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, "fence_i");
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        tu.write_mem(
            traits<ARCH>::FENCE,
            tu.constant(1, 64U),
            tu.trunc(tu.constant(imm, 32U), 32));
        tu.close_scope();
        tu.store(tu.constant(std::numeric_limits<uint32_t>::max(), 32),traits<ARCH>::LAST_BRANCH);
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 37);
        gen_trap_check(tu);
        return std::make_tuple(FLUSH);
    }
    
    /* instruction 38: ECALL */
    compile_ret_t __ecall(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("ECALL_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 38);
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, "ecall");
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        this->gen_raise_trap(tu, 0, 11);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 38);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 39: EBREAK */
    compile_ret_t __ebreak(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("EBREAK_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 39);
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, "ebreak");
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        this->gen_raise_trap(tu, 0, 3);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 39);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 40: URET */
    compile_ret_t __uret(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("URET_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 40);
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, "uret");
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        this->gen_leave_trap(tu, 0);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 40);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 41: SRET */
    compile_ret_t __sret(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRET_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 41);
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, "sret");
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        this->gen_leave_trap(tu, 1);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 41);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 42: MRET */
    compile_ret_t __mret(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("MRET_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 42);
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, "mret");
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        this->gen_leave_trap(tu, 3);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 42);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 43: WFI */
    compile_ret_t __wfi(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("WFI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 43);
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, "wfi");
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        this->gen_wait(tu, 1);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 43);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 44: SFENCE.VMA */
    compile_ret_t __sfence_vma(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SFENCE_VMA_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 44);
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, "sfence.vma");
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        tu.write_mem(
            traits<ARCH>::FENCE,
            tu.constant(2, 64U),
            tu.trunc(tu.constant(rs1, 32U), 32));
        tu.write_mem(
            traits<ARCH>::FENCE,
            tu.constant(3, 64U),
            tu.trunc(tu.constant(rs2, 32U), 32));
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 44);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 45: CSRRW */
    compile_ret_t __csrrw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 45);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {rs1}", fmt::arg("mnemonic", "csrrw"),
            	fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto rs_val_val = tu.assignment(tu.load(rs1 + traits<ARCH>::X0, 0), 32);
        if(rd != 0){
            auto csr_val_val = tu.assignment(tu.read_mem(traits<ARCH>::CSR, tu.constant(csr, 16U), 32), 32);
            tu.write_mem(
                traits<ARCH>::CSR,
                tu.constant(csr, 16U),
                tu.trunc(rs_val_val, 32));
            tu.store(csr_val_val, rd + traits<ARCH>::X0);
        } else {
            tu.write_mem(
                traits<ARCH>::CSR,
                tu.constant(csr, 16U),
                tu.trunc(rs_val_val, 32));
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 45);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 46: CSRRS */
    compile_ret_t __csrrs(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRS_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 46);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {rs1}", fmt::arg("mnemonic", "csrrs"),
            	fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto xrd_val = tu.assignment(tu.read_mem(traits<ARCH>::CSR, tu.constant(csr, 16U), 32), 32);
        auto xrs1_val = tu.assignment(tu.load(rs1 + traits<ARCH>::X0, 0), 32);
        if(rd != 0){
            tu.store(xrd_val, rd + traits<ARCH>::X0);
        }
        if(rs1 != 0){
            tu.write_mem(
                traits<ARCH>::CSR,
                tu.constant(csr, 16U),
                tu.trunc(tu.l_or(
                    xrd_val,
                    xrs1_val), 32));
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 46);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 47: CSRRC */
    compile_ret_t __csrrc(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRC_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 47);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {rs1}", fmt::arg("mnemonic", "csrrc"),
            	fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto xrd_val = tu.assignment(tu.read_mem(traits<ARCH>::CSR, tu.constant(csr, 16U), 32), 32);
        auto xrs1_val = tu.assignment(tu.load(rs1 + traits<ARCH>::X0, 0), 32);
        if(rd != 0){
            tu.store(xrd_val, rd + traits<ARCH>::X0);
        }
        if(rs1 != 0){
            tu.write_mem(
                traits<ARCH>::CSR,
                tu.constant(csr, 16U),
                tu.trunc(tu.l_and(
                    xrd_val,
                    tu.l_not(xrs1_val)), 32));
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 47);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 48: CSRRWI */
    compile_ret_t __csrrwi(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRWI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 48);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t zimm = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {zimm:#0x}", fmt::arg("mnemonic", "csrrwi"),
            	fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("zimm", zimm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu.store(tu.read_mem(traits<ARCH>::CSR, tu.constant(csr, 16U), 32), rd + traits<ARCH>::X0);
        }
        tu.write_mem(
            traits<ARCH>::CSR,
            tu.constant(csr, 16U),
            tu.trunc(tu.ext(
                tu.constant(zimm, 32U),
                32,
                true), 32));
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 48);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 49: CSRRSI */
    compile_ret_t __csrrsi(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRSI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 49);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t zimm = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {zimm:#0x}", fmt::arg("mnemonic", "csrrsi"),
            	fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("zimm", zimm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto res_val = tu.assignment(tu.read_mem(traits<ARCH>::CSR, tu.constant(csr, 16U), 32), 32);
        if(zimm != 0){
            tu.write_mem(
                traits<ARCH>::CSR,
                tu.constant(csr, 16U),
                tu.trunc(tu.l_or(
                    res_val,
                    tu.ext(
                        tu.constant(zimm, 32U),
                        32,
                        true)), 32));
        }
        if(rd != 0){
            tu.store(res_val, rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 49);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 50: CSRRCI */
    compile_ret_t __csrrci(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRCI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 50);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t zimm = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {zimm:#0x}", fmt::arg("mnemonic", "csrrci"),
            	fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("zimm", zimm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto res_val = tu.assignment(tu.read_mem(traits<ARCH>::CSR, tu.constant(csr, 16U), 32), 32);
        if(rd != 0){
            tu.store(res_val, rd + traits<ARCH>::X0);
        }
        if(zimm != 0){
            tu.write_mem(
                traits<ARCH>::CSR,
                tu.constant(csr, 16U),
                tu.trunc(tu.l_and(
                    res_val,
                    tu.l_not(tu.ext(
                        tu.constant(zimm, 32U),
                        32,
                        true))), 32));
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 50);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 51: MUL */
    compile_ret_t __mul(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("MUL_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 51);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mul"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            auto res_val = tu.assignment(tu.mul(
                tu.ext(
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    64,
                    true),
                tu.ext(
                    tu.load(rs2 + traits<ARCH>::X0, 0),
                    64,
                    true)), 64);
            tu.store(tu.ext(
                res_val,
                32,
                true), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 51);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 52: MULH */
    compile_ret_t __mulh(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("MULH_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 52);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mulh"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            auto res_val = tu.assignment(tu.mul(
                tu.ext(
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    64,
                    false),
                tu.ext(
                    tu.load(rs2 + traits<ARCH>::X0, 0),
                    64,
                    false)), 64);
            tu.store(tu.ext(
                tu.lshr(
                    res_val,
                    tu.constant(32, 32U)),
                32,
                true), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 52);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 53: MULHSU */
    compile_ret_t __mulhsu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("MULHSU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 53);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mulhsu"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            auto res_val = tu.assignment(tu.mul(
                tu.ext(
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    64,
                    false),
                tu.ext(
                    tu.load(rs2 + traits<ARCH>::X0, 0),
                    64,
                    true)), 64);
            tu.store(tu.ext(
                tu.lshr(
                    res_val,
                    tu.constant(32, 32U)),
                32,
                true), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 53);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 54: MULHU */
    compile_ret_t __mulhu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("MULHU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 54);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mulhu"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            auto res_val = tu.assignment(tu.mul(
                tu.ext(
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    64,
                    true),
                tu.ext(
                    tu.load(rs2 + traits<ARCH>::X0, 0),
                    64,
                    true)), 64);
            tu.store(tu.ext(
                tu.lshr(
                    res_val,
                    tu.constant(32, 32U)),
                32,
                true), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 54);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 55: DIV */
    compile_ret_t __div(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("DIV_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 55);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "div"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu(  " if({}) {{", tu.icmp(
                ICmpInst::ICMP_NE,
                tu.load(rs2 + traits<ARCH>::X0, 0),
                tu.constant(0, 32U)));
            uint32_t M1_val = - 1;
            uint8_t XLM1_val = 32 - 1;
            uint32_t ONE_val = 1;
            uint32_t MMIN_val = ONE_val << XLM1_val;
            tu(  " if({}) {{", tu.b_and(
                tu.icmp(
                    ICmpInst::ICMP_EQ,
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    tu.constant(MMIN_val, 32U)),
                tu.icmp(
                    ICmpInst::ICMP_EQ,
                    tu.load(rs2 + traits<ARCH>::X0, 0),
                    tu.constant(M1_val, 32U))));
            tu.store(tu.constant(MMIN_val, 32U), rd + traits<ARCH>::X0);
            tu("  }} else {{");
            tu.store(tu.sdiv(
                tu.ext(
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    32, false),
                tu.ext(
                    tu.load(rs2 + traits<ARCH>::X0, 0),
                    32, false)), rd + traits<ARCH>::X0);
            tu.close_scope();
            tu("  }} else {{");
            tu.store(tu.neg(tu.constant(1, 32U)), rd + traits<ARCH>::X0);
            tu.close_scope();
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 55);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 56: DIVU */
    compile_ret_t __divu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("DIVU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 56);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "divu"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu(  " if({}) {{", tu.icmp(
                ICmpInst::ICMP_NE,
                tu.load(rs2 + traits<ARCH>::X0, 0),
                tu.constant(0, 32U)));
            tu.store(tu.udiv(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                tu.load(rs2 + traits<ARCH>::X0, 0)), rd + traits<ARCH>::X0);
            tu("  }} else {{");
            tu.store(tu.neg(tu.constant(1, 32U)), rd + traits<ARCH>::X0);
            tu.close_scope();
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 56);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 57: REM */
    compile_ret_t __rem(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("REM_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 57);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "rem"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu(  " if({}) {{", tu.icmp(
                ICmpInst::ICMP_NE,
                tu.load(rs2 + traits<ARCH>::X0, 0),
                tu.constant(0, 32U)));
            uint32_t M1_val = - 1;
            uint32_t XLM1_val = 32 - 1;
            uint32_t ONE_val = 1;
            uint32_t MMIN_val = ONE_val << XLM1_val;
            tu(  " if({}) {{", tu.b_and(
                tu.icmp(
                    ICmpInst::ICMP_EQ,
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    tu.constant(MMIN_val, 32U)),
                tu.icmp(
                    ICmpInst::ICMP_EQ,
                    tu.load(rs2 + traits<ARCH>::X0, 0),
                    tu.constant(M1_val, 32U))));
            tu.store(tu.constant(0, 32U), rd + traits<ARCH>::X0);
            tu("  }} else {{");
            tu.store(tu.srem(
                tu.ext(
                    tu.load(rs1 + traits<ARCH>::X0, 0),
                    32, false),
                tu.ext(
                    tu.load(rs2 + traits<ARCH>::X0, 0),
                    32, false)), rd + traits<ARCH>::X0);
            tu.close_scope();
            tu("  }} else {{");
            tu.store(tu.load(rs1 + traits<ARCH>::X0, 0), rd + traits<ARCH>::X0);
            tu.close_scope();
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 57);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 58: REMU */
    compile_ret_t __remu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("REMU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 58);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "remu"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            tu(  " if({}) {{", tu.icmp(
                ICmpInst::ICMP_NE,
                tu.load(rs2 + traits<ARCH>::X0, 0),
                tu.constant(0, 32U)));
            tu.store(tu.urem(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                tu.load(rs2 + traits<ARCH>::X0, 0)), rd + traits<ARCH>::X0);
            tu("  }} else {{");
            tu.store(tu.load(rs1 + traits<ARCH>::X0, 0), rd + traits<ARCH>::X0);
            tu.close_scope();
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 58);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 59: LR.W */
    compile_ret_t __lr_w(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LR_W_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 59);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "lr.w"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        if(rd != 0){
            auto offs_val = tu.assignment(tu.load(rs1 + traits<ARCH>::X0, 0), 32);
            tu.store(tu.ext(
                tu.read_mem(traits<ARCH>::MEM, offs_val, 32),
                32,
                false), rd + traits<ARCH>::X0);
            tu.write_mem(
                traits<ARCH>::RES,
                offs_val,
                tu.trunc(tu.ext(
                    tu.neg(tu.constant(1, 8U)),
                    32,
                    false), 32));
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 59);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 60: SC.W */
    compile_ret_t __sc_w(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SC_W_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 60);
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
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.load(rs1 + traits<ARCH>::X0, 0), 32);
        auto res1_val = tu.assignment(tu.read_mem(traits<ARCH>::RES, offs_val, 32), 32);
        tu(  " if({}) {{", tu.icmp(
            ICmpInst::ICMP_NE,
            res1_val,
            tu.constant(0, 32U)));
        tu.write_mem(
            traits<ARCH>::MEM,
            offs_val,
            tu.trunc(tu.load(rs2 + traits<ARCH>::X0, 0), 32));
        tu.close_scope();
        if(rd != 0){
            tu.store(tu.choose(
                tu.icmp(
                    ICmpInst::ICMP_NE,
                    res1_val,
                    tu.ext(
                        tu.constant(0, 32U),
                        32,
                        true)),
                tu.constant(0, 32U),
                tu.constant(1, 32U)), rd + traits<ARCH>::X0);
        }
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 60);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 61: AMOSWAP.W */
    compile_ret_t __amoswap_w(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("AMOSWAP_W_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 61);
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
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.load(rs1 + traits<ARCH>::X0, 0), 32);
        if(rd != 0){
            tu.store(tu.ext(
                tu.read_mem(traits<ARCH>::MEM, offs_val, 32),
                32,
                false), rd + traits<ARCH>::X0);
        }
        tu.write_mem(
            traits<ARCH>::MEM,
            offs_val,
            tu.trunc(tu.load(rs2 + traits<ARCH>::X0, 0), 32));
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 61);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 62: AMOADD.W */
    compile_ret_t __amoadd_w(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("AMOADD_W_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 62);
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
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.load(rs1 + traits<ARCH>::X0, 0), 32);
        auto res1_val = tu.assignment(tu.ext(
            tu.read_mem(traits<ARCH>::MEM, offs_val, 32),
            32,
            false), 32);
        if(rd != 0){
            tu.store(res1_val, rd + traits<ARCH>::X0);
        }
        auto res2_val = tu.assignment(tu.add(
            res1_val,
            tu.load(rs2 + traits<ARCH>::X0, 0)), 32);
        tu.write_mem(
            traits<ARCH>::MEM,
            offs_val,
            tu.trunc(res2_val, 32));
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 62);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 63: AMOXOR.W */
    compile_ret_t __amoxor_w(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("AMOXOR_W_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 63);
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
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.load(rs1 + traits<ARCH>::X0, 0), 32);
        auto res1_val = tu.assignment(tu.ext(
            tu.read_mem(traits<ARCH>::MEM, offs_val, 32),
            32,
            false), 32);
        if(rd != 0){
            tu.store(res1_val, rd + traits<ARCH>::X0);
        }
        auto res2_val = tu.assignment(tu.l_xor(
            res1_val,
            tu.load(rs2 + traits<ARCH>::X0, 0)), 32);
        tu.write_mem(
            traits<ARCH>::MEM,
            offs_val,
            tu.trunc(res2_val, 32));
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 63);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 64: AMOAND.W */
    compile_ret_t __amoand_w(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("AMOAND_W_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 64);
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
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.load(rs1 + traits<ARCH>::X0, 0), 32);
        auto res1_val = tu.assignment(tu.ext(
            tu.read_mem(traits<ARCH>::MEM, offs_val, 32),
            32,
            false), 32);
        if(rd != 0){
            tu.store(res1_val, rd + traits<ARCH>::X0);
        }
        auto res2_val = tu.assignment(tu.l_and(
            res1_val,
            tu.load(rs2 + traits<ARCH>::X0, 0)), 32);
        tu.write_mem(
            traits<ARCH>::MEM,
            offs_val,
            tu.trunc(res2_val, 32));
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 64);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 65: AMOOR.W */
    compile_ret_t __amoor_w(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("AMOOR_W_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 65);
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
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.load(rs1 + traits<ARCH>::X0, 0), 32);
        auto res1_val = tu.assignment(tu.ext(
            tu.read_mem(traits<ARCH>::MEM, offs_val, 32),
            32,
            false), 32);
        if(rd != 0){
            tu.store(res1_val, rd + traits<ARCH>::X0);
        }
        auto res2_val = tu.assignment(tu.l_or(
            res1_val,
            tu.load(rs2 + traits<ARCH>::X0, 0)), 32);
        tu.write_mem(
            traits<ARCH>::MEM,
            offs_val,
            tu.trunc(res2_val, 32));
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 65);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 66: AMOMIN.W */
    compile_ret_t __amomin_w(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("AMOMIN_W_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 66);
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
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.load(rs1 + traits<ARCH>::X0, 0), 32);
        auto res1_val = tu.assignment(tu.ext(
            tu.read_mem(traits<ARCH>::MEM, offs_val, 32),
            32,
            false), 32);
        if(rd != 0){
            tu.store(res1_val, rd + traits<ARCH>::X0);
        }
        auto res2_val = tu.assignment(tu.choose(
            tu.icmp(
                ICmpInst::ICMP_SGT,
                tu.ext(
                    res1_val,
                    32, false),
                tu.ext(
                    tu.load(rs2 + traits<ARCH>::X0, 0),
                    32, false)),
            tu.load(rs2 + traits<ARCH>::X0, 0),
            res1_val), 32);
        tu.write_mem(
            traits<ARCH>::MEM,
            offs_val,
            tu.trunc(res2_val, 32));
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 66);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 67: AMOMAX.W */
    compile_ret_t __amomax_w(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("AMOMAX_W_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 67);
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
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.load(rs1 + traits<ARCH>::X0, 0), 32);
        auto res1_val = tu.assignment(tu.ext(
            tu.read_mem(traits<ARCH>::MEM, offs_val, 32),
            32,
            false), 32);
        if(rd != 0){
            tu.store(res1_val, rd + traits<ARCH>::X0);
        }
        auto res2_val = tu.assignment(tu.choose(
            tu.icmp(
                ICmpInst::ICMP_SLT,
                tu.ext(
                    res1_val,
                    32, false),
                tu.ext(
                    tu.load(rs2 + traits<ARCH>::X0, 0),
                    32, false)),
            tu.load(rs2 + traits<ARCH>::X0, 0),
            res1_val), 32);
        tu.write_mem(
            traits<ARCH>::MEM,
            offs_val,
            tu.trunc(res2_val, 32));
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 67);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 68: AMOMINU.W */
    compile_ret_t __amominu_w(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("AMOMINU_W_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 68);
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
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.load(rs1 + traits<ARCH>::X0, 0), 32);
        auto res1_val = tu.assignment(tu.ext(
            tu.read_mem(traits<ARCH>::MEM, offs_val, 32),
            32,
            false), 32);
        if(rd != 0){
            tu.store(res1_val, rd + traits<ARCH>::X0);
        }
        auto res2_val = tu.assignment(tu.choose(
            tu.icmp(
                ICmpInst::ICMP_UGT,
                res1_val,
                tu.load(rs2 + traits<ARCH>::X0, 0)),
            tu.load(rs2 + traits<ARCH>::X0, 0),
            res1_val), 32);
        tu.write_mem(
            traits<ARCH>::MEM,
            offs_val,
            tu.trunc(res2_val, 32));
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 68);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 69: AMOMAXU.W */
    compile_ret_t __amomaxu_w(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("AMOMAXU_W_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 69);
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
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.load(rs1 + traits<ARCH>::X0, 0), 32);
        auto res1_val = tu.assignment(tu.ext(
            tu.read_mem(traits<ARCH>::MEM, offs_val, 32),
            32,
            false), 32);
        if(rd != 0){
            tu.store(res1_val, rd + traits<ARCH>::X0);
        }
        auto res2_val = tu.assignment(tu.choose(
            tu.icmp(
                ICmpInst::ICMP_ULT,
                res1_val,
                tu.load(rs2 + traits<ARCH>::X0, 0)),
            tu.load(rs2 + traits<ARCH>::X0, 0),
            res1_val), 32);
        tu.write_mem(
            traits<ARCH>::MEM,
            offs_val,
            tu.trunc(res2_val, 32));
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 69);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 70: JALR */
    compile_ret_t __jalr(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("JALR_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 70);
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm:#0x}", fmt::arg("mnemonic", "jalr"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        auto new_pc_val = tu.assignment(tu.add(
            tu.ext(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                32, false),
            tu.constant(imm, 32U)), 32);
        if(rd != 0){
            tu.store(tu.add(
                cur_pc_val,
                tu.constant(4, 32U)), rd + traits<ARCH>::X0);
        }
        auto PC_val_v = tu.assignment("PC_val", tu.l_and(
            new_pc_val,
            tu.l_not(tu.constant(0x1, 32U))), 32);
        tu.store(PC_val_v, traits<ARCH>::NEXT_PC);
        tu.store(tu.constant(std::numeric_limits<uint32_t>::max(), 32U), traits<ARCH>::LAST_BRANCH);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 70);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 71: C.ADDI4SPN */
    compile_ret_t __c_addi4spn(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_ADDI4SPN_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 71);
        uint8_t rd = ((bit_sub<2,3>(instr)));
        uint16_t imm = ((bit_sub<5,1>(instr) << 3) | (bit_sub<6,1>(instr) << 2) | (bit_sub<7,4>(instr) << 6) | (bit_sub<11,2>(instr) << 4));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c.addi4spn"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        if(imm == 0){
            this->gen_raise_trap(tu, 0, 2);
        }
        tu.store(tu.add(
            tu.load(2 + traits<ARCH>::X0, 0),
            tu.constant(imm, 32U)), rd + 8 + traits<ARCH>::X0);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 71);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 72: C.LW */
    compile_ret_t __c_lw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_LW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 72);
        uint8_t rd = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {uimm:#05x}({rs1})", fmt::arg("mnemonic", "c.lw"),
            	fmt::arg("rd", name(8+rd)), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.add(
            tu.load(rs1 + 8 + traits<ARCH>::X0, 0),
            tu.constant(uimm, 32U)), 32);
        tu.store(tu.ext(
            tu.read_mem(traits<ARCH>::MEM, offs_val, 32),
            32,
            false), rd + 8 + traits<ARCH>::X0);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 72);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 73: C.SW */
    compile_ret_t __c_sw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_SW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 73);
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {uimm:#05x}({rs1})", fmt::arg("mnemonic", "c.sw"),
            	fmt::arg("rs2", name(8+rs2)), fmt::arg("uimm", uimm), fmt::arg("rs1", name(8+rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.add(
            tu.load(rs1 + 8 + traits<ARCH>::X0, 0),
            tu.constant(uimm, 32U)), 32);
        tu.write_mem(
            traits<ARCH>::MEM,
            offs_val,
            tu.trunc(tu.load(rs2 + 8 + traits<ARCH>::X0, 0), 32));
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 73);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 74: C.ADDI */
    compile_ret_t __c_addi(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_ADDI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 74);
        int8_t imm = signextend<int8_t,6>((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.addi"),
            	fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        tu.store(tu.add(
            tu.ext(
                tu.load(rs1 + traits<ARCH>::X0, 0),
                32, false),
            tu.constant(imm, 32U)), rs1 + traits<ARCH>::X0);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 74);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 75: C.NOP */
    compile_ret_t __c_nop(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_NOP_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 75);
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, "c.nop");
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        tu.close_scope();
        /* TODO: describe operations for C.NOP ! */
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 75);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 76: C.JAL */
    compile_ret_t __c_jal(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_JAL_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 76);
        int16_t imm = signextend<int16_t,12>((bit_sub<2,1>(instr) << 5) | (bit_sub<3,3>(instr) << 1) | (bit_sub<6,1>(instr) << 7) | (bit_sub<7,1>(instr) << 6) | (bit_sub<8,1>(instr) << 10) | (bit_sub<9,2>(instr) << 8) | (bit_sub<11,1>(instr) << 4) | (bit_sub<12,1>(instr) << 11));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {imm:#05x}", fmt::arg("mnemonic", "c.jal"),
            	fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        tu.store(tu.add(
            cur_pc_val,
            tu.constant(2, 32U)), 1 + traits<ARCH>::X0);
        auto PC_val_v = tu.assignment("PC_val", tu.add(
            tu.ext(
                cur_pc_val,
                32, false),
            tu.constant(imm, 32U)), 32);
        tu.store(PC_val_v, traits<ARCH>::NEXT_PC);
        auto is_cont_v = tu.choose(
            tu.icmp(ICmpInst::ICMP_NE, tu.ext(PC_val_v, 32U, true), tu.constant(pc.val, 32U)),
            tu.constant(0U, 32), tu.constant(1U, 32));
        tu.store(is_cont_v, traits<ARCH>::LAST_BRANCH);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 76);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 77: C.LI */
    compile_ret_t __c_li(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_LI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 77);
        int8_t imm = signextend<int8_t,6>((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c.li"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        if(rd == 0){
            this->gen_raise_trap(tu, 0, 2);
        }
        tu.store(tu.constant(imm, 32U), rd + traits<ARCH>::X0);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 77);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 78: C.LUI */
    compile_ret_t __c_lui(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_LUI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 78);
        int32_t imm = signextend<int32_t,18>((bit_sub<2,5>(instr) << 12) | (bit_sub<12,1>(instr) << 17));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c.lui"),
            	fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        if(rd == 0){
            this->gen_raise_trap(tu, 0, 2);
        }
        if(imm == 0){
            this->gen_raise_trap(tu, 0, 2);
        }
        tu.store(tu.constant(imm, 32U), rd + traits<ARCH>::X0);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 78);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 79: C.ADDI16SP */
    compile_ret_t __c_addi16sp(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_ADDI16SP_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 79);
        int16_t imm = signextend<int16_t,10>((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 7) | (bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 4) | (bit_sub<12,1>(instr) << 9));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {imm:#05x}", fmt::arg("mnemonic", "c.addi16sp"),
            	fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        tu.store(tu.add(
            tu.ext(
                tu.load(2 + traits<ARCH>::X0, 0),
                32, false),
            tu.constant(imm, 32U)), 2 + traits<ARCH>::X0);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 79);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 80: C.SRLI */
    compile_ret_t __c_srli(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_SRLI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 80);
        uint8_t shamt = ((bit_sub<2,5>(instr)));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {shamt}", fmt::arg("mnemonic", "c.srli"),
            	fmt::arg("rs1", name(8+rs1)), fmt::arg("shamt", shamt));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        uint8_t rs1_idx_val = rs1 + 8;
        tu.store(tu.lshr(
            tu.load(rs1_idx_val + traits<ARCH>::X0, 0),
            tu.constant(shamt, 32U)), rs1_idx_val + traits<ARCH>::X0);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 80);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 81: C.SRAI */
    compile_ret_t __c_srai(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_SRAI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 81);
        uint8_t shamt = ((bit_sub<2,5>(instr)));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {shamt}", fmt::arg("mnemonic", "c.srai"),
            	fmt::arg("rs1", name(8+rs1)), fmt::arg("shamt", shamt));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        uint8_t rs1_idx_val = rs1 + 8;
        tu.store(tu.ashr(
            tu.load(rs1_idx_val + traits<ARCH>::X0, 0),
            tu.constant(shamt, 32U)), rs1_idx_val + traits<ARCH>::X0);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 81);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 82: C.ANDI */
    compile_ret_t __c_andi(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_ANDI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 82);
        int8_t imm = signextend<int8_t,6>((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.andi"),
            	fmt::arg("rs1", name(8+rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        uint8_t rs1_idx_val = rs1 + 8;
        tu.store(tu.l_and(
            tu.ext(
                tu.load(rs1_idx_val + traits<ARCH>::X0, 0),
                32, false),
            tu.constant(imm, 32U)), rs1_idx_val + traits<ARCH>::X0);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 82);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 83: C.SUB */
    compile_ret_t __c_sub(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_SUB_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 83);
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.sub"),
            	fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        uint8_t rd_idx_val = rd + 8;
        tu.store(tu.sub(
             tu.load(rd_idx_val + traits<ARCH>::X0, 0),
             tu.load(rs2 + 8 + traits<ARCH>::X0, 0)), rd_idx_val + traits<ARCH>::X0);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 83);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 84: C.XOR */
    compile_ret_t __c_xor(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_XOR_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 84);
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.xor"),
            	fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        uint8_t rd_idx_val = rd + 8;
        tu.store(tu.l_xor(
            tu.load(rd_idx_val + traits<ARCH>::X0, 0),
            tu.load(rs2 + 8 + traits<ARCH>::X0, 0)), rd_idx_val + traits<ARCH>::X0);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 84);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 85: C.OR */
    compile_ret_t __c_or(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_OR_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 85);
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.or"),
            	fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        uint8_t rd_idx_val = rd + 8;
        tu.store(tu.l_or(
            tu.load(rd_idx_val + traits<ARCH>::X0, 0),
            tu.load(rs2 + 8 + traits<ARCH>::X0, 0)), rd_idx_val + traits<ARCH>::X0);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 85);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 86: C.AND */
    compile_ret_t __c_and(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_AND_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 86);
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.and"),
            	fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        uint8_t rd_idx_val = rd + 8;
        tu.store(tu.l_and(
            tu.load(rd_idx_val + traits<ARCH>::X0, 0),
            tu.load(rs2 + 8 + traits<ARCH>::X0, 0)), rd_idx_val + traits<ARCH>::X0);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 86);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 87: C.J */
    compile_ret_t __c_j(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_J_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 87);
        int16_t imm = signextend<int16_t,12>((bit_sub<2,1>(instr) << 5) | (bit_sub<3,3>(instr) << 1) | (bit_sub<6,1>(instr) << 7) | (bit_sub<7,1>(instr) << 6) | (bit_sub<8,1>(instr) << 10) | (bit_sub<9,2>(instr) << 8) | (bit_sub<11,1>(instr) << 4) | (bit_sub<12,1>(instr) << 11));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {imm:#05x}", fmt::arg("mnemonic", "c.j"),
            	fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        auto PC_val_v = tu.assignment("PC_val", tu.add(
            tu.ext(
                cur_pc_val,
                32, false),
            tu.constant(imm, 32U)), 32);
        tu.store(PC_val_v, traits<ARCH>::NEXT_PC);
        auto is_cont_v = tu.choose(
            tu.icmp(ICmpInst::ICMP_NE, tu.ext(PC_val_v, 32U, true), tu.constant(pc.val, 32U)),
            tu.constant(0U, 32), tu.constant(1U, 32));
        tu.store(is_cont_v, traits<ARCH>::LAST_BRANCH);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 87);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 88: C.BEQZ */
    compile_ret_t __c_beqz(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_BEQZ_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 88);
        int16_t imm = signextend<int16_t,9>((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 1) | (bit_sub<5,2>(instr) << 6) | (bit_sub<10,2>(instr) << 3) | (bit_sub<12,1>(instr) << 8));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.beqz"),
            	fmt::arg("rs1", name(8+rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        auto PC_val_v = tu.assignment("PC_val", tu.choose(
            tu.icmp(
                ICmpInst::ICMP_EQ,
                tu.load(rs1 + 8 + traits<ARCH>::X0, 0),
                tu.constant(0, 32U)),
            tu.add(
                tu.ext(
                    cur_pc_val,
                    32, false),
                tu.constant(imm, 32U)),
            tu.add(
                cur_pc_val,
                tu.constant(2, 32U))), 32);
        tu.store(PC_val_v, traits<ARCH>::NEXT_PC);
        auto is_cont_v = tu.choose(
            tu.icmp(ICmpInst::ICMP_NE, tu.ext(PC_val_v, 32U, true), tu.constant(pc.val, 32U)),
            tu.constant(0U, 32), tu.constant(1U, 32));
        tu.store(is_cont_v, traits<ARCH>::LAST_BRANCH);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 88);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 89: C.BNEZ */
    compile_ret_t __c_bnez(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_BNEZ_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 89);
        int16_t imm = signextend<int16_t,9>((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 1) | (bit_sub<5,2>(instr) << 6) | (bit_sub<10,2>(instr) << 3) | (bit_sub<12,1>(instr) << 8));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.bnez"),
            	fmt::arg("rs1", name(8+rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        auto PC_val_v = tu.assignment("PC_val", tu.choose(
            tu.icmp(
                ICmpInst::ICMP_NE,
                tu.load(rs1 + 8 + traits<ARCH>::X0, 0),
                tu.constant(0, 32U)),
            tu.add(
                tu.ext(
                    cur_pc_val,
                    32, false),
                tu.constant(imm, 32U)),
            tu.add(
                cur_pc_val,
                tu.constant(2, 32U))), 32);
        tu.store(PC_val_v, traits<ARCH>::NEXT_PC);
        auto is_cont_v = tu.choose(
            tu.icmp(ICmpInst::ICMP_NE, tu.ext(PC_val_v, 32U, true), tu.constant(pc.val, 32U)),
            tu.constant(0U, 32), tu.constant(1U, 32));
        tu.store(is_cont_v, traits<ARCH>::LAST_BRANCH);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 89);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 90: C.SLLI */
    compile_ret_t __c_slli(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_SLLI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 90);
        uint8_t shamt = ((bit_sub<2,5>(instr)));
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {shamt}", fmt::arg("mnemonic", "c.slli"),
            	fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        if(rs1 == 0){
            this->gen_raise_trap(tu, 0, 2);
        }
        tu.store(tu.shl(
            tu.load(rs1 + traits<ARCH>::X0, 0),
            tu.constant(shamt, 32U)), rs1 + traits<ARCH>::X0);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 90);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 91: C.LWSP */
    compile_ret_t __c_lwsp(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_LWSP_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 91);
        uint8_t uimm = ((bit_sub<2,2>(instr) << 6) | (bit_sub<4,3>(instr) << 2) | (bit_sub<12,1>(instr) << 5));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, sp, {uimm:#05x}", fmt::arg("mnemonic", "c.lwsp"),
            	fmt::arg("rd", name(rd)), fmt::arg("uimm", uimm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.add(
            tu.load(2 + traits<ARCH>::X0, 0),
            tu.constant(uimm, 32U)), 32);
        tu.store(tu.ext(
            tu.read_mem(traits<ARCH>::MEM, offs_val, 32),
            32,
            false), rd + traits<ARCH>::X0);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 91);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 92: C.MV */
    compile_ret_t __c_mv(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_MV_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 92);
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.mv"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        tu.store(tu.load(rs2 + traits<ARCH>::X0, 0), rd + traits<ARCH>::X0);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 92);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 93: C.JR */
    compile_ret_t __c_jr(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_JR_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 93);
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}", fmt::arg("mnemonic", "c.jr"),
            	fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        auto PC_val_v = tu.assignment("PC_val", tu.load(rs1 + traits<ARCH>::X0, 0), 32);
        tu.store(PC_val_v, traits<ARCH>::NEXT_PC);
        tu.store(tu.constant(std::numeric_limits<uint32_t>::max(), 32U), traits<ARCH>::LAST_BRANCH);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 93);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 94: C.ADD */
    compile_ret_t __c_add(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_ADD_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 94);
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.add"),
            	fmt::arg("rd", name(rd)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        tu.store(tu.add(
            tu.load(rd + traits<ARCH>::X0, 0),
            tu.load(rs2 + traits<ARCH>::X0, 0)), rd + traits<ARCH>::X0);
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 94);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 95: C.JALR */
    compile_ret_t __c_jalr(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_JALR_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 95);
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}", fmt::arg("mnemonic", "c.jalr"),
            	fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        tu.store(tu.add(
            cur_pc_val,
            tu.constant(2, 32U)), 1 + traits<ARCH>::X0);
        auto PC_val_v = tu.assignment("PC_val", tu.load(rs1 + traits<ARCH>::X0, 0), 32);
        tu.store(PC_val_v, traits<ARCH>::NEXT_PC);
        tu.store(tu.constant(std::numeric_limits<uint32_t>::max(), 32U), traits<ARCH>::LAST_BRANCH);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 95);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 96: C.EBREAK */
    compile_ret_t __c_ebreak(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_EBREAK_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 96);
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, "c.ebreak");
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        this->gen_raise_trap(tu, 0, 3);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 96);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 97: C.SWSP */
    compile_ret_t __c_swsp(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("C_SWSP_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 97);
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t uimm = ((bit_sub<7,2>(instr) << 6) | (bit_sub<9,4>(instr) << 2));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {uimm:#05x}(sp)", fmt::arg("mnemonic", "c.swsp"),
            	fmt::arg("rs2", name(rs2)), fmt::arg("uimm", uimm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        auto offs_val = tu.assignment(tu.add(
            tu.load(2 + traits<ARCH>::X0, 0),
            tu.constant(uimm, 32U)), 32);
        tu.write_mem(
            traits<ARCH>::MEM,
            offs_val,
            tu.trunc(tu.load(rs2 + traits<ARCH>::X0, 0), 32));
        tu.close_scope();
        gen_set_pc(tu, pc, traits<ARCH>::NEXT_PC);
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 97);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 98: DII */
    compile_ret_t __dii(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("DII_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 98);
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, "dii");
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+2;
        tu.open_scope();
        this->gen_raise_trap(tu, 0, 2);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 98);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /****************************************************************************
     * end opcode definitions
     ****************************************************************************/
    compile_ret_t illegal_intruction(virt_addr_t &pc, code_word_t instr, tu_builder& tu) {
        vm_impl::gen_sync(tu, iss::PRE_SYNC, instr_descr.size());
        pc = pc + ((instr & 3) == 3 ? 4 : 2);
        gen_raise_trap(tu, 0, 2);     // illegal instruction trap
        vm_impl::gen_sync(tu, iss::POST_SYNC, instr_descr.size());
        vm_impl::gen_trap_check(tu);
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
vm_impl<ARCH>::gen_single_inst_behavior(virt_addr_t &pc, unsigned int &inst_cnt, tu_builder& tu) {
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
    return (this->*f)(pc, insn, tu);
}

template <typename ARCH> void vm_impl<ARCH>::gen_raise_trap(tu_builder& tu, uint16_t trap_id, uint16_t cause) {
    tu("  *trap_state = {:#x};", 0x80 << 24 | (cause << 16) | trap_id);
    tu.store(tu.constant(std::numeric_limits<uint32_t>::max(), 32),traits<ARCH>::LAST_BRANCH);
}

template <typename ARCH> void vm_impl<ARCH>::gen_leave_trap(tu_builder& tu, unsigned lvl) {
    tu("leave_trap(core_ptr, {});", lvl);
    tu.store(tu.read_mem(traits<ARCH>::CSR, (lvl << 8) + 0x41, traits<ARCH>::XLEN),traits<ARCH>::NEXT_PC);
    tu.store(tu.constant(std::numeric_limits<uint32_t>::max(), 32),traits<ARCH>::LAST_BRANCH);
}

template <typename ARCH> void vm_impl<ARCH>::gen_wait(tu_builder& tu, unsigned type) {
}

template <typename ARCH> void vm_impl<ARCH>::gen_trap_behavior(tu_builder& tu) {
    tu("trap_entry:");
    tu("enter_trap(core_ptr, *trap_state, *pc);");
    tu.store(tu.constant(std::numeric_limits<uint32_t>::max(),32),traits<ARCH>::LAST_BRANCH);
    tu("return *next_pc;");
}

} // namespace mnrv32

template <>
std::unique_ptr<vm_if> create<arch::rv32imac>(arch::rv32imac *core, unsigned short port, bool dump) {
    auto ret = new rv32imac::vm_impl<arch::rv32imac>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret, port);
    return std::unique_ptr<vm_if>(ret);
}
}
} // namespace iss
