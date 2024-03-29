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

#include <iss/arch/tgf01.h>
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
namespace tgf01 {
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

    const std::array<InstructionDesriptor, 52> instr_descr = {{
         /* entries are: size, valid value, valid mask, function ptr */
        /* instruction LUI, encoding '.........................0110111' */
        {32, 0b00000000000000000000000000110111, 0b00000000000000000000000001111111, &this_class::__lui},
        /* instruction AUIPC, encoding '.........................0010111' */
        {32, 0b00000000000000000000000000010111, 0b00000000000000000000000001111111, &this_class::__auipc},
        /* instruction JAL, encoding '.........................1101111' */
        {32, 0b00000000000000000000000001101111, 0b00000000000000000000000001111111, &this_class::__jal},
        /* instruction JALR, encoding '.................000.....1100111' */
        {32, 0b00000000000000000000000001100111, 0b00000000000000000111000001111111, &this_class::__jalr},
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
    
    /* instruction 3: JALR */
    compile_ret_t __jalr(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("JALR_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 3);
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
        auto align_val = tu.assignment(tu.l_and(
            new_pc_val,
            tu.constant(0x2, 32U)), 32);
        tu(  " if({}) {{", tu.icmp(
            ICmpInst::ICMP_NE,
            align_val,
            tu.constant(0, 32U)));
        this->gen_raise_trap(tu, 0, 0);
        tu("  }} else {{");
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
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 3);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 4: BEQ */
    compile_ret_t __beq(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BEQ_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 4);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 4);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 5: BNE */
    compile_ret_t __bne(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BNE_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 5);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 5);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 6: BLT */
    compile_ret_t __blt(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BLT_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 6);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 6);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 7: BGE */
    compile_ret_t __bge(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BGE_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 7);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 7);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 8: BLTU */
    compile_ret_t __bltu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BLTU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 8);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 8);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 9: BGEU */
    compile_ret_t __bgeu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BGEU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 9);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 9);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 10: LB */
    compile_ret_t __lb(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LB_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 10);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 10);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 11: LH */
    compile_ret_t __lh(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LH_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 11);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 11);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 12: LW */
    compile_ret_t __lw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 12);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 12);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 13: LBU */
    compile_ret_t __lbu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LBU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 13);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 13);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 14: LHU */
    compile_ret_t __lhu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LHU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 14);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 14);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 15: SB */
    compile_ret_t __sb(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SB_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 15);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 15);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 16: SH */
    compile_ret_t __sh(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SH_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 16);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 16);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 17: SW */
    compile_ret_t __sw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 17);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 17);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 18: ADDI */
    compile_ret_t __addi(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("ADDI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 18);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 18);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 19: SLTI */
    compile_ret_t __slti(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLTI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 19);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 19);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 20: SLTIU */
    compile_ret_t __sltiu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLTIU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 20);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 20);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 21: XORI */
    compile_ret_t __xori(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("XORI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 21);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 21);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 22: ORI */
    compile_ret_t __ori(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("ORI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 22);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 22);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 23: ANDI */
    compile_ret_t __andi(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("ANDI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 23);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 23);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 24: SLLI */
    compile_ret_t __slli(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLLI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 24);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 24);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 25: SRLI */
    compile_ret_t __srli(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRLI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 25);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 25);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 26: SRAI */
    compile_ret_t __srai(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRAI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 26);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 26);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 27: ADD */
    compile_ret_t __add(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("ADD_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 27);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 27);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 28: SUB */
    compile_ret_t __sub(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SUB_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 28);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 28);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 29: SLL */
    compile_ret_t __sll(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLL_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 29);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 29);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 30: SLT */
    compile_ret_t __slt(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLT_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 30);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 30);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 31: SLTU */
    compile_ret_t __sltu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLTU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 31);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 31);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 32: XOR */
    compile_ret_t __xor(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("XOR_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 32);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 32);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 33: SRL */
    compile_ret_t __srl(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRL_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 33);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 33);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 34: SRA */
    compile_ret_t __sra(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRA_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 34);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 34);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 35: OR */
    compile_ret_t __or(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("OR_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 35);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 35);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 36: AND */
    compile_ret_t __and(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("AND_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 36);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 36);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 37: FENCE */
    compile_ret_t __fence(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("FENCE_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 37);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 37);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 38: FENCE_I */
    compile_ret_t __fence_i(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("FENCE_I_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 38);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 38);
        gen_trap_check(tu);
        return std::make_tuple(FLUSH);
    }
    
    /* instruction 39: ECALL */
    compile_ret_t __ecall(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("ECALL_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 39);
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, "ecall");
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        this->gen_raise_trap(tu, 0, 11);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 39);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 40: EBREAK */
    compile_ret_t __ebreak(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("EBREAK_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 40);
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, "ebreak");
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        this->gen_raise_trap(tu, 0, 3);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 40);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 41: URET */
    compile_ret_t __uret(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("URET_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 41);
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, "uret");
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        this->gen_leave_trap(tu, 0);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 41);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 42: SRET */
    compile_ret_t __sret(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRET_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 42);
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, "sret");
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        this->gen_leave_trap(tu, 1);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 42);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 43: MRET */
    compile_ret_t __mret(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("MRET_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 43);
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, "mret");
        }
        auto cur_pc_val = tu.constant(pc.val, arch::traits<ARCH>::reg_bit_widths[traits<ARCH>::PC]);
        pc=pc+4;
        tu.open_scope();
        this->gen_leave_trap(tu, 3);
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 43);
        gen_trap_check(tu);
        return std::make_tuple(BRANCH);
    }
    
    /* instruction 44: WFI */
    compile_ret_t __wfi(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("WFI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 44);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 44);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 45: SFENCE.VMA */
    compile_ret_t __sfence_vma(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SFENCE_VMA_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 45);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 45);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 46: CSRRW */
    compile_ret_t __csrrw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 46);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 46);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 47: CSRRS */
    compile_ret_t __csrrs(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRS_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 47);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 47);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 48: CSRRC */
    compile_ret_t __csrrc(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRC_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 48);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 48);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 49: CSRRWI */
    compile_ret_t __csrrwi(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRWI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 49);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 49);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 50: CSRRSI */
    compile_ret_t __csrrsi(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRSI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 50);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 50);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
    }
    
    /* instruction 51: CSRRCI */
    compile_ret_t __csrrci(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRCI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC, 51);
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
        vm_base<ARCH>::gen_sync(tu, POST_SYNC, 51);
        gen_trap_check(tu);
        return std::make_tuple(CONT);
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
std::unique_ptr<vm_if> create<arch::tgf01>(arch::tgf01 *core, unsigned short port, bool dump) {
    auto ret = new tgf01::vm_impl<arch::tgf01>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret, port);
    return std::unique_ptr<vm_if>(ret);
}
}
} // namespace iss
