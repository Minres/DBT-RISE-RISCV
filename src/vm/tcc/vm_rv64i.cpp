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
#include <iss/arch/rv64i.h>
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/server.h>
#include <iss/iss.h>
#include <iss/tcc/vm_base.h>
#include <util/logging.h>
#include <sstream>
#include <iss/instruction_decoder.h>

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#include <array>
#include <iss/debugger/riscv_target_adapter.h>

namespace iss {
namespace tcc {
namespace rv64i {
using namespace iss::arch;
using namespace iss::debugger;

template <typename ARCH> class vm_impl : public iss::tcc::vm_base<ARCH> {
public:
    using traits = arch::traits<ARCH>;
    using super       = typename iss::tcc::vm_base<ARCH>;
    using virt_addr_t = typename super::virt_addr_t;
    using phys_addr_t = typename super::phys_addr_t;
    using code_word_t = typename super::code_word_t;
    using mem_type_e  = typename traits::mem_type_e;    
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

    inline const char *name(size_t index){return traits::reg_aliases.at(index);}

    void setup_module(std::string m) override {
        super::setup_module(m);
    }

    compile_ret_t gen_single_inst_behavior(virt_addr_t &, unsigned int &, tu_builder&) override;

    void gen_trap_behavior(tu_builder& tu) override;

    void gen_raise_trap(tu_builder& tu, uint16_t trap_id, uint16_t cause);

    void gen_leave_trap(tu_builder& tu, unsigned lvl);

    inline void gen_set_tval(tu_builder& tu, uint64_t new_tval);

    inline void gen_set_tval(tu_builder& tu, value new_tval);

    inline void gen_trap_check(tu_builder& tu) {
        tu("if(*trap_state!=0) goto trap_entry;");
    }

    inline void gen_set_pc(tu_builder& tu, virt_addr_t pc, unsigned reg_num) {
        switch(reg_num){
        case traits::NEXT_PC:
            tu("*next_pc = {:#x};", pc.val);
            break;
        case traits::PC:
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

    
    template<unsigned W, typename U, typename S = typename std::make_signed<U>::type>
    inline S sext(U from) {
        auto mask = (1ULL<<W) - 1;
        auto sign_mask = 1ULL<<(W-1);
        return (from & mask) | ((from & sign_mask) ? ~mask : 0);
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

    const std::array<instruction_descriptor, 61> instr_descr = {{
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
        {32, 0b00000000000000000001000000010011, 0b11111100000000000111000001111111, &this_class::__slli},
        /* instruction SRLI, encoding '0b00000000000000000101000000010011' */
        {32, 0b00000000000000000101000000010011, 0b11111100000000000111000001111111, &this_class::__srli},
        /* instruction SRAI, encoding '0b01000000000000000101000000010011' */
        {32, 0b01000000000000000101000000010011, 0b11111100000000000111000001111111, &this_class::__srai},
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
        /* instruction LWU, encoding '0b00000000000000000110000000000011' */
        {32, 0b00000000000000000110000000000011, 0b00000000000000000111000001111111, &this_class::__lwu},
        /* instruction LD, encoding '0b00000000000000000011000000000011' */
        {32, 0b00000000000000000011000000000011, 0b00000000000000000111000001111111, &this_class::__ld},
        /* instruction SD, encoding '0b00000000000000000011000000100011' */
        {32, 0b00000000000000000011000000100011, 0b00000000000000000111000001111111, &this_class::__sd},
        /* instruction ADDIW, encoding '0b00000000000000000000000000011011' */
        {32, 0b00000000000000000000000000011011, 0b00000000000000000111000001111111, &this_class::__addiw},
        /* instruction SLLIW, encoding '0b00000000000000000001000000011011' */
        {32, 0b00000000000000000001000000011011, 0b11111110000000000111000001111111, &this_class::__slliw},
        /* instruction SRLIW, encoding '0b00000000000000000101000000011011' */
        {32, 0b00000000000000000101000000011011, 0b11111110000000000111000001111111, &this_class::__srliw},
        /* instruction SRAIW, encoding '0b01000000000000000101000000011011' */
        {32, 0b01000000000000000101000000011011, 0b11111110000000000111000001111111, &this_class::__sraiw},
        /* instruction ADDW, encoding '0b00000000000000000000000000111011' */
        {32, 0b00000000000000000000000000111011, 0b11111110000000000111000001111111, &this_class::__addw},
        /* instruction SUBW, encoding '0b01000000000000000000000000111011' */
        {32, 0b01000000000000000000000000111011, 0b11111110000000000111000001111111, &this_class::__subw},
        /* instruction SLLW, encoding '0b00000000000000000001000000111011' */
        {32, 0b00000000000000000001000000111011, 0b11111110000000000111000001111111, &this_class::__sllw},
        /* instruction SRLW, encoding '0b00000000000000000101000000111011' */
        {32, 0b00000000000000000101000000111011, 0b11111110000000000111000001111111, &this_class::__srlw},
        /* instruction SRAW, encoding '0b01000000000000000101000000111011' */
        {32, 0b01000000000000000101000000111011, 0b11111110000000000111000001111111, &this_class::__sraw},
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
    }};

    //needs to be declared after instr_descr
    decoder instr_decoder;
 
    /* instruction definitions */
    /* instruction 0: LUI */
    compile_ret_t __lui(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LUI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,0);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint32_t imm = ((bit_sub<12,20>(instr) << 12));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "lui"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.constant((uint64_t)((int32_t)imm),64));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,0);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 1: AUIPC */
    compile_ret_t __auipc(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("AUIPC_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,1);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint32_t imm = ((bit_sub<12,20>(instr) << 12));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#08x}", fmt::arg("mnemonic", "auipc"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.constant((uint64_t)(PC+(int32_t)imm),64));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,1);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 2: JAL */
    compile_ret_t __jal(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("JAL_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,2);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint32_t imm = ((bit_sub<12,8>(instr) << 12) | (bit_sub<20,1>(instr) << 11) | (bit_sub<21,10>(instr) << 1) | (bit_sub<31,1>(instr) << 20));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#0x}", fmt::arg("mnemonic", "jal"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(NO_JUMP),32));
        if(rd>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto new_pc = (uint64_t)(PC+(int32_t)sext<21>(imm));
        	if(new_pc%static_cast<uint32_t>(traits:: INSTR_ALIGNMENT)){ this->gen_set_tval(tu, new_pc);
        	this->gen_raise_trap(tu, 0, 0);
        	}
        	else{
        		if(rd!=0) {
        		    tu.store(rd + traits::X0, tu.constant((uint64_t)(PC+4),64));
        		}
        		auto PC_val_v = tu.assignment("PC_val", new_pc,64);
        		tu.store(traits::NEXT_PC, PC_val_v);
        		tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(KNOWN_JUMP), 2));
        	}
        }
        auto returnValue = std::make_tuple(BRANCH);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,2);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 3: JALR */
    compile_ret_t __jalr(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("JALR_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,3);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm:#0x}", fmt::arg("mnemonic", "jalr"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(NO_JUMP),32));
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto addr_mask = (uint64_t)- 2;
        	auto new_pc = tu.assignment(tu.ext((tu.bitwise_and(
        	   (tu.add(
        	      tu.load(rs1 + traits::X0, 0),
        	      tu.constant((int16_t)sext<12>(imm),16))),
        	   tu.constant(addr_mask,64))),64,false),64);
        	tu.open_if(tu.urem(
        	   new_pc,
        	   tu.constant(static_cast<uint32_t>(traits:: INSTR_ALIGNMENT),32)));
        	this->gen_set_tval(tu, new_pc);
        	this->gen_raise_trap(tu, 0, 0);
        	tu.open_else();
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.constant((uint64_t)(PC+4),64));
        	}
        	auto PC_val_v = tu.assignment("PC_val", new_pc,64);
        	tu.store(traits::NEXT_PC, PC_val_v);
        	tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(UNKNOWN_JUMP), 2));
        	tu.close_scope();
        }
        auto returnValue = std::make_tuple(BRANCH);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,3);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 4: BEQ */
    compile_ret_t __beq(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BEQ_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,4);
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "beq"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(NO_JUMP),32));
        if(rs2>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	tu.open_if(tu.icmp(ICmpInst::ICMP_EQ,
        	   tu.load(rs1 + traits::X0, 0),
        	   tu.load(rs2 + traits::X0, 0)));
        	auto new_pc = (uint64_t)(PC+(int16_t)sext<13>(imm));
        	if(new_pc%static_cast<uint32_t>(traits:: INSTR_ALIGNMENT)){ this->gen_set_tval(tu, new_pc);
        	this->gen_raise_trap(tu, 0, 0);
        	}
        	else{
        		auto PC_val_v = tu.assignment("PC_val", new_pc,64);
        		tu.store(traits::NEXT_PC, PC_val_v);
        		tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(KNOWN_JUMP), 2));
        	}
        	tu.close_scope();
        }
        auto returnValue = std::make_tuple(BRANCH);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,4);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 5: BNE */
    compile_ret_t __bne(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BNE_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,5);
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bne"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(NO_JUMP),32));
        if(rs2>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	tu.open_if(tu.icmp(ICmpInst::ICMP_NE,
        	   tu.load(rs1 + traits::X0, 0),
        	   tu.load(rs2 + traits::X0, 0)));
        	auto new_pc = (uint64_t)(PC+(int16_t)sext<13>(imm));
        	if(new_pc%static_cast<uint32_t>(traits:: INSTR_ALIGNMENT)){ this->gen_set_tval(tu, new_pc);
        	this->gen_raise_trap(tu, 0, 0);
        	}
        	else{
        		auto PC_val_v = tu.assignment("PC_val", new_pc,64);
        		tu.store(traits::NEXT_PC, PC_val_v);
        		tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(KNOWN_JUMP), 2));
        	}
        	tu.close_scope();
        }
        auto returnValue = std::make_tuple(BRANCH);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,5);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 6: BLT */
    compile_ret_t __blt(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BLT_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,6);
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "blt"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(NO_JUMP),32));
        if(rs2>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	tu.open_if(tu.icmp(ICmpInst::ICMP_SLT,
        	   tu.ext(tu.load(rs1 + traits::X0, 0),64,true),
        	   tu.ext(tu.load(rs2 + traits::X0, 0),64,true)));
        	auto new_pc = (uint64_t)(PC+(int16_t)sext<13>(imm));
        	if(new_pc%static_cast<uint32_t>(traits:: INSTR_ALIGNMENT)){ this->gen_set_tval(tu, new_pc);
        	this->gen_raise_trap(tu, 0, 0);
        	}
        	else{
        		auto PC_val_v = tu.assignment("PC_val", new_pc,64);
        		tu.store(traits::NEXT_PC, PC_val_v);
        		tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(KNOWN_JUMP), 2));
        	}
        	tu.close_scope();
        }
        auto returnValue = std::make_tuple(BRANCH);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,6);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 7: BGE */
    compile_ret_t __bge(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BGE_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,7);
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bge"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(NO_JUMP),32));
        if(rs2>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	tu.open_if(tu.icmp(ICmpInst::ICMP_SGE,
        	   tu.ext(tu.load(rs1 + traits::X0, 0),64,true),
        	   tu.ext(tu.load(rs2 + traits::X0, 0),64,true)));
        	auto new_pc = (uint64_t)(PC+(int16_t)sext<13>(imm));
        	if(new_pc%static_cast<uint32_t>(traits:: INSTR_ALIGNMENT)){ this->gen_set_tval(tu, new_pc);
        	this->gen_raise_trap(tu, 0, 0);
        	}
        	else{
        		auto PC_val_v = tu.assignment("PC_val", new_pc,64);
        		tu.store(traits::NEXT_PC, PC_val_v);
        		tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(KNOWN_JUMP), 2));
        	}
        	tu.close_scope();
        }
        auto returnValue = std::make_tuple(BRANCH);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,7);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 8: BLTU */
    compile_ret_t __bltu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BLTU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,8);
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bltu"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(NO_JUMP),32));
        if(rs2>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	tu.open_if(tu.icmp(ICmpInst::ICMP_ULT,
        	   tu.load(rs1 + traits::X0, 0),
        	   tu.load(rs2 + traits::X0, 0)));
        	auto new_pc = (uint64_t)(PC+(int16_t)sext<13>(imm));
        	if(new_pc%static_cast<uint32_t>(traits:: INSTR_ALIGNMENT)){ this->gen_set_tval(tu, new_pc);
        	this->gen_raise_trap(tu, 0, 0);
        	}
        	else{
        		auto PC_val_v = tu.assignment("PC_val", new_pc,64);
        		tu.store(traits::NEXT_PC, PC_val_v);
        		tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(KNOWN_JUMP), 2));
        	}
        	tu.close_scope();
        }
        auto returnValue = std::make_tuple(BRANCH);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,8);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 9: BGEU */
    compile_ret_t __bgeu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("BGEU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,9);
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bgeu"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(NO_JUMP),32));
        if(rs2>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	tu.open_if(tu.icmp(ICmpInst::ICMP_UGE,
        	   tu.load(rs1 + traits::X0, 0),
        	   tu.load(rs2 + traits::X0, 0)));
        	auto new_pc = (uint64_t)(PC+(int16_t)sext<13>(imm));
        	if(new_pc%static_cast<uint32_t>(traits:: INSTR_ALIGNMENT)){ this->gen_set_tval(tu, new_pc);
        	this->gen_raise_trap(tu, 0, 0);
        	}
        	else{
        		auto PC_val_v = tu.assignment("PC_val", new_pc,64);
        		tu.store(traits::NEXT_PC, PC_val_v);
        		tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(KNOWN_JUMP), 2));
        	}
        	tu.close_scope();
        }
        auto returnValue = std::make_tuple(BRANCH);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,9);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 10: LB */
    compile_ret_t __lb(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LB_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,10);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lb"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto load_address = tu.assignment(tu.ext((tu.add(
        	   tu.load(rs1 + traits::X0, 0),
        	   tu.constant((int16_t)sext<12>(imm),16))),64,false),64);
        	auto res = tu.assignment(tu.ext(tu.read_mem(traits::MEM, load_address, 8),8,true),8);
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.ext(res,64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,10);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 11: LH */
    compile_ret_t __lh(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LH_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,11);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lh"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto load_address = tu.assignment(tu.ext((tu.add(
        	   tu.load(rs1 + traits::X0, 0),
        	   tu.constant((int16_t)sext<12>(imm),16))),64,false),64);
        	auto res = tu.assignment(tu.ext(tu.read_mem(traits::MEM, load_address, 16),16,true),16);
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.ext(res,64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,11);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 12: LW */
    compile_ret_t __lw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,12);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lw"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto load_address = tu.assignment(tu.ext((tu.add(
        	   tu.load(rs1 + traits::X0, 0),
        	   tu.constant((int16_t)sext<12>(imm),16))),64,false),64);
        	auto res = tu.assignment(tu.ext(tu.read_mem(traits::MEM, load_address, 32),32,true),32);
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.ext(res,64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,12);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 13: LBU */
    compile_ret_t __lbu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LBU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,13);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lbu"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto load_address = tu.assignment(tu.ext((tu.add(
        	   tu.load(rs1 + traits::X0, 0),
        	   tu.constant((int16_t)sext<12>(imm),16))),64,false),64);
        	auto res = tu.assignment(tu.read_mem(traits::MEM, load_address, 8),8);
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.ext(res,64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,13);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 14: LHU */
    compile_ret_t __lhu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LHU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,14);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lhu"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto load_address = tu.assignment(tu.ext((tu.add(
        	   tu.load(rs1 + traits::X0, 0),
        	   tu.constant((int16_t)sext<12>(imm),16))),64,false),64);
        	auto res = tu.assignment(tu.read_mem(traits::MEM, load_address, 16),16);
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.ext(res,64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,14);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 15: SB */
    compile_ret_t __sb(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SB_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,15);
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sb"),
                fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rs2>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto store_address = tu.assignment(tu.ext((tu.add(
        	   tu.load(rs1 + traits::X0, 0),
        	   tu.constant((int16_t)sext<12>(imm),16))),64,false),64);
        	tu.write_mem(traits::MEM, store_address, tu.ext(tu.load(rs2 + traits::X0, 0),8,false));
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,15);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 16: SH */
    compile_ret_t __sh(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SH_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,16);
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sh"),
                fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rs2>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto store_address = tu.assignment(tu.ext((tu.add(
        	   tu.load(rs1 + traits::X0, 0),
        	   tu.constant((int16_t)sext<12>(imm),16))),64,false),64);
        	tu.write_mem(traits::MEM, store_address, tu.ext(tu.load(rs2 + traits::X0, 0),16,false));
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,16);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 17: SW */
    compile_ret_t __sw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,17);
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sw"),
                fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rs2>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto store_address = tu.assignment(tu.ext((tu.add(
        	   tu.load(rs1 + traits::X0, 0),
        	   tu.constant((int16_t)sext<12>(imm),16))),64,false),64);
        	tu.write_mem(traits::MEM, store_address, tu.ext(tu.load(rs2 + traits::X0, 0),32,false));
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,17);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 18: ADDI */
    compile_ret_t __addi(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("ADDI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,18);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "addi"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.ext((tu.add(
        	       tu.load(rs1 + traits::X0, 0),
        	       tu.constant((int16_t)sext<12>(imm),16))),64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,18);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 19: SLTI */
    compile_ret_t __slti(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLTI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,19);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "slti"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.conditionalAssignment((tu.icmp(ICmpInst::ICMP_SLT,
        	       tu.ext(tu.load(rs1 + traits::X0, 0),64,true),
        	       tu.constant((int16_t)sext<12>(imm),16))), tu.constant(1,8),tu.constant(0,8)));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,19);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 20: SLTIU */
    compile_ret_t __sltiu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLTIU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,20);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "sltiu"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.conditionalAssignment((tu.icmp(ICmpInst::ICMP_ULT,
        	       tu.load(rs1 + traits::X0, 0),
        	       tu.constant((uint64_t)((int16_t)sext<12>(imm)),64))), tu.constant(1,8),tu.constant(0,8)));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,20);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 21: XORI */
    compile_ret_t __xori(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("XORI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,21);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "xori"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.bitwise_xor(
        	       tu.load(rs1 + traits::X0, 0),
        	       tu.constant((uint64_t)((int16_t)sext<12>(imm)),64)));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,21);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 22: ORI */
    compile_ret_t __ori(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("ORI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,22);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "ori"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.bitwise_or(
        	       tu.load(rs1 + traits::X0, 0),
        	       tu.constant((uint64_t)((int16_t)sext<12>(imm)),64)));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,22);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 23: ANDI */
    compile_ret_t __andi(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("ANDI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,23);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "andi"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.bitwise_and(
        	       tu.load(rs1 + traits::X0, 0),
        	       tu.constant((uint64_t)((int16_t)sext<12>(imm)),64)));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,23);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 24: SLLI */
    compile_ret_t __slli(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLLI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,24);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,6>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "slli"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.shl(
        	       tu.load(rs1 + traits::X0, 0),
        	       tu.constant(shamt,8)));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,24);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 25: SRLI */
    compile_ret_t __srli(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRLI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,25);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,6>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srli"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.lshr(
        	       tu.load(rs1 + traits::X0, 0),
        	       tu.constant(shamt,8)));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,25);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 26: SRAI */
    compile_ret_t __srai(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRAI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,26);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,6>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srai"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.ext((tu.ashr(
        	       (tu.ext(tu.load(rs1 + traits::X0, 0),64,true)),
        	       tu.constant(shamt,8))),64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,26);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 27: ADD */
    compile_ret_t __add(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("ADD_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,27);
        uint64_t PC = pc.val;
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
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)||rs2>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.ext((tu.add(
        	       tu.load(rs1 + traits::X0, 0),
        	       tu.load(rs2 + traits::X0, 0))),64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,27);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 28: SUB */
    compile_ret_t __sub(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SUB_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,28);
        uint64_t PC = pc.val;
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
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)||rs2>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.ext((tu.sub(
        	       tu.load(rs1 + traits::X0, 0),
        	       tu.load(rs2 + traits::X0, 0))),64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,28);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 29: SLL */
    compile_ret_t __sll(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLL_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,29);
        uint64_t PC = pc.val;
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
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)||rs2>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.shl(
        	       tu.load(rs1 + traits::X0, 0),
        	       (tu.bitwise_and(
        	          tu.load(rs2 + traits::X0, 0),
        	          tu.constant((static_cast<uint32_t>(traits:: XLEN)-1),64)))));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,29);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 30: SLT */
    compile_ret_t __slt(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLT_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,30);
        uint64_t PC = pc.val;
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
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)||rs2>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.conditionalAssignment(tu.icmp(ICmpInst::ICMP_SLT,
        	       tu.ext(tu.load(rs1 + traits::X0, 0),64,true),
        	       tu.ext(tu.load(rs2 + traits::X0, 0),64,true)), tu.constant(1,8),tu.constant(0,8)));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,30);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 31: SLTU */
    compile_ret_t __sltu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLTU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,31);
        uint64_t PC = pc.val;
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
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)||rs2>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.conditionalAssignment(tu.icmp(ICmpInst::ICMP_ULT,
        	       tu.load(rs1 + traits::X0, 0),
        	       tu.load(rs2 + traits::X0, 0)), tu.constant(1,8),tu.constant(0,8)));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,31);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 32: XOR */
    compile_ret_t __xor(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("XOR_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,32);
        uint64_t PC = pc.val;
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
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)||rs2>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.bitwise_xor(
        	       tu.load(rs1 + traits::X0, 0),
        	       tu.load(rs2 + traits::X0, 0)));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,32);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 33: SRL */
    compile_ret_t __srl(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRL_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,33);
        uint64_t PC = pc.val;
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
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)||rs2>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.lshr(
        	       tu.load(rs1 + traits::X0, 0),
        	       (tu.bitwise_and(
        	          tu.load(rs2 + traits::X0, 0),
        	          tu.constant((static_cast<uint32_t>(traits:: XLEN)-1),64)))));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,33);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 34: SRA */
    compile_ret_t __sra(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRA_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,34);
        uint64_t PC = pc.val;
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
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)||rs2>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.ext((tu.ashr(
        	       tu.ext(tu.load(rs1 + traits::X0, 0),64,true),
        	       (tu.bitwise_and(
        	          tu.load(rs2 + traits::X0, 0),
        	          tu.constant((static_cast<uint32_t>(traits:: XLEN)-1),64))))),64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,34);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 35: OR */
    compile_ret_t __or(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("OR_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,35);
        uint64_t PC = pc.val;
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
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)||rs2>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.bitwise_or(
        	       tu.load(rs1 + traits::X0, 0),
        	       tu.load(rs2 + traits::X0, 0)));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,35);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 36: AND */
    compile_ret_t __and(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("AND_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,36);
        uint64_t PC = pc.val;
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
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)||rs2>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.bitwise_and(
        	       tu.load(rs1 + traits::X0, 0),
        	       tu.load(rs2 + traits::X0, 0)));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,36);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 37: FENCE */
    compile_ret_t __fence(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("FENCE_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,37);
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
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        tu.write_mem(traits::FENCE, static_cast<uint32_t>(traits:: fence), tu.constant((uint8_t)pred<<4|succ,8));
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,37);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 38: ECALL */
    compile_ret_t __ecall(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("ECALL_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,38);
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "ecall";
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(NO_JUMP),32));
        this->gen_raise_trap(tu, 0, 11);
        auto returnValue = std::make_tuple(TRAP);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,38);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 39: EBREAK */
    compile_ret_t __ebreak(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("EBREAK_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,39);
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "ebreak";
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(NO_JUMP),32));
        this->gen_raise_trap(tu, 0, 3);
        auto returnValue = std::make_tuple(TRAP);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,39);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 40: MRET */
    compile_ret_t __mret(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("MRET_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,40);
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "mret";
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(NO_JUMP),32));
        this->gen_leave_trap(tu, 3);
        auto returnValue = std::make_tuple(TRAP);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,40);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 41: WFI */
    compile_ret_t __wfi(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("WFI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,41);
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "wfi";
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        tu.callf("wait", tu.constant(1,8));
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,41);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 42: LWU */
    compile_ret_t __lwu(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LWU_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,42);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lwu"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto offs = tu.assignment(tu.ext((tu.add(
        	   tu.load(rs1 + traits::X0, 0),
        	   tu.constant((int16_t)sext<12>(imm),16))),64,false),64);
        	auto res = tu.assignment(tu.ext(tu.read_mem(traits::MEM, offs, 32),32,false),32);
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.ext(res,64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,42);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 43: LD */
    compile_ret_t __ld(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("LD_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,43);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "ld"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto offs = tu.assignment(tu.ext((tu.add(
        	   tu.load(rs1 + traits::X0, 0),
        	   tu.constant((int16_t)sext<12>(imm),16))),64,false),64);
        	auto res = tu.assignment(tu.ext(tu.read_mem(traits::MEM, offs, 64),64,true),64);
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, tu.ext(res,64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,43);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 44: SD */
    compile_ret_t __sd(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SD_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,44);
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sd"),
                fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rs2>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto offs = tu.assignment(tu.ext((tu.add(
        	   tu.load(rs1 + traits::X0, 0),
        	   tu.constant((int16_t)sext<12>(imm),16))),64,false),64);
        	tu.write_mem(traits::MEM, offs, tu.ext(tu.load(rs2 + traits::X0, 0),64,false));
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,44);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 45: ADDIW */
    compile_ret_t __addiw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("ADDIW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,45);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "addiw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0){ auto res = tu.assignment(tu.ext((tu.add(
        	   tu.load(rs1 + traits::X0, 0),
        	   tu.constant((int16_t)sext<12>(imm),16))),32,true),32);
        	tu.store(rd + traits::X0, tu.ext(res,64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,45);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 46: SLLIW */
    compile_ret_t __slliw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLLIW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,46);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "slliw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0){ auto sh_val = tu.assignment(tu.shl(
        	   (tu.ext(tu.load(rs1 + traits::X0, 0),32,false)),
        	   tu.constant(shamt,8)),32);
        	tu.store(rd + traits::X0, tu.ext(tu.ext(sh_val,32,true),64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,46);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 47: SRLIW */
    compile_ret_t __srliw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRLIW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,47);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srliw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0){ auto sh_val = tu.assignment(tu.lshr(
        	   (tu.ext(tu.load(rs1 + traits::X0, 0),32,false)),
        	   tu.constant(shamt,8)),32);
        	tu.store(rd + traits::X0, tu.ext(tu.ext(sh_val,32,true),64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,47);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 48: SRAIW */
    compile_ret_t __sraiw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRAIW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,48);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "sraiw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0){ auto sh_val = tu.assignment(tu.ashr(
        	   (tu.ext(tu.load(rs1 + traits::X0, 0),32,true)),
        	   tu.constant(shamt,8)),32);
        	tu.store(rd + traits::X0, tu.ext(sh_val,64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,48);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 49: ADDW */
    compile_ret_t __addw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("ADDW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,49);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "addw";
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)||rs2>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0){ auto res = tu.assignment(tu.ext((tu.add(
        	   tu.ext(tu.load(rs1 + traits::X0, 0),32,true),
        	   tu.ext(tu.load(rs2 + traits::X0, 0),32,true))),32,true),32);
        	tu.store(rd + traits::X0, tu.ext(res,64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,49);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 50: SUBW */
    compile_ret_t __subw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SUBW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,50);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "subw";
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)||rs2>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0){ auto res = tu.assignment(tu.ext((tu.sub(
        	   tu.ext(tu.load(rs1 + traits::X0, 0),32,true),
        	   tu.ext(tu.load(rs2 + traits::X0, 0),32,true))),32,true),32);
        	tu.store(rd + traits::X0, tu.ext(res,64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,50);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 51: SLLW */
    compile_ret_t __sllw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SLLW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,51);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sllw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)||rs2>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0){ auto count = tu.assignment(tu.bitwise_and(
        	   tu.ext(tu.load(rs2 + traits::X0, 0),32,false),
        	   tu.constant(31,8)),32);
        	auto sh_val = tu.assignment(tu.shl(
        	   (tu.ext(tu.load(rs1 + traits::X0, 0),32,false)),
        	   count),32);
        	tu.store(rd + traits::X0, tu.ext(tu.ext(sh_val,32,true),64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,51);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 52: SRLW */
    compile_ret_t __srlw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRLW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,52);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "srlw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)||rs2>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0){ auto count = tu.assignment(tu.bitwise_and(
        	   tu.ext(tu.load(rs2 + traits::X0, 0),32,false),
        	   tu.constant(31,8)),32);
        	auto sh_val = tu.assignment(tu.lshr(
        	   (tu.ext(tu.load(rs1 + traits::X0, 0),32,false)),
        	   count),32);
        	tu.store(rd + traits::X0, tu.ext(tu.ext(sh_val,32,true),64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,52);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 53: SRAW */
    compile_ret_t __sraw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("SRAW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,53);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sraw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)||rs2>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	if(rd!=0){ auto count = tu.assignment(tu.bitwise_and(
        	   tu.ext(tu.load(rs2 + traits::X0, 0),32,false),
        	   tu.constant(31,8)),32);
        	auto sh_val = tu.assignment(tu.ashr(
        	   (tu.ext(tu.load(rs1 + traits::X0, 0),32,true)),
        	   count),32);
        	tu.store(rd + traits::X0, tu.ext(sh_val,64,false));
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,53);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 54: CSRRW */
    compile_ret_t __csrrw(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRW_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,54);
        uint64_t PC = pc.val;
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
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto xrs1 = tu.assignment(tu.load(rs1 + traits::X0, 0),64);
        	if(rd!=0){ auto xrd = tu.assignment(tu.read_mem(traits::CSR, csr, 64),64);
        	tu.write_mem(traits::CSR, csr, xrs1);
        	tu.store(rd + traits::X0, xrd);
        	}
        	else{
        		tu.write_mem(traits::CSR, csr, xrs1);
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,54);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 55: CSRRS */
    compile_ret_t __csrrs(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRS_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,55);
        uint64_t PC = pc.val;
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
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto xrd = tu.assignment(tu.read_mem(traits::CSR, csr, 64),64);
        	auto xrs1 = tu.assignment(tu.load(rs1 + traits::X0, 0),64);
        	if(rs1!=0) {
        	    tu.write_mem(traits::CSR, csr, tu.bitwise_or(
        	       xrd,
        	       xrs1));
        	}
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, xrd);
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,55);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 56: CSRRC */
    compile_ret_t __csrrc(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRC_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,56);
        uint64_t PC = pc.val;
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
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)||rs1>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto xrd = tu.assignment(tu.read_mem(traits::CSR, csr, 64),64);
        	auto xrs1 = tu.assignment(tu.load(rs1 + traits::X0, 0),64);
        	if(rs1!=0) {
        	    tu.write_mem(traits::CSR, csr, tu.bitwise_and(
        	       xrd,
        	       tu.logical_neg(xrs1)));
        	}
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, xrd);
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,56);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 57: CSRRWI */
    compile_ret_t __csrrwi(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRWI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,57);
        uint64_t PC = pc.val;
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
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto xrd = tu.assignment(tu.read_mem(traits::CSR, csr, 64),64);
        	tu.write_mem(traits::CSR, csr, tu.constant((uint64_t)zimm,64));
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, xrd);
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,57);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 58: CSRRSI */
    compile_ret_t __csrrsi(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRSI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,58);
        uint64_t PC = pc.val;
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
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto xrd = tu.assignment(tu.read_mem(traits::CSR, csr, 64),64);
        	if(zimm!=0) {
        	    tu.write_mem(traits::CSR, csr, tu.bitwise_or(
        	       xrd,
        	       tu.constant((uint64_t)zimm,64)));
        	}
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, xrd);
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,58);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 59: CSRRCI */
    compile_ret_t __csrrci(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("CSRRCI_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,59);
        uint64_t PC = pc.val;
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
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        if(rd>=static_cast<uint32_t>(traits:: RFS)) {
            this->gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
        	auto xrd = tu.assignment(tu.read_mem(traits::CSR, csr, 64),64);
        	if(zimm!=0) {
        	    tu.write_mem(traits::CSR, csr, tu.bitwise_and(
        	       xrd,
        	       tu.constant(~ ((uint64_t)zimm),64)));
        	}
        	if(rd!=0) {
        	    tu.store(rd + traits::X0, xrd);
        	}
        }
        auto returnValue = std::make_tuple(CONT);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,59);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /* instruction 60: FENCE_I */
    compile_ret_t __fence_i(virt_addr_t& pc, code_word_t instr, tu_builder& tu){
        tu("FENCE_I_{:#010x}:", pc.val);
        vm_base<ARCH>::gen_sync(tu, PRE_SYNC,60);
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rd}, {imm}", fmt::arg("mnemonic", "fence_i"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, mnemonic);
        }
        auto cur_pc_val = tu.constant(pc.val, traits::reg_bit_widths[traits::PC]);
        pc=pc+ 4;
        gen_set_pc(tu, pc, traits::NEXT_PC);
        tu.open_scope();
        this->gen_set_tval(tu, instr);
        tu.write_mem(traits::FENCE, static_cast<uint32_t>(traits:: fencei), tu.constant(imm,16));
        auto returnValue = std::make_tuple(FLUSH);
        
        tu.close_scope();
        vm_base<ARCH>::gen_sync(tu, POST_SYNC,60);
        gen_trap_check(tu);        
        return returnValue;
    }
    
    /****************************************************************************
     * end opcode definitions
     ****************************************************************************/
    compile_ret_t illegal_instruction(virt_addr_t &pc, code_word_t instr, tu_builder& tu) {
        vm_impl::gen_sync(tu, iss::PRE_SYNC, instr_descr.size());
        if(this->disass_enabled){
            /* generate console output when executing the command */
            tu("print_disass(core_ptr, {:#x}, \"{}\");", pc.val, std::string("illegal_instruction"));
        }
        pc = pc + ((instr & 3) == 3 ? 4 : 2);
        gen_raise_trap(tu, 0, static_cast<int32_t>(traits:: RV_CAUSE_ILLEGAL_INSTRUCTION));
        this->gen_set_tval(tu, instr);
        vm_impl::gen_sync(tu, iss::POST_SYNC, instr_descr.size());
        vm_impl::gen_trap_check(tu);
        return ILLEGAL_INSTR;
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
std::tuple<continuation_e>
vm_impl<ARCH>::gen_single_inst_behavior(virt_addr_t &pc, unsigned int &inst_cnt, tu_builder& tu) {
    // we fetch at max 4 byte, alignment is 2
    enum {TRAP_ID=1<<16};
    code_word_t instr = 0;
    phys_addr_t paddr(pc);
    if(this->core.has_mmu())
        paddr = this->core.virt2phys(pc);
        auto res = this->core.read(paddr, 4, reinterpret_cast<uint8_t*>(&instr));
    if (res != iss::Ok)
        return ILLEGAL_FETCH;
    if (instr == 0x0000006f || (instr&0xffff)==0xa001) 
        return JUMP_TO_SELF;
    ++inst_cnt;
    uint32_t inst_index = instr_decoder.decode_instr(instr);
    compile_func f = nullptr;
    if(inst_index < instr_descr.size())
        f = instr_descr[inst_index].op;
    if (f == nullptr) {
        f = &this_class::illegal_instruction;
    }
    return (this->*f)(pc, instr, tu);
}

template <typename ARCH> void vm_impl<ARCH>::gen_raise_trap(tu_builder& tu, uint16_t trap_id, uint16_t cause) {
    tu("  *trap_state = {:#x};", 0x80 << 24 | (cause << 16) | trap_id);
}

template <typename ARCH> void vm_impl<ARCH>::gen_leave_trap(tu_builder& tu, unsigned lvl) {
    tu("leave_trap(core_ptr, {});", lvl);
    tu.store(traits::NEXT_PC, tu.read_mem(traits::CSR, (lvl << 8) + 0x41, traits::XLEN));
    tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(UNKNOWN_JUMP), 32));
}

template <typename ARCH> void vm_impl<ARCH>::gen_set_tval(tu_builder& tu, uint64_t new_tval) {
    tu(fmt::format("tval = {};", new_tval));
}
template <typename ARCH> void vm_impl<ARCH>::gen_set_tval(tu_builder& tu, value new_tval) {
    tu(fmt::format("tval = {};", new_tval.str));
}

template <typename ARCH> void vm_impl<ARCH>::gen_trap_behavior(tu_builder& tu) {
    tu("trap_entry:");
    this->gen_sync(tu, POST_SYNC, -1);    
    tu("enter_trap(core_ptr, *trap_state, *pc, tval);");
    tu.store(traits::LAST_BRANCH, tu.constant(static_cast<int>(UNKNOWN_JUMP),32));
    tu("return *next_pc;");
}


} // namespace rv64i

template <>
std::unique_ptr<vm_if> create<arch::rv64i>(arch::rv64i *core, unsigned short port, bool dump) {
    auto ret = new rv64i::vm_impl<arch::rv64i>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret, port);
    return std::unique_ptr<vm_if>(ret);
}
} // namesapce tcc
} // namespace iss

#include <iss/arch/riscv_hart_m_p.h>
#include <iss/arch/riscv_hart_mu_p.h>
#include <iss/factory.h>
namespace iss {
namespace {
volatile std::array<bool, 2> dummy = {
        core_factory::instance().register_creator("rv64i|m_p|tcc", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_m_p<iss::arch::rv64i>();
		    auto vm = new tcc::rv64i::vm_impl<arch::rv64i>(*cpu, false);
		    if (port != 0) debugger::server<debugger::gdb_session>::run_server(vm, port);
            if(init_data){
                auto* cb = reinterpret_cast<semihosting_cb_t<arch::traits<arch::rv64i>::reg_t>*>(init_data);
                cpu->set_semihosting_callback(*cb);
            }
            return {cpu_ptr{cpu}, vm_ptr{vm}};
        }),
        core_factory::instance().register_creator("rv64i|mu_p|tcc", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_mu_p<iss::arch::rv64i>();
		    auto vm = new tcc::rv64i::vm_impl<arch::rv64i>(*cpu, false);
		    if (port != 0) debugger::server<debugger::gdb_session>::run_server(vm, port);
            if(init_data){
                auto* cb = reinterpret_cast<semihosting_cb_t<arch::traits<arch::rv64i>::reg_t>*>(init_data);
                cpu->set_semihosting_callback(*cb);
            }
            return {cpu_ptr{cpu}, vm_ptr{vm}};
        })
};
}
}
// clang-format on
