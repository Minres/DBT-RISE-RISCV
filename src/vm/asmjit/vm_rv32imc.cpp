/*******************************************************************************
 * Copyright (C) 2017, 2023 MINRES Technologies GmbH
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
#include <iss/arch/rv32imc.h>
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/server.h>
#include <iss/iss.h>
#include <iss/asmjit/vm_base.h>
#include <asmjit/asmjit.h>
#include <util/logging.h>

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#include <array>
#include <iss/debugger/riscv_target_adapter.h>

namespace iss {
namespace asmjit {


namespace rv32imc {
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
using super::get_reg;
    using super::get_reg_for;
    using super::load_reg_from_mem;
    using super::write_reg_to_mem;
    using super::gen_ext;
    using super::gen_read_mem;
    using super::gen_write_mem;
    using super::gen_wait;
    using super::gen_leave;
    using super::gen_operation;
   
    using this_class = vm_impl<ARCH>;
    using compile_func = continuation_e (this_class::*)(virt_addr_t&, code_word_t, jit_holder&);

    continuation_e gen_single_inst_behavior(virt_addr_t&, unsigned int &, jit_holder&) override;
    void gen_block_prologue(jit_holder& jh) override;
    void gen_block_epilogue(jit_holder& jh) override;
    inline const char *name(size_t index){return traits::reg_aliases.at(index);}

    void gen_instr_prologue(jit_holder& jh);
    void gen_instr_epilogue(jit_holder& jh);
    inline void gen_raise(jit_holder& jh, uint16_t trap_id, uint16_t cause);

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
        size_t length;
        uint32_t value;
        uint32_t mask;
        compile_func op;
    };
    struct decoding_tree_node{
        std::vector<instruction_descriptor> instrs;
        std::vector<decoding_tree_node*> children;
        uint32_t submask = std::numeric_limits<uint32_t>::max();
        uint32_t value;
        decoding_tree_node(uint32_t value) : value(value){}
    };

    decoding_tree_node* root {nullptr};

    const std::array<instruction_descriptor, 87> instr_descr = {{
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
    }};
 
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
        this->gen_sync(jh, PRE_SYNC, 0);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     (uint32_t)((int32_t)imm));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 0);
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
        this->gen_sync(jh, PRE_SYNC, 1);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     (uint32_t)(PC+(int32_t)imm));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 1);
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
        this->gen_sync(jh, PRE_SYNC, 2);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(imm%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                gen_raise(jh, 0, 0);
            }
            else{
                if(rd!=0){
                    cc.mov(get_ptr_for(jh, traits::X0+ rd),
                         (uint32_t)(PC+4));
                }
                auto PC_val_v = (uint32_t)(PC+(int32_t)sext<21>(imm));
                cc.mov(jh.next_pc, PC_val_v);
                cc.mov(get_ptr_for(jh, traits::LAST_BRANCH), 32U);
            }
        }
        auto returnValue = BRANCH;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 2);
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
        this->gen_sync(jh, PRE_SYNC, 3);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto addr_mask = (uint32_t)- 2;
            auto new_pc = gen_ext(jh, 
                (gen_operation(jh, band, (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), addr_mask)
                ), 32, true);
            auto label_merge = cc.newLabel();
            cc.cmp(gen_operation(jh, urem, new_pc, static_cast<uint32_t>(traits::INSTR_ALIGNMENT))
            ,0);
            auto label_else = cc.newLabel();
            cc.je(label_else);
            {
                gen_raise(jh, 0, 0);
            }
            cc.jmp(label_merge);
            cc.bind(label_else);
                {
                    if(rd!=0){
                        cc.mov(get_ptr_for(jh, traits::X0+ rd),
                             (uint32_t)(PC+4));
                    }
                    auto PC_val_v = new_pc;
                    cc.mov(jh.next_pc, PC_val_v);
                    cc.mov(get_ptr_for(jh, traits::LAST_BRANCH), 32U);
                }
            cc.bind(label_merge);
        }
        auto returnValue = BRANCH;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 3);
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
        this->gen_sync(jh, PRE_SYNC, 4);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto label_merge = cc.newLabel();
            cc.cmp(gen_operation(jh, eq, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
            ,0);
            cc.je(label_merge);
            {
                if(imm%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_raise(jh, 0, 0);
                }
                else{
                    auto PC_val_v = (uint32_t)(PC+(int16_t)sext<13>(imm));
                    cc.mov(jh.next_pc, PC_val_v);
                    cc.mov(get_ptr_for(jh, traits::LAST_BRANCH), 32U);
                }
            }
            cc.bind(label_merge);
        }
        auto returnValue = BRANCH;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 4);
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
        this->gen_sync(jh, PRE_SYNC, 5);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto label_merge = cc.newLabel();
            cc.cmp(gen_operation(jh, ne, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
            ,0);
            cc.je(label_merge);
            {
                if(imm%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_raise(jh, 0, 0);
                }
                else{
                    auto PC_val_v = (uint32_t)(PC+(int16_t)sext<13>(imm));
                    cc.mov(jh.next_pc, PC_val_v);
                    cc.mov(get_ptr_for(jh, traits::LAST_BRANCH), 32U);
                }
            }
            cc.bind(label_merge);
        }
        auto returnValue = BRANCH;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 5);
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
        this->gen_sync(jh, PRE_SYNC, 6);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto label_merge = cc.newLabel();
            cc.cmp(gen_operation(jh, lt, gen_ext(jh, 
                load_reg_from_mem(jh, traits::X0 + rs1), 32, false), gen_ext(jh, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false))
            ,0);
            cc.je(label_merge);
            {
                if(imm%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_raise(jh, 0, 0);
                }
                else{
                    auto PC_val_v = (uint32_t)(PC+(int16_t)sext<13>(imm));
                    cc.mov(jh.next_pc, PC_val_v);
                    cc.mov(get_ptr_for(jh, traits::LAST_BRANCH), 32U);
                }
            }
            cc.bind(label_merge);
        }
        auto returnValue = BRANCH;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 6);
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
        this->gen_sync(jh, PRE_SYNC, 7);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto label_merge = cc.newLabel();
            cc.cmp(gen_operation(jh, gte, gen_ext(jh, 
                load_reg_from_mem(jh, traits::X0 + rs1), 32, false), gen_ext(jh, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false))
            ,0);
            cc.je(label_merge);
            {
                if(imm%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_raise(jh, 0, 0);
                }
                else{
                    auto PC_val_v = (uint32_t)(PC+(int16_t)sext<13>(imm));
                    cc.mov(jh.next_pc, PC_val_v);
                    cc.mov(get_ptr_for(jh, traits::LAST_BRANCH), 32U);
                }
            }
            cc.bind(label_merge);
        }
        auto returnValue = BRANCH;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 7);
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
        this->gen_sync(jh, PRE_SYNC, 8);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto label_merge = cc.newLabel();
            cc.cmp(gen_operation(jh, ltu, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
            ,0);
            cc.je(label_merge);
            {
                if(imm%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_raise(jh, 0, 0);
                }
                else{
                    auto PC_val_v = (uint32_t)(PC+(int16_t)sext<13>(imm));
                    cc.mov(jh.next_pc, PC_val_v);
                    cc.mov(get_ptr_for(jh, traits::LAST_BRANCH), 32U);
                }
            }
            cc.bind(label_merge);
        }
        auto returnValue = BRANCH;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 8);
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
        this->gen_sync(jh, PRE_SYNC, 9);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto label_merge = cc.newLabel();
            cc.cmp(gen_operation(jh, gteu, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
            ,0);
            cc.je(label_merge);
            {
                if(imm%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_raise(jh, 0, 0);
                }
                else{
                    auto PC_val_v = (uint32_t)(PC+(int16_t)sext<13>(imm));
                    cc.mov(jh.next_pc, PC_val_v);
                    cc.mov(get_ptr_for(jh, traits::LAST_BRANCH), 32U);
                }
            }
            cc.bind(label_merge);
        }
        auto returnValue = BRANCH;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 9);
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
        this->gen_sync(jh, PRE_SYNC, 10);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto load_address = gen_ext(jh, 
                (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            auto res = gen_ext(jh, 
                gen_read_mem(jh, traits::MEM, load_address, 1), 8, false);
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, 
                         res, 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 10);
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
        this->gen_sync(jh, PRE_SYNC, 11);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto load_address = gen_ext(jh, 
                (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            auto res = gen_ext(jh, 
                gen_read_mem(jh, traits::MEM, load_address, 2), 16, false);
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, 
                         res, 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 11);
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
        this->gen_sync(jh, PRE_SYNC, 12);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto load_address = gen_ext(jh, 
                (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            auto res = gen_ext(jh, 
                gen_read_mem(jh, traits::MEM, load_address, 4), 32, false);
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, 
                         res, 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 12);
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
        this->gen_sync(jh, PRE_SYNC, 13);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto load_address = gen_ext(jh, 
                (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            auto res = gen_read_mem(jh, traits::MEM, load_address, 1);
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, 
                         res, 32, false));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 13);
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
        this->gen_sync(jh, PRE_SYNC, 14);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto load_address = gen_ext(jh, 
                (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            auto res = gen_read_mem(jh, traits::MEM, load_address, 2);
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, 
                         res, 32, false));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 14);
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
        this->gen_sync(jh, PRE_SYNC, 15);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto store_address = gen_ext(jh, 
                (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            gen_write_mem(jh, traits::MEM, store_address, gen_ext(jh, 
                load_reg_from_mem(jh, traits::X0 + rs2), 8, false), 1);
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 15);
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
        this->gen_sync(jh, PRE_SYNC, 16);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto store_address = gen_ext(jh, 
                (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            gen_write_mem(jh, traits::MEM, store_address, gen_ext(jh, 
                load_reg_from_mem(jh, traits::X0 + rs2), 16, false), 2);
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 16);
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
        this->gen_sync(jh, PRE_SYNC, 17);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto store_address = gen_ext(jh, 
                (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                ), 32, true);
            gen_write_mem(jh, traits::MEM, store_address, gen_ext(jh, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false), 4);
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 17);
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
        this->gen_sync(jh, PRE_SYNC, 18);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, 
                         (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))
                         ), 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 18);
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
        this->gen_sync(jh, PRE_SYNC, 19);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                {
                auto label_then = cc.newLabel();
                auto label_merge = cc.newLabel();
                auto tmp_reg = get_reg_for(jh, 1);
                cc.cmp(gen_ext(jh, 
                    load_reg_from_mem(jh, traits::X0 + rs1), 32, true), (int16_t)sext<12>(imm));
                cc.jl(label_then);
                cc.mov(tmp_reg,0);
                cc.jmp(label_merge);
                cc.bind(label_then);
                cc.mov(tmp_reg,1);
                cc.bind(label_merge);
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, tmp_reg
                     , 32, false)
                );
                }
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 19);
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
        this->gen_sync(jh, PRE_SYNC, 20);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                {
                auto label_then = cc.newLabel();
                auto label_merge = cc.newLabel();
                auto tmp_reg = get_reg_for(jh, 1);
                cc.cmp(load_reg_from_mem(jh, traits::X0 + rs1), (uint32_t)((int16_t)sext<12>(imm)));
                cc.jb(label_then);
                cc.mov(tmp_reg,0);
                cc.jmp(label_merge);
                cc.bind(label_then);
                cc.mov(tmp_reg,1);
                cc.bind(label_merge);
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, tmp_reg
                     , 32, false)
                );
                }
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 20);
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
        this->gen_sync(jh, PRE_SYNC, 21);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_operation(jh, bxor, load_reg_from_mem(jh, traits::X0 + rs1), (uint32_t)((int16_t)sext<12>(imm)))
                     );
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 21);
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
        this->gen_sync(jh, PRE_SYNC, 22);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_operation(jh, bor, load_reg_from_mem(jh, traits::X0 + rs1), (uint32_t)((int16_t)sext<12>(imm)))
                     );
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 22);
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
        this->gen_sync(jh, PRE_SYNC, 23);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_operation(jh, band, load_reg_from_mem(jh, traits::X0 + rs1), (uint32_t)((int16_t)sext<12>(imm)))
                     );
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 23);
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
        this->gen_sync(jh, PRE_SYNC, 24);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_operation(jh, shl, load_reg_from_mem(jh, traits::X0 + rs1), shamt)
                     );
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 24);
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
        this->gen_sync(jh, PRE_SYNC, 25);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_operation(jh, shr, load_reg_from_mem(jh, traits::X0 + rs1), shamt)
                     );
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 25);
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
        this->gen_sync(jh, PRE_SYNC, 26);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, 
                         (gen_operation(jh, sar, gen_ext(jh, 
                             load_reg_from_mem(jh, traits::X0 + rs1), 32, true), shamt)
                         ), 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 26);
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
        this->gen_sync(jh, PRE_SYNC, 27);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, 
                         (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
                         ), 32, false));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 27);
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
        this->gen_sync(jh, PRE_SYNC, 28);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, 
                         (gen_operation(jh, sub, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
                         ), 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 28);
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
        this->gen_sync(jh, PRE_SYNC, 29);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, gen_operation(jh, shl, load_reg_from_mem(jh, traits::X0 + rs1), (gen_operation(jh, band, load_reg_from_mem(jh, traits::X0 + rs2), (static_cast<uint32_t>(traits::XLEN)-1))
                     ))
                     , 32, false));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 29);
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
        this->gen_sync(jh, PRE_SYNC, 30);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                {
                auto label_then = cc.newLabel();
                auto label_merge = cc.newLabel();
                auto tmp_reg = get_reg_for(jh, 1);
                cc.cmp(gen_ext(jh, 
                    load_reg_from_mem(jh, traits::X0 + rs1), 32, true), gen_ext(jh, 
                    load_reg_from_mem(jh, traits::X0 + rs2), 32, true));
                cc.jl(label_then);
                cc.mov(tmp_reg,0);
                cc.jmp(label_merge);
                cc.bind(label_then);
                cc.mov(tmp_reg,1);
                cc.bind(label_merge);
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, tmp_reg
                     , 32, false)
                );
                }
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 30);
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
        this->gen_sync(jh, PRE_SYNC, 31);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                {
                auto label_then = cc.newLabel();
                auto label_merge = cc.newLabel();
                auto tmp_reg = get_reg_for(jh, 1);
                cc.cmp(load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2));
                cc.jb(label_then);
                cc.mov(tmp_reg,0);
                cc.jmp(label_merge);
                cc.bind(label_then);
                cc.mov(tmp_reg,1);
                cc.bind(label_merge);
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, tmp_reg
                     , 32, false)
                );
                }
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 31);
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
        this->gen_sync(jh, PRE_SYNC, 32);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_operation(jh, bxor, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
                     );
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 32);
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
        this->gen_sync(jh, PRE_SYNC, 33);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, gen_operation(jh, shr, load_reg_from_mem(jh, traits::X0 + rs1), (gen_operation(jh, band, load_reg_from_mem(jh, traits::X0 + rs2), (static_cast<uint32_t>(traits::XLEN)-1))
                     ))
                     , 32, false));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 33);
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
        this->gen_sync(jh, PRE_SYNC, 34);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, 
                         (gen_ext(jh, gen_operation(jh, sar, gen_ext(jh, 
                             load_reg_from_mem(jh, traits::X0 + rs1), 32, true), (gen_operation(jh, band, load_reg_from_mem(jh, traits::X0 + rs2), (static_cast<uint32_t>(traits::XLEN)-1))
                         ))
                         , 32, true)), 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 34);
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
        this->gen_sync(jh, PRE_SYNC, 35);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_operation(jh, bor, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
                     );
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 35);
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
        this->gen_sync(jh, PRE_SYNC, 36);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_operation(jh, band, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
                     );
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 36);
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
        this->gen_sync(jh, PRE_SYNC, 37);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        gen_write_mem(jh, traits::FENCE, static_cast<uint32_t>(traits::fence), (uint8_t)pred<<4|succ, 4);
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 37);
    	return returnValue;        
    }
    
    /* instruction 38: ECALL */
    continuation_e __ecall(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate disass */
            
            //This disass is not yet implemented
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
        this->gen_sync(jh, PRE_SYNC, 38);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        gen_raise(jh, 0, 11);
        auto returnValue = TRAP;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 38);
    	return returnValue;        
    }
    
    /* instruction 39: EBREAK */
    continuation_e __ebreak(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate disass */
            
            //This disass is not yet implemented
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
        this->gen_sync(jh, PRE_SYNC, 39);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        gen_raise(jh, 0, 3);
        auto returnValue = TRAP;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 39);
    	return returnValue;        
    }
    
    /* instruction 40: MRET */
    continuation_e __mret(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate disass */
            
            //This disass is not yet implemented
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
        this->gen_sync(jh, PRE_SYNC, 40);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        gen_leave(jh, 3);
        auto returnValue = TRAP;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 40);
    	return returnValue;        
    }
    
    /* instruction 41: WFI */
    continuation_e __wfi(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate disass */
            
            //This disass is not yet implemented
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
        this->gen_sync(jh, PRE_SYNC, 41);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        gen_wait(jh, 1);
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 41);
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
        this->gen_sync(jh, PRE_SYNC, 42);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto xrs1 = load_reg_from_mem(jh, traits::X0 + rs1);
            if(rd!=0){
                auto xrd = gen_read_mem(jh, traits::CSR, csr, 4);
                gen_write_mem(jh, traits::CSR, csr, xrs1, 4);
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     xrd);
            }
            else{
                gen_write_mem(jh, traits::CSR, csr, xrs1, 4);
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 42);
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
        this->gen_sync(jh, PRE_SYNC, 43);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 4);
            auto xrs1 = load_reg_from_mem(jh, traits::X0 + rs1);
            if(rs1!=0){
                gen_write_mem(jh, traits::CSR, csr, gen_operation(jh, bor, xrd, xrs1)
                , 4);
            }
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     xrd);
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 43);
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
        this->gen_sync(jh, PRE_SYNC, 44);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 4);
            auto xrs1 = load_reg_from_mem(jh, traits::X0 + rs1);
            if(rs1!=0){
                gen_write_mem(jh, traits::CSR, csr, gen_operation(jh, band, xrd, gen_operation(jh, bnot, xrs1))
                , 4);
            }
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     xrd);
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 44);
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
        this->gen_sync(jh, PRE_SYNC, 45);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 4);
            gen_write_mem(jh, traits::CSR, csr, (uint32_t)zimm, 4);
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     xrd);
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 45);
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
        this->gen_sync(jh, PRE_SYNC, 46);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 4);
            if(zimm!=0){
                gen_write_mem(jh, traits::CSR, csr, gen_operation(jh, bor, xrd, (uint32_t)zimm)
                , 4);
            }
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     xrd);
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 46);
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
        this->gen_sync(jh, PRE_SYNC, 47);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 4);
            if(zimm!=0){
                gen_write_mem(jh, traits::CSR, csr, gen_operation(jh, band, xrd, ~ ((uint32_t)zimm))
                , 4);
            }
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     xrd);
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 47);
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
        this->gen_sync(jh, PRE_SYNC, 48);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        gen_write_mem(jh, traits::FENCE, static_cast<uint32_t>(traits::fencei), imm, 4);
        auto returnValue = FLUSH;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 48);
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
        this->gen_sync(jh, PRE_SYNC, 49);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto res = gen_ext(jh, 
                (gen_operation(jh, imul, gen_ext(jh, 
                    gen_ext(jh, 
                        load_reg_from_mem(jh, traits::X0 + rs1), 32, true), 64, true), gen_ext(jh, 
                    gen_ext(jh, 
                        load_reg_from_mem(jh, traits::X0 + rs2), 32, true), 64, true))
                ), 64, true);
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, 
                         res, 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 49);
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
        this->gen_sync(jh, PRE_SYNC, 50);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto res = gen_ext(jh, 
                (gen_operation(jh, imul, gen_ext(jh, 
                    gen_ext(jh, 
                        load_reg_from_mem(jh, traits::X0 + rs1), 32, true), 64, true), gen_ext(jh, 
                    gen_ext(jh, 
                        load_reg_from_mem(jh, traits::X0 + rs2), 32, true), 64, true))
                ), 64, true);
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, 
                         (gen_operation(jh, sar, res, static_cast<uint32_t>(traits::XLEN))
                         ), 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 50);
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
        this->gen_sync(jh, PRE_SYNC, 51);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto res = gen_ext(jh, 
                (gen_operation(jh, imul, gen_ext(jh, 
                    gen_ext(jh, 
                        load_reg_from_mem(jh, traits::X0 + rs1), 32, true), 64, true), gen_ext(jh, 
                    load_reg_from_mem(jh, traits::X0 + rs2), 64, false))
                ), 64, true);
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, 
                         (gen_operation(jh, sar, res, static_cast<uint32_t>(traits::XLEN))
                         ), 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 51);
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
        this->gen_sync(jh, PRE_SYNC, 52);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto res = gen_ext(jh, 
                (gen_operation(jh, mul, gen_ext(jh, 
                    load_reg_from_mem(jh, traits::X0 + rs1), 64, false), gen_ext(jh, 
                    load_reg_from_mem(jh, traits::X0 + rs2), 64, false))
                ), 64, false);
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, 
                         (gen_operation(jh, shr, res, static_cast<uint32_t>(traits::XLEN))
                         ), 32, false));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 52);
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
        this->gen_sync(jh, PRE_SYNC, 53);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto dividend = gen_ext(jh, 
                load_reg_from_mem(jh, traits::X0 + rs1), 32, false);
            auto divisor = gen_ext(jh, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false);
            if(rd!=0){
                auto label_merge = cc.newLabel();
                cc.cmp(gen_operation(jh, ne, divisor, 0)
                ,0);
                auto label_else = cc.newLabel();
                cc.je(label_else);
                {
                    auto MMIN = ((uint32_t)1)<<(static_cast<uint32_t>(traits::XLEN)-1);
                    auto label_merge = cc.newLabel();
                    cc.cmp(gen_operation(jh, land, gen_operation(jh, eq, load_reg_from_mem(jh, traits::X0 + rs1), MMIN)
                    , gen_operation(jh, eq, divisor, - 1)
                    )
                    ,0);
                    auto label_else = cc.newLabel();
                    cc.je(label_else);
                    {
                        cc.mov(get_ptr_for(jh, traits::X0+ rd),
                             MMIN);
                    }
                    cc.jmp(label_merge);
                    cc.bind(label_else);
                        {
                            cc.mov(get_ptr_for(jh, traits::X0+ rd),
                                 gen_ext(jh, 
                                     (gen_operation(jh, idiv, dividend, divisor)
                                     ), 32, true));
                        }
                    cc.bind(label_merge);
                }
                cc.jmp(label_merge);
                cc.bind(label_else);
                    {
                        cc.mov(get_ptr_for(jh, traits::X0+ rd),
                             (uint32_t)- 1);
                    }
                cc.bind(label_merge);
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 53);
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
        this->gen_sync(jh, PRE_SYNC, 54);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto label_merge = cc.newLabel();
            cc.cmp(gen_operation(jh, ne, load_reg_from_mem(jh, traits::X0 + rs2), 0)
            ,0);
            auto label_else = cc.newLabel();
            cc.je(label_else);
            {
                if(rd!=0){
                    cc.mov(get_ptr_for(jh, traits::X0+ rd),
                         gen_ext(jh, 
                             (gen_operation(jh, div, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
                             ), 32, false));
                }
            }
            cc.jmp(label_merge);
            cc.bind(label_else);
                {
                    if(rd!=0){
                        cc.mov(get_ptr_for(jh, traits::X0+ rd),
                             (uint32_t)- 1);
                    }
                }
            cc.bind(label_merge);
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 54);
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
        this->gen_sync(jh, PRE_SYNC, 55);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto label_merge = cc.newLabel();
            cc.cmp(gen_operation(jh, ne, load_reg_from_mem(jh, traits::X0 + rs2), 0)
            ,0);
            auto label_else = cc.newLabel();
            cc.je(label_else);
            {
                auto MMIN = (uint32_t)1<<(static_cast<uint32_t>(traits::XLEN)-1);
                auto label_merge = cc.newLabel();
                cc.cmp(gen_operation(jh, land, gen_operation(jh, eq, load_reg_from_mem(jh, traits::X0 + rs1), MMIN)
                , gen_operation(jh, eq, gen_ext(jh, 
                    load_reg_from_mem(jh, traits::X0 + rs2), 32, false), - 1)
                )
                ,0);
                auto label_else = cc.newLabel();
                cc.je(label_else);
                {
                    if(rd!=0){
                        cc.mov(get_ptr_for(jh, traits::X0+ rd),
                             gen_ext(jh, 0, 32, false)
                        );
                    }
                }
                cc.jmp(label_merge);
                cc.bind(label_else);
                    {
                        if(rd!=0){
                            cc.mov(get_ptr_for(jh, traits::X0+ rd),
                                 gen_ext(jh, 
                                     (gen_operation(jh, srem, gen_ext(jh, 
                                         load_reg_from_mem(jh, traits::X0 + rs1), 32, false), gen_ext(jh, 
                                         load_reg_from_mem(jh, traits::X0 + rs2), 32, false))
                                     ), 32, true));
                        }
                    }
                cc.bind(label_merge);
            }
            cc.jmp(label_merge);
            cc.bind(label_else);
                {
                    if(rd!=0){
                        cc.mov(get_ptr_for(jh, traits::X0+ rd),
                             load_reg_from_mem(jh, traits::X0 + rs1));
                    }
                }
            cc.bind(label_merge);
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 55);
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
        this->gen_sync(jh, PRE_SYNC, 56);
        cc.mov(jh.pc, pc.val);
        pc = pc+4;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto label_merge = cc.newLabel();
            cc.cmp(gen_operation(jh, ne, load_reg_from_mem(jh, traits::X0 + rs2), 0)
            ,0);
            auto label_else = cc.newLabel();
            cc.je(label_else);
            {
                if(rd!=0){
                    cc.mov(get_ptr_for(jh, traits::X0+ rd),
                         gen_operation(jh, urem, load_reg_from_mem(jh, traits::X0 + rs1), load_reg_from_mem(jh, traits::X0 + rs2))
                         );
                }
            }
            cc.jmp(label_merge);
            cc.bind(label_else);
                {
                    if(rd!=0){
                        cc.mov(get_ptr_for(jh, traits::X0+ rd),
                             load_reg_from_mem(jh, traits::X0 + rs1));
                    }
                }
            cc.bind(label_merge);
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 56);
    	return returnValue;        
    }
    
    /* instruction 57: C__ADDI4SPN */
    continuation_e __c__addi4spn(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<2,3>(instr)));
        uint16_t imm = ((bit_sub<5,1>(instr) << 3) | (bit_sub<6,1>(instr) << 2) | (bit_sub<7,4>(instr) << 6) | (bit_sub<11,2>(instr) << 4));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c__addi4spn"),
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
        this->gen_sync(jh, PRE_SYNC, 57);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(imm){
            cc.mov(get_ptr_for(jh, traits::X0+ rd+8),
                 gen_ext(jh, 
                     (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + 2), imm)
                     ), 32, false));
        }
        else{
            gen_raise(jh, 0, 2);
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 57);
    	return returnValue;        
    }
    
    /* instruction 58: C__LW */
    continuation_e __c__lw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {uimm:#05x}({rs1})", fmt::arg("mnemonic", "c__lw"),
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
        this->gen_sync(jh, PRE_SYNC, 58);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        auto offs = gen_ext(jh, 
            (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + rs1+8), uimm)
            ), 32, false);
        cc.mov(get_ptr_for(jh, traits::X0+ rd+8),
             gen_ext(jh, 
                 gen_ext(jh, 
                     gen_read_mem(jh, traits::MEM, offs, 4), 32, false), 32, true));
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 58);
    	return returnValue;        
    }
    
    /* instruction 59: C__SW */
    continuation_e __c__sw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {uimm:#05x}({rs1})", fmt::arg("mnemonic", "c__sw"),
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
        this->gen_sync(jh, PRE_SYNC, 59);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        auto offs = gen_ext(jh, 
            (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + rs1+8), uimm)
            ), 32, false);
        gen_write_mem(jh, traits::MEM, offs, gen_ext(jh, 
            load_reg_from_mem(jh, traits::X0 + rs2+8), 32, false), 4);
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 59);
    	return returnValue;        
    }
    
    /* instruction 60: C__ADDI */
    continuation_e __c__addi(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t imm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c__addi"),
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
        this->gen_sync(jh, PRE_SYNC, 60);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rs1!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rs1),
                     gen_ext(jh, 
                         (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + rs1), (int8_t)sext<6>(imm))
                         ), 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 60);
    	return returnValue;        
    }
    
    /* instruction 61: C__NOP */
    continuation_e __c__nop(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t nzimm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        if(this->disass_enabled){
            /* generate disass */
            
            //This disass is not yet implemented
            std::string mnemonic = "c__nop";
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
        this->gen_sync(jh, PRE_SYNC, 61);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 61);
    	return returnValue;        
    }
    
    /* instruction 62: C__JAL */
    continuation_e __c__jal(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,3>(instr) << 1) | (bit_sub<6,1>(instr) << 7) | (bit_sub<7,1>(instr) << 6) | (bit_sub<8,1>(instr) << 10) | (bit_sub<9,2>(instr) << 8) | (bit_sub<11,1>(instr) << 4) | (bit_sub<12,1>(instr) << 11));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {imm:#05x}", fmt::arg("mnemonic", "c__jal"),
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
        this->gen_sync(jh, PRE_SYNC, 62);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        cc.mov(get_ptr_for(jh, traits::X0+ 1),
             (uint32_t)(PC+2));
        auto PC_val_v = (uint32_t)(PC+(int16_t)sext<12>(imm));
        cc.mov(jh.next_pc, PC_val_v);
        cc.mov(get_ptr_for(jh, traits::LAST_BRANCH), 32U);
        auto returnValue = BRANCH;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 62);
    	return returnValue;        
    }
    
    /* instruction 63: C__LI */
    continuation_e __c__li(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t imm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c__li"),
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
        this->gen_sync(jh, PRE_SYNC, 63);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     (uint32_t)((int8_t)sext<6>(imm)));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 63);
    	return returnValue;        
    }
    
    /* instruction 64: C__LUI */
    continuation_e __c__lui(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint32_t imm = ((bit_sub<2,5>(instr) << 12) | (bit_sub<12,1>(instr) << 17));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c__lui"),
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
        this->gen_sync(jh, PRE_SYNC, 64);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(imm==0||rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        if(rd!=0){
            cc.mov(get_ptr_for(jh, traits::X0+ rd),
                 (uint32_t)((int32_t)sext<18>(imm)));
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 64);
    	return returnValue;        
    }
    
    /* instruction 65: C__ADDI16SP */
    continuation_e __c__addi16sp(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t nzimm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 7) | (bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 4) | (bit_sub<12,1>(instr) << 9));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {nzimm:#05x}", fmt::arg("mnemonic", "c__addi16sp"),
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
        this->gen_sync(jh, PRE_SYNC, 65);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(nzimm){
            cc.mov(get_ptr_for(jh, traits::X0+ 2),
                 gen_ext(jh, 
                     (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + 2), (int16_t)sext<10>(nzimm))
                     ), 32, true));
        }
        else{
            gen_raise(jh, 0, 2);
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 65);
    	return returnValue;        
    }
    
    /* instruction 66: __reserved_clui */
    continuation_e ____reserved_clui(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            //This disass is not yet implemented
            std::string mnemonic = "__reserved_clui";
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
        this->gen_sync(jh, PRE_SYNC, 66);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        gen_raise(jh, 0, 2);
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 66);
    	return returnValue;        
    }
    
    /* instruction 67: C__SRLI */
    continuation_e __c__srli(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t shamt = ((bit_sub<2,5>(instr)));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {shamt}", fmt::arg("mnemonic", "c__srli"),
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
        this->gen_sync(jh, PRE_SYNC, 67);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        cc.mov(get_ptr_for(jh, traits::X0+ rs1+8),
             gen_operation(jh, shr, load_reg_from_mem(jh, traits::X0 + rs1+8), shamt)
             );
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 67);
    	return returnValue;        
    }
    
    /* instruction 68: C__SRAI */
    continuation_e __c__srai(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t shamt = ((bit_sub<2,5>(instr)));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {shamt}", fmt::arg("mnemonic", "c__srai"),
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
        this->gen_sync(jh, PRE_SYNC, 68);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(shamt){
            cc.mov(get_ptr_for(jh, traits::X0+ rs1+8),
                 gen_ext(jh, 
                     (gen_operation(jh, sar, (gen_ext(jh, 
                         load_reg_from_mem(jh, traits::X0 + rs1+8), 32, false)), shamt)
                     ), 32, true));
        }
        else{
            if(static_cast<uint32_t>(traits::XLEN)==128){
                cc.mov(get_ptr_for(jh, traits::X0+ rs1+8),
                     gen_ext(jh, 
                         (gen_operation(jh, sar, (gen_ext(jh, 
                             load_reg_from_mem(jh, traits::X0 + rs1+8), 32, false)), 64)
                         ), 32, true));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 68);
    	return returnValue;        
    }
    
    /* instruction 69: C__ANDI */
    continuation_e __c__andi(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t imm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c__andi"),
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
        this->gen_sync(jh, PRE_SYNC, 69);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        cc.mov(get_ptr_for(jh, traits::X0+ rs1+8),
             gen_ext(jh, 
                 (gen_operation(jh, band, load_reg_from_mem(jh, traits::X0 + rs1+8), (int8_t)sext<6>(imm))
                 ), 32, true));
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 69);
    	return returnValue;        
    }
    
    /* instruction 70: C__SUB */
    continuation_e __c__sub(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c__sub"),
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
        this->gen_sync(jh, PRE_SYNC, 70);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        cc.mov(get_ptr_for(jh, traits::X0+ rd+8),
             gen_ext(jh, 
                 (gen_operation(jh, sub, load_reg_from_mem(jh, traits::X0 + rd+8), load_reg_from_mem(jh, traits::X0 + rs2+8))
                 ), 32, true));
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 70);
    	return returnValue;        
    }
    
    /* instruction 71: C__XOR */
    continuation_e __c__xor(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c__xor"),
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
        this->gen_sync(jh, PRE_SYNC, 71);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        cc.mov(get_ptr_for(jh, traits::X0+ rd+8),
             gen_operation(jh, bxor, load_reg_from_mem(jh, traits::X0 + rd+8), load_reg_from_mem(jh, traits::X0 + rs2+8))
             );
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 71);
    	return returnValue;        
    }
    
    /* instruction 72: C__OR */
    continuation_e __c__or(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c__or"),
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
        this->gen_sync(jh, PRE_SYNC, 72);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        cc.mov(get_ptr_for(jh, traits::X0+ rd+8),
             gen_operation(jh, bor, load_reg_from_mem(jh, traits::X0 + rd+8), load_reg_from_mem(jh, traits::X0 + rs2+8))
             );
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 72);
    	return returnValue;        
    }
    
    /* instruction 73: C__AND */
    continuation_e __c__and(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c__and"),
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
        this->gen_sync(jh, PRE_SYNC, 73);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        cc.mov(get_ptr_for(jh, traits::X0+ rd+8),
             gen_operation(jh, band, load_reg_from_mem(jh, traits::X0 + rd+8), load_reg_from_mem(jh, traits::X0 + rs2+8))
             );
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 73);
    	return returnValue;        
    }
    
    /* instruction 74: C__J */
    continuation_e __c__j(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,3>(instr) << 1) | (bit_sub<6,1>(instr) << 7) | (bit_sub<7,1>(instr) << 6) | (bit_sub<8,1>(instr) << 10) | (bit_sub<9,2>(instr) << 8) | (bit_sub<11,1>(instr) << 4) | (bit_sub<12,1>(instr) << 11));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {imm:#05x}", fmt::arg("mnemonic", "c__j"),
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
        this->gen_sync(jh, PRE_SYNC, 74);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        auto PC_val_v = (uint32_t)(PC+(int16_t)sext<12>(imm));
        cc.mov(jh.next_pc, PC_val_v);
        cc.mov(get_ptr_for(jh, traits::LAST_BRANCH), 32U);
        auto returnValue = BRANCH;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 74);
    	return returnValue;        
    }
    
    /* instruction 75: C__BEQZ */
    continuation_e __c__beqz(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 1) | (bit_sub<5,2>(instr) << 6) | (bit_sub<10,2>(instr) << 3) | (bit_sub<12,1>(instr) << 8));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c__beqz"),
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
        this->gen_sync(jh, PRE_SYNC, 75);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        auto label_merge = cc.newLabel();
        cc.cmp(gen_operation(jh, eq, load_reg_from_mem(jh, traits::X0 + rs1+8), 0)
        ,0);
        cc.je(label_merge);
        {
            auto PC_val_v = (uint32_t)(PC+(int16_t)sext<9>(imm));
            cc.mov(jh.next_pc, PC_val_v);
            cc.mov(get_ptr_for(jh, traits::LAST_BRANCH), 32U);
        }
        cc.bind(label_merge);
        auto returnValue = BRANCH;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 75);
    	return returnValue;        
    }
    
    /* instruction 76: C__BNEZ */
    continuation_e __c__bnez(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 1) | (bit_sub<5,2>(instr) << 6) | (bit_sub<10,2>(instr) << 3) | (bit_sub<12,1>(instr) << 8));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c__bnez"),
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
        this->gen_sync(jh, PRE_SYNC, 76);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        auto label_merge = cc.newLabel();
        cc.cmp(gen_operation(jh, ne, load_reg_from_mem(jh, traits::X0 + rs1+8), 0)
        ,0);
        cc.je(label_merge);
        {
            auto PC_val_v = (uint32_t)(PC+(int16_t)sext<9>(imm));
            cc.mov(jh.next_pc, PC_val_v);
            cc.mov(get_ptr_for(jh, traits::LAST_BRANCH), 32U);
        }
        cc.bind(label_merge);
        auto returnValue = BRANCH;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 76);
    	return returnValue;        
    }
    
    /* instruction 77: C__SLLI */
    continuation_e __c__slli(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t nzuimm = ((bit_sub<2,5>(instr)));
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {nzuimm}", fmt::arg("mnemonic", "c__slli"),
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
        this->gen_sync(jh, PRE_SYNC, 77);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rs1!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rs1),
                     gen_operation(jh, shl, load_reg_from_mem(jh, traits::X0 + rs1), nzuimm)
                     );
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 77);
    	return returnValue;        
    }
    
    /* instruction 78: C__LWSP */
    continuation_e __c__lwsp(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t uimm = ((bit_sub<2,2>(instr) << 6) | (bit_sub<4,3>(instr) << 2) | (bit_sub<12,1>(instr) << 5));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, sp, {uimm:#05x}", fmt::arg("mnemonic", "c__lwsp"),
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
        this->gen_sync(jh, PRE_SYNC, 78);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rd==0){
            gen_raise(jh, 0, 2);
        }
        else{
            auto offs = gen_ext(jh, 
                (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + 2), uimm)
                ), 32, false);
            cc.mov(get_ptr_for(jh, traits::X0+ rd),
                 gen_ext(jh, 
                     gen_ext(jh, 
                         gen_read_mem(jh, traits::MEM, offs, 4), 32, false), 32, true));
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 78);
    	return returnValue;        
    }
    
    /* instruction 79: C__MV */
    continuation_e __c__mv(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c__mv"),
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
        this->gen_sync(jh, PRE_SYNC, 79);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     load_reg_from_mem(jh, traits::X0 + rs2));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 79);
    	return returnValue;        
    }
    
    /* instruction 80: C__JR */
    continuation_e __c__jr(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}", fmt::arg("mnemonic", "c__jr"),
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
        this->gen_sync(jh, PRE_SYNC, 80);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs1&&rs1<static_cast<uint32_t>(traits::RFS)){
            auto PC_val_v = gen_operation(jh, band, load_reg_from_mem(jh, traits::X0 + rs1%static_cast<uint32_t>(traits::RFS)), ~ 1)
            ;
            cc.mov(jh.next_pc, PC_val_v);
            cc.mov(get_ptr_for(jh, traits::LAST_BRANCH), 32U);
        }
        else{
            gen_raise(jh, 0, 2);
        }
        auto returnValue = BRANCH;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 80);
    	return returnValue;        
    }
    
    /* instruction 81: __reserved_cmv */
    continuation_e ____reserved_cmv(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate disass */
            
            //This disass is not yet implemented
            std::string mnemonic = "__reserved_cmv";
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
        this->gen_sync(jh, PRE_SYNC, 81);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        gen_raise(jh, 0, 2);
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 81);
    	return returnValue;        
    }
    
    /* instruction 82: C__ADD */
    continuation_e __c__add(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c__add"),
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
        this->gen_sync(jh, PRE_SYNC, 82);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            if(rd!=0){
                cc.mov(get_ptr_for(jh, traits::X0+ rd),
                     gen_ext(jh, 
                         (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + rd), load_reg_from_mem(jh, traits::X0 + rs2))
                         ), 32, false));
            }
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 82);
    	return returnValue;        
    }
    
    /* instruction 83: C__JALR */
    continuation_e __c__jalr(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}", fmt::arg("mnemonic", "c__jalr"),
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
        this->gen_sync(jh, PRE_SYNC, 83);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto new_pc = load_reg_from_mem(jh, traits::X0 + rs1);
            cc.mov(get_ptr_for(jh, traits::X0+ 1),
                 (uint32_t)(PC+2));
            auto PC_val_v = gen_operation(jh, band, new_pc, ~ 1)
            ;
            cc.mov(jh.next_pc, PC_val_v);
            cc.mov(get_ptr_for(jh, traits::LAST_BRANCH), 32U);
        }
        auto returnValue = BRANCH;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 83);
    	return returnValue;        
    }
    
    /* instruction 84: C__EBREAK */
    continuation_e __c__ebreak(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate disass */
            
            //This disass is not yet implemented
            std::string mnemonic = "c__ebreak";
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
        this->gen_sync(jh, PRE_SYNC, 84);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        gen_raise(jh, 0, 3);
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 84);
    	return returnValue;        
    }
    
    /* instruction 85: C__SWSP */
    continuation_e __c__swsp(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint8_t uimm = ((bit_sub<7,2>(instr) << 6) | (bit_sub<9,4>(instr) << 2));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {uimm:#05x}(sp)", fmt::arg("mnemonic", "c__swsp"),
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
        this->gen_sync(jh, PRE_SYNC, 85);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, 2);
        }
        else{
            auto offs = gen_ext(jh, 
                (gen_operation(jh, add, load_reg_from_mem(jh, traits::X0 + 2), uimm)
                ), 32, false);
            gen_write_mem(jh, traits::MEM, offs, gen_ext(jh, 
                load_reg_from_mem(jh, traits::X0 + rs2), 32, false), 4);
        }
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 85);
    	return returnValue;        
    }
    
    /* instruction 86: DII */
    continuation_e __dii(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate disass */
            
            //This disass is not yet implemented
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
        this->gen_sync(jh, PRE_SYNC, 86);
        cc.mov(jh.pc, pc.val);
        pc = pc+2;
        cc.mov(jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        /*generate behavior*/
        gen_raise(jh, 0, 2);
        auto returnValue = CONT;
        
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, 86);
    	return returnValue;        
    }
    
    /****************************************************************************
     * end opcode definitions
     ****************************************************************************/
    continuation_e illegal_intruction(virt_addr_t &pc, code_word_t instr, jit_holder& jh ) {
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("illegal_intruction{:#x}:",pc.val).c_str());
        this->gen_sync(jh, PRE_SYNC, instr_descr.size());
        pc = pc + ((instr & 3) == 3 ? 4 : 2);
        gen_instr_prologue(jh);
        cc.comment("//behavior:");
        gen_instr_epilogue(jh);
        this->gen_sync(jh, POST_SYNC, instr_descr.size());
        return BRANCH;
    }    
    //decoding functionality

    void populate_decoding_tree(decoding_tree_node* root){
        //create submask
        for(auto instr: root->instrs){
            root->submask &= instr.mask;
        }
        //put each instr according to submask&encoding into children
        for(auto instr: root->instrs){
            bool foundMatch = false;
            for(auto child: root->children){
                //use value as identifying trait
                if(child->value == (instr.value&root->submask)){
                    child->instrs.push_back(instr);
                    foundMatch = true;
                }
            }
            if(!foundMatch){
                decoding_tree_node* child = new decoding_tree_node(instr.value&root->submask);
                child->instrs.push_back(instr);
                root->children.push_back(child);
            }
        }
        root->instrs.clear();
        //call populate_decoding_tree for all children
        if(root->children.size() >1)
            for(auto child: root->children){
                populate_decoding_tree(child);      
            }
        else{
            //sort instrs by value of the mask, this works bc we want to have the least restrictive one last
            std::sort(root->children[0]->instrs.begin(), root->children[0]->instrs.end(), [](const instruction_descriptor& instr1, const instruction_descriptor& instr2) {
            return instr1.mask > instr2.mask;
            }); 
        }
    }
    compile_func decode_instr(decoding_tree_node* node, code_word_t word){
        if(!node->children.size()){
            if(node->instrs.size() == 1) return node->instrs[0].op;
            for(auto instr : node->instrs){
                if((instr.mask&word) == instr.value) return instr.op;
            }
        }
        else{
            for(auto child : node->children){
                if (child->value == (node->submask&word)){
                    return decode_instr(child, word);
                }  
            }  
        }
        return nullptr;
    }
};

template <typename ARCH> vm_impl<ARCH>::vm_impl() { this(new ARCH()); }

template <typename ARCH>
vm_impl<ARCH>::vm_impl(ARCH &core, unsigned core_id, unsigned cluster_id)
: vm_base<ARCH>(core, core_id, cluster_id) {
    root = new decoding_tree_node(std::numeric_limits<uint32_t>::max());
    for(auto instr: instr_descr){
        root->instrs.push_back(instr);
    }
    populate_decoding_tree(root);
}

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
        throw trap_access(TRAP_ID, pc.val);
    if (instr == 0x0000006f || (instr&0xffff)==0xa001)
        throw simulation_stopped(0); // 'J 0' or 'C.J 0'
    ++inst_cnt;
    auto f = decode_instr(root, instr);
    if (f == nullptr) 
        f = &this_class::illegal_intruction;
    return (this->*f)(pc, instr, jh);
}
template <typename ARCH>
void vm_impl<ARCH>::gen_instr_prologue(jit_holder& jh) {
    auto& cc = jh.cc;

    cc.comment("//gen_instr_prologue");
    cc.inc(get_ptr_for(jh, traits::ICOUNT));

    x86::Gp current_trap_state = get_reg_for(jh, traits::TRAP_STATE);
    cc.mov(current_trap_state, get_ptr_for(jh, traits::TRAP_STATE));
    cc.mov(get_ptr_for(jh, traits::PENDING_TRAP), current_trap_state);

}
template <typename ARCH>
void vm_impl<ARCH>::gen_instr_epilogue(jit_holder& jh) {
    auto& cc = jh.cc;

    cc.comment("//gen_instr_epilogue");
    x86::Gp current_trap_state = get_reg_for(jh, traits::TRAP_STATE);
    cc.mov(current_trap_state, get_ptr_for(jh, traits::TRAP_STATE));
    cc.cmp(current_trap_state, 0);
    cc.jne(jh.trap_entry);
}
template <typename ARCH>
void vm_impl<ARCH>::gen_block_prologue(jit_holder& jh){

    jh.pc = load_reg_from_mem(jh, traits::PC);
    jh.next_pc = load_reg_from_mem(jh, traits::NEXT_PC);
}
template <typename ARCH>
void vm_impl<ARCH>::gen_block_epilogue(jit_holder& jh){
    x86::Compiler& cc = jh.cc;
    cc.comment("//gen_block_epilogue");
    cc.ret(jh.next_pc);

    cc.bind(jh.trap_entry);
    this->write_back(jh);
    this->gen_sync(jh, POST_SYNC, -1);

    x86::Gp current_trap_state = get_reg_for(jh, traits::TRAP_STATE);
    cc.mov(current_trap_state, get_ptr_for(jh, traits::TRAP_STATE));

    x86::Gp current_pc = get_reg_for(jh, traits::PC);
    cc.mov(current_pc, get_ptr_for(jh, traits::PC));

    x86::Gp instr = cc.newInt32("instr");
    cc.mov(instr, 0); // FIXME:this is not correct
    cc.comment("//enter trap call;");
    InvokeNode* call_enter_trap;
    cc.invoke(&call_enter_trap, &enter_trap, FuncSignature::build<uint64_t, void*, uint64_t, uint64_t, uint64_t>());
    call_enter_trap->setArg(0, jh.arch_if_ptr);
    call_enter_trap->setArg(1, current_trap_state);
    call_enter_trap->setArg(2, current_pc);
    call_enter_trap->setArg(3, instr);

    x86::Gp current_next_pc = get_reg_for(jh, traits::NEXT_PC);
    cc.mov(current_next_pc, get_ptr_for(jh, traits::NEXT_PC));
    cc.mov(jh.next_pc, current_next_pc);

    cc.mov(get_ptr_for(jh, traits::LAST_BRANCH), std::numeric_limits<uint32_t>::max());
    cc.ret(jh.next_pc);
}
template <typename ARCH>
inline void vm_impl<ARCH>::gen_raise(jit_holder& jh, uint16_t trap_id, uint16_t cause) {
    auto& cc = jh.cc;
    cc.comment("//gen_raise");
    auto tmp1 = get_reg_for(jh, traits::TRAP_STATE);
    cc.mov(tmp1, 0x80ULL << 24 | (cause << 16) | trap_id);
    cc.mov(get_ptr_for(jh, traits::TRAP_STATE), tmp1);
    cc.mov(jh.next_pc, std::numeric_limits<uint32_t>::max());
}

} // namespace rv32imc

template <>
std::unique_ptr<vm_if> create<arch::rv32imc>(arch::rv32imc *core, unsigned short port, bool dump) {
    auto ret = new rv32imc::vm_impl<arch::rv32imc>(*core, dump);
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
        core_factory::instance().register_creator("rv32imc|m_p|asmjit", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_m_p<iss::arch::rv32imc>();
		    auto vm = new asmjit::rv32imc::vm_impl<arch::rv32imc>(*cpu, false);
		    if (port != 0) debugger::server<debugger::gdb_session>::run_server(vm, port);
            if(init_data){
                auto* cb = reinterpret_cast<semihosting_cb_t<arch::traits<arch::rv32imc>::reg_t>*>(init_data);
                cpu->set_semihosting_callback(*cb);
            }
            return {cpu_ptr{cpu}, vm_ptr{vm}};
        }),
        core_factory::instance().register_creator("rv32imc|mu_p|asmjit", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_mu_p<iss::arch::rv32imc>();
		    auto vm = new asmjit::rv32imc::vm_impl<arch::rv32imc>(*cpu, false);
		    if (port != 0) debugger::server<debugger::gdb_session>::run_server(vm, port);
            if(init_data){
                auto* cb = reinterpret_cast<semihosting_cb_t<arch::traits<arch::rv32imc>::reg_t>*>(init_data);
                cpu->set_semihosting_callback(*cb);
            }
            return {cpu_ptr{cpu}, vm_ptr{vm}};
        })
};
}
}
// clang-format on
