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
#include <iss/arch/rv32i.h>
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/server.h>
#include <iss/iss.h>
#include <iss/asmjit/vm_base.h>
#include <asmjit/asmjit.h>
#include <util/logging.h>
#include <iss/instruction_decoder.h>

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#include <array>
#include <iss/debugger/riscv_target_adapter.h>

#ifndef _MSC_VER
using int128_t = __int128;
using uint128_t = unsigned __int128;
namespace std {
template <> struct make_unsigned<__int128> { typedef unsigned __int128 type; };
template <> class __make_unsigned_selector<__int128 unsigned, false, false> {
public:
    typedef unsigned __int128 __type;
};
template <> struct is_signed<int128_t> { static constexpr bool value = true; };
template <> struct is_signed<uint128_t> { static constexpr bool value = false; };
template <> struct is_unsigned<int128_t> { static constexpr bool value = false; };
template <> struct is_unsigned<uint128_t> { static constexpr bool value = true; };
} // namespace std
#endif

namespace iss {
namespace asmjit {


namespace rv32i {
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

    continuation_e gen_single_inst_behavior(virt_addr_t&, jit_holder&) override;
    enum globals_e {TVAL = 0, GLOBALS_SIZE};
    void gen_block_prologue(jit_holder& jh) override;
    void gen_block_epilogue(jit_holder& jh) override;
    inline const char *name(size_t index){return traits::reg_aliases.at(index);}

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
    inline void raise(uint16_t trap_id, uint16_t cause){
        auto trap_val =  0x80ULL << 24 | (cause << 16) | trap_id;
        this->core.reg.trap_state = trap_val;
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

    const std::array<instruction_descriptor, 49> instr_descr = {{
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
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
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto new_pc = (uint32_t)(PC+(int32_t)sext<21>(imm));
            if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                gen_set_tval(jh, new_pc);
                gen_raise(jh, 0, 0);
            }
            else {
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
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto addr_mask = (uint32_t)- 2;
            auto new_pc = gen_ext(cc, 
                (gen_operation(cc, band, (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), addr_mask)), 32, true);
            {
            auto label_merge = cc.newLabel();
            auto cond =  gen_operation(cc, urem, new_pc, static_cast<uint32_t>(traits::INSTR_ALIGNMENT));
            cmp(cc, cond, 0);
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
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            {
            auto label_merge = cc.newLabel();
            auto cond =  gen_operation(cc, eq, load_reg_from_mem_Gp(jh, traits::X0 + rs1), load_reg_from_mem_Gp(jh, traits::X0 + rs2));
            cmp(cc, cond, 0);
            cc.je(label_merge);
            {
                auto new_pc = (uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_set_tval(jh, new_pc);
                    gen_raise(jh, 0, 0);
                }
                else {
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
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            {
            auto label_merge = cc.newLabel();
            auto cond =  gen_operation(cc, ne, load_reg_from_mem_Gp(jh, traits::X0 + rs1), load_reg_from_mem_Gp(jh, traits::X0 + rs2));
            cmp(cc, cond, 0);
            cc.je(label_merge);
            {
                auto new_pc = (uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_set_tval(jh, new_pc);
                    gen_raise(jh, 0, 0);
                }
                else {
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
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            {
            auto label_merge = cc.newLabel();
            auto cond =  gen_operation(cc, lt, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false), gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false));
            cmp(cc, cond, 0);
            cc.je(label_merge);
            {
                auto new_pc = (uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_set_tval(jh, new_pc);
                    gen_raise(jh, 0, 0);
                }
                else {
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
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            {
            auto label_merge = cc.newLabel();
            auto cond =  gen_operation(cc, gte, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false), gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false));
            cmp(cc, cond, 0);
            cc.je(label_merge);
            {
                auto new_pc = (uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_set_tval(jh, new_pc);
                    gen_raise(jh, 0, 0);
                }
                else {
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
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            {
            auto label_merge = cc.newLabel();
            auto cond =  gen_operation(cc, ltu, load_reg_from_mem_Gp(jh, traits::X0 + rs1), load_reg_from_mem_Gp(jh, traits::X0 + rs2));
            cmp(cc, cond, 0);
            cc.je(label_merge);
            {
                auto new_pc = (uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_set_tval(jh, new_pc);
                    gen_raise(jh, 0, 0);
                }
                else {
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
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            {
            auto label_merge = cc.newLabel();
            auto cond =  gen_operation(cc, gteu, load_reg_from_mem_Gp(jh, traits::X0 + rs1), load_reg_from_mem_Gp(jh, traits::X0 + rs2));
            cmp(cc, cond, 0);
            cc.je(label_merge);
            {
                auto new_pc = (uint32_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                    gen_set_tval(jh, new_pc);
                    gen_raise(jh, 0, 0);
                }
                else {
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto load_address = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 32, true);
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto load_address = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 32, true);
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto load_address = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 32, true);
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto load_address = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 32, true);
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto load_address = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 32, true);
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
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto store_address = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 32, true);
            gen_write_mem(jh, traits::MEM, store_address, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 8, false), 1);
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
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto store_address = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 32, true);
            gen_write_mem(jh, traits::MEM, store_address, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 16, false), 2);
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
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto store_address = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 32, true);
            gen_write_mem(jh, traits::MEM, store_address, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false), 4);
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 32, true));
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                auto label_then1 = cc.newLabel();
                auto label_merge1 = cc.newLabel();
                auto tmp_reg1 = get_reg_Gp(cc, 8, false);
                cmp(cc, gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, true), (int16_t)sext<12>(imm));
                cc.jl(label_then1);
                mov(cc, tmp_reg1,0);
                cc.jmp(label_merge1);
                cc.bind(label_then1);
                mov(cc, tmp_reg1, 1);
                cc.bind(label_merge1);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, tmp_reg1
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                auto label_then2 = cc.newLabel();
                auto label_merge2 = cc.newLabel();
                auto tmp_reg2 = get_reg_Gp(cc, 8, false);
                cmp(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (uint32_t)((int16_t)sext<12>(imm)));
                cc.jb(label_then2);
                mov(cc, tmp_reg2,0);
                cc.jmp(label_merge2);
                cc.bind(label_then2);
                mov(cc, tmp_reg2, 1);
                cc.bind(label_merge2);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, tmp_reg2
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, bxor, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (uint32_t)((int16_t)sext<12>(imm))));
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, bor, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (uint32_t)((int16_t)sext<12>(imm))));
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, band, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (uint32_t)((int16_t)sext<12>(imm))));
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, shl, load_reg_from_mem_Gp(jh, traits::X0 + rs1), shamt));
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, shr, load_reg_from_mem_Gp(jh, traits::X0 + rs1), shamt));
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, (gen_operation(cc, sar, gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, true), shamt)), 32, false));
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), load_reg_from_mem_Gp(jh, traits::X0 + rs2))), 32, false));
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_operation(cc, sub, load_reg_from_mem_Gp(jh, traits::X0 + rs1), load_reg_from_mem_Gp(jh, traits::X0 + rs2))), 32, true));
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, shl, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (gen_operation(cc, band, load_reg_from_mem_Gp(jh, traits::X0 + rs2), (static_cast<uint32_t>(traits::XLEN)-1)))));
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                auto label_then3 = cc.newLabel();
                auto label_merge3 = cc.newLabel();
                auto tmp_reg3 = get_reg_Gp(cc, 8, false);
                cmp(cc, gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, true), gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, true));
                cc.jl(label_then3);
                mov(cc, tmp_reg3,0);
                cc.jmp(label_merge3);
                cc.bind(label_then3);
                mov(cc, tmp_reg3, 1);
                cc.bind(label_merge3);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, tmp_reg3
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                auto label_then4 = cc.newLabel();
                auto label_merge4 = cc.newLabel();
                auto tmp_reg4 = get_reg_Gp(cc, 8, false);
                cmp(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs1), load_reg_from_mem_Gp(jh, traits::X0 + rs2));
                cc.jb(label_then4);
                mov(cc, tmp_reg4,0);
                cc.jmp(label_merge4);
                cc.bind(label_then4);
                mov(cc, tmp_reg4, 1);
                cc.bind(label_merge4);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, tmp_reg4
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, bxor, load_reg_from_mem_Gp(jh, traits::X0 + rs1), load_reg_from_mem_Gp(jh, traits::X0 + rs2)));
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, shr, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (gen_operation(cc, band, load_reg_from_mem_Gp(jh, traits::X0 + rs2), (static_cast<uint32_t>(traits::XLEN)-1)))));
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_operation(cc, sar, gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, true), (gen_operation(cc, band, load_reg_from_mem_Gp(jh, traits::X0 + rs2), (static_cast<uint32_t>(traits::XLEN)-1))))), 32, true));
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, bor, load_reg_from_mem_Gp(jh, traits::X0 + rs1), load_reg_from_mem_Gp(jh, traits::X0 + rs2)));
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_operation(cc, band, load_reg_from_mem_Gp(jh, traits::X0 + rs1), load_reg_from_mem_Gp(jh, traits::X0 + rs2)));
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
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
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
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
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
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
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
        /*generate behavior*/
        InvokeNode* call_wait_5;
        cc.invoke(&call_wait_5,  &wait, FuncSignature::build<void, uint32_t>());
        setArg(call_wait_5, 0, 1);
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto xrs1 = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            if(rd!=0){
                auto xrd = gen_read_mem(jh, traits::CSR, csr, 4);
                gen_write_mem(jh, traits::CSR, csr, xrs1, 4);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      xrd);
            }
            else {
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 4);
            auto xrs1 = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            if(rs1!=0){
                gen_write_mem(jh, traits::CSR, csr, gen_operation(cc, bor, xrd, xrs1), 4);
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 4);
            auto xrs1 = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            if(rs1!=0){
                gen_write_mem(jh, traits::CSR, csr, gen_operation(cc, band, xrd, gen_operation(cc, bnot, xrs1)), 4);
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 4);
            if(zimm!=0){
                gen_write_mem(jh, traits::CSR, csr, gen_operation(cc, bor, xrd, (uint32_t)zimm), 4);
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 4);
            if(zimm!=0){
                gen_write_mem(jh, traits::CSR, csr, gen_operation(cc, band, xrd, ~ ((uint32_t)zimm)), 4);
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
        /*generate behavior*/
        gen_write_mem(jh, traits::FENCE, static_cast<uint32_t>(traits::fencei), imm, 4);
        auto returnValue = FLUSH;
        
        gen_sync(jh, POST_SYNC, 48);
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
continuation_e vm_impl<ARCH>::gen_single_inst_behavior(virt_addr_t &pc, jit_holder& jh) {
    enum {TRAP_ID=1<<16};
    code_word_t instr = 0;
    phys_addr_t paddr(pc);
    auto *const data = (uint8_t *)&instr;
    auto res = this->core.read(paddr, 4, data);
    if (res != iss::Ok)
        return ILLEGAL_FETCH;
    if (instr == 0x0000006f || (instr&0xffff)==0xa001)
        return JUMP_TO_SELF;
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

    x86_reg_t current_trap_state = get_reg_for(cc, traits::TRAP_STATE);
    mov(cc, current_trap_state, get_ptr_for(jh, traits::TRAP_STATE));
    mov(cc, get_ptr_for(jh, traits::PENDING_TRAP), current_trap_state);
    cc.comment("//Instruction prologue end");

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
    cc.inc(get_ptr_for(jh, traits::CYCLE));
    cc.comment("//Instruction epilogue end");

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
std::unique_ptr<vm_if> create<arch::rv32i>(arch::rv32i *core, unsigned short port, bool dump) {
    auto ret = new rv32i::vm_impl<arch::rv32i>(*core, dump);
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
        core_factory::instance().register_creator("rv32i|m_p|asmjit", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_m_p<iss::arch::rv32i>();
		    auto vm = new asmjit::rv32i::vm_impl<arch::rv32i>(*cpu, false);
		    if (port != 0) debugger::server<debugger::gdb_session>::run_server(vm, port);
            if(init_data){
                auto* cb = reinterpret_cast<semihosting_cb_t<arch::traits<arch::rv32i>::reg_t>*>(init_data);
                cpu->set_semihosting_callback(*cb);
            }
            return {cpu_ptr{cpu}, vm_ptr{vm}};
        }),
        core_factory::instance().register_creator("rv32i|mu_p|asmjit", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_mu_p<iss::arch::rv32i>();
		    auto vm = new asmjit::rv32i::vm_impl<arch::rv32i>(*cpu, false);
		    if (port != 0) debugger::server<debugger::gdb_session>::run_server(vm, port);
            if(init_data){
                auto* cb = reinterpret_cast<semihosting_cb_t<arch::traits<arch::rv32i>::reg_t>*>(init_data);
                cpu->set_semihosting_callback(*cb);
            }
            return {cpu_ptr{cpu}, vm_ptr{vm}};
        })
};
}
}
// clang-format on
