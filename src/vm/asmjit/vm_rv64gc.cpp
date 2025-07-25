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
#include <iss/arch/rv64gc.h>
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


namespace rv64gc {
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
    inline void raise(uint16_t trap_id, uint16_t cause){
        auto trap_val =  0x80ULL << 24 | (cause << 16) | trap_id;
        this->core.reg.trap_state = trap_val;
    }


    static uint64_t _NaNBox32(void* vm_impl_ptr , uint32_t val){
         return reinterpret_cast<this_class*>(vm_impl_ptr)->NaNBox32(val);
    }
    uint64_t NaNBox32(uint32_t val){
        if(traits::FLEN == 32) {
            return (uint64_t)val;
        }
        else {
            uint64_t box = ~ ((uint64_t)0);
            return (uint64_t)(((uint128_t)box<<32)|val);
        }
    }
    static uint8_t _get_rm(void* vm_impl_ptr , uint8_t rm){
         return reinterpret_cast<this_class*>(vm_impl_ptr)->get_rm(rm);
    }
    uint8_t get_rm(uint8_t rm){
        auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+::iss::arch::traits<ARCH>::reg_byte_offsets[::iss::arch::traits<ARCH>::FCSR]); 
        uint8_t rm_eff = rm == 7? bit_sub<5, 7-5+1>(*FCSR) : rm;
        if(rm_eff > 4) {
            raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
        }
        return rm_eff;
    }
    static uint64_t _NaNBox64(void* vm_impl_ptr , uint64_t val){
         return reinterpret_cast<this_class*>(vm_impl_ptr)->NaNBox64(val);
    }
    uint64_t NaNBox64(uint64_t val){
        if(traits::FLEN == 64) {
            return (uint64_t)val;
        }
        else {
            uint64_t box = ~ ((uint64_t)0);
            return (uint64_t)(((uint128_t)box<<64)|val);
        }
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

    const std::array<instruction_descriptor, 198> instr_descr = {{
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
        /* instruction MULW, encoding '0b00000010000000000000000000111011' */
        {32, 0b00000010000000000000000000111011, 0b11111110000000000111000001111111, &this_class::__mulw},
        /* instruction DIVW, encoding '0b00000010000000000100000000111011' */
        {32, 0b00000010000000000100000000111011, 0b11111110000000000111000001111111, &this_class::__divw},
        /* instruction DIVUW, encoding '0b00000010000000000101000000111011' */
        {32, 0b00000010000000000101000000111011, 0b11111110000000000111000001111111, &this_class::__divuw},
        /* instruction REMW, encoding '0b00000010000000000110000000111011' */
        {32, 0b00000010000000000110000000111011, 0b11111110000000000111000001111111, &this_class::__remw},
        /* instruction REMUW, encoding '0b00000010000000000111000000111011' */
        {32, 0b00000010000000000111000000111011, 0b11111110000000000111000001111111, &this_class::__remuw},
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
        /* instruction LRD, encoding '0b00010000000000000011000000101111' */
        {32, 0b00010000000000000011000000101111, 0b11111001111100000111000001111111, &this_class::__lrd},
        /* instruction SCD, encoding '0b00011000000000000011000000101111' */
        {32, 0b00011000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__scd},
        /* instruction AMOSWAPD, encoding '0b00001000000000000011000000101111' */
        {32, 0b00001000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amoswapd},
        /* instruction AMOADDD, encoding '0b00000000000000000011000000101111' */
        {32, 0b00000000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amoaddd},
        /* instruction AMOXORD, encoding '0b00100000000000000011000000101111' */
        {32, 0b00100000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amoxord},
        /* instruction AMOANDD, encoding '0b01100000000000000011000000101111' */
        {32, 0b01100000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amoandd},
        /* instruction AMOORD, encoding '0b01000000000000000011000000101111' */
        {32, 0b01000000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amoord},
        /* instruction AMOMIND, encoding '0b10000000000000000011000000101111' */
        {32, 0b10000000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amomind},
        /* instruction AMOMAXD, encoding '0b10100000000000000011000000101111' */
        {32, 0b10100000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amomaxd},
        /* instruction AMOMINUD, encoding '0b11000000000000000011000000101111' */
        {32, 0b11000000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amominud},
        /* instruction AMOMAXUD, encoding '0b11100000000000000011000000101111' */
        {32, 0b11100000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amomaxud},
        /* instruction C__ADDI4SPN, encoding '0b0000000000000000' */
        {16, 0b0000000000000000, 0b1110000000000011, &this_class::__c__addi4spn},
        /* instruction C__LW, encoding '0b0100000000000000' */
        {16, 0b0100000000000000, 0b1110000000000011, &this_class::__c__lw},
        /* instruction C__LD, encoding '0b0110000000000000' */
        {16, 0b0110000000000000, 0b1110000000000011, &this_class::__c__ld},
        /* instruction C__SW, encoding '0b1100000000000000' */
        {16, 0b1100000000000000, 0b1110000000000011, &this_class::__c__sw},
        /* instruction C__SD, encoding '0b1110000000000000' */
        {16, 0b1110000000000000, 0b1110000000000011, &this_class::__c__sd},
        /* instruction C__ADDI, encoding '0b0000000000000001' */
        {16, 0b0000000000000001, 0b1110000000000011, &this_class::__c__addi},
        /* instruction C__NOP, encoding '0b0000000000000001' */
        {16, 0b0000000000000001, 0b1110111110000011, &this_class::__c__nop},
        /* instruction C__ADDIW, encoding '0b0010000000000001' */
        {16, 0b0010000000000001, 0b1110000000000011, &this_class::__c__addiw},
        /* instruction C__LI, encoding '0b0100000000000001' */
        {16, 0b0100000000000001, 0b1110000000000011, &this_class::__c__li},
        /* instruction C__LUI, encoding '0b0110000000000001' */
        {16, 0b0110000000000001, 0b1110000000000011, &this_class::__c__lui},
        /* instruction C__ADDI16SP, encoding '0b0110000100000001' */
        {16, 0b0110000100000001, 0b1110111110000011, &this_class::__c__addi16sp},
        /* instruction __reserved_clui, encoding '0b0110000000000001' */
        {16, 0b0110000000000001, 0b1111000001111111, &this_class::____reserved_clui},
        /* instruction C__SRLI, encoding '0b1000000000000001' */
        {16, 0b1000000000000001, 0b1110110000000011, &this_class::__c__srli},
        /* instruction C__SRAI, encoding '0b1000010000000001' */
        {16, 0b1000010000000001, 0b1110110000000011, &this_class::__c__srai},
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
        /* instruction C__SUBW, encoding '0b1001110000000001' */
        {16, 0b1001110000000001, 0b1111110001100011, &this_class::__c__subw},
        /* instruction C__ADDW, encoding '0b1001110000100001' */
        {16, 0b1001110000100001, 0b1111110001100011, &this_class::__c__addw},
        /* instruction C__J, encoding '0b1010000000000001' */
        {16, 0b1010000000000001, 0b1110000000000011, &this_class::__c__j},
        /* instruction C__BEQZ, encoding '0b1100000000000001' */
        {16, 0b1100000000000001, 0b1110000000000011, &this_class::__c__beqz},
        /* instruction C__BNEZ, encoding '0b1110000000000001' */
        {16, 0b1110000000000001, 0b1110000000000011, &this_class::__c__bnez},
        /* instruction C__SLLI, encoding '0b0000000000000010' */
        {16, 0b0000000000000010, 0b1110000000000011, &this_class::__c__slli},
        /* instruction C__LWSP, encoding '0b0100000000000010' */
        {16, 0b0100000000000010, 0b1110000000000011, &this_class::__c__lwsp},
        /* instruction C__LDSP, encoding '0b0110000000000010' */
        {16, 0b0110000000000010, 0b1110000000000011, &this_class::__c__ldsp},
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
        /* instruction C__SDSP, encoding '0b1110000000000010' */
        {16, 0b1110000000000010, 0b1110000000000011, &this_class::__c__sdsp},
        /* instruction DII, encoding '0b0000000000000000' */
        {16, 0b0000000000000000, 0b1111111111111111, &this_class::__dii},
        /* instruction FLW, encoding '0b00000000000000000010000000000111' */
        {32, 0b00000000000000000010000000000111, 0b00000000000000000111000001111111, &this_class::__flw},
        /* instruction FSW, encoding '0b00000000000000000010000000100111' */
        {32, 0b00000000000000000010000000100111, 0b00000000000000000111000001111111, &this_class::__fsw},
        /* instruction FADD__S, encoding '0b00000000000000000000000001010011' */
        {32, 0b00000000000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fadd__s},
        /* instruction FSUB__S, encoding '0b00001000000000000000000001010011' */
        {32, 0b00001000000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fsub__s},
        /* instruction FMUL__S, encoding '0b00010000000000000000000001010011' */
        {32, 0b00010000000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fmul__s},
        /* instruction FDIV__S, encoding '0b00011000000000000000000001010011' */
        {32, 0b00011000000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fdiv__s},
        /* instruction FMIN__S, encoding '0b00101000000000000000000001010011' */
        {32, 0b00101000000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fmin__s},
        /* instruction FMAX__S, encoding '0b00101000000000000001000001010011' */
        {32, 0b00101000000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__fmax__s},
        /* instruction FSQRT__S, encoding '0b01011000000000000000000001010011' */
        {32, 0b01011000000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fsqrt__s},
        /* instruction FMADD__S, encoding '0b00000000000000000000000001000011' */
        {32, 0b00000000000000000000000001000011, 0b00000110000000000000000001111111, &this_class::__fmadd__s},
        /* instruction FMSUB__S, encoding '0b00000000000000000000000001000111' */
        {32, 0b00000000000000000000000001000111, 0b00000110000000000000000001111111, &this_class::__fmsub__s},
        /* instruction FNMADD__S, encoding '0b00000000000000000000000001001111' */
        {32, 0b00000000000000000000000001001111, 0b00000110000000000000000001111111, &this_class::__fnmadd__s},
        /* instruction FNMSUB__S, encoding '0b00000000000000000000000001001011' */
        {32, 0b00000000000000000000000001001011, 0b00000110000000000000000001111111, &this_class::__fnmsub__s},
        /* instruction FCVT__W__S, encoding '0b11000000000000000000000001010011' */
        {32, 0b11000000000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__w__s},
        /* instruction FCVT__WU__S, encoding '0b11000000000100000000000001010011' */
        {32, 0b11000000000100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__wu__s},
        /* instruction FCVT__L__S, encoding '0b11000000001000000000000001010011' */
        {32, 0b11000000001000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__l__s},
        /* instruction FCVT__LU__S, encoding '0b11000000001100000000000001010011' */
        {32, 0b11000000001100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__lu__s},
        /* instruction FCVT__S__W, encoding '0b11010000000000000000000001010011' */
        {32, 0b11010000000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__s__w},
        /* instruction FCVT__S__WU, encoding '0b11010000000100000000000001010011' */
        {32, 0b11010000000100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__s__wu},
        /* instruction FCVT__S__L, encoding '0b11010000001000000000000001010011' */
        {32, 0b11010000001000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__s__l},
        /* instruction FCVT__S__LU, encoding '0b11010000001100000000000001010011' */
        {32, 0b11010000001100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__s__lu},
        /* instruction FSGNJ__S, encoding '0b00100000000000000000000001010011' */
        {32, 0b00100000000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnj__s},
        /* instruction FSGNJN__S, encoding '0b00100000000000000001000001010011' */
        {32, 0b00100000000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnjn__s},
        /* instruction FSGNJX__S, encoding '0b00100000000000000010000001010011' */
        {32, 0b00100000000000000010000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnjx__s},
        /* instruction FMV__X__W, encoding '0b11100000000000000000000001010011' */
        {32, 0b11100000000000000000000001010011, 0b11111111111100000111000001111111, &this_class::__fmv__x__w},
        /* instruction FMV__W__X, encoding '0b11110000000000000000000001010011' */
        {32, 0b11110000000000000000000001010011, 0b11111111111100000111000001111111, &this_class::__fmv__w__x},
        /* instruction FEQ__S, encoding '0b10100000000000000010000001010011' */
        {32, 0b10100000000000000010000001010011, 0b11111110000000000111000001111111, &this_class::__feq__s},
        /* instruction FLT__S, encoding '0b10100000000000000001000001010011' */
        {32, 0b10100000000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__flt__s},
        /* instruction FLE__S, encoding '0b10100000000000000000000001010011' */
        {32, 0b10100000000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fle__s},
        /* instruction FCLASS__S, encoding '0b11100000000000000001000001010011' */
        {32, 0b11100000000000000001000001010011, 0b11111111111100000111000001111111, &this_class::__fclass__s},
        /* instruction FLD, encoding '0b00000000000000000011000000000111' */
        {32, 0b00000000000000000011000000000111, 0b00000000000000000111000001111111, &this_class::__fld},
        /* instruction FSD, encoding '0b00000000000000000011000000100111' */
        {32, 0b00000000000000000011000000100111, 0b00000000000000000111000001111111, &this_class::__fsd},
        /* instruction FADD__D, encoding '0b00000010000000000000000001010011' */
        {32, 0b00000010000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fadd__d},
        /* instruction FSUB__D, encoding '0b00001010000000000000000001010011' */
        {32, 0b00001010000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fsub__d},
        /* instruction FMUL__D, encoding '0b00010010000000000000000001010011' */
        {32, 0b00010010000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fmul__d},
        /* instruction FDIV__D, encoding '0b00011010000000000000000001010011' */
        {32, 0b00011010000000000000000001010011, 0b11111110000000000000000001111111, &this_class::__fdiv__d},
        /* instruction FMIN__D, encoding '0b00101010000000000000000001010011' */
        {32, 0b00101010000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fmin__d},
        /* instruction FMAX__D, encoding '0b00101010000000000001000001010011' */
        {32, 0b00101010000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__fmax__d},
        /* instruction FSQRT__D, encoding '0b01011010000000000000000001010011' */
        {32, 0b01011010000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fsqrt__d},
        /* instruction FMADD__D, encoding '0b00000010000000000000000001000011' */
        {32, 0b00000010000000000000000001000011, 0b00000110000000000000000001111111, &this_class::__fmadd__d},
        /* instruction FMSUB__D, encoding '0b00000010000000000000000001000111' */
        {32, 0b00000010000000000000000001000111, 0b00000110000000000000000001111111, &this_class::__fmsub__d},
        /* instruction FNMADD__D, encoding '0b00000010000000000000000001001111' */
        {32, 0b00000010000000000000000001001111, 0b00000110000000000000000001111111, &this_class::__fnmadd__d},
        /* instruction FNMSUB__D, encoding '0b00000010000000000000000001001011' */
        {32, 0b00000010000000000000000001001011, 0b00000110000000000000000001111111, &this_class::__fnmsub__d},
        /* instruction FCVT__W__D, encoding '0b11000010000000000000000001010011' */
        {32, 0b11000010000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__w__d},
        /* instruction FCVT__WU__D, encoding '0b11000010000100000000000001010011' */
        {32, 0b11000010000100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__wu__d},
        /* instruction FCVT__L__D, encoding '0b11000010001000000000000001010011' */
        {32, 0b11000010001000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__l__d},
        /* instruction FCVT__LU__D, encoding '0b11000010001100000000000001010011' */
        {32, 0b11000010001100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__lu__d},
        /* instruction FCVT__D__W, encoding '0b11010010000000000000000001010011' */
        {32, 0b11010010000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__d__w},
        /* instruction FCVT__D__WU, encoding '0b11010010000100000000000001010011' */
        {32, 0b11010010000100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__d__wu},
        /* instruction FCVT__D__L, encoding '0b11010010001000000000000001010011' */
        {32, 0b11010010001000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__d__l},
        /* instruction FCVT__D__LU, encoding '0b11010010001100000000000001010011' */
        {32, 0b11010010001100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__d__lu},
        /* instruction FCVT__S__D, encoding '0b01000000000100000000000001010011' */
        {32, 0b01000000000100000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__s__d},
        /* instruction FCVT__D__S, encoding '0b01000010000000000000000001010011' */
        {32, 0b01000010000000000000000001010011, 0b11111111111100000000000001111111, &this_class::__fcvt__d__s},
        /* instruction FSGNJ__D, encoding '0b00100010000000000000000001010011' */
        {32, 0b00100010000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnj__d},
        /* instruction FSGNJN__D, encoding '0b00100010000000000001000001010011' */
        {32, 0b00100010000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnjn__d},
        /* instruction FSGNJX__D, encoding '0b00100010000000000010000001010011' */
        {32, 0b00100010000000000010000001010011, 0b11111110000000000111000001111111, &this_class::__fsgnjx__d},
        /* instruction FMV__X__D, encoding '0b11100010000000000000000001010011' */
        {32, 0b11100010000000000000000001010011, 0b11111111111100000111000001111111, &this_class::__fmv__x__d},
        /* instruction FMV__D__X, encoding '0b11110010000000000000000001010011' */
        {32, 0b11110010000000000000000001010011, 0b11111111111100000111000001111111, &this_class::__fmv__d__x},
        /* instruction FEQ__D, encoding '0b10100010000000000010000001010011' */
        {32, 0b10100010000000000010000001010011, 0b11111110000000000111000001111111, &this_class::__feq__d},
        /* instruction FLT__D, encoding '0b10100010000000000001000001010011' */
        {32, 0b10100010000000000001000001010011, 0b11111110000000000111000001111111, &this_class::__flt__d},
        /* instruction FLE__D, encoding '0b10100010000000000000000001010011' */
        {32, 0b10100010000000000000000001010011, 0b11111110000000000111000001111111, &this_class::__fle__d},
        /* instruction FCLASS__D, encoding '0b11100010000000000001000001010011' */
        {32, 0b11100010000000000001000001010011, 0b11111111111100000111000001111111, &this_class::__fclass__d},
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
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                auto value_reg = get_reg_Gp(cc, 64, false);
                cc.movabs(value_reg, (uint64_t)((int32_t)imm));
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                    value_reg);
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
                auto value_reg = get_reg_Gp(cc, 64, false);
                cc.movabs(value_reg, (uint64_t)(PC+(int32_t)imm));
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                    value_reg);
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
            auto new_pc = (uint64_t)(PC+(int32_t)sext<21>(imm));
            if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){
                gen_set_tval(jh, new_pc);
                gen_raise(jh, 0, 0);
            }
            else {
                if(rd!=0){
                    auto value_reg = get_reg_Gp(cc, 64, false);
                    cc.movabs(value_reg, (uint64_t)(PC+4));
                    mov(cc, get_ptr_for(jh, traits::X0+ rd),
                        value_reg);
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
            auto addr_mask = (uint64_t)- 2;
            auto new_pc = gen_ext(cc, 
                (gen_operation(cc, band, (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), addr_mask)), 64, true);
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
                        auto value_reg = get_reg_Gp(cc, 64, false);
                        cc.movabs(value_reg, (uint64_t)(PC+4));
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                            value_reg);
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
                auto new_pc = (uint64_t)(PC+(int16_t)sext<13>(imm));
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
                auto new_pc = (uint64_t)(PC+(int16_t)sext<13>(imm));
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
                load_reg_from_mem_Gp(jh, traits::X0 + rs1), 64, false), gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 64, false));
            cmp(cc, cond, 0);
            cc.je(label_merge);
            {
                auto new_pc = (uint64_t)(PC+(int16_t)sext<13>(imm));
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
                load_reg_from_mem_Gp(jh, traits::X0 + rs1), 64, false), gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 64, false));
            cmp(cc, cond, 0);
            cc.je(label_merge);
            {
                auto new_pc = (uint64_t)(PC+(int16_t)sext<13>(imm));
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
                auto new_pc = (uint64_t)(PC+(int16_t)sext<13>(imm));
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
                auto new_pc = (uint64_t)(PC+(int16_t)sext<13>(imm));
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
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 64, true);
            auto res = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, load_address, 1), 8, false);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, true));
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
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 64, true);
            auto res = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, load_address, 2), 16, false);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, true));
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
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 64, true);
            auto res = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, load_address, 4), 32, false);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, true));
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
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 64, true);
            auto res = gen_read_mem(jh, traits::MEM, load_address, 1);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, false));
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
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 64, true);
            auto res = gen_read_mem(jh, traits::MEM, load_address, 2);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, false));
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
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 64, true);
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
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 64, true);
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
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 64, true);
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
                          (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 64, true));
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
                cmp(cc, gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs1), 64, true), (int16_t)sext<12>(imm));
                cc.jl(label_then1);
                mov(cc, tmp_reg1,0);
                cc.jmp(label_merge1);
                cc.bind(label_then1);
                mov(cc, tmp_reg1, 1);
                cc.bind(label_merge1);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, tmp_reg1
                      , 64, false)
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
                cmp(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (uint64_t)((int16_t)sext<12>(imm)));
                cc.jb(label_then2);
                mov(cc, tmp_reg2,0);
                cc.jmp(label_merge2);
                cc.bind(label_then2);
                mov(cc, tmp_reg2, 1);
                cc.bind(label_merge2);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, tmp_reg2
                      , 64, false)
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
                      gen_operation(cc, bxor, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (uint64_t)((int16_t)sext<12>(imm))));
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
                      gen_operation(cc, bor, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (uint64_t)((int16_t)sext<12>(imm))));
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
                      gen_operation(cc, band, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (uint64_t)((int16_t)sext<12>(imm))));
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
        uint8_t shamt = ((bit_sub<20,6>(instr)));
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
        uint8_t shamt = ((bit_sub<20,6>(instr)));
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
        uint8_t shamt = ((bit_sub<20,6>(instr)));
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
                      gen_ext(cc, 
                          (gen_operation(cc, sar, (gen_ext(cc, 
                              load_reg_from_mem_Gp(jh, traits::X0 + rs1), 64, false)), shamt)), 64, true));
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
                          (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), load_reg_from_mem_Gp(jh, traits::X0 + rs2))), 64, false));
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
                          (gen_operation(cc, sub, load_reg_from_mem_Gp(jh, traits::X0 + rs1), load_reg_from_mem_Gp(jh, traits::X0 + rs2))), 64, true));
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
                cmp(cc, gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs1), 64, true), gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs2), 64, true));
                cc.jl(label_then3);
                mov(cc, tmp_reg3,0);
                cc.jmp(label_merge3);
                cc.bind(label_then3);
                mov(cc, tmp_reg3, 1);
                cc.bind(label_merge3);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, tmp_reg3
                      , 64, false)
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
                      , 64, false)
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
                          (gen_operation(cc, sar, gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs1), 64, true), (gen_operation(cc, band, load_reg_from_mem_Gp(jh, traits::X0 + rs2), (static_cast<uint32_t>(traits::XLEN)-1))))), 64, true));
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
        gen_write_mem(jh, traits::FENCE, static_cast<uint32_t>(traits::fence), (uint8_t)pred<<4|succ, 8);
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
    
    /* instruction 42: LWU */
    continuation_e __lwu(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lwu"),
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
        cc.comment(fmt::format("LWU_{:#x}:",pc.val).c_str());
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
            auto offs = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 64, true);
            auto res = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, offs, 4), 32, false);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, false));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 42);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 43: LD */
    continuation_e __ld(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "ld"),
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
        cc.comment(fmt::format("LD_{:#x}:",pc.val).c_str());
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
            auto offs = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 64, true);
            auto res = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, offs, 8), 64, false);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 43);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 44: SD */
    continuation_e __sd(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sd"),
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
        cc.comment(fmt::format("SD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 44);
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
            auto offs = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 64, true);
            gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 64, false), 8);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 44);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 45: ADDIW */
    continuation_e __addiw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "addiw"),
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
        cc.comment(fmt::format("ADDIW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 45);
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
                auto res = gen_ext(cc, 
                    (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 32, true);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 45);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 46: SLLIW */
    continuation_e __slliw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "slliw"),
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
        cc.comment(fmt::format("SLLIW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 46);
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
                auto sh_val = gen_operation(cc, shl, (gen_ext(cc, 
                    load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false)), shamt);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          gen_ext(cc, sh_val, 32, true), 64, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 46);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 47: SRLIW */
    continuation_e __srliw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srliw"),
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
        cc.comment(fmt::format("SRLIW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 47);
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
                auto sh_val = gen_operation(cc, shr, (gen_ext(cc, 
                    load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false)), shamt);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          gen_ext(cc, sh_val, 32, true), 64, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 47);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 48: SRAIW */
    continuation_e __sraiw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "sraiw"),
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
        cc.comment(fmt::format("SRAIW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 48);
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
                auto sh_val = gen_operation(cc, sar, (gen_ext(cc, 
                    load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false)), shamt);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          sh_val, 64, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 48);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 49: ADDW */
    continuation_e __addw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            //No disass specified, using instruction name
            std::string mnemonic = "addw";
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("ADDW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 49);
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
                auto res = gen_ext(cc, 
                    (gen_operation(cc, add, gen_ext(cc, 
                        load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false), gen_ext(cc, 
                        load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false))), 32, true);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 49);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 50: SUBW */
    continuation_e __subw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            //No disass specified, using instruction name
            std::string mnemonic = "subw";
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("SUBW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 50);
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
                auto res = gen_ext(cc, 
                    (gen_operation(cc, sub, gen_ext(cc, 
                        load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false), gen_ext(cc, 
                        load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false))), 32, true);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 50);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 51: SLLW */
    continuation_e __sllw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sllw"),
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
        cc.comment(fmt::format("SLLW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 51);
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
                auto count = gen_operation(cc, band, gen_ext(cc, 
                    load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false), 31);
                auto sh_val = gen_operation(cc, shl, (gen_ext(cc, 
                    load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false)), count);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          gen_ext(cc, sh_val, 32, true), 64, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 51);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 52: SRLW */
    continuation_e __srlw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "srlw"),
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
        cc.comment(fmt::format("SRLW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 52);
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
                auto count = gen_operation(cc, band, gen_ext(cc, 
                    load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false), 31);
                auto sh_val = gen_operation(cc, shr, (gen_ext(cc, 
                    load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false)), count);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          gen_ext(cc, sh_val, 32, true), 64, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 52);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 53: SRAW */
    continuation_e __sraw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sraw"),
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
        cc.comment(fmt::format("SRAW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 53);
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
                auto count = gen_operation(cc, band, gen_ext(cc, 
                    load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false), 31);
                auto sh_val = gen_operation(cc, sar, (gen_ext(cc, 
                    load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false)), count);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          sh_val, 64, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 53);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 54: CSRRW */
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
        gen_sync(jh, PRE_SYNC, 54);
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
                auto xrd = gen_read_mem(jh, traits::CSR, csr, 8);
                gen_write_mem(jh, traits::CSR, csr, xrs1, 8);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      xrd);
            }
            else {
                gen_write_mem(jh, traits::CSR, csr, xrs1, 8);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 54);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 55: CSRRS */
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
        gen_sync(jh, PRE_SYNC, 55);
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
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 8);
            auto xrs1 = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            if(rs1!=0){
                gen_write_mem(jh, traits::CSR, csr, gen_operation(cc, bor, xrd, xrs1), 8);
            }
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      xrd);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 55);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 56: CSRRC */
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
        gen_sync(jh, PRE_SYNC, 56);
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
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 8);
            auto xrs1 = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            if(rs1!=0){
                gen_write_mem(jh, traits::CSR, csr, gen_operation(cc, band, xrd, gen_operation(cc, bnot, xrs1)), 8);
            }
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      xrd);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 56);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 57: CSRRWI */
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
        gen_sync(jh, PRE_SYNC, 57);
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
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 8);
            gen_write_mem(jh, traits::CSR, csr, (uint64_t)zimm, 8);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      xrd);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 57);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 58: CSRRSI */
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
        gen_sync(jh, PRE_SYNC, 58);
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
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 8);
            if(zimm!=0){
                gen_write_mem(jh, traits::CSR, csr, gen_operation(cc, bor, xrd, (uint64_t)zimm), 8);
            }
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      xrd);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 58);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 59: CSRRCI */
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
        gen_sync(jh, PRE_SYNC, 59);
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
            auto xrd = gen_read_mem(jh, traits::CSR, csr, 8);
            if(zimm!=0){
                gen_write_mem(jh, traits::CSR, csr, gen_operation(cc, band, xrd, ~ ((uint64_t)zimm)), 8);
            }
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      xrd);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 59);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 60: FENCE_I */
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
        gen_sync(jh, PRE_SYNC, 60);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        gen_write_mem(jh, traits::FENCE, static_cast<uint32_t>(traits::fencei), imm, 8);
        auto returnValue = FLUSH;
        
        gen_sync(jh, POST_SYNC, 60);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 61: MUL */
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
        gen_sync(jh, PRE_SYNC, 61);
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
            auto res = gen_operation(cc, smul, gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs1), 64, true), gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs2), 64, true));
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 61);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 62: MULH */
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
        gen_sync(jh, PRE_SYNC, 62);
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
            auto res = gen_operation(cc, smul, gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs1), 64, true), gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs2), 64, true));
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_operation(cc, sar, res, static_cast<uint32_t>(traits::XLEN))), 64, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 62);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 63: MULHSU */
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
        gen_sync(jh, PRE_SYNC, 63);
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
            auto res = gen_operation(cc, sumul, gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs1), 64, true), load_reg_from_mem_Gp(jh, traits::X0 + rs2));
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_operation(cc, sar, res, static_cast<uint32_t>(traits::XLEN))), 64, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 63);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 64: MULHU */
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
        gen_sync(jh, PRE_SYNC, 64);
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
            auto res = gen_operation(cc, umul, load_reg_from_mem_Gp(jh, traits::X0 + rs1), load_reg_from_mem_Gp(jh, traits::X0 + rs2));
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_operation(cc, shr, res, static_cast<uint32_t>(traits::XLEN))), 64, false));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 64);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 65: DIV */
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
        gen_sync(jh, PRE_SYNC, 65);
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
            auto dividend = gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs1), 64, true);
            auto divisor = gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs2), 64, true);
            if(rd!=0){
                {
                auto label_merge = cc.newLabel();
                auto cond =  gen_operation(cc, ne, divisor, 0);
                cmp(cc, cond, 0);
                auto label_else = cc.newLabel();
                cc.je(label_else);
                {
                    auto MMIN = ((uint64_t)1)<<(static_cast<uint32_t>(traits::XLEN)-1);
                    {
                    auto label_merge = cc.newLabel();
                    auto cond =  gen_operation(cc, land, gen_operation(cc, eq, load_reg_from_mem_Gp(jh, traits::X0 + rs1), MMIN), gen_operation(cc, eq, divisor, - 1));
                    cmp(cc, cond, 0);
                    auto label_else = cc.newLabel();
                    cc.je(label_else);
                    {
                        auto value_reg = get_reg_Gp(cc, 64, false);
                        cc.movabs(value_reg, MMIN);
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                            value_reg);
                    }
                    cc.jmp(label_merge);
                    cc.bind(label_else);
                        {
                            mov(cc, get_ptr_for(jh, traits::X0+ rd),
                                  gen_ext(cc, 
                                      (gen_operation(cc, sdiv, dividend, divisor)), 64, true));
                        }
                    cc.bind(label_merge);
                    }
                }
                cc.jmp(label_merge);
                cc.bind(label_else);
                    {
                        auto value_reg = get_reg_Gp(cc, 64, false);
                        cc.movabs(value_reg, (uint64_t)- 1);
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                            value_reg);
                    }
                cc.bind(label_merge);
                }
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 65);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 66: DIVU */
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
        gen_sync(jh, PRE_SYNC, 66);
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
            {
            auto label_merge = cc.newLabel();
            auto cond =  gen_operation(cc, ne, load_reg_from_mem_Gp(jh, traits::X0 + rs2), 0);
            cmp(cc, cond, 0);
            auto label_else = cc.newLabel();
            cc.je(label_else);
            {
                if(rd!=0){
                    mov(cc, get_ptr_for(jh, traits::X0+ rd),
                          gen_operation(cc, udiv, load_reg_from_mem_Gp(jh, traits::X0 + rs1), load_reg_from_mem_Gp(jh, traits::X0 + rs2)));
                }
            }
            cc.jmp(label_merge);
            cc.bind(label_else);
                {
                    if(rd!=0){
                        auto value_reg = get_reg_Gp(cc, 64, false);
                        cc.movabs(value_reg, (uint64_t)- 1);
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                            value_reg);
                    }
                }
            cc.bind(label_merge);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 66);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 67: REM */
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
        gen_sync(jh, PRE_SYNC, 67);
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
            {
            auto label_merge = cc.newLabel();
            auto cond =  gen_operation(cc, ne, load_reg_from_mem_Gp(jh, traits::X0 + rs2), 0);
            cmp(cc, cond, 0);
            auto label_else = cc.newLabel();
            cc.je(label_else);
            {
                auto MMIN = (uint64_t)1<<(static_cast<uint32_t>(traits::XLEN)-1);
                {
                auto label_merge = cc.newLabel();
                auto cond =  gen_operation(cc, land, gen_operation(cc, eq, load_reg_from_mem_Gp(jh, traits::X0 + rs1), MMIN), gen_operation(cc, eq, gen_ext(cc, 
                    load_reg_from_mem_Gp(jh, traits::X0 + rs2), 64, false), - 1));
                cmp(cc, cond, 0);
                auto label_else = cc.newLabel();
                cc.je(label_else);
                {
                    if(rd!=0){
                        auto value_reg = get_reg_Gp(cc, 64, false);
                        cc.movabs(value_reg, 0);
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                            value_reg);
                    }
                }
                cc.jmp(label_merge);
                cc.bind(label_else);
                    {
                        if(rd!=0){
                            mov(cc, get_ptr_for(jh, traits::X0+ rd),
                                  gen_ext(cc, (gen_operation(cc, srem, gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs1), 64, true), gen_ext(cc, load_reg_from_mem_Gp(jh, traits::X0 + rs2), 64, true))), 64, false));
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
                              load_reg_from_mem_Gp(jh, traits::X0 + rs1));
                    }
                }
            cc.bind(label_merge);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 67);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 68: REMU */
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
        gen_sync(jh, PRE_SYNC, 68);
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
            {
            auto label_merge = cc.newLabel();
            auto cond =  gen_operation(cc, ne, load_reg_from_mem_Gp(jh, traits::X0 + rs2), 0);
            cmp(cc, cond, 0);
            auto label_else = cc.newLabel();
            cc.je(label_else);
            {
                if(rd!=0){
                    mov(cc, get_ptr_for(jh, traits::X0+ rd),
                          gen_operation(cc, urem, load_reg_from_mem_Gp(jh, traits::X0 + rs1), load_reg_from_mem_Gp(jh, traits::X0 + rs2)));
                }
            }
            cc.jmp(label_merge);
            cc.bind(label_else);
                {
                    if(rd!=0){
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                              load_reg_from_mem_Gp(jh, traits::X0 + rs1));
                    }
                }
            cc.bind(label_merge);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 68);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 69: MULW */
    continuation_e __mulw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "mulw"),
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
        cc.comment(fmt::format("MULW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 69);
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
                auto resw = gen_ext(cc, 
                    (gen_operation(cc, smul, gen_ext(cc, 
                        load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false), gen_ext(cc, 
                        load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false))), 32, true);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, gen_ext(cc, 
                          resw, 64, true), 64, false));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 69);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 70: DIVW */
    continuation_e __divw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "divw"),
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
        cc.comment(fmt::format("DIVW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 70);
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
            auto dividend = gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false);
            auto divisor = gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false);
            {
            auto label_merge = cc.newLabel();
            auto cond =  gen_operation(cc, ne, divisor, 0);
            cmp(cc, cond, 0);
            auto label_else = cc.newLabel();
            cc.je(label_else);
            {
                auto MMIN = (int32_t)1<<31;
                {
                auto label_merge = cc.newLabel();
                auto cond =  gen_operation(cc, land, gen_operation(cc, eq, dividend, MMIN), gen_operation(cc, eq, divisor, - 1));
                cmp(cc, cond, 0);
                auto label_else = cc.newLabel();
                cc.je(label_else);
                {
                    if(rd!=0){
                        auto value_reg = get_reg_Gp(cc, 64, false);
                        cc.movabs(value_reg, (uint64_t)- 1<<31);
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                            value_reg);
                    }
                }
                cc.jmp(label_merge);
                cc.bind(label_else);
                    {
                        if(rd!=0){
                            mov(cc, get_ptr_for(jh, traits::X0+ rd),
                                  gen_ext(cc, 
                                      (gen_operation(cc, sdiv, dividend, divisor)), 64, true));
                        }
                    }
                cc.bind(label_merge);
                }
            }
            cc.jmp(label_merge);
            cc.bind(label_else);
                {
                    if(rd!=0){
                        auto value_reg = get_reg_Gp(cc, 64, false);
                        cc.movabs(value_reg, (uint64_t)- 1);
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                            value_reg);
                    }
                }
            cc.bind(label_merge);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 70);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 71: DIVUW */
    continuation_e __divuw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "divuw"),
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
        cc.comment(fmt::format("DIVUW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 71);
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
            auto divisor = gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false);
            {
            auto label_merge = cc.newLabel();
            auto cond =  gen_operation(cc, ne, divisor, 0);
            cmp(cc, cond, 0);
            auto label_else = cc.newLabel();
            cc.je(label_else);
            {
                auto res = gen_ext(cc, 
                    (gen_operation(cc, udiv, gen_ext(cc, 
                        load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false), divisor)), 32, false);
                if(rd!=0){
                    mov(cc, get_ptr_for(jh, traits::X0+ rd),
                          gen_ext(cc, gen_ext(cc, 
                              res, 64, true), 64, false));
                }
            }
            cc.jmp(label_merge);
            cc.bind(label_else);
                {
                    if(rd!=0){
                        auto value_reg = get_reg_Gp(cc, 64, false);
                        cc.movabs(value_reg, (uint64_t)- 1);
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                            value_reg);
                    }
                }
            cc.bind(label_merge);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 71);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 72: REMW */
    continuation_e __remw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "remw"),
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
        cc.comment(fmt::format("REMW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 72);
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
            {
            auto label_merge = cc.newLabel();
            auto cond =  gen_operation(cc, ne, (gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false)), 0);
            cmp(cc, cond, 0);
            auto label_else = cc.newLabel();
            cc.je(label_else);
            {
                auto SMIN = (int32_t)1<<31;
                {
                auto label_merge = cc.newLabel();
                auto cond =  gen_operation(cc, land, gen_operation(cc, eq, gen_ext(cc, 
                    load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false), SMIN), gen_operation(cc, eq, gen_ext(cc, 
                    load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false), - 1));
                cmp(cc, cond, 0);
                auto label_else = cc.newLabel();
                cc.je(label_else);
                {
                    if(rd!=0){
                        auto value_reg = get_reg_Gp(cc, 64, false);
                        cc.movabs(value_reg, 0);
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                            value_reg);
                    }
                }
                cc.jmp(label_merge);
                cc.bind(label_else);
                    {
                        if(rd!=0){
                            mov(cc, get_ptr_for(jh, traits::X0+ rd),
                                  gen_ext(cc, 
                                      (gen_operation(cc, srem, gen_ext(cc, 
                                          load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false), gen_ext(cc, 
                                          load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false))), 64, true));
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
                              gen_ext(cc, 
                                  (gen_ext(cc, 
                                      load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false)), 64, true));
                    }
                }
            cc.bind(label_merge);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 72);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 73: REMUW */
    continuation_e __remuw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "remuw"),
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
        cc.comment(fmt::format("REMUW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 73);
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
            auto divisor = gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false);
            {
            auto label_merge = cc.newLabel();
            auto cond =  gen_operation(cc, ne, divisor, 0);
            cmp(cc, cond, 0);
            auto label_else = cc.newLabel();
            cc.je(label_else);
            {
                auto res = gen_ext(cc, 
                    (gen_operation(cc, urem, gen_ext(cc, 
                        load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false), divisor)), 32, false);
                if(rd!=0){
                    mov(cc, get_ptr_for(jh, traits::X0+ rd),
                          gen_ext(cc, gen_ext(cc, 
                              res, 64, true), 64, false));
                }
            }
            cc.jmp(label_merge);
            cc.bind(label_else);
                {
                    auto res = gen_ext(cc, 
                        load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false);
                    if(rd!=0){
                        mov(cc, get_ptr_for(jh, traits::X0+ rd),
                              gen_ext(cc, gen_ext(cc, 
                                  res, 64, true), 64, false));
                    }
                }
            cc.bind(label_merge);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 73);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 74: LRW */
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
        gen_sync(jh, PRE_SYNC, 74);
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
                auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_ext(cc, 
                              gen_read_mem(jh, traits::MEM, offs, 4), 32, false)), 64, true));
                gen_write_mem(jh, traits::RES, offs, (uint8_t)- 1, 1);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 74);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 75: SCW */
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
        gen_sync(jh, PRE_SYNC, 75);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::RES, offs, 1);
            {
            auto label_merge = cc.newLabel();
            auto cond =  gen_operation(cc, ne, res1, 0);
            cmp(cc, cond, 0);
            cc.je(label_merge);
            {
                gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
                    load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false), 4);
            }
            cc.bind(label_merge);
            }
            if(rd!=0){
                auto label_then6 = cc.newLabel();
                auto label_merge6 = cc.newLabel();
                auto tmp_reg6 = get_reg_Gp(cc, 8, false);
                cmp(cc, res1, 0);
                cc.jne(label_then6);
                mov(cc, tmp_reg6,1);
                cc.jmp(label_merge6);
                cc.bind(label_then6);
                mov(cc, tmp_reg6, 0);
                cc.bind(label_merge6);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, tmp_reg6
                      , 64, false)
                );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 75);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 76: AMOSWAPW */
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
        gen_sync(jh, PRE_SYNC, 76);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res = load_reg_from_mem_Gp(jh, traits::X0 + rs2);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_ext(cc, 
                              gen_read_mem(jh, traits::MEM, offs, 4), 32, false)), 64, true));
            }
            gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
                res, 32, false), 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 76);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 77: AMOADDW */
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
        gen_sync(jh, PRE_SYNC, 77);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res1 = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, offs, 4), 32, false);
            auto res2 = gen_operation(cc, add, res1, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false));
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res1, 64, true));
            }
            gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
                res2, 32, true), 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 77);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 78: AMOXORW */
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
        gen_sync(jh, PRE_SYNC, 78);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::MEM, offs, 4);
            auto res2 = gen_operation(cc, bxor, res1, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false));
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, gen_ext(cc, 
                          gen_ext(cc, res1, 32, true), 64, true), 64, false));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 78);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 79: AMOANDW */
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
        gen_sync(jh, PRE_SYNC, 79);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::MEM, offs, 4);
            auto res2 = gen_operation(cc, band, res1, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false));
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, gen_ext(cc, 
                          gen_ext(cc, res1, 32, true), 64, true), 64, false));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 79);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 80: AMOORW */
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
        gen_sync(jh, PRE_SYNC, 80);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::MEM, offs, 4);
            auto res2 = gen_operation(cc, bor, res1, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false));
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, gen_ext(cc, 
                          gen_ext(cc, res1, 32, true), 64, true), 64, false));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 80);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 81: AMOMINW */
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
        gen_sync(jh, PRE_SYNC, 81);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res1 = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, offs, 4), 32, false);
            auto label_then7 = cc.newLabel();
            auto label_merge7 = cc.newLabel();
            auto tmp_reg7 = get_reg_Gp(cc, 32, false);
            cmp(cc, res1, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false));
            cc.jg(label_then7);
            mov(cc, tmp_reg7,gen_ext(cc, res1, 32, false));
            cc.jmp(label_merge7);
            cc.bind(label_then7);
            mov(cc, tmp_reg7, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false));
            cc.bind(label_merge7);
            auto res2 = tmp_reg7
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res1, 64, true));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 81);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 82: AMOMAXW */
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
        gen_sync(jh, PRE_SYNC, 82);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res1 = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, offs, 4), 32, false);
            auto label_then8 = cc.newLabel();
            auto label_merge8 = cc.newLabel();
            auto tmp_reg8 = get_reg_Gp(cc, 32, false);
            cmp(cc, res1, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false));
            cc.jl(label_then8);
            mov(cc, tmp_reg8,gen_ext(cc, res1, 32, false));
            cc.jmp(label_merge8);
            cc.bind(label_then8);
            mov(cc, tmp_reg8, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false));
            cc.bind(label_merge8);
            auto res2 = tmp_reg8
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res1, 64, true));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 82);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 83: AMOMINUW */
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
        gen_sync(jh, PRE_SYNC, 83);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::MEM, offs, 4);
            auto label_then9 = cc.newLabel();
            auto label_merge9 = cc.newLabel();
            auto tmp_reg9 = get_reg_Gp(cc, 32, false);
            cmp(cc, res1, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false));
            cc.ja(label_then9);
            mov(cc, tmp_reg9,res1);
            cc.jmp(label_merge9);
            cc.bind(label_then9);
            mov(cc, tmp_reg9, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false));
            cc.bind(label_merge9);
            auto res2 = tmp_reg9
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, gen_ext(cc, 
                          gen_ext(cc, res1, 32, true), 64, true), 64, false));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 83);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 84: AMOMAXUW */
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
        gen_sync(jh, PRE_SYNC, 84);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::MEM, offs, 4);
            auto label_then10 = cc.newLabel();
            auto label_merge10 = cc.newLabel();
            auto tmp_reg10 = get_reg_Gp(cc, 32, false);
            cmp(cc, res1, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false));
            cc.jb(label_then10);
            mov(cc, tmp_reg10,res1);
            cc.jmp(label_merge10);
            cc.bind(label_then10);
            mov(cc, tmp_reg10, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false));
            cc.bind(label_merge10);
            auto res2 = tmp_reg10
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, gen_ext(cc, 
                          gen_ext(cc, res1, 32, true), 64, true), 64, false));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 84);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 85: LRD */
    continuation_e __lrd(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "lrd"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("LRD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 85);
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
                auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_ext(cc, 
                              gen_read_mem(jh, traits::MEM, offs, 8), 64, false)), 64, true));
                gen_write_mem(jh, traits::RES, offs, (uint8_t)- 1, 1);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 85);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 86: SCD */
    continuation_e __scd(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "scd"),
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
        cc.comment(fmt::format("SCD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 86);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res = gen_read_mem(jh, traits::RES, offs, 1);
            {
            auto label_merge = cc.newLabel();
            auto cond =  gen_operation(cc, ne, res, 0);
            cmp(cc, cond, 0);
            cc.je(label_merge);
            {
                gen_write_mem(jh, traits::MEM, offs, load_reg_from_mem_Gp(jh, traits::X0 + rs2), 8);
            }
            cc.bind(label_merge);
            }
            if(rd!=0){
                auto label_then11 = cc.newLabel();
                auto label_merge11 = cc.newLabel();
                auto tmp_reg11 = get_reg_Gp(cc, 8, false);
                cmp(cc, res, 0);
                cc.jne(label_then11);
                mov(cc, tmp_reg11,1);
                cc.jmp(label_merge11);
                cc.bind(label_then11);
                mov(cc, tmp_reg11, 0);
                cc.bind(label_merge11);
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, tmp_reg11
                      , 64, false)
                );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 86);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 87: AMOSWAPD */
    continuation_e __amoswapd(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoswapd"),
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
        cc.comment(fmt::format("AMOSWAPD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 87);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res = load_reg_from_mem_Gp(jh, traits::X0 + rs2);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_ext(cc, 
                              gen_read_mem(jh, traits::MEM, offs, 8), 64, false)), 64, true));
            }
            gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
                res, 64, false), 8);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 87);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 88: AMOADDD */
    continuation_e __amoaddd(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoaddd"),
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
        cc.comment(fmt::format("AMOADDD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 88);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::MEM, offs, 8);
            auto res2 = gen_ext(cc, 
                (gen_operation(cc, add, res1, load_reg_from_mem_Gp(jh, traits::X0 + rs2))), 64, false);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res1, 64, false));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 8);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 88);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 89: AMOXORD */
    continuation_e __amoxord(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoxord"),
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
        cc.comment(fmt::format("AMOXORD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 89);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::MEM, offs, 8);
            auto res2 = gen_operation(cc, bxor, res1, load_reg_from_mem_Gp(jh, traits::X0 + rs2));
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, gen_ext(cc, 
                          res1, 64, false), 64, false));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 8);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 89);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 90: AMOANDD */
    continuation_e __amoandd(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoandd"),
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
        cc.comment(fmt::format("AMOANDD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 90);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::MEM, offs, 8);
            auto res2 = gen_operation(cc, band, res1, load_reg_from_mem_Gp(jh, traits::X0 + rs2));
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res1, 64, false));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 8);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 90);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 91: AMOORD */
    continuation_e __amoord(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoord"),
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
        cc.comment(fmt::format("AMOORD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 91);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::MEM, offs, 8);
            auto res2 = gen_operation(cc, bor, res1, load_reg_from_mem_Gp(jh, traits::X0 + rs2));
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res1, 64, false));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 8);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 91);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 92: AMOMIND */
    continuation_e __amomind(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amomind"),
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
        cc.comment(fmt::format("AMOMIND_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 92);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res1 = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, offs, 8), 64, false);
            auto label_then12 = cc.newLabel();
            auto label_merge12 = cc.newLabel();
            auto tmp_reg12 = get_reg_Gp(cc, 64, false);
            cmp(cc, res1, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 64, false));
            cc.jg(label_then12);
            mov(cc, tmp_reg12,gen_ext(cc, res1, 64, false));
            cc.jmp(label_merge12);
            cc.bind(label_then12);
            mov(cc, tmp_reg12, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 64, false));
            cc.bind(label_merge12);
            auto res2 = tmp_reg12
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res1, 64, true));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 8);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 92);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 93: AMOMAXD */
    continuation_e __amomaxd(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amomaxd"),
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
        cc.comment(fmt::format("AMOMAXD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 93);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res1 = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, offs, 8), 64, false);
            auto label_then13 = cc.newLabel();
            auto label_merge13 = cc.newLabel();
            auto tmp_reg13 = get_reg_Gp(cc, 64, false);
            cmp(cc, res1, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 64, false));
            cc.jl(label_then13);
            mov(cc, tmp_reg13,gen_ext(cc, res1, 64, false));
            cc.jmp(label_merge13);
            cc.bind(label_then13);
            mov(cc, tmp_reg13, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 64, false));
            cc.bind(label_merge13);
            auto res2 = tmp_reg13
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res1, 64, true));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 8);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 93);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 94: AMOMINUD */
    continuation_e __amominud(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amominud"),
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
        cc.comment(fmt::format("AMOMINUD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 94);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::MEM, offs, 8);
            auto label_then14 = cc.newLabel();
            auto label_merge14 = cc.newLabel();
            auto tmp_reg14 = get_reg_Gp(cc, 64, false);
            cmp(cc, res1, load_reg_from_mem_Gp(jh, traits::X0 + rs2));
            cc.ja(label_then14);
            mov(cc, tmp_reg14,res1);
            cc.jmp(label_merge14);
            cc.bind(label_then14);
            mov(cc, tmp_reg14, load_reg_from_mem_Gp(jh, traits::X0 + rs2));
            cc.bind(label_merge14);
            auto res2 = tmp_reg14
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res1, 64, false));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 8);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 94);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 95: AMOMAXUD */
    continuation_e __amomaxud(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rl = ((bit_sub<25,1>(instr)));
        uint8_t aq = ((bit_sub<26,1>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amomaxud"),
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
        cc.comment(fmt::format("AMOMAXUD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 95);
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
            auto offs = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto res1 = gen_read_mem(jh, traits::MEM, offs, 8);
            auto label_then15 = cc.newLabel();
            auto label_merge15 = cc.newLabel();
            auto tmp_reg15 = get_reg_Gp(cc, 64, false);
            cmp(cc, res1, load_reg_from_mem_Gp(jh, traits::X0 + rs2));
            cc.jb(label_then15);
            mov(cc, tmp_reg15,res1);
            cc.jmp(label_merge15);
            cc.bind(label_then15);
            mov(cc, tmp_reg15, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 64, false));
            cc.bind(label_merge15);
            auto res2 = tmp_reg15
            ;
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res1, 64, false));
            }
            gen_write_mem(jh, traits::MEM, offs, res2, 8);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 95);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 96: C__ADDI4SPN */
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
        gen_sync(jh, PRE_SYNC, 96);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(imm){
            mov(cc, get_ptr_for(jh, traits::X0+ rd+8),
                  gen_ext(cc, 
                      (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + 2), imm)), 64, false));
        }
        else {
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 96);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 97: C__LW */
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
        gen_sync(jh, PRE_SYNC, 97);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1+8), uimm)), 64, false);
        mov(cc, get_ptr_for(jh, traits::X0+ rd+8),
              gen_ext(cc, 
                  gen_ext(cc, 
                      gen_read_mem(jh, traits::MEM, offs, 4), 32, false), 64, true));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 97);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 98: C__LD */
    continuation_e __c__ld(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,2>(instr) << 6) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {uimm},({rs1})", fmt::arg("mnemonic", "c.ld"),
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
        cc.comment(fmt::format("C__LD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 98);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1+8), uimm)), 64, false);
        mov(cc, get_ptr_for(jh, traits::X0+ rd+8),
              gen_ext(cc, 
                  gen_read_mem(jh, traits::MEM, offs, 8), 64, false));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 98);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 99: C__SW */
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
        gen_sync(jh, PRE_SYNC, 99);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1+8), uimm)), 64, false);
        gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
            load_reg_from_mem_Gp(jh, traits::X0 + rs2+8), 32, false), 4);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 99);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 100: C__SD */
    continuation_e __c__sd(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t uimm = ((bit_sub<5,2>(instr) << 6) | (bit_sub<10,3>(instr) << 3));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {uimm},({rs1})", fmt::arg("mnemonic", "c.sd"),
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
        cc.comment(fmt::format("C__SD_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 100);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1+8), uimm)), 64, false);
        gen_write_mem(jh, traits::MEM, offs, load_reg_from_mem_Gp(jh, traits::X0 + rs2+8), 8);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 100);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 101: C__ADDI */
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
        gen_sync(jh, PRE_SYNC, 101);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rs1!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rs1),
                      gen_ext(cc, 
                          (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int8_t)sext<6>(imm))), 64, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 101);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 102: C__NOP */
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
        gen_sync(jh, PRE_SYNC, 102);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 102);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 103: C__ADDIW */
    continuation_e __c__addiw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t imm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.addiw"),
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
        cc.comment(fmt::format("C__ADDIW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 103);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)||rs1==0){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rs1!=0){
                auto res = gen_ext(cc, 
                    (gen_operation(cc, add, gen_ext(cc, 
                        load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false), (int8_t)sext<6>(imm))), 32, true);
                mov(cc, get_ptr_for(jh, traits::X0+ rs1),
                      gen_ext(cc, 
                          res, 64, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 103);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 104: C__LI */
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
        gen_sync(jh, PRE_SYNC, 104);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                auto value_reg = get_reg_Gp(cc, 64, false);
                cc.movabs(value_reg, (uint64_t)((int8_t)sext<6>(imm)));
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                    value_reg);
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 104);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 105: C__LUI */
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
        gen_sync(jh, PRE_SYNC, 105);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(imm==0||rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        if(rd!=0){
            auto value_reg = get_reg_Gp(cc, 64, false);
            cc.movabs(value_reg, (uint64_t)((int32_t)sext<18>(imm)));
            mov(cc, get_ptr_for(jh, traits::X0+ rd),
                value_reg);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 105);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 106: C__ADDI16SP */
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
        gen_sync(jh, PRE_SYNC, 106);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(nzimm){
            mov(cc, get_ptr_for(jh, traits::X0+ 2),
                  gen_ext(cc, 
                      (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + 2), (int16_t)sext<10>(nzimm))), 64, true));
        }
        else {
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 106);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 107: __reserved_clui */
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
        gen_sync(jh, PRE_SYNC, 107);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 107);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 108: C__SRLI */
    continuation_e __c__srli(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t nzuimm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rs1 = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {nzuimm}", fmt::arg("mnemonic", "c.srli"),
                fmt::arg("rs1", name(8+rs1)), fmt::arg("nzuimm", nzuimm));
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
        gen_sync(jh, PRE_SYNC, 108);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::X0+ rs1+8),
              gen_operation(cc, shr, load_reg_from_mem_Gp(jh, traits::X0 + rs1+8), nzuimm));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 108);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 109: C__SRAI */
    continuation_e __c__srai(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t shamt = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
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
        gen_sync(jh, PRE_SYNC, 109);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::X0+ rs1+8),
              gen_ext(cc, 
                  (gen_operation(cc, sar, (gen_ext(cc, 
                      load_reg_from_mem_Gp(jh, traits::X0 + rs1+8), 64, false)), shamt)), 64, true));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 109);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 110: C__ANDI */
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
        gen_sync(jh, PRE_SYNC, 110);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::X0+ rs1+8),
              gen_ext(cc, 
                  (gen_operation(cc, band, load_reg_from_mem_Gp(jh, traits::X0 + rs1+8), (int8_t)sext<6>(imm))), 64, true));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 110);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 111: C__SUB */
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
        gen_sync(jh, PRE_SYNC, 111);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::X0+ rd+8),
              gen_ext(cc, 
                  (gen_operation(cc, sub, load_reg_from_mem_Gp(jh, traits::X0 + rd+8), load_reg_from_mem_Gp(jh, traits::X0 + rs2+8))), 64, true));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 111);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 112: C__XOR */
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
        gen_sync(jh, PRE_SYNC, 112);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::X0+ rd+8),
              gen_operation(cc, bxor, load_reg_from_mem_Gp(jh, traits::X0 + rd+8), load_reg_from_mem_Gp(jh, traits::X0 + rs2+8)));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 112);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 113: C__OR */
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
        gen_sync(jh, PRE_SYNC, 113);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::X0+ rd+8),
              gen_operation(cc, bor, load_reg_from_mem_Gp(jh, traits::X0 + rd+8), load_reg_from_mem_Gp(jh, traits::X0 + rs2+8)));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 113);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 114: C__AND */
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
        gen_sync(jh, PRE_SYNC, 114);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::X0+ rd+8),
              gen_operation(cc, band, load_reg_from_mem_Gp(jh, traits::X0 + rd+8), load_reg_from_mem_Gp(jh, traits::X0 + rs2+8)));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 114);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 115: C__SUBW */
    continuation_e __c__subw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rd}, {rs2}", fmt::arg("mnemonic", "c.subw"),
                fmt::arg("rd", name(8+rd)), fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__SUBW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 115);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        auto res = gen_ext(cc, 
            (gen_operation(cc, sub, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rd+8), 32, false), gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2+8), 32, false))), 32, true);
        mov(cc, get_ptr_for(jh, traits::X0+ rd+8),
              gen_ext(cc, 
                  res, 64, true));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 115);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 116: C__ADDW */
    continuation_e __c__addw(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,3>(instr)));
        uint8_t rd = ((bit_sub<7,3>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rd}, {rs2}", fmt::arg("mnemonic", "c.addw"),
                fmt::arg("rd", name(8+rd)), fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("C__ADDW_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 116);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        auto res = gen_ext(cc, 
            (gen_operation(cc, add, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rd+8), 32, false), gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2+8), 32, false))), 32, true);
        mov(cc, get_ptr_for(jh, traits::X0+ rd+8),
              gen_ext(cc, 
                  res, 64, true));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 116);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 117: C__J */
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
        gen_sync(jh, PRE_SYNC, 117);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        auto PC_val_v = (uint64_t)(PC+(int16_t)sext<12>(imm));
        mov(cc, jh.next_pc, PC_val_v);
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(KNOWN_JUMP));
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 117);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 118: C__BEQZ */
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
        gen_sync(jh, PRE_SYNC, 118);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        {
        auto label_merge = cc.newLabel();
        auto cond =  gen_operation(cc, eq, load_reg_from_mem_Gp(jh, traits::X0 + rs1+8), 0);
        cmp(cc, cond, 0);
        cc.je(label_merge);
        {
            auto PC_val_v = (uint64_t)(PC+(int16_t)sext<9>(imm));
            mov(cc, jh.next_pc, PC_val_v);
            mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(KNOWN_JUMP));
        }
        cc.bind(label_merge);
        }
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 118);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 119: C__BNEZ */
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
        gen_sync(jh, PRE_SYNC, 119);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        {
        auto label_merge = cc.newLabel();
        auto cond =  gen_operation(cc, ne, load_reg_from_mem_Gp(jh, traits::X0 + rs1+8), 0);
        cmp(cc, cond, 0);
        cc.je(label_merge);
        {
            auto PC_val_v = (uint64_t)(PC+(int16_t)sext<9>(imm));
            mov(cc, jh.next_pc, PC_val_v);
            mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(KNOWN_JUMP));
        }
        cc.bind(label_merge);
        }
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 119);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 120: C__SLLI */
    continuation_e __c__slli(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t shamt = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
        uint8_t rs1 = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {shamt}", fmt::arg("mnemonic", "c.slli"),
                fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
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
        gen_sync(jh, PRE_SYNC, 120);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rs1!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rs1),
                      gen_operation(cc, shl, load_reg_from_mem_Gp(jh, traits::X0 + rs1), shamt));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 120);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 121: C__LWSP */
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
        gen_sync(jh, PRE_SYNC, 121);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rd==0){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto offs = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + 2), uimm)), 64, false);
            mov(cc, get_ptr_for(jh, traits::X0+ rd),
                  gen_ext(cc, 
                      gen_ext(cc, 
                          gen_read_mem(jh, traits::MEM, offs, 4), 32, false), 64, true));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 121);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 122: C__LDSP */
    continuation_e __c__ldsp(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint16_t uimm = ((bit_sub<2,3>(instr) << 6) | (bit_sub<5,2>(instr) << 3) | (bit_sub<12,1>(instr) << 5));
        uint8_t rd = ((bit_sub<7,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {uimm}(sp)", fmt::arg("mnemonic", "c.ldsp"),
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
        cc.comment(fmt::format("C__LDSP_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 122);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rd==0){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto offs = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + 2), uimm)), 64, false);
            auto res = gen_ext(cc, 
                gen_read_mem(jh, traits::MEM, offs, 8), 64, false);
            mov(cc, get_ptr_for(jh, traits::X0+ rd),
                  res);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 122);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 123: C__MV */
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
        gen_sync(jh, PRE_SYNC, 123);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      load_reg_from_mem_Gp(jh, traits::X0 + rs2));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 123);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 124: C__JR */
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
        gen_sync(jh, PRE_SYNC, 124);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rs1&&rs1<static_cast<uint32_t>(traits::RFS)){
            auto addr_mask = (uint64_t)- 2;
            auto PC_val_v = gen_operation(cc, band, load_reg_from_mem_Gp(jh, traits::X0 + rs1%static_cast<uint32_t>(traits::RFS)), addr_mask);
            mov(cc, jh.next_pc, PC_val_v);
            mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(UNKNOWN_JUMP));
        }
        else {
            gen_raise(jh, 0, 2);
        }
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 124);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 125: __reserved_cmv */
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
        gen_sync(jh, PRE_SYNC, 125);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        gen_raise(jh, 0, 2);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 125);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 126: C__ADD */
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
        gen_sync(jh, PRE_SYNC, 126);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rd), load_reg_from_mem_Gp(jh, traits::X0 + rs2))), 64, false));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 126);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 127: C__JALR */
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
        gen_sync(jh, PRE_SYNC, 127);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(NO_JUMP));
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto addr_mask = (uint64_t)- 2;
            auto new_pc = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto value_reg = get_reg_Gp(cc, 64, false);
            cc.movabs(value_reg, (uint64_t)(PC+2));
            mov(cc, get_ptr_for(jh, traits::X0+ 1),
                value_reg);
            auto PC_val_v = gen_operation(cc, band, new_pc, addr_mask);
            mov(cc, jh.next_pc, PC_val_v);
            mov(cc, get_ptr_for(jh, traits::LAST_BRANCH), static_cast<int>(UNKNOWN_JUMP));
        }
        auto returnValue = BRANCH;
        
        gen_sync(jh, POST_SYNC, 127);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 128: C__EBREAK */
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
        gen_sync(jh, PRE_SYNC, 128);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        gen_raise(jh, 0, 3);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 128);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 129: C__SWSP */
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
        gen_sync(jh, PRE_SYNC, 129);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto offs = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + 2), uimm)), 64, false);
            gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 32, false), 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 129);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 130: C__SDSP */
    continuation_e __c__sdsp(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rs2 = ((bit_sub<2,5>(instr)));
        uint16_t uimm = ((bit_sub<7,3>(instr) << 6) | (bit_sub<10,3>(instr) << 3));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {uimm}(sp)", fmt::arg("mnemonic", "c.sdsp"),
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
        cc.comment(fmt::format("C__SDSP_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 130);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto offs = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + 2), uimm)), 64, false);
            gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs2), 64, false), 8);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 130);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 131: DII */
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
        gen_sync(jh, PRE_SYNC, 131);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 131);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 132: FLW */
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
        gen_sync(jh, PRE_SYNC, 132);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto offs = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 32, true);
            InvokeNode* call_NaNBox32_16;
            auto NaNBox32_16_arg0 = gen_read_mem(jh, traits::MEM, offs, 4);
            x86::Gp ret_val_NaNBox32_16 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_NaNBox32_16, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  ret_val_NaNBox32_16);
            setArg(call_NaNBox32_16, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_NaNBox32_16, 1, NaNBox32_16_arg0);
            setRet(call_NaNBox32_16, 0, ret_val_NaNBox32_16);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 132);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 133: FSW */
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
        gen_sync(jh, PRE_SYNC, 133);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto offs = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 32, true);
            gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::F0 + rs2), 32, false), 4);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 133);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 134: FADD__S */
    continuation_e __fadd__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {rm}", fmt::arg("mnemonic", "fadd.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rm", rm));
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
        gen_sync(jh, PRE_SYNC, 134);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_s_19;
        auto unbox_s_19_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_19 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_19,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_20;
        auto unbox_s_20_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_20 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_20,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_21;
        x86::Gp ret_val_get_rm_21 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_21, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fadd_s_18;
        auto fadd_s_18_arg0 = ret_val_unbox_s_19;
        auto fadd_s_18_arg1 = ret_val_unbox_s_20;
        auto fadd_s_18_arg2 = ret_val_get_rm_21;
        x86::Gp ret_val_fadd_s_18 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fadd_s_18,  &fadd_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint8_t>());
        InvokeNode* call_NaNBox32_17;
        auto NaNBox32_17_arg0 = ret_val_fadd_s_18;
        x86::Gp ret_val_NaNBox32_17 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox32_17, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox32_17);
        setArg(call_unbox_s_19, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_19, 1, unbox_s_19_arg1);
        setRet(call_unbox_s_19, 0, ret_val_unbox_s_19);
        setArg(call_unbox_s_20, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_20, 1, unbox_s_20_arg1);
        setRet(call_unbox_s_20, 0, ret_val_unbox_s_20);
        setArg(call_get_rm_21, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_21, 1, rm);
        setRet(call_get_rm_21, 0, ret_val_get_rm_21);
        setArg(call_fadd_s_18, 0, fadd_s_18_arg0);
        setArg(call_fadd_s_18, 1, fadd_s_18_arg1);
        setArg(call_fadd_s_18, 2, fadd_s_18_arg2);
        setRet(call_fadd_s_18, 0, ret_val_fadd_s_18);
        setArg(call_NaNBox32_17, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox32_17, 1, NaNBox32_17_arg0);
        setRet(call_NaNBox32_17, 0, ret_val_NaNBox32_17);
        InvokeNode* call_fget_flags_22;
        x86::Gp ret_val_fget_flags_22 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_22,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_22;
        setRet(call_fget_flags_22, 0, ret_val_fget_flags_22);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 134);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 135: FSUB__S */
    continuation_e __fsub__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {rm}", fmt::arg("mnemonic", "fsub.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rm", rm));
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
        gen_sync(jh, PRE_SYNC, 135);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_s_25;
        auto unbox_s_25_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_25 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_25,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_26;
        auto unbox_s_26_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_26 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_26,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_27;
        x86::Gp ret_val_get_rm_27 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_27, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fsub_s_24;
        auto fsub_s_24_arg0 = ret_val_unbox_s_25;
        auto fsub_s_24_arg1 = ret_val_unbox_s_26;
        auto fsub_s_24_arg2 = ret_val_get_rm_27;
        x86::Gp ret_val_fsub_s_24 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fsub_s_24,  &fsub_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint8_t>());
        InvokeNode* call_NaNBox32_23;
        auto NaNBox32_23_arg0 = ret_val_fsub_s_24;
        x86::Gp ret_val_NaNBox32_23 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox32_23, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox32_23);
        setArg(call_unbox_s_25, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_25, 1, unbox_s_25_arg1);
        setRet(call_unbox_s_25, 0, ret_val_unbox_s_25);
        setArg(call_unbox_s_26, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_26, 1, unbox_s_26_arg1);
        setRet(call_unbox_s_26, 0, ret_val_unbox_s_26);
        setArg(call_get_rm_27, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_27, 1, rm);
        setRet(call_get_rm_27, 0, ret_val_get_rm_27);
        setArg(call_fsub_s_24, 0, fsub_s_24_arg0);
        setArg(call_fsub_s_24, 1, fsub_s_24_arg1);
        setArg(call_fsub_s_24, 2, fsub_s_24_arg2);
        setRet(call_fsub_s_24, 0, ret_val_fsub_s_24);
        setArg(call_NaNBox32_23, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox32_23, 1, NaNBox32_23_arg0);
        setRet(call_NaNBox32_23, 0, ret_val_NaNBox32_23);
        InvokeNode* call_fget_flags_28;
        x86::Gp ret_val_fget_flags_28 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_28,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_28;
        setRet(call_fget_flags_28, 0, ret_val_fget_flags_28);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 135);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 136: FMUL__S */
    continuation_e __fmul__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {rm}", fmt::arg("mnemonic", "fmul.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rm", rm));
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
        gen_sync(jh, PRE_SYNC, 136);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_s_31;
        auto unbox_s_31_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_31 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_31,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_32;
        auto unbox_s_32_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_32 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_32,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_33;
        x86::Gp ret_val_get_rm_33 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_33, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fmul_s_30;
        auto fmul_s_30_arg0 = ret_val_unbox_s_31;
        auto fmul_s_30_arg1 = ret_val_unbox_s_32;
        auto fmul_s_30_arg2 = ret_val_get_rm_33;
        x86::Gp ret_val_fmul_s_30 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fmul_s_30,  &fmul_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint8_t>());
        InvokeNode* call_NaNBox32_29;
        auto NaNBox32_29_arg0 = ret_val_fmul_s_30;
        x86::Gp ret_val_NaNBox32_29 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox32_29, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox32_29);
        setArg(call_unbox_s_31, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_31, 1, unbox_s_31_arg1);
        setRet(call_unbox_s_31, 0, ret_val_unbox_s_31);
        setArg(call_unbox_s_32, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_32, 1, unbox_s_32_arg1);
        setRet(call_unbox_s_32, 0, ret_val_unbox_s_32);
        setArg(call_get_rm_33, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_33, 1, rm);
        setRet(call_get_rm_33, 0, ret_val_get_rm_33);
        setArg(call_fmul_s_30, 0, fmul_s_30_arg0);
        setArg(call_fmul_s_30, 1, fmul_s_30_arg1);
        setArg(call_fmul_s_30, 2, fmul_s_30_arg2);
        setRet(call_fmul_s_30, 0, ret_val_fmul_s_30);
        setArg(call_NaNBox32_29, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox32_29, 1, NaNBox32_29_arg0);
        setRet(call_NaNBox32_29, 0, ret_val_NaNBox32_29);
        InvokeNode* call_fget_flags_34;
        x86::Gp ret_val_fget_flags_34 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_34,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_34;
        setRet(call_fget_flags_34, 0, ret_val_fget_flags_34);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 136);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 137: FDIV__S */
    continuation_e __fdiv__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {rm}", fmt::arg("mnemonic", "fdiv.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rm", rm));
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
        gen_sync(jh, PRE_SYNC, 137);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_s_37;
        auto unbox_s_37_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_37 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_37,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_38;
        auto unbox_s_38_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_38 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_38,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_39;
        x86::Gp ret_val_get_rm_39 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_39, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fdiv_s_36;
        auto fdiv_s_36_arg0 = ret_val_unbox_s_37;
        auto fdiv_s_36_arg1 = ret_val_unbox_s_38;
        auto fdiv_s_36_arg2 = ret_val_get_rm_39;
        x86::Gp ret_val_fdiv_s_36 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fdiv_s_36,  &fdiv_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint8_t>());
        InvokeNode* call_NaNBox32_35;
        auto NaNBox32_35_arg0 = ret_val_fdiv_s_36;
        x86::Gp ret_val_NaNBox32_35 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox32_35, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox32_35);
        setArg(call_unbox_s_37, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_37, 1, unbox_s_37_arg1);
        setRet(call_unbox_s_37, 0, ret_val_unbox_s_37);
        setArg(call_unbox_s_38, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_38, 1, unbox_s_38_arg1);
        setRet(call_unbox_s_38, 0, ret_val_unbox_s_38);
        setArg(call_get_rm_39, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_39, 1, rm);
        setRet(call_get_rm_39, 0, ret_val_get_rm_39);
        setArg(call_fdiv_s_36, 0, fdiv_s_36_arg0);
        setArg(call_fdiv_s_36, 1, fdiv_s_36_arg1);
        setArg(call_fdiv_s_36, 2, fdiv_s_36_arg2);
        setRet(call_fdiv_s_36, 0, ret_val_fdiv_s_36);
        setArg(call_NaNBox32_35, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox32_35, 1, NaNBox32_35_arg0);
        setRet(call_NaNBox32_35, 0, ret_val_NaNBox32_35);
        InvokeNode* call_fget_flags_40;
        x86::Gp ret_val_fget_flags_40 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_40,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_40;
        setRet(call_fget_flags_40, 0, ret_val_fget_flags_40);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 137);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 138: FMIN__S */
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
        gen_sync(jh, PRE_SYNC, 138);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_s_43;
        auto unbox_s_43_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_43 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_43,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_44;
        auto unbox_s_44_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_44 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_44,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_fsel_s_42;
        auto fsel_s_42_arg0 = ret_val_unbox_s_43;
        auto fsel_s_42_arg1 = ret_val_unbox_s_44;
        x86::Gp ret_val_fsel_s_42 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fsel_s_42,  &fsel_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t>());
        InvokeNode* call_NaNBox32_41;
        auto NaNBox32_41_arg0 = ret_val_fsel_s_42;
        x86::Gp ret_val_NaNBox32_41 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox32_41, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox32_41);
        setArg(call_unbox_s_43, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_43, 1, unbox_s_43_arg1);
        setRet(call_unbox_s_43, 0, ret_val_unbox_s_43);
        setArg(call_unbox_s_44, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_44, 1, unbox_s_44_arg1);
        setRet(call_unbox_s_44, 0, ret_val_unbox_s_44);
        setArg(call_fsel_s_42, 0, fsel_s_42_arg0);
        setArg(call_fsel_s_42, 1, fsel_s_42_arg1);
        setArg(call_fsel_s_42, 2, 0);
        setRet(call_fsel_s_42, 0, ret_val_fsel_s_42);
        setArg(call_NaNBox32_41, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox32_41, 1, NaNBox32_41_arg0);
        setRet(call_NaNBox32_41, 0, ret_val_NaNBox32_41);
        InvokeNode* call_fget_flags_45;
        x86::Gp ret_val_fget_flags_45 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_45,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_45;
        setRet(call_fget_flags_45, 0, ret_val_fget_flags_45);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 138);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 139: FMAX__S */
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
        gen_sync(jh, PRE_SYNC, 139);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_s_48;
        auto unbox_s_48_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_48 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_48,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_49;
        auto unbox_s_49_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_49 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_49,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_fsel_s_47;
        auto fsel_s_47_arg0 = ret_val_unbox_s_48;
        auto fsel_s_47_arg1 = ret_val_unbox_s_49;
        x86::Gp ret_val_fsel_s_47 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fsel_s_47,  &fsel_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t>());
        InvokeNode* call_NaNBox32_46;
        auto NaNBox32_46_arg0 = ret_val_fsel_s_47;
        x86::Gp ret_val_NaNBox32_46 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox32_46, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox32_46);
        setArg(call_unbox_s_48, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_48, 1, unbox_s_48_arg1);
        setRet(call_unbox_s_48, 0, ret_val_unbox_s_48);
        setArg(call_unbox_s_49, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_49, 1, unbox_s_49_arg1);
        setRet(call_unbox_s_49, 0, ret_val_unbox_s_49);
        setArg(call_fsel_s_47, 0, fsel_s_47_arg0);
        setArg(call_fsel_s_47, 1, fsel_s_47_arg1);
        setArg(call_fsel_s_47, 2, 1);
        setRet(call_fsel_s_47, 0, ret_val_fsel_s_47);
        setArg(call_NaNBox32_46, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox32_46, 1, NaNBox32_46_arg0);
        setRet(call_NaNBox32_46, 0, ret_val_NaNBox32_46);
        InvokeNode* call_fget_flags_50;
        x86::Gp ret_val_fget_flags_50 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_50,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_50;
        setRet(call_fget_flags_50, 0, ret_val_fget_flags_50);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 139);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 140: FSQRT__S */
    continuation_e __fsqrt__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fsqrt.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rm", rm));
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
        gen_sync(jh, PRE_SYNC, 140);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_s_53;
        auto unbox_s_53_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_53 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_53,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_54;
        x86::Gp ret_val_get_rm_54 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_54, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fsqrt_s_52;
        auto fsqrt_s_52_arg0 = ret_val_unbox_s_53;
        auto fsqrt_s_52_arg1 = ret_val_get_rm_54;
        x86::Gp ret_val_fsqrt_s_52 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fsqrt_s_52,  &fsqrt_s, FuncSignature::build<uint32_t, uint32_t, uint8_t>());
        InvokeNode* call_NaNBox32_51;
        auto NaNBox32_51_arg0 = ret_val_fsqrt_s_52;
        x86::Gp ret_val_NaNBox32_51 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox32_51, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox32_51);
        setArg(call_unbox_s_53, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_53, 1, unbox_s_53_arg1);
        setRet(call_unbox_s_53, 0, ret_val_unbox_s_53);
        setArg(call_get_rm_54, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_54, 1, rm);
        setRet(call_get_rm_54, 0, ret_val_get_rm_54);
        setArg(call_fsqrt_s_52, 0, fsqrt_s_52_arg0);
        setArg(call_fsqrt_s_52, 1, fsqrt_s_52_arg1);
        setRet(call_fsqrt_s_52, 0, ret_val_fsqrt_s_52);
        setArg(call_NaNBox32_51, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox32_51, 1, NaNBox32_51_arg0);
        setRet(call_NaNBox32_51, 0, ret_val_NaNBox32_51);
        InvokeNode* call_fget_flags_55;
        x86::Gp ret_val_fget_flags_55 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_55,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_55;
        setRet(call_fget_flags_55, 0, ret_val_fget_flags_55);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 140);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 141: FMADD__S */
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
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {rs3}, {rm}", fmt::arg("mnemonic", "fmadd.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)), fmt::arg("rm", rm));
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
        gen_sync(jh, PRE_SYNC, 141);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_s_58;
        auto unbox_s_58_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_58 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_58,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_59;
        auto unbox_s_59_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_59 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_59,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_60;
        auto unbox_s_60_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs3);
        x86::Gp ret_val_unbox_s_60 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_60,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_61;
        x86::Gp ret_val_get_rm_61 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_61, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fmadd_s_57;
        auto fmadd_s_57_arg0 = ret_val_unbox_s_58;
        auto fmadd_s_57_arg1 = ret_val_unbox_s_59;
        auto fmadd_s_57_arg2 = ret_val_unbox_s_60;
        auto fmadd_s_57_arg4 = ret_val_get_rm_61;
        x86::Gp ret_val_fmadd_s_57 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fmadd_s_57,  &fmadd_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint8_t>());
        InvokeNode* call_NaNBox32_56;
        auto NaNBox32_56_arg0 = ret_val_fmadd_s_57;
        x86::Gp ret_val_NaNBox32_56 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox32_56, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox32_56);
        setArg(call_unbox_s_58, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_58, 1, unbox_s_58_arg1);
        setRet(call_unbox_s_58, 0, ret_val_unbox_s_58);
        setArg(call_unbox_s_59, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_59, 1, unbox_s_59_arg1);
        setRet(call_unbox_s_59, 0, ret_val_unbox_s_59);
        setArg(call_unbox_s_60, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_60, 1, unbox_s_60_arg1);
        setRet(call_unbox_s_60, 0, ret_val_unbox_s_60);
        setArg(call_get_rm_61, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_61, 1, rm);
        setRet(call_get_rm_61, 0, ret_val_get_rm_61);
        setArg(call_fmadd_s_57, 0, fmadd_s_57_arg0);
        setArg(call_fmadd_s_57, 1, fmadd_s_57_arg1);
        setArg(call_fmadd_s_57, 2, fmadd_s_57_arg2);
        setArg(call_fmadd_s_57, 3, 0);
        setArg(call_fmadd_s_57, 4, fmadd_s_57_arg4);
        setRet(call_fmadd_s_57, 0, ret_val_fmadd_s_57);
        setArg(call_NaNBox32_56, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox32_56, 1, NaNBox32_56_arg0);
        setRet(call_NaNBox32_56, 0, ret_val_NaNBox32_56);
        InvokeNode* call_fget_flags_62;
        x86::Gp ret_val_fget_flags_62 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_62,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_62;
        setRet(call_fget_flags_62, 0, ret_val_fget_flags_62);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 141);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 142: FMSUB__S */
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
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {rs3}, {rm}", fmt::arg("mnemonic", "fmsub.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)), fmt::arg("rm", rm));
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
        gen_sync(jh, PRE_SYNC, 142);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_s_65;
        auto unbox_s_65_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_65 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_65,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_66;
        auto unbox_s_66_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_66 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_66,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_67;
        auto unbox_s_67_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs3);
        x86::Gp ret_val_unbox_s_67 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_67,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_68;
        x86::Gp ret_val_get_rm_68 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_68, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fmadd_s_64;
        auto fmadd_s_64_arg0 = ret_val_unbox_s_65;
        auto fmadd_s_64_arg1 = ret_val_unbox_s_66;
        auto fmadd_s_64_arg2 = ret_val_unbox_s_67;
        auto fmadd_s_64_arg4 = ret_val_get_rm_68;
        x86::Gp ret_val_fmadd_s_64 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fmadd_s_64,  &fmadd_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint8_t>());
        InvokeNode* call_NaNBox32_63;
        auto NaNBox32_63_arg0 = ret_val_fmadd_s_64;
        x86::Gp ret_val_NaNBox32_63 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox32_63, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox32_63);
        setArg(call_unbox_s_65, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_65, 1, unbox_s_65_arg1);
        setRet(call_unbox_s_65, 0, ret_val_unbox_s_65);
        setArg(call_unbox_s_66, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_66, 1, unbox_s_66_arg1);
        setRet(call_unbox_s_66, 0, ret_val_unbox_s_66);
        setArg(call_unbox_s_67, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_67, 1, unbox_s_67_arg1);
        setRet(call_unbox_s_67, 0, ret_val_unbox_s_67);
        setArg(call_get_rm_68, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_68, 1, rm);
        setRet(call_get_rm_68, 0, ret_val_get_rm_68);
        setArg(call_fmadd_s_64, 0, fmadd_s_64_arg0);
        setArg(call_fmadd_s_64, 1, fmadd_s_64_arg1);
        setArg(call_fmadd_s_64, 2, fmadd_s_64_arg2);
        setArg(call_fmadd_s_64, 3, 1);
        setArg(call_fmadd_s_64, 4, fmadd_s_64_arg4);
        setRet(call_fmadd_s_64, 0, ret_val_fmadd_s_64);
        setArg(call_NaNBox32_63, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox32_63, 1, NaNBox32_63_arg0);
        setRet(call_NaNBox32_63, 0, ret_val_NaNBox32_63);
        InvokeNode* call_fget_flags_69;
        x86::Gp ret_val_fget_flags_69 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_69,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_69;
        setRet(call_fget_flags_69, 0, ret_val_fget_flags_69);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 142);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 143: FNMADD__S */
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
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {rs3}, {rm}", fmt::arg("mnemonic", "fnmadd.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)), fmt::arg("rm", rm));
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
        gen_sync(jh, PRE_SYNC, 143);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_s_72;
        auto unbox_s_72_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_72 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_72,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_73;
        auto unbox_s_73_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_73 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_73,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_74;
        auto unbox_s_74_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs3);
        x86::Gp ret_val_unbox_s_74 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_74,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_75;
        x86::Gp ret_val_get_rm_75 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_75, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fmadd_s_71;
        auto fmadd_s_71_arg0 = ret_val_unbox_s_72;
        auto fmadd_s_71_arg1 = ret_val_unbox_s_73;
        auto fmadd_s_71_arg2 = ret_val_unbox_s_74;
        auto fmadd_s_71_arg4 = ret_val_get_rm_75;
        x86::Gp ret_val_fmadd_s_71 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fmadd_s_71,  &fmadd_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint8_t>());
        InvokeNode* call_NaNBox32_70;
        auto NaNBox32_70_arg0 = ret_val_fmadd_s_71;
        x86::Gp ret_val_NaNBox32_70 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox32_70, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox32_70);
        setArg(call_unbox_s_72, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_72, 1, unbox_s_72_arg1);
        setRet(call_unbox_s_72, 0, ret_val_unbox_s_72);
        setArg(call_unbox_s_73, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_73, 1, unbox_s_73_arg1);
        setRet(call_unbox_s_73, 0, ret_val_unbox_s_73);
        setArg(call_unbox_s_74, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_74, 1, unbox_s_74_arg1);
        setRet(call_unbox_s_74, 0, ret_val_unbox_s_74);
        setArg(call_get_rm_75, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_75, 1, rm);
        setRet(call_get_rm_75, 0, ret_val_get_rm_75);
        setArg(call_fmadd_s_71, 0, fmadd_s_71_arg0);
        setArg(call_fmadd_s_71, 1, fmadd_s_71_arg1);
        setArg(call_fmadd_s_71, 2, fmadd_s_71_arg2);
        setArg(call_fmadd_s_71, 3, 2);
        setArg(call_fmadd_s_71, 4, fmadd_s_71_arg4);
        setRet(call_fmadd_s_71, 0, ret_val_fmadd_s_71);
        setArg(call_NaNBox32_70, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox32_70, 1, NaNBox32_70_arg0);
        setRet(call_NaNBox32_70, 0, ret_val_NaNBox32_70);
        InvokeNode* call_fget_flags_76;
        x86::Gp ret_val_fget_flags_76 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_76,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_76;
        setRet(call_fget_flags_76, 0, ret_val_fget_flags_76);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 143);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 144: FNMSUB__S */
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
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {rs3}, {rm}", fmt::arg("mnemonic", "fnmsub.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)), fmt::arg("rm", rm));
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
        gen_sync(jh, PRE_SYNC, 144);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_s_79;
        auto unbox_s_79_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_79 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_79,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_80;
        auto unbox_s_80_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_80 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_80,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_81;
        auto unbox_s_81_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs3);
        x86::Gp ret_val_unbox_s_81 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_81,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_82;
        x86::Gp ret_val_get_rm_82 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_82, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fmadd_s_78;
        auto fmadd_s_78_arg0 = ret_val_unbox_s_79;
        auto fmadd_s_78_arg1 = ret_val_unbox_s_80;
        auto fmadd_s_78_arg2 = ret_val_unbox_s_81;
        auto fmadd_s_78_arg4 = ret_val_get_rm_82;
        x86::Gp ret_val_fmadd_s_78 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fmadd_s_78,  &fmadd_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint8_t>());
        InvokeNode* call_NaNBox32_77;
        auto NaNBox32_77_arg0 = ret_val_fmadd_s_78;
        x86::Gp ret_val_NaNBox32_77 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox32_77, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox32_77);
        setArg(call_unbox_s_79, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_79, 1, unbox_s_79_arg1);
        setRet(call_unbox_s_79, 0, ret_val_unbox_s_79);
        setArg(call_unbox_s_80, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_80, 1, unbox_s_80_arg1);
        setRet(call_unbox_s_80, 0, ret_val_unbox_s_80);
        setArg(call_unbox_s_81, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_81, 1, unbox_s_81_arg1);
        setRet(call_unbox_s_81, 0, ret_val_unbox_s_81);
        setArg(call_get_rm_82, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_82, 1, rm);
        setRet(call_get_rm_82, 0, ret_val_get_rm_82);
        setArg(call_fmadd_s_78, 0, fmadd_s_78_arg0);
        setArg(call_fmadd_s_78, 1, fmadd_s_78_arg1);
        setArg(call_fmadd_s_78, 2, fmadd_s_78_arg2);
        setArg(call_fmadd_s_78, 3, 3);
        setArg(call_fmadd_s_78, 4, fmadd_s_78_arg4);
        setRet(call_fmadd_s_78, 0, ret_val_fmadd_s_78);
        setArg(call_NaNBox32_77, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox32_77, 1, NaNBox32_77_arg0);
        setRet(call_NaNBox32_77, 0, ret_val_NaNBox32_77);
        InvokeNode* call_fget_flags_83;
        x86::Gp ret_val_fget_flags_83 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_83,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_83;
        setRet(call_fget_flags_83, 0, ret_val_fget_flags_83);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 144);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 145: FCVT__W__S */
    continuation_e __fcvt__w__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.w.s"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rm", rm));
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
        gen_sync(jh, PRE_SYNC, 145);
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
            InvokeNode* call_unbox_s_85;
            auto unbox_s_85_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_s_85 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_unbox_s_85,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
            InvokeNode* call_get_rm_86;
            x86::Gp ret_val_get_rm_86 = get_reg_Gp(cc, 8, false);
            cc.invoke(&call_get_rm_86, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
            InvokeNode* call_f32toi32_84;
            auto f32toi32_84_arg0 = ret_val_unbox_s_85;
            auto f32toi32_84_arg1 = ret_val_get_rm_86;
            x86::Gp ret_val_f32toi32_84 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_f32toi32_84,  &f32toi32, FuncSignature::build<uint32_t, uint32_t, uint8_t>());
            auto res = gen_ext(cc, 
                gen_ext(cc, 
                    ret_val_f32toi32_84, 32, false), 64, true);
            setArg(call_unbox_s_85, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_s_85, 1, unbox_s_85_arg1);
            setRet(call_unbox_s_85, 0, ret_val_unbox_s_85);
            setArg(call_get_rm_86, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_get_rm_86, 1, rm);
            setRet(call_get_rm_86, 0, ret_val_get_rm_86);
            setArg(call_f32toi32_84, 0, f32toi32_84_arg0);
            setArg(call_f32toi32_84, 1, f32toi32_84_arg1);
            setRet(call_f32toi32_84, 0, ret_val_f32toi32_84);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, true));
            }
            InvokeNode* call_fget_flags_87;
            x86::Gp ret_val_fget_flags_87 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_87,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_87;
            setRet(call_fget_flags_87, 0, ret_val_fget_flags_87);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 145);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 146: FCVT__WU__S */
    continuation_e __fcvt__wu__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.wu.s"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rm", rm));
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
        gen_sync(jh, PRE_SYNC, 146);
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
            InvokeNode* call_unbox_s_89;
            auto unbox_s_89_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_s_89 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_unbox_s_89,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
            InvokeNode* call_get_rm_90;
            x86::Gp ret_val_get_rm_90 = get_reg_Gp(cc, 8, false);
            cc.invoke(&call_get_rm_90, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
            InvokeNode* call_f32toui32_88;
            auto f32toui32_88_arg0 = ret_val_unbox_s_89;
            auto f32toui32_88_arg1 = ret_val_get_rm_90;
            x86::Gp ret_val_f32toui32_88 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_f32toui32_88,  &f32toui32, FuncSignature::build<uint32_t, uint32_t, uint8_t>());
            auto res = gen_ext(cc, 
                gen_ext(cc, 
                    ret_val_f32toui32_88, 32, false), 64, true);
            setArg(call_unbox_s_89, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_s_89, 1, unbox_s_89_arg1);
            setRet(call_unbox_s_89, 0, ret_val_unbox_s_89);
            setArg(call_get_rm_90, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_get_rm_90, 1, rm);
            setRet(call_get_rm_90, 0, ret_val_get_rm_90);
            setArg(call_f32toui32_88, 0, f32toui32_88_arg0);
            setArg(call_f32toui32_88, 1, f32toui32_88_arg1);
            setRet(call_f32toui32_88, 0, ret_val_f32toui32_88);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, true));
            }
            InvokeNode* call_fget_flags_91;
            x86::Gp ret_val_fget_flags_91 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_91,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_91;
            setRet(call_fget_flags_91, 0, ret_val_fget_flags_91);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 146);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 147: FCVT__L__S */
    continuation_e __fcvt__l__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.l.s"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__L__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 147);
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
            InvokeNode* call_unbox_s_93;
            auto unbox_s_93_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_s_93 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_unbox_s_93,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
            InvokeNode* call_get_rm_94;
            x86::Gp ret_val_get_rm_94 = get_reg_Gp(cc, 8, false);
            cc.invoke(&call_get_rm_94, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
            InvokeNode* call_f32toi64_92;
            auto f32toi64_92_arg0 = ret_val_unbox_s_93;
            auto f32toi64_92_arg1 = ret_val_get_rm_94;
            x86::Gp ret_val_f32toi64_92 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_f32toi64_92,  &f32toi64, FuncSignature::build<uint64_t, uint32_t, uint8_t>());
            auto res = gen_ext(cc, 
                gen_ext(cc, 
                    ret_val_f32toi64_92, 64, false), 64, true);
            setArg(call_unbox_s_93, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_s_93, 1, unbox_s_93_arg1);
            setRet(call_unbox_s_93, 0, ret_val_unbox_s_93);
            setArg(call_get_rm_94, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_get_rm_94, 1, rm);
            setRet(call_get_rm_94, 0, ret_val_get_rm_94);
            setArg(call_f32toi64_92, 0, f32toi64_92_arg0);
            setArg(call_f32toi64_92, 1, f32toi64_92_arg1);
            setRet(call_f32toi64_92, 0, ret_val_f32toi64_92);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, true));
            }
            InvokeNode* call_fget_flags_95;
            x86::Gp ret_val_fget_flags_95 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_95,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_95;
            setRet(call_fget_flags_95, 0, ret_val_fget_flags_95);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 147);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 148: FCVT__LU__S */
    continuation_e __fcvt__lu__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.lu.s"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__LU__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 148);
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
            InvokeNode* call_unbox_s_97;
            auto unbox_s_97_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_s_97 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_unbox_s_97,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
            InvokeNode* call_get_rm_98;
            x86::Gp ret_val_get_rm_98 = get_reg_Gp(cc, 8, false);
            cc.invoke(&call_get_rm_98, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
            InvokeNode* call_f32toui64_96;
            auto f32toui64_96_arg0 = ret_val_unbox_s_97;
            auto f32toui64_96_arg1 = ret_val_get_rm_98;
            x86::Gp ret_val_f32toui64_96 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_f32toui64_96,  &f32toui64, FuncSignature::build<uint64_t, uint32_t, uint8_t>());
            auto res = ret_val_f32toui64_96;
            setArg(call_unbox_s_97, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_s_97, 1, unbox_s_97_arg1);
            setRet(call_unbox_s_97, 0, ret_val_unbox_s_97);
            setArg(call_get_rm_98, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_get_rm_98, 1, rm);
            setRet(call_get_rm_98, 0, ret_val_get_rm_98);
            setArg(call_f32toui64_96, 0, f32toui64_96_arg0);
            setArg(call_f32toui64_96, 1, f32toui64_96_arg1);
            setRet(call_f32toui64_96, 0, ret_val_f32toui64_96);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      res);
            }
            InvokeNode* call_fget_flags_99;
            x86::Gp ret_val_fget_flags_99 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_99,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_99;
            setRet(call_fget_flags_99, 0, ret_val_fget_flags_99);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 148);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 149: FCVT__S__W */
    continuation_e __fcvt__s__w(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.s.w"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rm", rm));
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
        gen_sync(jh, PRE_SYNC, 149);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            InvokeNode* call_get_rm_102;
            x86::Gp ret_val_get_rm_102 = get_reg_Gp(cc, 8, false);
            cc.invoke(&call_get_rm_102, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
            InvokeNode* call_i32tof32_101;
            auto i32tof32_101_arg0 = gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false);
            auto i32tof32_101_arg1 = ret_val_get_rm_102;
            x86::Gp ret_val_i32tof32_101 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_i32tof32_101,  &i32tof32, FuncSignature::build<uint32_t, uint32_t, uint8_t>());
            InvokeNode* call_NaNBox32_100;
            auto NaNBox32_100_arg0 = ret_val_i32tof32_101;
            x86::Gp ret_val_NaNBox32_100 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_NaNBox32_100, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  ret_val_NaNBox32_100);
            setArg(call_get_rm_102, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_get_rm_102, 1, rm);
            setRet(call_get_rm_102, 0, ret_val_get_rm_102);
            setArg(call_i32tof32_101, 0, i32tof32_101_arg0);
            setArg(call_i32tof32_101, 1, i32tof32_101_arg1);
            setRet(call_i32tof32_101, 0, ret_val_i32tof32_101);
            setArg(call_NaNBox32_100, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_NaNBox32_100, 1, NaNBox32_100_arg0);
            setRet(call_NaNBox32_100, 0, ret_val_NaNBox32_100);
            InvokeNode* call_fget_flags_103;
            x86::Gp ret_val_fget_flags_103 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_103,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_103;
            setRet(call_fget_flags_103, 0, ret_val_fget_flags_103);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 149);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 150: FCVT__S__WU */
    continuation_e __fcvt__s__wu(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.s.wu"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rm", rm));
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
        gen_sync(jh, PRE_SYNC, 150);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            InvokeNode* call_get_rm_106;
            x86::Gp ret_val_get_rm_106 = get_reg_Gp(cc, 8, false);
            cc.invoke(&call_get_rm_106, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
            InvokeNode* call_ui32tof32_105;
            auto ui32tof32_105_arg0 = gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false);
            auto ui32tof32_105_arg1 = ret_val_get_rm_106;
            x86::Gp ret_val_ui32tof32_105 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_ui32tof32_105,  &ui32tof32, FuncSignature::build<uint32_t, uint32_t, uint8_t>());
            InvokeNode* call_NaNBox32_104;
            auto NaNBox32_104_arg0 = ret_val_ui32tof32_105;
            x86::Gp ret_val_NaNBox32_104 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_NaNBox32_104, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  ret_val_NaNBox32_104);
            setArg(call_get_rm_106, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_get_rm_106, 1, rm);
            setRet(call_get_rm_106, 0, ret_val_get_rm_106);
            setArg(call_ui32tof32_105, 0, ui32tof32_105_arg0);
            setArg(call_ui32tof32_105, 1, ui32tof32_105_arg1);
            setRet(call_ui32tof32_105, 0, ret_val_ui32tof32_105);
            setArg(call_NaNBox32_104, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_NaNBox32_104, 1, NaNBox32_104_arg0);
            setRet(call_NaNBox32_104, 0, ret_val_NaNBox32_104);
            InvokeNode* call_fget_flags_107;
            x86::Gp ret_val_fget_flags_107 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_107,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_107;
            setRet(call_fget_flags_107, 0, ret_val_fget_flags_107);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 150);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 151: FCVT__S__L */
    continuation_e __fcvt__s__l(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.s.l"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__S__L_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 151);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            InvokeNode* call_get_rm_110;
            x86::Gp ret_val_get_rm_110 = get_reg_Gp(cc, 8, false);
            cc.invoke(&call_get_rm_110, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
            InvokeNode* call_i64tof32_109;
            auto i64tof32_109_arg0 = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto i64tof32_109_arg1 = ret_val_get_rm_110;
            x86::Gp ret_val_i64tof32_109 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_i64tof32_109,  &i64tof32, FuncSignature::build<uint32_t, uint64_t, uint8_t>());
            InvokeNode* call_NaNBox32_108;
            auto NaNBox32_108_arg0 = ret_val_i64tof32_109;
            x86::Gp ret_val_NaNBox32_108 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_NaNBox32_108, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  ret_val_NaNBox32_108);
            setArg(call_get_rm_110, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_get_rm_110, 1, rm);
            setRet(call_get_rm_110, 0, ret_val_get_rm_110);
            setArg(call_i64tof32_109, 0, i64tof32_109_arg0);
            setArg(call_i64tof32_109, 1, i64tof32_109_arg1);
            setRet(call_i64tof32_109, 0, ret_val_i64tof32_109);
            setArg(call_NaNBox32_108, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_NaNBox32_108, 1, NaNBox32_108_arg0);
            setRet(call_NaNBox32_108, 0, ret_val_NaNBox32_108);
            InvokeNode* call_fget_flags_111;
            x86::Gp ret_val_fget_flags_111 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_111,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_111;
            setRet(call_fget_flags_111, 0, ret_val_fget_flags_111);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 151);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 152: FCVT__S__LU */
    continuation_e __fcvt__s__lu(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.s.lu"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__S__LU_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 152);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            InvokeNode* call_get_rm_114;
            x86::Gp ret_val_get_rm_114 = get_reg_Gp(cc, 8, false);
            cc.invoke(&call_get_rm_114, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
            InvokeNode* call_ui64tof32_113;
            auto ui64tof32_113_arg0 = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto ui64tof32_113_arg1 = ret_val_get_rm_114;
            x86::Gp ret_val_ui64tof32_113 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_ui64tof32_113,  &ui64tof32, FuncSignature::build<uint32_t, uint64_t, uint8_t>());
            InvokeNode* call_NaNBox32_112;
            auto NaNBox32_112_arg0 = ret_val_ui64tof32_113;
            x86::Gp ret_val_NaNBox32_112 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_NaNBox32_112, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  ret_val_NaNBox32_112);
            setArg(call_get_rm_114, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_get_rm_114, 1, rm);
            setRet(call_get_rm_114, 0, ret_val_get_rm_114);
            setArg(call_ui64tof32_113, 0, ui64tof32_113_arg0);
            setArg(call_ui64tof32_113, 1, ui64tof32_113_arg1);
            setRet(call_ui64tof32_113, 0, ret_val_ui64tof32_113);
            setArg(call_NaNBox32_112, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_NaNBox32_112, 1, NaNBox32_112_arg0);
            setRet(call_NaNBox32_112, 0, ret_val_NaNBox32_112);
            InvokeNode* call_fget_flags_115;
            x86::Gp ret_val_fget_flags_115 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_115,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_115;
            setRet(call_fget_flags_115, 0, ret_val_fget_flags_115);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 152);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 153: FSGNJ__S */
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
        gen_sync(jh, PRE_SYNC, 153);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_s_117;
        auto unbox_s_117_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_117 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_117,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_118;
        auto unbox_s_118_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_118 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_118,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_NaNBox32_116;
        auto NaNBox32_116_arg0 = gen_operation(cc, bor, gen_operation(cc, shl, gen_ext(cc, gen_slice(cc, ret_val_unbox_s_117, 31, 31-31+1), 32, false), 31), gen_ext(cc, gen_slice(cc, ret_val_unbox_s_118, 0, 30-0+1), 32, false));
        x86::Gp ret_val_NaNBox32_116 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox32_116, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox32_116);
        setArg(call_unbox_s_117, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_117, 1, unbox_s_117_arg1);
        setRet(call_unbox_s_117, 0, ret_val_unbox_s_117);
        setArg(call_unbox_s_118, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_118, 1, unbox_s_118_arg1);
        setRet(call_unbox_s_118, 0, ret_val_unbox_s_118);
        setArg(call_NaNBox32_116, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox32_116, 1, NaNBox32_116_arg0);
        setRet(call_NaNBox32_116, 0, ret_val_NaNBox32_116);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 153);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 154: FSGNJN__S */
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
        gen_sync(jh, PRE_SYNC, 154);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_s_120;
        auto unbox_s_120_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_120 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_120,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_121;
        auto unbox_s_121_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_121 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_121,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_NaNBox32_119;
        auto NaNBox32_119_arg0 = gen_operation(cc, bor, gen_operation(cc, shl, gen_ext(cc, gen_operation(cc, bnot, gen_slice(cc, ret_val_unbox_s_120, 31, 31-31+1)), 32, false), 31), gen_ext(cc, gen_slice(cc, ret_val_unbox_s_121, 0, 30-0+1), 32, false));
        x86::Gp ret_val_NaNBox32_119 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox32_119, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox32_119);
        setArg(call_unbox_s_120, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_120, 1, unbox_s_120_arg1);
        setRet(call_unbox_s_120, 0, ret_val_unbox_s_120);
        setArg(call_unbox_s_121, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_121, 1, unbox_s_121_arg1);
        setRet(call_unbox_s_121, 0, ret_val_unbox_s_121);
        setArg(call_NaNBox32_119, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox32_119, 1, NaNBox32_119_arg0);
        setRet(call_NaNBox32_119, 0, ret_val_NaNBox32_119);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 154);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 155: FSGNJX__S */
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
        gen_sync(jh, PRE_SYNC, 155);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_s_123;
        auto unbox_s_123_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_s_123 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_123,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_s_124;
        auto unbox_s_124_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_124 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_124,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_NaNBox32_122;
        auto NaNBox32_122_arg0 = gen_operation(cc, bxor, (gen_operation(cc, band, ret_val_unbox_s_123, ((uint32_t)1<<31))), ret_val_unbox_s_124);
        x86::Gp ret_val_NaNBox32_122 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox32_122, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox32_122);
        setArg(call_unbox_s_123, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_123, 1, unbox_s_123_arg1);
        setRet(call_unbox_s_123, 0, ret_val_unbox_s_123);
        setArg(call_unbox_s_124, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_124, 1, unbox_s_124_arg1);
        setRet(call_unbox_s_124, 0, ret_val_unbox_s_124);
        setArg(call_NaNBox32_122, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox32_122, 1, NaNBox32_122_arg0);
        setRet(call_NaNBox32_122, 0, ret_val_NaNBox32_122);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 155);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 156: FMV__X__W */
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
        gen_sync(jh, PRE_SYNC, 156);
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
                      gen_ext(cc, 
                          gen_ext(cc, 
                              gen_ext(cc, 
                                  load_reg_from_mem_Gp(jh, traits::F0 + rs1), 32, false), 64, true), 64, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 156);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 157: FMV__W__X */
    continuation_e __fmv__w__x(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "fmv.w.x"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)));
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
        gen_sync(jh, PRE_SYNC, 157);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            InvokeNode* call_NaNBox32_125;
            auto NaNBox32_125_arg0 = gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false);
            x86::Gp ret_val_NaNBox32_125 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_NaNBox32_125, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  ret_val_NaNBox32_125);
            setArg(call_NaNBox32_125, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_NaNBox32_125, 1, NaNBox32_125_arg0);
            setRet(call_NaNBox32_125, 0, ret_val_NaNBox32_125);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 157);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 158: FEQ__S */
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
        gen_sync(jh, PRE_SYNC, 158);
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
            InvokeNode* call_unbox_s_127;
            auto unbox_s_127_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_s_127 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_unbox_s_127,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
            InvokeNode* call_unbox_s_128;
            auto unbox_s_128_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
            x86::Gp ret_val_unbox_s_128 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_unbox_s_128,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
            InvokeNode* call_fcmp_s_126;
            auto fcmp_s_126_arg0 = ret_val_unbox_s_127;
            auto fcmp_s_126_arg1 = ret_val_unbox_s_128;
            x86::Gp ret_val_fcmp_s_126 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fcmp_s_126,  &fcmp_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t>());
            auto res = ret_val_fcmp_s_126;
            setArg(call_unbox_s_127, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_s_127, 1, unbox_s_127_arg1);
            setRet(call_unbox_s_127, 0, ret_val_unbox_s_127);
            setArg(call_unbox_s_128, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_s_128, 1, unbox_s_128_arg1);
            setRet(call_unbox_s_128, 0, ret_val_unbox_s_128);
            setArg(call_fcmp_s_126, 0, fcmp_s_126_arg0);
            setArg(call_fcmp_s_126, 1, fcmp_s_126_arg1);
            setArg(call_fcmp_s_126, 2, 0);
            setRet(call_fcmp_s_126, 0, ret_val_fcmp_s_126);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, res, 64, false)
                );
            }
            InvokeNode* call_fget_flags_129;
            x86::Gp ret_val_fget_flags_129 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_129,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_129;
            setRet(call_fget_flags_129, 0, ret_val_fget_flags_129);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 158);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 159: FLT__S */
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
        gen_sync(jh, PRE_SYNC, 159);
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
            InvokeNode* call_unbox_s_131;
            auto unbox_s_131_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_s_131 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_unbox_s_131,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
            InvokeNode* call_unbox_s_132;
            auto unbox_s_132_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
            x86::Gp ret_val_unbox_s_132 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_unbox_s_132,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
            InvokeNode* call_fcmp_s_130;
            auto fcmp_s_130_arg0 = ret_val_unbox_s_131;
            auto fcmp_s_130_arg1 = ret_val_unbox_s_132;
            x86::Gp ret_val_fcmp_s_130 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fcmp_s_130,  &fcmp_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t>());
            auto res = ret_val_fcmp_s_130;
            setArg(call_unbox_s_131, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_s_131, 1, unbox_s_131_arg1);
            setRet(call_unbox_s_131, 0, ret_val_unbox_s_131);
            setArg(call_unbox_s_132, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_s_132, 1, unbox_s_132_arg1);
            setRet(call_unbox_s_132, 0, ret_val_unbox_s_132);
            setArg(call_fcmp_s_130, 0, fcmp_s_130_arg0);
            setArg(call_fcmp_s_130, 1, fcmp_s_130_arg1);
            setArg(call_fcmp_s_130, 2, 2);
            setRet(call_fcmp_s_130, 0, ret_val_fcmp_s_130);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, res, 64, false)
                );
            }
            InvokeNode* call_fget_flags_133;
            x86::Gp ret_val_fget_flags_133 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_133,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_133;
            setRet(call_fget_flags_133, 0, ret_val_fget_flags_133);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 159);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 160: FLE__S */
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
        gen_sync(jh, PRE_SYNC, 160);
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
            InvokeNode* call_unbox_s_135;
            auto unbox_s_135_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_s_135 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_unbox_s_135,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
            InvokeNode* call_unbox_s_136;
            auto unbox_s_136_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
            x86::Gp ret_val_unbox_s_136 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_unbox_s_136,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
            InvokeNode* call_fcmp_s_134;
            auto fcmp_s_134_arg0 = ret_val_unbox_s_135;
            auto fcmp_s_134_arg1 = ret_val_unbox_s_136;
            x86::Gp ret_val_fcmp_s_134 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fcmp_s_134,  &fcmp_s, FuncSignature::build<uint32_t, uint32_t, uint32_t, uint32_t>());
            auto res = ret_val_fcmp_s_134;
            setArg(call_unbox_s_135, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_s_135, 1, unbox_s_135_arg1);
            setRet(call_unbox_s_135, 0, ret_val_unbox_s_135);
            setArg(call_unbox_s_136, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_s_136, 1, unbox_s_136_arg1);
            setRet(call_unbox_s_136, 0, ret_val_unbox_s_136);
            setArg(call_fcmp_s_134, 0, fcmp_s_134_arg0);
            setArg(call_fcmp_s_134, 1, fcmp_s_134_arg1);
            setArg(call_fcmp_s_134, 2, 1);
            setRet(call_fcmp_s_134, 0, ret_val_fcmp_s_134);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, res, 64, false)
                );
            }
            InvokeNode* call_fget_flags_137;
            x86::Gp ret_val_fget_flags_137 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_137,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_137;
            setRet(call_fget_flags_137, 0, ret_val_fget_flags_137);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 160);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 161: FCLASS__S */
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
        gen_sync(jh, PRE_SYNC, 161);
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
            InvokeNode* call_unbox_s_139;
            auto unbox_s_139_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_s_139 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_unbox_s_139,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
            InvokeNode* call_fclass_s_138;
            auto fclass_s_138_arg0 = ret_val_unbox_s_139;
            x86::Gp ret_val_fclass_s_138 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fclass_s_138,  &fclass_s, FuncSignature::build<uint32_t, uint32_t>());
            auto res = ret_val_fclass_s_138;
            setArg(call_unbox_s_139, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_s_139, 1, unbox_s_139_arg1);
            setRet(call_unbox_s_139, 0, ret_val_unbox_s_139);
            setArg(call_fclass_s_138, 0, fclass_s_138_arg0);
            setRet(call_fclass_s_138, 0, ret_val_fclass_s_138);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, res, 64, false)
                );
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 161);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 162: FLD */
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
        gen_sync(jh, PRE_SYNC, 162);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto offs = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 64, true);
            InvokeNode* call_NaNBox64_140;
            auto NaNBox64_140_arg0 = gen_read_mem(jh, traits::MEM, offs, 8);
            x86::Gp ret_val_NaNBox64_140 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_NaNBox64_140, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  ret_val_NaNBox64_140);
            setArg(call_NaNBox64_140, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_NaNBox64_140, 1, NaNBox64_140_arg0);
            setRet(call_NaNBox64_140, 0, ret_val_NaNBox64_140);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 162);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 163: FSD */
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
        gen_sync(jh, PRE_SYNC, 163);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            auto offs = gen_ext(cc, 
                (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1), (int16_t)sext<12>(imm))), 64, true);
            gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::F0 + rs2), 64, false), 8);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 163);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 164: FADD__D */
    continuation_e __fadd__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {rm}", fmt::arg("mnemonic", "fadd.d"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FADD__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 164);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_d_143;
        auto unbox_d_143_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_d_143 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_143,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_d_144;
        auto unbox_d_144_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_d_144 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_144,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_145;
        x86::Gp ret_val_get_rm_145 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_145, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fadd_d_142;
        auto fadd_d_142_arg0 = ret_val_unbox_d_143;
        auto fadd_d_142_arg1 = ret_val_unbox_d_144;
        auto fadd_d_142_arg2 = ret_val_get_rm_145;
        x86::Gp ret_val_fadd_d_142 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_fadd_d_142,  &fadd_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint8_t>());
        InvokeNode* call_NaNBox64_141;
        auto NaNBox64_141_arg0 = ret_val_fadd_d_142;
        x86::Gp ret_val_NaNBox64_141 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox64_141, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox64_141);
        setArg(call_unbox_d_143, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_143, 1, unbox_d_143_arg1);
        setRet(call_unbox_d_143, 0, ret_val_unbox_d_143);
        setArg(call_unbox_d_144, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_144, 1, unbox_d_144_arg1);
        setRet(call_unbox_d_144, 0, ret_val_unbox_d_144);
        setArg(call_get_rm_145, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_145, 1, rm);
        setRet(call_get_rm_145, 0, ret_val_get_rm_145);
        setArg(call_fadd_d_142, 0, fadd_d_142_arg0);
        setArg(call_fadd_d_142, 1, fadd_d_142_arg1);
        setArg(call_fadd_d_142, 2, fadd_d_142_arg2);
        setRet(call_fadd_d_142, 0, ret_val_fadd_d_142);
        setArg(call_NaNBox64_141, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox64_141, 1, NaNBox64_141_arg0);
        setRet(call_NaNBox64_141, 0, ret_val_NaNBox64_141);
        InvokeNode* call_fget_flags_146;
        x86::Gp ret_val_fget_flags_146 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_146,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_146;
        setRet(call_fget_flags_146, 0, ret_val_fget_flags_146);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 164);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 165: FSUB__D */
    continuation_e __fsub__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {rm}", fmt::arg("mnemonic", "fsub.d"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FSUB__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 165);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_d_149;
        auto unbox_d_149_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_d_149 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_149,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_d_150;
        auto unbox_d_150_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_d_150 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_150,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_151;
        x86::Gp ret_val_get_rm_151 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_151, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fsub_d_148;
        auto fsub_d_148_arg0 = ret_val_unbox_d_149;
        auto fsub_d_148_arg1 = ret_val_unbox_d_150;
        auto fsub_d_148_arg2 = ret_val_get_rm_151;
        x86::Gp ret_val_fsub_d_148 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_fsub_d_148,  &fsub_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint8_t>());
        InvokeNode* call_NaNBox64_147;
        auto NaNBox64_147_arg0 = ret_val_fsub_d_148;
        x86::Gp ret_val_NaNBox64_147 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox64_147, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox64_147);
        setArg(call_unbox_d_149, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_149, 1, unbox_d_149_arg1);
        setRet(call_unbox_d_149, 0, ret_val_unbox_d_149);
        setArg(call_unbox_d_150, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_150, 1, unbox_d_150_arg1);
        setRet(call_unbox_d_150, 0, ret_val_unbox_d_150);
        setArg(call_get_rm_151, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_151, 1, rm);
        setRet(call_get_rm_151, 0, ret_val_get_rm_151);
        setArg(call_fsub_d_148, 0, fsub_d_148_arg0);
        setArg(call_fsub_d_148, 1, fsub_d_148_arg1);
        setArg(call_fsub_d_148, 2, fsub_d_148_arg2);
        setRet(call_fsub_d_148, 0, ret_val_fsub_d_148);
        setArg(call_NaNBox64_147, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox64_147, 1, NaNBox64_147_arg0);
        setRet(call_NaNBox64_147, 0, ret_val_NaNBox64_147);
        InvokeNode* call_fget_flags_152;
        x86::Gp ret_val_fget_flags_152 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_152,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_152;
        setRet(call_fget_flags_152, 0, ret_val_fget_flags_152);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 165);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 166: FMUL__D */
    continuation_e __fmul__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {rm}", fmt::arg("mnemonic", "fmul.d"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FMUL__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 166);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_d_155;
        auto unbox_d_155_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_d_155 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_155,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_d_156;
        auto unbox_d_156_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_d_156 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_156,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_157;
        x86::Gp ret_val_get_rm_157 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_157, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fmul_d_154;
        auto fmul_d_154_arg0 = ret_val_unbox_d_155;
        auto fmul_d_154_arg1 = ret_val_unbox_d_156;
        auto fmul_d_154_arg2 = ret_val_get_rm_157;
        x86::Gp ret_val_fmul_d_154 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_fmul_d_154,  &fmul_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint8_t>());
        InvokeNode* call_NaNBox64_153;
        auto NaNBox64_153_arg0 = ret_val_fmul_d_154;
        x86::Gp ret_val_NaNBox64_153 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox64_153, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox64_153);
        setArg(call_unbox_d_155, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_155, 1, unbox_d_155_arg1);
        setRet(call_unbox_d_155, 0, ret_val_unbox_d_155);
        setArg(call_unbox_d_156, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_156, 1, unbox_d_156_arg1);
        setRet(call_unbox_d_156, 0, ret_val_unbox_d_156);
        setArg(call_get_rm_157, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_157, 1, rm);
        setRet(call_get_rm_157, 0, ret_val_get_rm_157);
        setArg(call_fmul_d_154, 0, fmul_d_154_arg0);
        setArg(call_fmul_d_154, 1, fmul_d_154_arg1);
        setArg(call_fmul_d_154, 2, fmul_d_154_arg2);
        setRet(call_fmul_d_154, 0, ret_val_fmul_d_154);
        setArg(call_NaNBox64_153, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox64_153, 1, NaNBox64_153_arg0);
        setRet(call_NaNBox64_153, 0, ret_val_NaNBox64_153);
        InvokeNode* call_fget_flags_158;
        x86::Gp ret_val_fget_flags_158 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_158,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_158;
        setRet(call_fget_flags_158, 0, ret_val_fget_flags_158);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 166);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 167: FDIV__D */
    continuation_e __fdiv__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {rm}", fmt::arg("mnemonic", "fdiv.d"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FDIV__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 167);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_d_161;
        auto unbox_d_161_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_d_161 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_161,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_d_162;
        auto unbox_d_162_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_d_162 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_162,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_163;
        x86::Gp ret_val_get_rm_163 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_163, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fdiv_d_160;
        auto fdiv_d_160_arg0 = ret_val_unbox_d_161;
        auto fdiv_d_160_arg1 = ret_val_unbox_d_162;
        auto fdiv_d_160_arg2 = ret_val_get_rm_163;
        x86::Gp ret_val_fdiv_d_160 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_fdiv_d_160,  &fdiv_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint8_t>());
        InvokeNode* call_NaNBox64_159;
        auto NaNBox64_159_arg0 = ret_val_fdiv_d_160;
        x86::Gp ret_val_NaNBox64_159 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox64_159, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox64_159);
        setArg(call_unbox_d_161, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_161, 1, unbox_d_161_arg1);
        setRet(call_unbox_d_161, 0, ret_val_unbox_d_161);
        setArg(call_unbox_d_162, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_162, 1, unbox_d_162_arg1);
        setRet(call_unbox_d_162, 0, ret_val_unbox_d_162);
        setArg(call_get_rm_163, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_163, 1, rm);
        setRet(call_get_rm_163, 0, ret_val_get_rm_163);
        setArg(call_fdiv_d_160, 0, fdiv_d_160_arg0);
        setArg(call_fdiv_d_160, 1, fdiv_d_160_arg1);
        setArg(call_fdiv_d_160, 2, fdiv_d_160_arg2);
        setRet(call_fdiv_d_160, 0, ret_val_fdiv_d_160);
        setArg(call_NaNBox64_159, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox64_159, 1, NaNBox64_159_arg0);
        setRet(call_NaNBox64_159, 0, ret_val_NaNBox64_159);
        InvokeNode* call_fget_flags_164;
        x86::Gp ret_val_fget_flags_164 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_164,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_164;
        setRet(call_fget_flags_164, 0, ret_val_fget_flags_164);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 167);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 168: FMIN__D */
    continuation_e __fmin__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fmin.d"),
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
        cc.comment(fmt::format("FMIN__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 168);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_d_167;
        auto unbox_d_167_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_d_167 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_167,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_d_168;
        auto unbox_d_168_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_d_168 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_168,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_fsel_d_166;
        auto fsel_d_166_arg0 = ret_val_unbox_d_167;
        auto fsel_d_166_arg1 = ret_val_unbox_d_168;
        x86::Gp ret_val_fsel_d_166 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_fsel_d_166,  &fsel_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint32_t>());
        InvokeNode* call_NaNBox64_165;
        auto NaNBox64_165_arg0 = ret_val_fsel_d_166;
        x86::Gp ret_val_NaNBox64_165 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox64_165, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox64_165);
        setArg(call_unbox_d_167, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_167, 1, unbox_d_167_arg1);
        setRet(call_unbox_d_167, 0, ret_val_unbox_d_167);
        setArg(call_unbox_d_168, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_168, 1, unbox_d_168_arg1);
        setRet(call_unbox_d_168, 0, ret_val_unbox_d_168);
        setArg(call_fsel_d_166, 0, fsel_d_166_arg0);
        setArg(call_fsel_d_166, 1, fsel_d_166_arg1);
        setArg(call_fsel_d_166, 2, 0);
        setRet(call_fsel_d_166, 0, ret_val_fsel_d_166);
        setArg(call_NaNBox64_165, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox64_165, 1, NaNBox64_165_arg0);
        setRet(call_NaNBox64_165, 0, ret_val_NaNBox64_165);
        InvokeNode* call_fget_flags_169;
        x86::Gp ret_val_fget_flags_169 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_169,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_169;
        setRet(call_fget_flags_169, 0, ret_val_fget_flags_169);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 168);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 169: FMAX__D */
    continuation_e __fmax__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fmax.d"),
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
        cc.comment(fmt::format("FMAX__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 169);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_d_172;
        auto unbox_d_172_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_d_172 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_172,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_d_173;
        auto unbox_d_173_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_d_173 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_173,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_fsel_d_171;
        auto fsel_d_171_arg0 = ret_val_unbox_d_172;
        auto fsel_d_171_arg1 = ret_val_unbox_d_173;
        x86::Gp ret_val_fsel_d_171 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_fsel_d_171,  &fsel_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint32_t>());
        InvokeNode* call_NaNBox64_170;
        auto NaNBox64_170_arg0 = ret_val_fsel_d_171;
        x86::Gp ret_val_NaNBox64_170 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox64_170, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox64_170);
        setArg(call_unbox_d_172, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_172, 1, unbox_d_172_arg1);
        setRet(call_unbox_d_172, 0, ret_val_unbox_d_172);
        setArg(call_unbox_d_173, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_173, 1, unbox_d_173_arg1);
        setRet(call_unbox_d_173, 0, ret_val_unbox_d_173);
        setArg(call_fsel_d_171, 0, fsel_d_171_arg0);
        setArg(call_fsel_d_171, 1, fsel_d_171_arg1);
        setArg(call_fsel_d_171, 2, 1);
        setRet(call_fsel_d_171, 0, ret_val_fsel_d_171);
        setArg(call_NaNBox64_170, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox64_170, 1, NaNBox64_170_arg0);
        setRet(call_NaNBox64_170, 0, ret_val_NaNBox64_170);
        InvokeNode* call_fget_flags_174;
        x86::Gp ret_val_fget_flags_174 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_174,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_174;
        setRet(call_fget_flags_174, 0, ret_val_fget_flags_174);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 169);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 170: FSQRT__D */
    continuation_e __fsqrt__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fsqrt.d"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FSQRT__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 170);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_d_177;
        auto unbox_d_177_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_d_177 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_177,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_178;
        x86::Gp ret_val_get_rm_178 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_178, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fsqrt_d_176;
        auto fsqrt_d_176_arg0 = ret_val_unbox_d_177;
        auto fsqrt_d_176_arg1 = ret_val_get_rm_178;
        x86::Gp ret_val_fsqrt_d_176 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_fsqrt_d_176,  &fsqrt_d, FuncSignature::build<uint64_t, uint64_t, uint8_t>());
        InvokeNode* call_NaNBox64_175;
        auto NaNBox64_175_arg0 = ret_val_fsqrt_d_176;
        x86::Gp ret_val_NaNBox64_175 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox64_175, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox64_175);
        setArg(call_unbox_d_177, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_177, 1, unbox_d_177_arg1);
        setRet(call_unbox_d_177, 0, ret_val_unbox_d_177);
        setArg(call_get_rm_178, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_178, 1, rm);
        setRet(call_get_rm_178, 0, ret_val_get_rm_178);
        setArg(call_fsqrt_d_176, 0, fsqrt_d_176_arg0);
        setArg(call_fsqrt_d_176, 1, fsqrt_d_176_arg1);
        setRet(call_fsqrt_d_176, 0, ret_val_fsqrt_d_176);
        setArg(call_NaNBox64_175, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox64_175, 1, NaNBox64_175_arg0);
        setRet(call_NaNBox64_175, 0, ret_val_NaNBox64_175);
        InvokeNode* call_fget_flags_179;
        x86::Gp ret_val_fget_flags_179 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_179,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_179;
        setRet(call_fget_flags_179, 0, ret_val_fget_flags_179);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 170);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 171: FMADD__D */
    continuation_e __fmadd__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rs3 = ((bit_sub<27,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {rs3}, {rm}", fmt::arg("mnemonic", "fmadd.d"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FMADD__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 171);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_d_182;
        auto unbox_d_182_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_d_182 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_182,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_d_183;
        auto unbox_d_183_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_d_183 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_183,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_d_184;
        auto unbox_d_184_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs3);
        x86::Gp ret_val_unbox_d_184 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_184,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_185;
        x86::Gp ret_val_get_rm_185 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_185, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fmadd_d_181;
        auto fmadd_d_181_arg0 = ret_val_unbox_d_182;
        auto fmadd_d_181_arg1 = ret_val_unbox_d_183;
        auto fmadd_d_181_arg2 = ret_val_unbox_d_184;
        auto fmadd_d_181_arg4 = ret_val_get_rm_185;
        x86::Gp ret_val_fmadd_d_181 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_fmadd_d_181,  &fmadd_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint8_t>());
        InvokeNode* call_NaNBox64_180;
        auto NaNBox64_180_arg0 = ret_val_fmadd_d_181;
        x86::Gp ret_val_NaNBox64_180 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox64_180, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox64_180);
        setArg(call_unbox_d_182, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_182, 1, unbox_d_182_arg1);
        setRet(call_unbox_d_182, 0, ret_val_unbox_d_182);
        setArg(call_unbox_d_183, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_183, 1, unbox_d_183_arg1);
        setRet(call_unbox_d_183, 0, ret_val_unbox_d_183);
        setArg(call_unbox_d_184, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_184, 1, unbox_d_184_arg1);
        setRet(call_unbox_d_184, 0, ret_val_unbox_d_184);
        setArg(call_get_rm_185, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_185, 1, rm);
        setRet(call_get_rm_185, 0, ret_val_get_rm_185);
        setArg(call_fmadd_d_181, 0, fmadd_d_181_arg0);
        setArg(call_fmadd_d_181, 1, fmadd_d_181_arg1);
        setArg(call_fmadd_d_181, 2, fmadd_d_181_arg2);
        setArg(call_fmadd_d_181, 3, 0);
        setArg(call_fmadd_d_181, 4, fmadd_d_181_arg4);
        setRet(call_fmadd_d_181, 0, ret_val_fmadd_d_181);
        setArg(call_NaNBox64_180, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox64_180, 1, NaNBox64_180_arg0);
        setRet(call_NaNBox64_180, 0, ret_val_NaNBox64_180);
        InvokeNode* call_fget_flags_186;
        x86::Gp ret_val_fget_flags_186 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_186,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_186;
        setRet(call_fget_flags_186, 0, ret_val_fget_flags_186);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 171);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 172: FMSUB__D */
    continuation_e __fmsub__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rs3 = ((bit_sub<27,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {rs3}, {rm}", fmt::arg("mnemonic", "fmsub.d"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FMSUB__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 172);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_d_189;
        auto unbox_d_189_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_d_189 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_189,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_d_190;
        auto unbox_d_190_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_d_190 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_190,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_d_191;
        auto unbox_d_191_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs3);
        x86::Gp ret_val_unbox_d_191 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_191,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_192;
        x86::Gp ret_val_get_rm_192 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_192, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fmadd_d_188;
        auto fmadd_d_188_arg0 = ret_val_unbox_d_189;
        auto fmadd_d_188_arg1 = ret_val_unbox_d_190;
        auto fmadd_d_188_arg2 = ret_val_unbox_d_191;
        auto fmadd_d_188_arg4 = ret_val_get_rm_192;
        x86::Gp ret_val_fmadd_d_188 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_fmadd_d_188,  &fmadd_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint8_t>());
        InvokeNode* call_NaNBox64_187;
        auto NaNBox64_187_arg0 = ret_val_fmadd_d_188;
        x86::Gp ret_val_NaNBox64_187 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox64_187, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
        auto res = ret_val_NaNBox64_187;
        setArg(call_unbox_d_189, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_189, 1, unbox_d_189_arg1);
        setRet(call_unbox_d_189, 0, ret_val_unbox_d_189);
        setArg(call_unbox_d_190, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_190, 1, unbox_d_190_arg1);
        setRet(call_unbox_d_190, 0, ret_val_unbox_d_190);
        setArg(call_unbox_d_191, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_191, 1, unbox_d_191_arg1);
        setRet(call_unbox_d_191, 0, ret_val_unbox_d_191);
        setArg(call_get_rm_192, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_192, 1, rm);
        setRet(call_get_rm_192, 0, ret_val_get_rm_192);
        setArg(call_fmadd_d_188, 0, fmadd_d_188_arg0);
        setArg(call_fmadd_d_188, 1, fmadd_d_188_arg1);
        setArg(call_fmadd_d_188, 2, fmadd_d_188_arg2);
        setArg(call_fmadd_d_188, 3, 1);
        setArg(call_fmadd_d_188, 4, fmadd_d_188_arg4);
        setRet(call_fmadd_d_188, 0, ret_val_fmadd_d_188);
        setArg(call_NaNBox64_187, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox64_187, 1, NaNBox64_187_arg0);
        setRet(call_NaNBox64_187, 0, ret_val_NaNBox64_187);
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              res);
        InvokeNode* call_fget_flags_193;
        x86::Gp ret_val_fget_flags_193 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_193,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_193;
        setRet(call_fget_flags_193, 0, ret_val_fget_flags_193);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 172);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 173: FNMADD__D */
    continuation_e __fnmadd__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rs3 = ((bit_sub<27,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {rs3}, {rm}", fmt::arg("mnemonic", "fnmadd.d"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FNMADD__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 173);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_d_196;
        auto unbox_d_196_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_d_196 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_196,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_d_197;
        auto unbox_d_197_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_d_197 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_197,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_d_198;
        auto unbox_d_198_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs3);
        x86::Gp ret_val_unbox_d_198 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_198,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_199;
        x86::Gp ret_val_get_rm_199 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_199, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fmadd_d_195;
        auto fmadd_d_195_arg0 = ret_val_unbox_d_196;
        auto fmadd_d_195_arg1 = ret_val_unbox_d_197;
        auto fmadd_d_195_arg2 = ret_val_unbox_d_198;
        auto fmadd_d_195_arg4 = ret_val_get_rm_199;
        x86::Gp ret_val_fmadd_d_195 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_fmadd_d_195,  &fmadd_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint8_t>());
        InvokeNode* call_NaNBox64_194;
        auto NaNBox64_194_arg0 = ret_val_fmadd_d_195;
        x86::Gp ret_val_NaNBox64_194 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox64_194, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox64_194);
        setArg(call_unbox_d_196, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_196, 1, unbox_d_196_arg1);
        setRet(call_unbox_d_196, 0, ret_val_unbox_d_196);
        setArg(call_unbox_d_197, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_197, 1, unbox_d_197_arg1);
        setRet(call_unbox_d_197, 0, ret_val_unbox_d_197);
        setArg(call_unbox_d_198, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_198, 1, unbox_d_198_arg1);
        setRet(call_unbox_d_198, 0, ret_val_unbox_d_198);
        setArg(call_get_rm_199, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_199, 1, rm);
        setRet(call_get_rm_199, 0, ret_val_get_rm_199);
        setArg(call_fmadd_d_195, 0, fmadd_d_195_arg0);
        setArg(call_fmadd_d_195, 1, fmadd_d_195_arg1);
        setArg(call_fmadd_d_195, 2, fmadd_d_195_arg2);
        setArg(call_fmadd_d_195, 3, 2);
        setArg(call_fmadd_d_195, 4, fmadd_d_195_arg4);
        setRet(call_fmadd_d_195, 0, ret_val_fmadd_d_195);
        setArg(call_NaNBox64_194, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox64_194, 1, NaNBox64_194_arg0);
        setRet(call_NaNBox64_194, 0, ret_val_NaNBox64_194);
        InvokeNode* call_fget_flags_200;
        x86::Gp ret_val_fget_flags_200 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_200,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_200;
        setRet(call_fget_flags_200, 0, ret_val_fget_flags_200);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 173);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 174: FNMSUB__D */
    continuation_e __fnmsub__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        uint8_t rs3 = ((bit_sub<27,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}, {rs3}, {rm}", fmt::arg("mnemonic", "fnmsub.d"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FNMSUB__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 174);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_d_203;
        auto unbox_d_203_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_d_203 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_203,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_d_204;
        auto unbox_d_204_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_d_204 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_204,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_d_205;
        auto unbox_d_205_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs3);
        x86::Gp ret_val_unbox_d_205 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_205,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_206;
        x86::Gp ret_val_get_rm_206 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_206, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_fmadd_d_202;
        auto fmadd_d_202_arg0 = ret_val_unbox_d_203;
        auto fmadd_d_202_arg1 = ret_val_unbox_d_204;
        auto fmadd_d_202_arg2 = ret_val_unbox_d_205;
        auto fmadd_d_202_arg4 = ret_val_get_rm_206;
        x86::Gp ret_val_fmadd_d_202 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_fmadd_d_202,  &fmadd_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint8_t>());
        InvokeNode* call_NaNBox64_201;
        auto NaNBox64_201_arg0 = ret_val_fmadd_d_202;
        x86::Gp ret_val_NaNBox64_201 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox64_201, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox64_201);
        setArg(call_unbox_d_203, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_203, 1, unbox_d_203_arg1);
        setRet(call_unbox_d_203, 0, ret_val_unbox_d_203);
        setArg(call_unbox_d_204, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_204, 1, unbox_d_204_arg1);
        setRet(call_unbox_d_204, 0, ret_val_unbox_d_204);
        setArg(call_unbox_d_205, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_205, 1, unbox_d_205_arg1);
        setRet(call_unbox_d_205, 0, ret_val_unbox_d_205);
        setArg(call_get_rm_206, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_206, 1, rm);
        setRet(call_get_rm_206, 0, ret_val_get_rm_206);
        setArg(call_fmadd_d_202, 0, fmadd_d_202_arg0);
        setArg(call_fmadd_d_202, 1, fmadd_d_202_arg1);
        setArg(call_fmadd_d_202, 2, fmadd_d_202_arg2);
        setArg(call_fmadd_d_202, 3, 3);
        setArg(call_fmadd_d_202, 4, fmadd_d_202_arg4);
        setRet(call_fmadd_d_202, 0, ret_val_fmadd_d_202);
        setArg(call_NaNBox64_201, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox64_201, 1, NaNBox64_201_arg0);
        setRet(call_NaNBox64_201, 0, ret_val_NaNBox64_201);
        InvokeNode* call_fget_flags_207;
        x86::Gp ret_val_fget_flags_207 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_207,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_207;
        setRet(call_fget_flags_207, 0, ret_val_fget_flags_207);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 174);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 175: FCVT__W__D */
    continuation_e __fcvt__w__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.w.d"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__W__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 175);
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
            InvokeNode* call_unbox_d_209;
            auto unbox_d_209_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_d_209 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_unbox_d_209,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
            InvokeNode* call_get_rm_210;
            x86::Gp ret_val_get_rm_210 = get_reg_Gp(cc, 8, false);
            cc.invoke(&call_get_rm_210, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
            InvokeNode* call_f64toi32_208;
            auto f64toi32_208_arg0 = ret_val_unbox_d_209;
            auto f64toi32_208_arg1 = ret_val_get_rm_210;
            x86::Gp ret_val_f64toi32_208 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_f64toi32_208,  &f64toi32, FuncSignature::build<uint32_t, uint64_t, uint8_t>());
            auto res = gen_ext(cc, 
                gen_ext(cc, 
                    ret_val_f64toi32_208, 32, false), 64, true);
            setArg(call_unbox_d_209, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_d_209, 1, unbox_d_209_arg1);
            setRet(call_unbox_d_209, 0, ret_val_unbox_d_209);
            setArg(call_get_rm_210, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_get_rm_210, 1, rm);
            setRet(call_get_rm_210, 0, ret_val_get_rm_210);
            setArg(call_f64toi32_208, 0, f64toi32_208_arg0);
            setArg(call_f64toi32_208, 1, f64toi32_208_arg1);
            setRet(call_f64toi32_208, 0, ret_val_f64toi32_208);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, true));
            }
            InvokeNode* call_fget_flags_211;
            x86::Gp ret_val_fget_flags_211 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_211,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_211;
            setRet(call_fget_flags_211, 0, ret_val_fget_flags_211);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 175);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 176: FCVT__WU__D */
    continuation_e __fcvt__wu__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.wu.d"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__WU__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 176);
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
            InvokeNode* call_unbox_d_213;
            auto unbox_d_213_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_d_213 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_unbox_d_213,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
            InvokeNode* call_get_rm_214;
            x86::Gp ret_val_get_rm_214 = get_reg_Gp(cc, 8, false);
            cc.invoke(&call_get_rm_214, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
            InvokeNode* call_f64toui32_212;
            auto f64toui32_212_arg0 = ret_val_unbox_d_213;
            auto f64toui32_212_arg1 = ret_val_get_rm_214;
            x86::Gp ret_val_f64toui32_212 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_f64toui32_212,  &f64toui32, FuncSignature::build<uint32_t, uint64_t, uint8_t>());
            auto res = gen_ext(cc, 
                gen_ext(cc, 
                    ret_val_f64toui32_212, 32, false), 64, true);
            setArg(call_unbox_d_213, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_d_213, 1, unbox_d_213_arg1);
            setRet(call_unbox_d_213, 0, ret_val_unbox_d_213);
            setArg(call_get_rm_214, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_get_rm_214, 1, rm);
            setRet(call_get_rm_214, 0, ret_val_get_rm_214);
            setArg(call_f64toui32_212, 0, f64toui32_212_arg0);
            setArg(call_f64toui32_212, 1, f64toui32_212_arg1);
            setRet(call_f64toui32_212, 0, ret_val_f64toui32_212);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, true));
            }
            InvokeNode* call_fget_flags_215;
            x86::Gp ret_val_fget_flags_215 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_215,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_215;
            setRet(call_fget_flags_215, 0, ret_val_fget_flags_215);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 176);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 177: FCVT__L__D */
    continuation_e __fcvt__l__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.l.d"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__L__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 177);
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
            InvokeNode* call_unbox_d_217;
            auto unbox_d_217_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_d_217 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_unbox_d_217,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
            InvokeNode* call_get_rm_218;
            x86::Gp ret_val_get_rm_218 = get_reg_Gp(cc, 8, false);
            cc.invoke(&call_get_rm_218, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
            InvokeNode* call_f64toi64_216;
            auto f64toi64_216_arg0 = ret_val_unbox_d_217;
            auto f64toi64_216_arg1 = ret_val_get_rm_218;
            x86::Gp ret_val_f64toi64_216 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_f64toi64_216,  &f64toi64, FuncSignature::build<uint64_t, uint64_t, uint8_t>());
            auto res = gen_ext(cc, 
                gen_ext(cc, 
                    ret_val_f64toi64_216, 64, false), 64, true);
            setArg(call_unbox_d_217, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_d_217, 1, unbox_d_217_arg1);
            setRet(call_unbox_d_217, 0, ret_val_unbox_d_217);
            setArg(call_get_rm_218, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_get_rm_218, 1, rm);
            setRet(call_get_rm_218, 0, ret_val_get_rm_218);
            setArg(call_f64toi64_216, 0, f64toi64_216_arg0);
            setArg(call_f64toi64_216, 1, f64toi64_216_arg1);
            setRet(call_f64toi64_216, 0, ret_val_f64toi64_216);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, true));
            }
            InvokeNode* call_fget_flags_219;
            x86::Gp ret_val_fget_flags_219 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_219,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_219;
            setRet(call_fget_flags_219, 0, ret_val_fget_flags_219);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 177);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 178: FCVT__LU__D */
    continuation_e __fcvt__lu__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.lu.d"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__LU__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 178);
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
            InvokeNode* call_unbox_d_221;
            auto unbox_d_221_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_d_221 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_unbox_d_221,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
            InvokeNode* call_get_rm_222;
            x86::Gp ret_val_get_rm_222 = get_reg_Gp(cc, 8, false);
            cc.invoke(&call_get_rm_222, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
            InvokeNode* call_f64toui64_220;
            auto f64toui64_220_arg0 = ret_val_unbox_d_221;
            auto f64toui64_220_arg1 = ret_val_get_rm_222;
            x86::Gp ret_val_f64toui64_220 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_f64toui64_220,  &f64toui64, FuncSignature::build<uint64_t, uint64_t, uint8_t>());
            auto res = ret_val_f64toui64_220;
            setArg(call_unbox_d_221, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_d_221, 1, unbox_d_221_arg1);
            setRet(call_unbox_d_221, 0, ret_val_unbox_d_221);
            setArg(call_get_rm_222, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_get_rm_222, 1, rm);
            setRet(call_get_rm_222, 0, ret_val_get_rm_222);
            setArg(call_f64toui64_220, 0, f64toui64_220_arg0);
            setArg(call_f64toui64_220, 1, f64toui64_220_arg1);
            setRet(call_f64toui64_220, 0, ret_val_f64toui64_220);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      res);
            }
            InvokeNode* call_fget_flags_223;
            x86::Gp ret_val_fget_flags_223 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_223,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_223;
            setRet(call_fget_flags_223, 0, ret_val_fget_flags_223);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 178);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 179: FCVT__D__W */
    continuation_e __fcvt__d__w(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.d.w"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__D__W_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 179);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            InvokeNode* call_get_rm_226;
            x86::Gp ret_val_get_rm_226 = get_reg_Gp(cc, 8, false);
            cc.invoke(&call_get_rm_226, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
            InvokeNode* call_i32tof64_225;
            auto i32tof64_225_arg0 = gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false);
            auto i32tof64_225_arg1 = ret_val_get_rm_226;
            x86::Gp ret_val_i32tof64_225 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_i32tof64_225,  &i32tof64, FuncSignature::build<uint64_t, uint32_t, uint8_t>());
            InvokeNode* call_NaNBox64_224;
            auto NaNBox64_224_arg0 = ret_val_i32tof64_225;
            x86::Gp ret_val_NaNBox64_224 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_NaNBox64_224, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  ret_val_NaNBox64_224);
            setArg(call_get_rm_226, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_get_rm_226, 1, rm);
            setRet(call_get_rm_226, 0, ret_val_get_rm_226);
            setArg(call_i32tof64_225, 0, i32tof64_225_arg0);
            setArg(call_i32tof64_225, 1, i32tof64_225_arg1);
            setRet(call_i32tof64_225, 0, ret_val_i32tof64_225);
            setArg(call_NaNBox64_224, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_NaNBox64_224, 1, NaNBox64_224_arg0);
            setRet(call_NaNBox64_224, 0, ret_val_NaNBox64_224);
            InvokeNode* call_fget_flags_227;
            x86::Gp ret_val_fget_flags_227 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_227,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_227;
            setRet(call_fget_flags_227, 0, ret_val_fget_flags_227);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 179);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 180: FCVT__D__WU */
    continuation_e __fcvt__d__wu(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.d.wu"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__D__WU_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 180);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            InvokeNode* call_get_rm_230;
            x86::Gp ret_val_get_rm_230 = get_reg_Gp(cc, 8, false);
            cc.invoke(&call_get_rm_230, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
            InvokeNode* call_ui32tof64_229;
            auto ui32tof64_229_arg0 = gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs1), 32, false);
            auto ui32tof64_229_arg1 = ret_val_get_rm_230;
            x86::Gp ret_val_ui32tof64_229 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_ui32tof64_229,  &ui32tof64, FuncSignature::build<uint64_t, uint32_t, uint8_t>());
            InvokeNode* call_NaNBox64_228;
            auto NaNBox64_228_arg0 = ret_val_ui32tof64_229;
            x86::Gp ret_val_NaNBox64_228 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_NaNBox64_228, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  ret_val_NaNBox64_228);
            setArg(call_get_rm_230, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_get_rm_230, 1, rm);
            setRet(call_get_rm_230, 0, ret_val_get_rm_230);
            setArg(call_ui32tof64_229, 0, ui32tof64_229_arg0);
            setArg(call_ui32tof64_229, 1, ui32tof64_229_arg1);
            setRet(call_ui32tof64_229, 0, ret_val_ui32tof64_229);
            setArg(call_NaNBox64_228, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_NaNBox64_228, 1, NaNBox64_228_arg0);
            setRet(call_NaNBox64_228, 0, ret_val_NaNBox64_228);
            InvokeNode* call_fget_flags_231;
            x86::Gp ret_val_fget_flags_231 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_231,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_231;
            setRet(call_fget_flags_231, 0, ret_val_fget_flags_231);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 180);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 181: FCVT__D__L */
    continuation_e __fcvt__d__l(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.d.l"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__D__L_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 181);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            InvokeNode* call_get_rm_234;
            x86::Gp ret_val_get_rm_234 = get_reg_Gp(cc, 8, false);
            cc.invoke(&call_get_rm_234, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
            InvokeNode* call_i64tof64_233;
            auto i64tof64_233_arg0 = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto i64tof64_233_arg1 = ret_val_get_rm_234;
            x86::Gp ret_val_i64tof64_233 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_i64tof64_233,  &i64tof64, FuncSignature::build<uint64_t, uint64_t, uint8_t>());
            InvokeNode* call_NaNBox64_232;
            auto NaNBox64_232_arg0 = ret_val_i64tof64_233;
            x86::Gp ret_val_NaNBox64_232 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_NaNBox64_232, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  ret_val_NaNBox64_232);
            setArg(call_get_rm_234, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_get_rm_234, 1, rm);
            setRet(call_get_rm_234, 0, ret_val_get_rm_234);
            setArg(call_i64tof64_233, 0, i64tof64_233_arg0);
            setArg(call_i64tof64_233, 1, i64tof64_233_arg1);
            setRet(call_i64tof64_233, 0, ret_val_i64tof64_233);
            setArg(call_NaNBox64_232, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_NaNBox64_232, 1, NaNBox64_232_arg0);
            setRet(call_NaNBox64_232, 0, ret_val_NaNBox64_232);
            InvokeNode* call_fget_flags_235;
            x86::Gp ret_val_fget_flags_235 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_235,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_235;
            setRet(call_fget_flags_235, 0, ret_val_fget_flags_235);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 181);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 182: FCVT__D__LU */
    continuation_e __fcvt__d__lu(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.d.lu"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__D__LU_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 182);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            InvokeNode* call_get_rm_238;
            x86::Gp ret_val_get_rm_238 = get_reg_Gp(cc, 8, false);
            cc.invoke(&call_get_rm_238, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
            InvokeNode* call_ui64tof64_237;
            auto ui64tof64_237_arg0 = load_reg_from_mem_Gp(jh, traits::X0 + rs1);
            auto ui64tof64_237_arg1 = ret_val_get_rm_238;
            x86::Gp ret_val_ui64tof64_237 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_ui64tof64_237,  &ui64tof64, FuncSignature::build<uint64_t, uint64_t, uint8_t>());
            InvokeNode* call_NaNBox64_236;
            auto NaNBox64_236_arg0 = ret_val_ui64tof64_237;
            x86::Gp ret_val_NaNBox64_236 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_NaNBox64_236, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  ret_val_NaNBox64_236);
            setArg(call_get_rm_238, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_get_rm_238, 1, rm);
            setRet(call_get_rm_238, 0, ret_val_get_rm_238);
            setArg(call_ui64tof64_237, 0, ui64tof64_237_arg0);
            setArg(call_ui64tof64_237, 1, ui64tof64_237_arg1);
            setRet(call_ui64tof64_237, 0, ret_val_ui64tof64_237);
            setArg(call_NaNBox64_236, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_NaNBox64_236, 1, NaNBox64_236_arg0);
            setRet(call_NaNBox64_236, 0, ret_val_NaNBox64_236);
            InvokeNode* call_fget_flags_239;
            x86::Gp ret_val_fget_flags_239 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_239,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_239;
            setRet(call_fget_flags_239, 0, ret_val_fget_flags_239);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 182);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 183: FCVT__S__D */
    continuation_e __fcvt__s__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.s.d"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__S__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 183);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_d_242;
        auto unbox_d_242_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_d_242 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_242,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_243;
        x86::Gp ret_val_get_rm_243 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_243, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_f64tof32_241;
        auto f64tof32_241_arg0 = ret_val_unbox_d_242;
        auto f64tof32_241_arg1 = ret_val_get_rm_243;
        x86::Gp ret_val_f64tof32_241 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_f64tof32_241,  &f64tof32, FuncSignature::build<uint32_t, uint64_t, uint8_t>());
        InvokeNode* call_NaNBox32_240;
        auto NaNBox32_240_arg0 = ret_val_f64tof32_241;
        x86::Gp ret_val_NaNBox32_240 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox32_240, (uintptr_t)&vm_impl::_NaNBox32, FuncSignature::build<uint64_t, uintptr_t, uint32_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox32_240);
        setArg(call_unbox_d_242, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_242, 1, unbox_d_242_arg1);
        setRet(call_unbox_d_242, 0, ret_val_unbox_d_242);
        setArg(call_get_rm_243, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_243, 1, rm);
        setRet(call_get_rm_243, 0, ret_val_get_rm_243);
        setArg(call_f64tof32_241, 0, f64tof32_241_arg0);
        setArg(call_f64tof32_241, 1, f64tof32_241_arg1);
        setRet(call_f64tof32_241, 0, ret_val_f64tof32_241);
        setArg(call_NaNBox32_240, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox32_240, 1, NaNBox32_240_arg0);
        setRet(call_NaNBox32_240, 0, ret_val_NaNBox32_240);
        InvokeNode* call_fget_flags_244;
        x86::Gp ret_val_fget_flags_244 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_fget_flags_244,  &fget_flags, FuncSignature::build<uint32_t>());
        auto flags = ret_val_fget_flags_244;
        setRet(call_fget_flags_244, 0, ret_val_fget_flags_244);
        mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 183);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 184: FCVT__D__S */
    continuation_e __fcvt__d__s(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rm = ((bit_sub<12,3>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rm}", fmt::arg("mnemonic", "fcvt.d.s"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rm", rm));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FCVT__D__S_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 184);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_s_247;
        auto unbox_s_247_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_s_247 = get_reg_Gp(cc, 32, false);
        cc.invoke(&call_unbox_s_247,  &unbox_s, FuncSignature::build<uint32_t, uint32_t, uint64_t>());
        InvokeNode* call_get_rm_248;
        x86::Gp ret_val_get_rm_248 = get_reg_Gp(cc, 8, false);
        cc.invoke(&call_get_rm_248, (uintptr_t)&vm_impl::_get_rm, FuncSignature::build<uint8_t, uintptr_t, uint8_t>());
        InvokeNode* call_f32tof64_246;
        auto f32tof64_246_arg0 = ret_val_unbox_s_247;
        auto f32tof64_246_arg1 = ret_val_get_rm_248;
        x86::Gp ret_val_f32tof64_246 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_f32tof64_246,  &f32tof64, FuncSignature::build<uint64_t, uint32_t, uint8_t>());
        InvokeNode* call_NaNBox64_245;
        auto NaNBox64_245_arg0 = ret_val_f32tof64_246;
        x86::Gp ret_val_NaNBox64_245 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox64_245, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox64_245);
        setArg(call_unbox_s_247, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_s_247, 1, unbox_s_247_arg1);
        setRet(call_unbox_s_247, 0, ret_val_unbox_s_247);
        setArg(call_get_rm_248, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_get_rm_248, 1, rm);
        setRet(call_get_rm_248, 0, ret_val_get_rm_248);
        setArg(call_f32tof64_246, 0, f32tof64_246_arg0);
        setArg(call_f32tof64_246, 1, f32tof64_246_arg1);
        setRet(call_f32tof64_246, 0, ret_val_f32tof64_246);
        setArg(call_NaNBox64_245, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox64_245, 1, NaNBox64_245_arg0);
        setRet(call_NaNBox64_245, 0, ret_val_NaNBox64_245);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 184);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 185: FSGNJ__D */
    continuation_e __fsgnj__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnj.d"),
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
        cc.comment(fmt::format("FSGNJ__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 185);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_d_250;
        auto unbox_d_250_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_d_250 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_250,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_d_251;
        auto unbox_d_251_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_d_251 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_251,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_NaNBox64_249;
        auto NaNBox64_249_arg0 = gen_operation(cc, bor, gen_operation(cc, shl, gen_ext(cc, gen_slice(cc, ret_val_unbox_d_250, 63, 63-63+1), 64, false), 63), gen_ext(cc, gen_slice(cc, ret_val_unbox_d_251, 0, 62-0+1), 64, false));
        x86::Gp ret_val_NaNBox64_249 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox64_249, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox64_249);
        setArg(call_unbox_d_250, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_250, 1, unbox_d_250_arg1);
        setRet(call_unbox_d_250, 0, ret_val_unbox_d_250);
        setArg(call_unbox_d_251, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_251, 1, unbox_d_251_arg1);
        setRet(call_unbox_d_251, 0, ret_val_unbox_d_251);
        setArg(call_NaNBox64_249, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox64_249, 1, NaNBox64_249_arg0);
        setRet(call_NaNBox64_249, 0, ret_val_NaNBox64_249);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 185);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 186: FSGNJN__D */
    continuation_e __fsgnjn__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnjn.d"),
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
        cc.comment(fmt::format("FSGNJN__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 186);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_d_253;
        auto unbox_d_253_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_d_253 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_253,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_d_254;
        auto unbox_d_254_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_d_254 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_254,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_NaNBox64_252;
        auto NaNBox64_252_arg0 = gen_operation(cc, bor, gen_operation(cc, shl, gen_ext(cc, gen_operation(cc, bnot, gen_slice(cc, ret_val_unbox_d_253, 63, 63-63+1)), 64, false), 63), gen_ext(cc, gen_slice(cc, ret_val_unbox_d_254, 0, 62-0+1), 64, false));
        x86::Gp ret_val_NaNBox64_252 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox64_252, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox64_252);
        setArg(call_unbox_d_253, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_253, 1, unbox_d_253_arg1);
        setRet(call_unbox_d_253, 0, ret_val_unbox_d_253);
        setArg(call_unbox_d_254, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_254, 1, unbox_d_254_arg1);
        setRet(call_unbox_d_254, 0, ret_val_unbox_d_254);
        setArg(call_NaNBox64_252, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox64_252, 1, NaNBox64_252_arg0);
        setRet(call_NaNBox64_252, 0, ret_val_NaNBox64_252);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 186);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 187: FSGNJX__D */
    continuation_e __fsgnjx__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnjx.d"),
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
        cc.comment(fmt::format("FSGNJX__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 187);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        InvokeNode* call_unbox_d_256;
        auto unbox_d_256_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
        x86::Gp ret_val_unbox_d_256 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_256,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_unbox_d_257;
        auto unbox_d_257_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
        x86::Gp ret_val_unbox_d_257 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_unbox_d_257,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
        InvokeNode* call_NaNBox64_255;
        auto NaNBox64_255_arg0 = gen_operation(cc, bxor, (gen_operation(cc, band, ret_val_unbox_d_256, ((uint64_t)1<<63))), ret_val_unbox_d_257);
        x86::Gp ret_val_NaNBox64_255 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox64_255, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox64_255);
        setArg(call_unbox_d_256, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_256, 1, unbox_d_256_arg1);
        setRet(call_unbox_d_256, 0, ret_val_unbox_d_256);
        setArg(call_unbox_d_257, 0, static_cast<uint32_t>(traits::FLEN));
        setArg(call_unbox_d_257, 1, unbox_d_257_arg1);
        setRet(call_unbox_d_257, 0, ret_val_unbox_d_257);
        setArg(call_NaNBox64_255, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox64_255, 1, NaNBox64_255_arg0);
        setRet(call_NaNBox64_255, 0, ret_val_NaNBox64_255);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 187);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 188: FMV__X__D */
    continuation_e __fmv__x__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "fmv.x.d"),
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
        cc.comment(fmt::format("FMV__X__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 188);
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
                      gen_ext(cc, 
                          gen_ext(cc, 
                              gen_ext(cc, 
                                  load_reg_from_mem_Gp(jh, traits::F0 + rs1), 64, false), 64, true), 64, true));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 188);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 189: FMV__D__X */
    continuation_e __fmv__d__x(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "fmv.d.x"),
                fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)));
            InvokeNode* call_print_disass;
            char* mnemonic_ptr = strdup(mnemonic.c_str());
            jh.disass_collection.push_back(mnemonic_ptr);
            jh.cc.invoke(&call_print_disass, &print_disass, FuncSignature::build<void, void *, uint64_t, char *>());
            call_print_disass->setArg(0, jh.arch_if_ptr);
            call_print_disass->setArg(1, pc.val);
            call_print_disass->setArg(2, mnemonic_ptr);

        }
        x86::Compiler& cc = jh.cc;
        cc.comment(fmt::format("FMV__D__X_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 189);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+4;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        if(rs1>=static_cast<uint32_t>(traits::RFS)){
            gen_raise(jh, 0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else {
            InvokeNode* call_NaNBox64_258;
            auto NaNBox64_258_arg0 = gen_ext(cc, 
                load_reg_from_mem_Gp(jh, traits::X0 + rs1), 64, false);
            x86::Gp ret_val_NaNBox64_258 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_NaNBox64_258, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
            mov(cc, get_ptr_for(jh, traits::F0+ rd),
                  ret_val_NaNBox64_258);
            setArg(call_NaNBox64_258, 0, reinterpret_cast<uintptr_t>(this));
            setArg(call_NaNBox64_258, 1, NaNBox64_258_arg0);
            setRet(call_NaNBox64_258, 0, ret_val_NaNBox64_258);
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 189);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 190: FEQ__D */
    continuation_e __feq__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "feq.d"),
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
        cc.comment(fmt::format("FEQ__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 190);
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
            InvokeNode* call_unbox_d_260;
            auto unbox_d_260_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_d_260 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_unbox_d_260,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
            InvokeNode* call_unbox_d_261;
            auto unbox_d_261_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
            x86::Gp ret_val_unbox_d_261 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_unbox_d_261,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
            InvokeNode* call_fcmp_d_259;
            auto fcmp_d_259_arg0 = ret_val_unbox_d_260;
            auto fcmp_d_259_arg1 = ret_val_unbox_d_261;
            x86::Gp ret_val_fcmp_d_259 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_fcmp_d_259,  &fcmp_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint32_t>());
            auto res = ret_val_fcmp_d_259;
            setArg(call_unbox_d_260, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_d_260, 1, unbox_d_260_arg1);
            setRet(call_unbox_d_260, 0, ret_val_unbox_d_260);
            setArg(call_unbox_d_261, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_d_261, 1, unbox_d_261_arg1);
            setRet(call_unbox_d_261, 0, ret_val_unbox_d_261);
            setArg(call_fcmp_d_259, 0, fcmp_d_259_arg0);
            setArg(call_fcmp_d_259, 1, fcmp_d_259_arg1);
            setArg(call_fcmp_d_259, 2, 0);
            setRet(call_fcmp_d_259, 0, ret_val_fcmp_d_259);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, false));
            }
            InvokeNode* call_fget_flags_262;
            x86::Gp ret_val_fget_flags_262 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_262,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_262;
            setRet(call_fget_flags_262, 0, ret_val_fget_flags_262);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 190);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 191: FLT__D */
    continuation_e __flt__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "flt.d"),
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
        cc.comment(fmt::format("FLT__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 191);
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
            InvokeNode* call_unbox_d_264;
            auto unbox_d_264_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_d_264 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_unbox_d_264,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
            InvokeNode* call_unbox_d_265;
            auto unbox_d_265_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
            x86::Gp ret_val_unbox_d_265 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_unbox_d_265,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
            InvokeNode* call_fcmp_d_263;
            auto fcmp_d_263_arg0 = ret_val_unbox_d_264;
            auto fcmp_d_263_arg1 = ret_val_unbox_d_265;
            x86::Gp ret_val_fcmp_d_263 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_fcmp_d_263,  &fcmp_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint32_t>());
            auto res = ret_val_fcmp_d_263;
            setArg(call_unbox_d_264, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_d_264, 1, unbox_d_264_arg1);
            setRet(call_unbox_d_264, 0, ret_val_unbox_d_264);
            setArg(call_unbox_d_265, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_d_265, 1, unbox_d_265_arg1);
            setRet(call_unbox_d_265, 0, ret_val_unbox_d_265);
            setArg(call_fcmp_d_263, 0, fcmp_d_263_arg0);
            setArg(call_fcmp_d_263, 1, fcmp_d_263_arg1);
            setArg(call_fcmp_d_263, 2, 2);
            setRet(call_fcmp_d_263, 0, ret_val_fcmp_d_263);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, false));
            }
            InvokeNode* call_fget_flags_266;
            x86::Gp ret_val_fget_flags_266 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_266,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_266;
            setRet(call_fget_flags_266, 0, ret_val_fget_flags_266);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 191);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 192: FLE__D */
    continuation_e __fle__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fle.d"),
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
        cc.comment(fmt::format("FLE__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 192);
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
            InvokeNode* call_unbox_d_268;
            auto unbox_d_268_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_d_268 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_unbox_d_268,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
            InvokeNode* call_unbox_d_269;
            auto unbox_d_269_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs2);
            x86::Gp ret_val_unbox_d_269 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_unbox_d_269,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
            InvokeNode* call_fcmp_d_267;
            auto fcmp_d_267_arg0 = ret_val_unbox_d_268;
            auto fcmp_d_267_arg1 = ret_val_unbox_d_269;
            x86::Gp ret_val_fcmp_d_267 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_fcmp_d_267,  &fcmp_d, FuncSignature::build<uint64_t, uint64_t, uint64_t, uint32_t>());
            auto res = ret_val_fcmp_d_267;
            setArg(call_unbox_d_268, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_d_268, 1, unbox_d_268_arg1);
            setRet(call_unbox_d_268, 0, ret_val_unbox_d_268);
            setArg(call_unbox_d_269, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_d_269, 1, unbox_d_269_arg1);
            setRet(call_unbox_d_269, 0, ret_val_unbox_d_269);
            setArg(call_fcmp_d_267, 0, fcmp_d_267_arg0);
            setArg(call_fcmp_d_267, 1, fcmp_d_267_arg1);
            setArg(call_fcmp_d_267, 2, 1);
            setRet(call_fcmp_d_267, 0, ret_val_fcmp_d_267);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, false));
            }
            InvokeNode* call_fget_flags_270;
            x86::Gp ret_val_fget_flags_270 = get_reg_Gp(cc, 32, false);
            cc.invoke(&call_fget_flags_270,  &fget_flags, FuncSignature::build<uint32_t>());
            auto flags = ret_val_fget_flags_270;
            setRet(call_fget_flags_270, 0, ret_val_fget_flags_270);
            mov(cc, load_reg_from_mem(jh, traits::FCSR), gen_operation(cc, bor, (gen_operation(cc, band, load_reg_from_mem(jh, traits::FCSR), ~ static_cast<uint32_t>(traits::FFLAG_MASK))), (gen_operation(cc, band, flags, static_cast<uint32_t>(traits::FFLAG_MASK)))));
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 192);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 193: FCLASS__D */
    continuation_e __fclass__d(virt_addr_t& pc, code_word_t instr, jit_holder& jh){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        if(this->disass_enabled){
            /* generate disass */
            
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "fclass.d"),
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
        cc.comment(fmt::format("FCLASS__D_{:#x}:",pc.val).c_str());
        gen_sync(jh, PRE_SYNC, 193);
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
            InvokeNode* call_unbox_d_272;
            auto unbox_d_272_arg1 = load_reg_from_mem_Gp(jh, traits::F0 + rs1);
            x86::Gp ret_val_unbox_d_272 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_unbox_d_272,  &unbox_d, FuncSignature::build<uint64_t, uint32_t, uint64_t>());
            InvokeNode* call_fclass_d_271;
            auto fclass_d_271_arg0 = ret_val_unbox_d_272;
            x86::Gp ret_val_fclass_d_271 = get_reg_Gp(cc, 64, false);
            cc.invoke(&call_fclass_d_271,  &fclass_d, FuncSignature::build<uint64_t, uint64_t>());
            auto res = ret_val_fclass_d_271;
            setArg(call_unbox_d_272, 0, static_cast<uint32_t>(traits::FLEN));
            setArg(call_unbox_d_272, 1, unbox_d_272_arg1);
            setRet(call_unbox_d_272, 0, ret_val_unbox_d_272);
            setArg(call_fclass_d_271, 0, fclass_d_271_arg0);
            setRet(call_fclass_d_271, 0, ret_val_fclass_d_271);
            if(rd!=0){
                mov(cc, get_ptr_for(jh, traits::X0+ rd),
                      gen_ext(cc, 
                          res, 64, false));
            }
        }
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 193);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 194: C__FLD */
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
        gen_sync(jh, PRE_SYNC, 194);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1+8), uimm)), 64, false);
        auto res = gen_ext(cc, 
            gen_read_mem(jh, traits::MEM, offs, 8), 64, false);
        InvokeNode* call_NaNBox64_273;
        auto NaNBox64_273_arg0 = res;
        x86::Gp ret_val_NaNBox64_273 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox64_273, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd+8),
              ret_val_NaNBox64_273);
        setArg(call_NaNBox64_273, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox64_273, 1, NaNBox64_273_arg0);
        setRet(call_NaNBox64_273, 0, ret_val_NaNBox64_273);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 194);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 195: C__FSD */
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
        gen_sync(jh, PRE_SYNC, 195);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + rs1+8), uimm)), 64, false);
        gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
            load_reg_from_mem_Gp(jh, traits::F0 + rs2+8), 64, false), 8);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 195);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 196: C__FLDSP */
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
        gen_sync(jh, PRE_SYNC, 196);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + 2), uimm)), 64, false);
        auto res = gen_ext(cc, 
            gen_read_mem(jh, traits::MEM, offs, 8), 64, false);
        InvokeNode* call_NaNBox64_274;
        auto NaNBox64_274_arg0 = res;
        x86::Gp ret_val_NaNBox64_274 = get_reg_Gp(cc, 64, false);
        cc.invoke(&call_NaNBox64_274, (uintptr_t)&vm_impl::_NaNBox64, FuncSignature::build<uint64_t, uintptr_t, uint64_t>());
        mov(cc, get_ptr_for(jh, traits::F0+ rd),
              ret_val_NaNBox64_274);
        setArg(call_NaNBox64_274, 0, reinterpret_cast<uintptr_t>(this));
        setArg(call_NaNBox64_274, 1, NaNBox64_274_arg0);
        setRet(call_NaNBox64_274, 0, ret_val_NaNBox64_274);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 196);
        gen_instr_epilogue(jh);
    	return returnValue;        
    }
    
    /* instruction 197: C__FSDSP */
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
        gen_sync(jh, PRE_SYNC, 197);
        mov(cc, jh.pc, pc.val);
        gen_set_tval(jh, instr);
        pc = pc+2;
        mov(cc, jh.next_pc, pc.val);

        gen_instr_prologue(jh);
        /*generate behavior*/
        auto offs = gen_ext(cc, 
            (gen_operation(cc, add, load_reg_from_mem_Gp(jh, traits::X0 + 2), uimm)), 64, false);
        gen_write_mem(jh, traits::MEM, offs, gen_ext(cc, 
            load_reg_from_mem_Gp(jh, traits::F0 + rs2), 64, false), 8);
        auto returnValue = CONT;
        
        gen_sync(jh, POST_SYNC, 197);
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
std::unique_ptr<vm_if> create<arch::rv64gc>(arch::rv64gc *core, unsigned short port, bool dump) {
    auto ret = new rv64gc::vm_impl<arch::rv64gc>(*core, dump);
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
        core_factory::instance().register_creator("rv64gc|m_p|asmjit", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_m_p<iss::arch::rv64gc>();
		    auto vm = new asmjit::rv64gc::vm_impl<arch::rv64gc>(*cpu, false);
		    if (port != 0) debugger::server<debugger::gdb_session>::run_server(vm, port);
            if(init_data){
                auto* cb = reinterpret_cast<semihosting_cb_t<arch::traits<arch::rv64gc>::reg_t>*>(init_data);
                cpu->set_semihosting_callback(*cb);
            }
            return {cpu_ptr{cpu}, vm_ptr{vm}};
        }),
        core_factory::instance().register_creator("rv64gc|mu_p|asmjit", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_mu_p<iss::arch::rv64gc>();
		    auto vm = new asmjit::rv64gc::vm_impl<arch::rv64gc>(*cpu, false);
		    if (port != 0) debugger::server<debugger::gdb_session>::run_server(vm, port);
            if(init_data){
                auto* cb = reinterpret_cast<semihosting_cb_t<arch::traits<arch::rv64gc>::reg_t>*>(init_data);
                cpu->set_semihosting_callback(*cb);
            }
            return {cpu_ptr{cpu}, vm_ptr{vm}};
        })
};
}
}
// clang-format on
