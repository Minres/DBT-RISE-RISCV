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
#include <cstdint>
#include <iss/arch/rv64gc.h>
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/server.h>
#include <iss/iss.h>
#include <iss/interp/vm_base.h>
#include <vm/fp_functions.h>
#include <util/logging.h>
#include <boost/coroutine2/all.hpp>
#include <functional>
#include <exception>
#include <vector>
#include <sstream>
#include <iss/instruction_decoder.h>


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
using namespace std::placeholders;

struct memory_access_exception : public std::exception{
    memory_access_exception(){}
};

template <typename ARCH> class vm_impl : public iss::interp::vm_base<ARCH> {
public:
    using traits = arch::traits<ARCH>;
    using super       = typename iss::interp::vm_base<ARCH>;
    using virt_addr_t = typename super::virt_addr_t;
    using phys_addr_t = typename super::phys_addr_t;
    using code_word_t = typename super::code_word_t;
    using addr_t      = typename super::addr_t;
    using reg_t       = typename traits::reg_t;
    using mem_type_e  = typename traits::mem_type_e;
    using opcode_e    = typename traits::opcode_e;
    
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

    inline const char *name(size_t index){return traits::reg_aliases.at(index);}

    inline const char *fname(size_t index){return index < 32?name(index+traits::F0):"illegal";}     


    virt_addr_t execute_inst(finish_cond_e cond, virt_addr_t start, uint64_t icount_limit) override;

    // some compile time constants

    inline void raise(uint16_t trap_id, uint16_t cause){
        auto trap_val =  0x80ULL << 24 | (cause << 16) | trap_id;
        this->core.reg.trap_state = trap_val;
    }

    inline void leave(unsigned lvl){
        this->core.leave_trap(lvl);
    }

    inline void wait(unsigned type){
        this->core.wait_until(type);
    }

    inline void set_tval(uint64_t new_tval){
        tval = new_tval;
    }

    uint64_t fetch_count{0};
    uint64_t tval{0};

    using yield_t = boost::coroutines2::coroutine<void>::push_type;
    using coro_t = boost::coroutines2::coroutine<void>::pull_type;
    std::vector<coro_t> spawn_blocks;

    template<unsigned W, typename U, typename S = typename std::make_signed<U>::type>
    inline S sext(U from) {
        auto mask = (1ULL<<W) - 1;
        auto sign_mask = 1ULL<<(W-1);
        return (from & mask) | ((from & sign_mask) ? ~mask : 0);
    }
    
    inline void process_spawn_blocks() {
        if(spawn_blocks.size()==0) return;
        for(auto it = std::begin(spawn_blocks); it!=std::end(spawn_blocks);)
             if(*it){
                 (*it)();
                 ++it;
             } else
                 spawn_blocks.erase(it);
    }

    uint8_t get_rm(uint8_t rm){
        auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]); 
        uint8_t rm_eff = rm == 7? bit_sub<5, 7-5+1>(*FCSR) : rm;
        if(rm_eff > 4) {
            raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
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
        typename arch::traits<ARCH>::opcode_e op;
    };

    const std::array<instruction_descriptor, 198> instr_descr = {{
         /* entries are: size, valid value, valid mask, function ptr */
        {32, 0b00000000000000000000000000110111, 0b00000000000000000000000001111111, arch::traits<ARCH>::opcode_e::LUI},
        {32, 0b00000000000000000000000000010111, 0b00000000000000000000000001111111, arch::traits<ARCH>::opcode_e::AUIPC},
        {32, 0b00000000000000000000000001101111, 0b00000000000000000000000001111111, arch::traits<ARCH>::opcode_e::JAL},
        {32, 0b00000000000000000000000001100111, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::JALR},
        {32, 0b00000000000000000000000001100011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::BEQ},
        {32, 0b00000000000000000001000001100011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::BNE},
        {32, 0b00000000000000000100000001100011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::BLT},
        {32, 0b00000000000000000101000001100011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::BGE},
        {32, 0b00000000000000000110000001100011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::BLTU},
        {32, 0b00000000000000000111000001100011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::BGEU},
        {32, 0b00000000000000000000000000000011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::LB},
        {32, 0b00000000000000000001000000000011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::LH},
        {32, 0b00000000000000000010000000000011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::LW},
        {32, 0b00000000000000000100000000000011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::LBU},
        {32, 0b00000000000000000101000000000011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::LHU},
        {32, 0b00000000000000000000000000100011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::SB},
        {32, 0b00000000000000000001000000100011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::SH},
        {32, 0b00000000000000000010000000100011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::SW},
        {32, 0b00000000000000000000000000010011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::ADDI},
        {32, 0b00000000000000000010000000010011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::SLTI},
        {32, 0b00000000000000000011000000010011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::SLTIU},
        {32, 0b00000000000000000100000000010011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::XORI},
        {32, 0b00000000000000000110000000010011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::ORI},
        {32, 0b00000000000000000111000000010011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::ANDI},
        {32, 0b00000000000000000001000000010011, 0b11111100000000000111000001111111, arch::traits<ARCH>::opcode_e::SLLI},
        {32, 0b00000000000000000101000000010011, 0b11111100000000000111000001111111, arch::traits<ARCH>::opcode_e::SRLI},
        {32, 0b01000000000000000101000000010011, 0b11111100000000000111000001111111, arch::traits<ARCH>::opcode_e::SRAI},
        {32, 0b00000000000000000000000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::ADD},
        {32, 0b01000000000000000000000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::SUB},
        {32, 0b00000000000000000001000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::SLL},
        {32, 0b00000000000000000010000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::SLT},
        {32, 0b00000000000000000011000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::SLTU},
        {32, 0b00000000000000000100000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::XOR},
        {32, 0b00000000000000000101000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::SRL},
        {32, 0b01000000000000000101000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::SRA},
        {32, 0b00000000000000000110000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::OR},
        {32, 0b00000000000000000111000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::AND},
        {32, 0b00000000000000000000000000001111, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::FENCE},
        {32, 0b00000000000000000000000001110011, 0b11111111111111111111111111111111, arch::traits<ARCH>::opcode_e::ECALL},
        {32, 0b00000000000100000000000001110011, 0b11111111111111111111111111111111, arch::traits<ARCH>::opcode_e::EBREAK},
        {32, 0b00110000001000000000000001110011, 0b11111111111111111111111111111111, arch::traits<ARCH>::opcode_e::MRET},
        {32, 0b00010000010100000000000001110011, 0b11111111111111111111111111111111, arch::traits<ARCH>::opcode_e::WFI},
        {32, 0b00000000000000000110000000000011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::LWU},
        {32, 0b00000000000000000011000000000011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::LD},
        {32, 0b00000000000000000011000000100011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::SD},
        {32, 0b00000000000000000000000000011011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::ADDIW},
        {32, 0b00000000000000000001000000011011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::SLLIW},
        {32, 0b00000000000000000101000000011011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::SRLIW},
        {32, 0b01000000000000000101000000011011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::SRAIW},
        {32, 0b00000000000000000000000000111011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::ADDW},
        {32, 0b01000000000000000000000000111011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::SUBW},
        {32, 0b00000000000000000001000000111011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::SLLW},
        {32, 0b00000000000000000101000000111011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::SRLW},
        {32, 0b01000000000000000101000000111011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::SRAW},
        {32, 0b00000000000000000001000001110011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::CSRRW},
        {32, 0b00000000000000000010000001110011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::CSRRS},
        {32, 0b00000000000000000011000001110011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::CSRRC},
        {32, 0b00000000000000000101000001110011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::CSRRWI},
        {32, 0b00000000000000000110000001110011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::CSRRSI},
        {32, 0b00000000000000000111000001110011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::CSRRCI},
        {32, 0b00000000000000000001000000001111, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::FENCE_I},
        {32, 0b00000010000000000000000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::MUL},
        {32, 0b00000010000000000001000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::MULH},
        {32, 0b00000010000000000010000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::MULHSU},
        {32, 0b00000010000000000011000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::MULHU},
        {32, 0b00000010000000000100000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::DIV},
        {32, 0b00000010000000000101000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::DIVU},
        {32, 0b00000010000000000110000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::REM},
        {32, 0b00000010000000000111000000110011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::REMU},
        {32, 0b00000010000000000000000000111011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::MULW},
        {32, 0b00000010000000000100000000111011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::DIVW},
        {32, 0b00000010000000000101000000111011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::DIVUW},
        {32, 0b00000010000000000110000000111011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::REMW},
        {32, 0b00000010000000000111000000111011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::REMUW},
        {32, 0b00010000000000000010000000101111, 0b11111001111100000111000001111111, arch::traits<ARCH>::opcode_e::LRW},
        {32, 0b00011000000000000010000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::SCW},
        {32, 0b00001000000000000010000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOSWAPW},
        {32, 0b00000000000000000010000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOADDW},
        {32, 0b00100000000000000010000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOXORW},
        {32, 0b01100000000000000010000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOANDW},
        {32, 0b01000000000000000010000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOORW},
        {32, 0b10000000000000000010000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOMINW},
        {32, 0b10100000000000000010000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOMAXW},
        {32, 0b11000000000000000010000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOMINUW},
        {32, 0b11100000000000000010000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOMAXUW},
        {32, 0b00010000000000000011000000101111, 0b11111001111100000111000001111111, arch::traits<ARCH>::opcode_e::LRD},
        {32, 0b00011000000000000011000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::SCD},
        {32, 0b00001000000000000011000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOSWAPD},
        {32, 0b00000000000000000011000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOADDD},
        {32, 0b00100000000000000011000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOXORD},
        {32, 0b01100000000000000011000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOANDD},
        {32, 0b01000000000000000011000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOORD},
        {32, 0b10000000000000000011000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOMIND},
        {32, 0b10100000000000000011000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOMAXD},
        {32, 0b11000000000000000011000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOMINUD},
        {32, 0b11100000000000000011000000101111, 0b11111000000000000111000001111111, arch::traits<ARCH>::opcode_e::AMOMAXUD},
        {16, 0b0000000000000000, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__ADDI4SPN},
        {16, 0b0100000000000000, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__LW},
        {16, 0b0110000000000000, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__LD},
        {16, 0b1100000000000000, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__SW},
        {16, 0b1110000000000000, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__SD},
        {16, 0b0000000000000001, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__ADDI},
        {16, 0b0000000000000001, 0b1110111110000011, arch::traits<ARCH>::opcode_e::C__NOP},
        {16, 0b0010000000000001, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__ADDIW},
        {16, 0b0100000000000001, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__LI},
        {16, 0b0110000000000001, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__LUI},
        {16, 0b0110000100000001, 0b1110111110000011, arch::traits<ARCH>::opcode_e::C__ADDI16SP},
        {16, 0b0110000000000001, 0b1111000001111111, arch::traits<ARCH>::opcode_e::__reserved_clui},
        {16, 0b1000000000000001, 0b1110110000000011, arch::traits<ARCH>::opcode_e::C__SRLI},
        {16, 0b1000010000000001, 0b1110110000000011, arch::traits<ARCH>::opcode_e::C__SRAI},
        {16, 0b1000100000000001, 0b1110110000000011, arch::traits<ARCH>::opcode_e::C__ANDI},
        {16, 0b1000110000000001, 0b1111110001100011, arch::traits<ARCH>::opcode_e::C__SUB},
        {16, 0b1000110000100001, 0b1111110001100011, arch::traits<ARCH>::opcode_e::C__XOR},
        {16, 0b1000110001000001, 0b1111110001100011, arch::traits<ARCH>::opcode_e::C__OR},
        {16, 0b1000110001100001, 0b1111110001100011, arch::traits<ARCH>::opcode_e::C__AND},
        {16, 0b1001110000000001, 0b1111110001100011, arch::traits<ARCH>::opcode_e::C__SUBW},
        {16, 0b1001110000100001, 0b1111110001100011, arch::traits<ARCH>::opcode_e::C__ADDW},
        {16, 0b1010000000000001, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__J},
        {16, 0b1100000000000001, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__BEQZ},
        {16, 0b1110000000000001, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__BNEZ},
        {16, 0b0000000000000010, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__SLLI},
        {16, 0b0100000000000010, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__LWSP},
        {16, 0b0110000000000010, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__LDSP},
        {16, 0b1000000000000010, 0b1111000000000011, arch::traits<ARCH>::opcode_e::C__MV},
        {16, 0b1000000000000010, 0b1111000001111111, arch::traits<ARCH>::opcode_e::C__JR},
        {16, 0b1000000000000010, 0b1111111111111111, arch::traits<ARCH>::opcode_e::__reserved_cmv},
        {16, 0b1001000000000010, 0b1111000000000011, arch::traits<ARCH>::opcode_e::C__ADD},
        {16, 0b1001000000000010, 0b1111000001111111, arch::traits<ARCH>::opcode_e::C__JALR},
        {16, 0b1001000000000010, 0b1111111111111111, arch::traits<ARCH>::opcode_e::C__EBREAK},
        {16, 0b1100000000000010, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__SWSP},
        {16, 0b1110000000000010, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__SDSP},
        {16, 0b0000000000000000, 0b1111111111111111, arch::traits<ARCH>::opcode_e::DII},
        {32, 0b00000000000000000010000000000111, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::FLW},
        {32, 0b00000000000000000010000000100111, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::FSW},
        {32, 0b00000000000000000000000001000011, 0b00000110000000000000000001111111, arch::traits<ARCH>::opcode_e::FMADD__S},
        {32, 0b00000000000000000000000001000111, 0b00000110000000000000000001111111, arch::traits<ARCH>::opcode_e::FMSUB__S},
        {32, 0b00000000000000000000000001001111, 0b00000110000000000000000001111111, arch::traits<ARCH>::opcode_e::FNMADD__S},
        {32, 0b00000000000000000000000001001011, 0b00000110000000000000000001111111, arch::traits<ARCH>::opcode_e::FNMSUB__S},
        {32, 0b00000000000000000000000001010011, 0b11111110000000000000000001111111, arch::traits<ARCH>::opcode_e::FADD__S},
        {32, 0b00001000000000000000000001010011, 0b11111110000000000000000001111111, arch::traits<ARCH>::opcode_e::FSUB__S},
        {32, 0b00010000000000000000000001010011, 0b11111110000000000000000001111111, arch::traits<ARCH>::opcode_e::FMUL__S},
        {32, 0b00011000000000000000000001010011, 0b11111110000000000000000001111111, arch::traits<ARCH>::opcode_e::FDIV__S},
        {32, 0b01011000000000000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FSQRT__S},
        {32, 0b00100000000000000000000001010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::FSGNJ__S},
        {32, 0b00100000000000000001000001010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::FSGNJN__S},
        {32, 0b00100000000000000010000001010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::FSGNJX__S},
        {32, 0b00101000000000000000000001010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::FMIN__S},
        {32, 0b00101000000000000001000001010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::FMAX__S},
        {32, 0b11000000000000000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT__W__S},
        {32, 0b11000000000100000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT__WU__S},
        {32, 0b10100000000000000010000001010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::FEQ__S},
        {32, 0b10100000000000000001000001010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::FLT__S},
        {32, 0b10100000000000000000000001010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::FLE__S},
        {32, 0b11100000000000000001000001010011, 0b11111111111100000111000001111111, arch::traits<ARCH>::opcode_e::FCLASS__S},
        {32, 0b11010000000000000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT__S__W},
        {32, 0b11010000000100000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT__S__WU},
        {32, 0b11100000000000000000000001010011, 0b11111111111100000111000001111111, arch::traits<ARCH>::opcode_e::FMV__X__W},
        {32, 0b11110000000000000000000001010011, 0b11111111111100000111000001111111, arch::traits<ARCH>::opcode_e::FMV__W__X},
        {32, 0b11000000001000000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT_L_S},
        {32, 0b11000000001100000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT_LU_S},
        {32, 0b11010000001000000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT_S_L},
        {32, 0b11010000001100000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT_S_LU},
        {32, 0b00000000000000000011000000000111, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::FLD},
        {32, 0b00000000000000000011000000100111, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::FSD},
        {32, 0b00000010000000000000000001000011, 0b00000110000000000000000001111111, arch::traits<ARCH>::opcode_e::FMADD_D},
        {32, 0b00000010000000000000000001000111, 0b00000110000000000000000001111111, arch::traits<ARCH>::opcode_e::FMSUB_D},
        {32, 0b00000010000000000000000001001111, 0b00000110000000000000000001111111, arch::traits<ARCH>::opcode_e::FNMADD_D},
        {32, 0b00000010000000000000000001001011, 0b00000110000000000000000001111111, arch::traits<ARCH>::opcode_e::FNMSUB_D},
        {32, 0b00000010000000000000000001010011, 0b11111110000000000000000001111111, arch::traits<ARCH>::opcode_e::FADD_D},
        {32, 0b00001010000000000000000001010011, 0b11111110000000000000000001111111, arch::traits<ARCH>::opcode_e::FSUB_D},
        {32, 0b00010010000000000000000001010011, 0b11111110000000000000000001111111, arch::traits<ARCH>::opcode_e::FMUL_D},
        {32, 0b00011010000000000000000001010011, 0b11111110000000000000000001111111, arch::traits<ARCH>::opcode_e::FDIV_D},
        {32, 0b01011010000000000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FSQRT_D},
        {32, 0b00100010000000000000000001010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::FSGNJ_D},
        {32, 0b00100010000000000001000001010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::FSGNJN_D},
        {32, 0b00100010000000000010000001010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::FSGNJX_D},
        {32, 0b00101010000000000000000001010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::FMIN_D},
        {32, 0b00101010000000000001000001010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::FMAX_D},
        {32, 0b01000000000100000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT_S_D},
        {32, 0b01000010000000000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT_D_S},
        {32, 0b10100010000000000010000001010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::FEQ_D},
        {32, 0b10100010000000000001000001010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::FLT_D},
        {32, 0b10100010000000000000000001010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::FLE_D},
        {32, 0b11100010000000000001000001010011, 0b11111111111100000111000001111111, arch::traits<ARCH>::opcode_e::FCLASS_D},
        {32, 0b11000010000000000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT_W_D},
        {32, 0b11000010000100000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT_WU_D},
        {32, 0b11010010000000000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT_D_W},
        {32, 0b11010010000100000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT_D_WU},
        {32, 0b11000010001000000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT_L_D},
        {32, 0b11000010001100000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT_LU_D},
        {32, 0b11010010001000000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT_D_L},
        {32, 0b11010010001100000000000001010011, 0b11111111111100000000000001111111, arch::traits<ARCH>::opcode_e::FCVT_D_LU},
        {32, 0b11100010000000000000000001010011, 0b11111111111100000111000001111111, arch::traits<ARCH>::opcode_e::FMV_X_D},
        {32, 0b11110010000000000000000001010011, 0b11111111111100000111000001111111, arch::traits<ARCH>::opcode_e::FMV_D_X},
        {16, 0b0010000000000000, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__FLD},
        {16, 0b1010000000000000, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__FSD},
        {16, 0b0010000000000010, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__FLDSP},
        {16, 0b1010000000000010, 0b1110000000000011, arch::traits<ARCH>::opcode_e::C__FSDSP},
    }};

    //needs to be declared after instr_descr
    decoder instr_decoder;

    iss::status fetch_ins(virt_addr_t pc, uint8_t * data){
        if(this->core.has_mmu()) {
            auto phys_pc = this->core.virt2phys(pc);
//            if ((pc.val & upper_bits) != ((pc.val + 2) & upper_bits)) { // we may cross a page boundary
//                if (this->core.read(phys_pc, 2, data) != iss::Ok) return iss::Err;
//                if ((data[0] & 0x3) == 0x3) // this is a 32bit instruction
//                    if (this->core.read(this->core.v2p(pc + 2), 2, data + 2) != iss::Ok)
//                        return iss::Err;
//            } else {
                if (this->core.read(phys_pc, 4, data) != iss::Ok)
                    return iss::Err;
//            }
        } else {
            if (this->core.read(phys_addr_t(pc.access, pc.space, pc.val), 4, data) != iss::Ok)
                return iss::Err;

        }
        return iss::Ok;
    }
};

template <typename CODE_WORD> void debug_fn(CODE_WORD insn) {
    volatile CODE_WORD x = insn;
    insn = 2 * x;
}

template <typename ARCH> vm_impl<ARCH>::vm_impl() { this(new ARCH()); }

// according to
// https://stackoverflow.com/questions/8871204/count-number-of-1s-in-binary-representation
#ifdef __GCC__
constexpr size_t bit_count(uint32_t u) { return __builtin_popcount(u); }
#elif __cplusplus < 201402L
constexpr size_t uCount(uint32_t u) { return u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111); }
constexpr size_t bit_count(uint32_t u) { return ((uCount(u) + (uCount(u) >> 3)) & 030707070707) % 63; }
#else
constexpr size_t bit_count(uint32_t u) {
    size_t uCount = u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111);
    return ((uCount + (uCount >> 3)) & 030707070707) % 63;
}
#endif

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

inline bool is_icount_limit_enabled(finish_cond_e cond){
    return (cond & finish_cond_e::ICOUNT_LIMIT) == finish_cond_e::ICOUNT_LIMIT;
}

inline bool is_fcount_limit_enabled(finish_cond_e cond){
    return (cond & finish_cond_e::FCOUNT_LIMIT) == finish_cond_e::FCOUNT_LIMIT;
}

inline bool is_jump_to_self_enabled(finish_cond_e cond){
    return (cond & finish_cond_e::JUMP_TO_SELF) == finish_cond_e::JUMP_TO_SELF;
}

template <typename ARCH>
typename vm_base<ARCH>::virt_addr_t vm_impl<ARCH>::execute_inst(finish_cond_e cond, virt_addr_t start, uint64_t count_limit){
    auto pc=start;
    auto* PC = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::PC]);
    auto* NEXT_PC = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::NEXT_PC]);
    auto& trap_state = this->core.reg.trap_state;
    auto& icount =  this->core.reg.icount;
    auto& cycle =  this->core.reg.cycle;
    auto& instret =  this->core.reg.instret;
    auto& instr =  this->core.reg.instruction;
    // we fetch at max 4 byte, alignment is 2
    auto *const data = reinterpret_cast<uint8_t*>(&instr);

    while(!this->core.should_stop() &&
            !(is_icount_limit_enabled(cond) && icount >= count_limit) &&
            !(is_fcount_limit_enabled(cond) && fetch_count >= count_limit)){
        if(this->debugging_enabled())
            this->tgt_adapter->check_continue(*PC);
        pc.val=*PC;
        if(fetch_ins(pc, data)!=iss::Ok){
            if(this->sync_exec && PRE_SYNC) this->do_sync(PRE_SYNC, std::numeric_limits<unsigned>::max());
            process_spawn_blocks();
            if(this->sync_exec && POST_SYNC) this->do_sync(PRE_SYNC, std::numeric_limits<unsigned>::max());
            pc.val = super::core.enter_trap(arch::traits<ARCH>::RV_CAUSE_FETCH_ACCESS<<16, pc.val, 0);
        } else {
            if (is_jump_to_self_enabled(cond) &&
                    (instr == 0x0000006f || (instr&0xffff)==0xa001)) throw simulation_stopped(0); // 'J 0' or 'C.J 0'
            uint32_t inst_index = instr_decoder.decode_instr(instr);
            opcode_e inst_id = arch::traits<ARCH>::opcode_e::MAX_OPCODE;;
            if(inst_index <instr_descr.size())
                inst_id = instr_descr[inst_index].op;

            // pre execution stuff
             this->core.reg.last_branch = 0;
            if(this->sync_exec && PRE_SYNC) this->do_sync(PRE_SYNC, static_cast<unsigned>(inst_id));
            try{
                switch(inst_id){
                case arch::traits<ARCH>::opcode_e::LUI: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint32_t imm = ((bit_sub<12,20>(instr) << 12));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "lui"),
                            fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)((int32_t)imm);
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AUIPC: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint32_t imm = ((bit_sub<12,20>(instr) << 12));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {imm:#08x}", fmt::arg("mnemonic", "auipc"),
                            fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)((uint128_t)(*PC ) + (uint128_t)((int32_t)imm ));
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::JAL: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint32_t imm = ((bit_sub<12,8>(instr) << 12) | (bit_sub<20,1>(instr) << 11) | (bit_sub<21,10>(instr) << 1) | (bit_sub<31,1>(instr) << 20));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {imm:#0x}", fmt::arg("mnemonic", "jal"),
                            fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t new_pc = (uint64_t)((uint128_t)(*PC ) + (uint128_t)((int32_t)sext<21>(imm) ));
                                        if(new_pc % (uint64_t)(traits::INSTR_ALIGNMENT )) {
                                            set_tval(new_pc);
                                            raise(0, 0);
                                        }
                                        else {
                                            if(rd != 0) {
                                                *(X+rd) = (uint64_t)((uint128_t)(*PC ) + (uint128_t)(4 ));
                                            }
                                            *NEXT_PC = new_pc;
                                            this->core.reg.last_branch = 1;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::JALR: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {imm:#0x}", fmt::arg("mnemonic", "jalr"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t addr_mask = (uint64_t)- 2;
                                        uint64_t new_pc = (uint64_t)(((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) )) & (int128_t)(addr_mask ));
                                        if(new_pc % (uint64_t)(traits::INSTR_ALIGNMENT )) {
                                            set_tval(new_pc);
                                            raise(0, 0);
                                        }
                                        else {
                                            if(rd != 0) {
                                                *(X+rd) = (uint64_t)((uint128_t)(*PC ) + (uint128_t)(4 ));
                                            }
                                            *NEXT_PC = new_pc;
                                            this->core.reg.last_branch = 1;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::BEQ: {
                    uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "beq"),
                            fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(*(X+rs1) == *(X+rs2)) {
                                            uint64_t new_pc = (uint64_t)((uint128_t)(*PC ) + (uint128_t)((int16_t)sext<13>(imm) ));
                                            if(new_pc % (uint64_t)(traits::INSTR_ALIGNMENT )) {
                                                set_tval(new_pc);
                                                raise(0, 0);
                                            }
                                            else {
                                                *NEXT_PC = new_pc;
                                                this->core.reg.last_branch = 1;
                                            }
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::BNE: {
                    uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bne"),
                            fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(*(X+rs1) != *(X+rs2)) {
                                            uint64_t new_pc = (uint64_t)((uint128_t)(*PC ) + (uint128_t)((int16_t)sext<13>(imm) ));
                                            if(new_pc % (uint64_t)(traits::INSTR_ALIGNMENT )) {
                                                set_tval(new_pc);
                                                raise(0, 0);
                                            }
                                            else {
                                                *NEXT_PC = new_pc;
                                                this->core.reg.last_branch = 1;
                                            }
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::BLT: {
                    uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "blt"),
                            fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if((int64_t)*(X+rs1) < (int64_t)*(X+rs2)) {
                                            uint64_t new_pc = (uint64_t)((uint128_t)(*PC ) + (uint128_t)((int16_t)sext<13>(imm) ));
                                            if(new_pc % (uint64_t)(traits::INSTR_ALIGNMENT )) {
                                                set_tval(new_pc);
                                                raise(0, 0);
                                            }
                                            else {
                                                *NEXT_PC = new_pc;
                                                this->core.reg.last_branch = 1;
                                            }
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::BGE: {
                    uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bge"),
                            fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if((int64_t)*(X+rs1) >= (int64_t)*(X+rs2)) {
                                            uint64_t new_pc = (uint64_t)((uint128_t)(*PC ) + (uint128_t)((int16_t)sext<13>(imm) ));
                                            if(new_pc % (uint64_t)(traits::INSTR_ALIGNMENT )) {
                                                set_tval(new_pc);
                                                raise(0, 0);
                                            }
                                            else {
                                                *NEXT_PC = new_pc;
                                                this->core.reg.last_branch = 1;
                                            }
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::BLTU: {
                    uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bltu"),
                            fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(*(X+rs1) < *(X+rs2)) {
                                            uint64_t new_pc = (uint64_t)((uint128_t)(*PC ) + (uint128_t)((int16_t)sext<13>(imm) ));
                                            if(new_pc % (uint64_t)(traits::INSTR_ALIGNMENT )) {
                                                set_tval(new_pc);
                                                raise(0, 0);
                                            }
                                            else {
                                                *NEXT_PC = new_pc;
                                                this->core.reg.last_branch = 1;
                                            }
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::BGEU: {
                    uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bgeu"),
                            fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(*(X+rs1) >= *(X+rs2)) {
                                            uint64_t new_pc = (uint64_t)((uint128_t)(*PC ) + (uint128_t)((int16_t)sext<13>(imm) ));
                                            if(new_pc % (uint64_t)(traits::INSTR_ALIGNMENT )) {
                                                set_tval(new_pc);
                                                raise(0, 0);
                                            }
                                            else {
                                                *NEXT_PC = new_pc;
                                                this->core.reg.last_branch = 1;
                                            }
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::LB: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lb"),
                            fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t load_address = (uint64_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                        int8_t res_87 = super::template read_mem<int8_t>(traits::MEM, load_address);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        int8_t res = (int8_t)res_87;
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::LH: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lh"),
                            fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t load_address = (uint64_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                        int16_t res_88 = super::template read_mem<int16_t>(traits::MEM, load_address);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        int16_t res = (int16_t)res_88;
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::LW: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lw"),
                            fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t load_address = (uint64_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                        int32_t res_89 = super::template read_mem<int32_t>(traits::MEM, load_address);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        int32_t res = (int32_t)res_89;
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::LBU: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lbu"),
                            fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t load_address = (uint64_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                        uint8_t res_90 = super::template read_mem<uint8_t>(traits::MEM, load_address);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint8_t res = res_90;
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::LHU: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lhu"),
                            fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t load_address = (uint64_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                        uint16_t res_91 = super::template read_mem<uint16_t>(traits::MEM, load_address);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint16_t res = res_91;
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SB: {
                    uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sb"),
                            fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t store_address = (uint64_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                        super::template write_mem<uint8_t>(traits::MEM, store_address, (uint8_t)*(X+rs2));
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SH: {
                    uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sh"),
                            fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t store_address = (uint64_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                        super::template write_mem<uint16_t>(traits::MEM, store_address, (uint16_t)*(X+rs2));
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SW: {
                    uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sw"),
                            fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t store_address = (uint64_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                        super::template write_mem<uint32_t>(traits::MEM, store_address, (uint32_t)*(X+rs2));
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::ADDI: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "addi"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SLTI: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "slti"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = ((int64_t)*(X+rs1) < (int16_t)sext<12>(imm))? 1 : 0;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SLTIU: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "sltiu"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (*(X+rs1) < (uint64_t)((int16_t)sext<12>(imm)))? 1 : 0;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::XORI: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "xori"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = *(X+rs1) ^ (uint64_t)((int16_t)sext<12>(imm));
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::ORI: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "ori"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = *(X+rs1) | (uint64_t)((int16_t)sext<12>(imm));
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::ANDI: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "andi"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = *(X+rs1) & (uint64_t)((int16_t)sext<12>(imm));
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SLLI: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t shamt = ((bit_sub<20,6>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "slli"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = *(X+rs1) << shamt;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SRLI: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t shamt = ((bit_sub<20,6>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srli"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = *(X+rs1) >> shamt;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SRAI: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t shamt = ((bit_sub<20,6>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srai"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)(((int64_t)*(X+rs1)) >> shamt);
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::ADD: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)((uint128_t)(*(X+rs1) ) + (uint128_t)(*(X+rs2) ));
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SUB: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)((uint128_t)(*(X+rs1) ) - (uint128_t)(*(X+rs2) ));
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SLL: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = *(X+rs1) << (*(X+rs2) & ((uint64_t)(traits::XLEN ) - (uint64_t)(1 )));
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SLT: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (int64_t)*(X+rs1) < (int64_t)*(X+rs2)? 1 : 0;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SLTU: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = *(X+rs1) < *(X+rs2)? 1 : 0;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::XOR: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = *(X+rs1) ^ *(X+rs2);
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SRL: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = *(X+rs1) >> (*(X+rs2) & ((uint64_t)(traits::XLEN ) - (uint64_t)(1 )));
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SRA: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)((int64_t)*(X+rs1) >> (*(X+rs2) & ((uint64_t)(traits::XLEN ) - (uint64_t)(1 ))));
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::OR: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = *(X+rs1) | *(X+rs2);
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AND: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = *(X+rs1) & *(X+rs2);
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FENCE: {
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
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    super::template write_mem<uint64_t>(traits::FENCE, traits::fence, (uint8_t)pred << 4 | succ);
                                    if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::ECALL: {
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        //No disass specified, using instruction name
                        std::string mnemonic = "ecall";
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    raise(0, 11);
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::EBREAK: {
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        //No disass specified, using instruction name
                        std::string mnemonic = "ebreak";
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    raise(0, 3);
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::MRET: {
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        //No disass specified, using instruction name
                        std::string mnemonic = "mret";
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    leave(3);
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::WFI: {
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        //No disass specified, using instruction name
                        std::string mnemonic = "wfi";
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    wait(1);
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::LWU: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lwu"),
                            fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = (uint64_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                        uint32_t res_92 = super::template read_mem<uint32_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint32_t res = (uint32_t)res_92;
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::LD: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "ld"),
                            fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = (uint64_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                        int64_t res_93 = super::template read_mem<int64_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        int64_t res = (int64_t)res_93;
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SD: {
                    uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sd"),
                            fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = (uint64_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                        super::template write_mem<uint64_t>(traits::MEM, offs, (uint64_t)*(X+rs2));
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::ADDIW: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "addiw"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            int32_t res = (int32_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                            *(X+rd) = (uint64_t)res;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SLLIW: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            uint32_t sh_val = ((uint32_t)*(X+rs1)) << shamt;
                                            *(X+rd) = (uint64_t)(int32_t)sh_val;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SRLIW: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            uint32_t sh_val = ((uint32_t)*(X+rs1)) >> shamt;
                                            *(X+rd) = (uint64_t)(int32_t)sh_val;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SRAIW: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            int32_t sh_val = ((int32_t)*(X+rs1)) >> shamt;
                                            *(X+rd) = (uint64_t)sh_val;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::ADDW: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        //No disass specified, using instruction name
                        std::string mnemonic = "addw";
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            int32_t res = (int32_t)((int64_t)((int32_t)*(X+rs1) ) + (int64_t)((int32_t)*(X+rs2) ));
                                            *(X+rd) = (uint64_t)res;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SUBW: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        //No disass specified, using instruction name
                        std::string mnemonic = "subw";
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            int32_t res = (int32_t)((int64_t)((int32_t)*(X+rs1) ) - (int64_t)((int32_t)*(X+rs2) ));
                                            *(X+rd) = (uint64_t)res;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SLLW: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            uint32_t count = (uint32_t)*(X+rs2) & (uint32_t)(31 );
                                            uint32_t sh_val = ((uint32_t)*(X+rs1)) << count;
                                            *(X+rd) = (uint64_t)(int32_t)sh_val;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SRLW: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            uint32_t count = (uint32_t)*(X+rs2) & (uint32_t)(31 );
                                            uint32_t sh_val = ((uint32_t)*(X+rs1)) >> count;
                                            *(X+rd) = (uint64_t)(int32_t)sh_val;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SRAW: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            uint32_t count = (uint32_t)*(X+rs2) & (uint32_t)(31 );
                                            int32_t sh_val = ((int32_t)*(X+rs1)) >> count;
                                            *(X+rd) = (uint64_t)sh_val;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::CSRRW: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t xrs1 = *(X+rs1);
                                        if(rd != 0) {
                                            uint64_t res_94 = super::template read_mem<uint64_t>(traits::CSR, csr);
                                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                            uint64_t xrd = res_94;
                                            super::template write_mem<uint64_t>(traits::CSR, csr, xrs1);
                                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                            *(X+rd) = xrd;
                                        }
                                        else {
                                            super::template write_mem<uint64_t>(traits::CSR, csr, xrs1);
                                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::CSRRS: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t res_95 = super::template read_mem<uint64_t>(traits::CSR, csr);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint64_t xrd = res_95;
                                        uint64_t xrs1 = *(X+rs1);
                                        if(rs1 != 0) {
                                            super::template write_mem<uint64_t>(traits::CSR, csr, xrd | xrs1);
                                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        }
                                        if(rd != 0) {
                                            *(X+rd) = xrd;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::CSRRC: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t res_96 = super::template read_mem<uint64_t>(traits::CSR, csr);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint64_t xrd = res_96;
                                        uint64_t xrs1 = *(X+rs1);
                                        if(rs1 != 0) {
                                            super::template write_mem<uint64_t>(traits::CSR, csr, xrd & ~ xrs1);
                                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        }
                                        if(rd != 0) {
                                            *(X+rd) = xrd;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::CSRRWI: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t res_97 = super::template read_mem<uint64_t>(traits::CSR, csr);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint64_t xrd = res_97;
                                        super::template write_mem<uint64_t>(traits::CSR, csr, (uint64_t)zimm);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        if(rd != 0) {
                                            *(X+rd) = xrd;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::CSRRSI: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t res_98 = super::template read_mem<uint64_t>(traits::CSR, csr);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint64_t xrd = res_98;
                                        if(zimm != 0) {
                                            super::template write_mem<uint64_t>(traits::CSR, csr, xrd | (uint64_t)zimm);
                                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        }
                                        if(rd != 0) {
                                            *(X+rd) = xrd;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::CSRRCI: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t res_99 = super::template read_mem<uint64_t>(traits::CSR, csr);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint64_t xrd = res_99;
                                        if(zimm != 0) {
                                            super::template write_mem<uint64_t>(traits::CSR, csr, xrd & ~ ((uint64_t)zimm));
                                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        }
                                        if(rd != 0) {
                                            *(X+rd) = xrd;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FENCE_I: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs1}, {rd}, {imm}", fmt::arg("mnemonic", "fence_i"),
                            fmt::arg("rs1", name(rs1)), fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    super::template write_mem<uint64_t>(traits::FENCE, traits::fencei, imm);
                                    if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::MUL: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        int128_t res = (int128_t)((int64_t)*(X+rs1) ) * (int128_t)((int64_t)*(X+rs2) );
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::MULH: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        int128_t res = (int128_t)((int64_t)*(X+rs1) ) * (int128_t)((int64_t)*(X+rs2) );
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)(res >> traits::XLEN);
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::MULHSU: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        int128_t res = (int128_t)((int64_t)*(X+rs1) ) * (int128_t)(*(X+rs2) );
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)(res >> traits::XLEN);
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::MULHU: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint128_t res = (uint128_t)(*(X+rs1) ) * (uint128_t)(*(X+rs2) );
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)(res >> traits::XLEN);
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::DIV: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        int64_t dividend = (int64_t)*(X+rs1);
                                        int64_t divisor = (int64_t)*(X+rs2);
                                        if(rd != 0) {
                                            if(divisor != 0) {
                                                uint64_t MMIN = ((uint64_t)1) << ((uint64_t)(traits::XLEN ) - (uint64_t)(1 ));
                                                if(*(X+rs1) == MMIN && divisor == - 1) {
                                                    *(X+rd) = MMIN;
                                                }
                                                else {
                                                    *(X+rd) = (uint64_t)(dividend / divisor);
                                                }
                                            }
                                            else {
                                                *(X+rd) = (uint64_t)- 1;
                                            }
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::DIVU: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(*(X+rs2) != 0) {
                                            if(rd != 0) {
                                                *(X+rd) = *(X+rs1) / *(X+rs2);
                                            }
                                        }
                                        else {
                                            if(rd != 0) {
                                                *(X+rd) = (uint64_t)- 1;
                                            }
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::REM: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(*(X+rs2) != 0) {
                                            uint64_t MMIN = (uint64_t)1 << ((uint64_t)(traits::XLEN ) - (uint64_t)(1 ));
                                            if(*(X+rs1) == MMIN && (int64_t)*(X+rs2) == - 1) {
                                                if(rd != 0) {
                                                    *(X+rd) = 0;
                                                }
                                            }
                                            else {
                                                if(rd != 0) {
                                                    *(X+rd) = ((uint64_t)((int64_t)*(X+rs1) % (int64_t)*(X+rs2)));
                                                }
                                            }
                                        }
                                        else {
                                            if(rd != 0) {
                                                *(X+rd) = *(X+rs1);
                                            }
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::REMU: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(*(X+rs2) != 0) {
                                            if(rd != 0) {
                                                *(X+rd) = *(X+rs1) % *(X+rs2);
                                            }
                                        }
                                        else {
                                            if(rd != 0) {
                                                *(X+rd) = *(X+rs1);
                                            }
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::MULW: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            int32_t resw = (int32_t)((int64_t)((int32_t)*(X+rs1) ) * (int64_t)((int32_t)*(X+rs2) ));
                                            *(X+rd) = ((uint64_t)(int64_t)resw);
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::DIVW: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        int32_t dividend = (int32_t)*(X+rs1);
                                        int32_t divisor = (int32_t)*(X+rs2);
                                        if(divisor != 0) {
                                            int32_t MMIN = (int32_t)1 << 31;
                                            if(dividend == MMIN && divisor == - 1) {
                                                if(rd != 0) {
                                                    *(X+rd) = (uint64_t)- 1 << 31;
                                                }
                                            }
                                            else {
                                                if(rd != 0) {
                                                    *(X+rd) = (uint64_t)(dividend / divisor);
                                                }
                                            }
                                        }
                                        else {
                                            if(rd != 0) {
                                                *(X+rd) = (uint64_t)- 1;
                                            }
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::DIVUW: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t divisor = (uint32_t)*(X+rs2);
                                        if(divisor != 0) {
                                            int32_t res = (int32_t)((uint32_t)*(X+rs1) / divisor);
                                            if(rd != 0) {
                                                *(X+rd) = ((uint64_t)(int64_t)res);
                                            }
                                        }
                                        else {
                                            if(rd != 0) {
                                                *(X+rd) = (uint64_t)- 1;
                                            }
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::REMW: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(((int32_t)*(X+rs2)) != 0) {
                                            int32_t SMIN = (int32_t)1 << 31;
                                            if((int32_t)*(X+rs1) == SMIN && (int32_t)*(X+rs2) == - 1) {
                                                if(rd != 0) {
                                                    *(X+rd) = 0;
                                                }
                                            }
                                            else {
                                                if(rd != 0) {
                                                    *(X+rd) = (uint64_t)((int32_t)*(X+rs1) % (int32_t)*(X+rs2));
                                                }
                                            }
                                        }
                                        else {
                                            if(rd != 0) {
                                                *(X+rd) = (uint64_t)((int32_t)*(X+rs1));
                                            }
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::REMUW: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t divisor = (uint32_t)*(X+rs2);
                                        if(divisor != 0) {
                                            int32_t res = (int32_t)((uint32_t)*(X+rs1) % divisor);
                                            if(rd != 0) {
                                                *(X+rd) = ((uint64_t)(int64_t)res);
                                            }
                                        }
                                        else {
                                            int32_t res = (int32_t)*(X+rs1);
                                            if(rd != 0) {
                                                *(X+rd) = ((uint64_t)(int64_t)res);
                                            }
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::LRW: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rl = ((bit_sub<25,1>(instr)));
                    uint8_t aq = ((bit_sub<26,1>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {aq}, {rl}", fmt::arg("mnemonic", "lrw"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("aq", name(aq)), fmt::arg("rl", name(rl)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            uint64_t offs = *(X+rs1);
                                            int32_t res_100 = super::template read_mem<int32_t>(traits::MEM, offs);
                                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                            *(X+rd) = (uint64_t)((int32_t)res_100);
                                            super::template write_mem<uint8_t>(traits::RES, offs, (uint8_t)- 1);
                                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SCW: {
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
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        uint8_t res_101 = super::template read_mem<uint8_t>(traits::RES, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint32_t res1 = res_101;
                                        if(res1 != 0) {
                                            super::template write_mem<uint32_t>(traits::MEM, offs, (uint32_t)*(X+rs2));
                                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        }
                                        if(rd != 0) {
                                            *(X+rd) = res1? 0 : 1;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOSWAPW: {
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
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        uint64_t res = *(X+rs2);
                                        if(rd != 0) {
                                            int32_t res_102 = super::template read_mem<int32_t>(traits::MEM, offs);
                                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                            *(X+rd) = (uint64_t)((int32_t)res_102);
                                        }
                                        super::template write_mem<uint32_t>(traits::MEM, offs, (uint32_t)res);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOADDW: {
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
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        int32_t res_103 = super::template read_mem<int32_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        int32_t res1 = (int32_t)res_103;
                                        int64_t res2 = (int64_t)(res1 ) + (int64_t)((int32_t)*(X+rs2) );
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res1;
                                        }
                                        super::template write_mem<uint32_t>(traits::MEM, offs, (uint32_t)res2);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOXORW: {
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
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        uint32_t res_104 = super::template read_mem<uint32_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint32_t res1 = res_104;
                                        uint32_t res2 = res1 ^ (uint32_t)*(X+rs2);
                                        if(rd != 0) {
                                            *(X+rd) = ((uint64_t)(int64_t)(int32_t)res1);
                                        }
                                        super::template write_mem<uint32_t>(traits::MEM, offs, res2);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOANDW: {
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
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        uint32_t res_105 = super::template read_mem<uint32_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint32_t res1 = res_105;
                                        uint32_t res2 = res1 & (uint32_t)*(X+rs2);
                                        if(rd != 0) {
                                            *(X+rd) = ((uint64_t)(int64_t)(int32_t)res1);
                                        }
                                        super::template write_mem<uint32_t>(traits::MEM, offs, res2);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOORW: {
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
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        uint32_t res_106 = super::template read_mem<uint32_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint32_t res1 = res_106;
                                        uint32_t res2 = res1 | (uint32_t)*(X+rs2);
                                        if(rd != 0) {
                                            *(X+rd) = ((uint64_t)(int64_t)(int32_t)res1);
                                        }
                                        super::template write_mem<uint32_t>(traits::MEM, offs, res2);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOMINW: {
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
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        int32_t res_107 = super::template read_mem<int32_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        int32_t res1 = (int32_t)res_107;
                                        uint32_t res2 = res1 > (int32_t)*(X+rs2)? (uint32_t)*(X+rs2) : ((uint32_t)res1);
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res1;
                                        }
                                        super::template write_mem<uint32_t>(traits::MEM, offs, res2);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOMAXW: {
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
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        int32_t res_108 = super::template read_mem<int32_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        int32_t res1 = (int32_t)res_108;
                                        uint32_t res2 = res1 < (int32_t)*(X+rs2)? (uint32_t)*(X+rs2) : ((uint32_t)res1);
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res1;
                                        }
                                        super::template write_mem<uint32_t>(traits::MEM, offs, res2);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOMINUW: {
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
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        uint32_t res_109 = super::template read_mem<uint32_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint32_t res1 = res_109;
                                        uint32_t res2 = res1 > (uint32_t)*(X+rs2)? (uint32_t)*(X+rs2) : res1;
                                        if(rd != 0) {
                                            *(X+rd) = ((uint64_t)(int64_t)(int32_t)res1);
                                        }
                                        super::template write_mem<uint32_t>(traits::MEM, offs, res2);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOMAXUW: {
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
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        uint32_t res_110 = super::template read_mem<uint32_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint32_t res1 = res_110;
                                        uint32_t res2 = res1 < (int32_t)*(X+rs2)? (uint32_t)*(X+rs2) : res1;
                                        if(rd != 0) {
                                            *(X+rd) = ((uint64_t)(int64_t)(int32_t)res1);
                                        }
                                        super::template write_mem<uint32_t>(traits::MEM, offs, res2);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::LRD: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rl = ((bit_sub<25,1>(instr)));
                    uint8_t aq = ((bit_sub<26,1>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "lrd"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            uint64_t offs = *(X+rs1);
                                            int64_t res_111 = super::template read_mem<int64_t>(traits::MEM, offs);
                                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                            *(X+rd) = (uint64_t)((int64_t)res_111);
                                            super::template write_mem<uint8_t>(traits::RES, offs, (uint8_t)- 1);
                                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SCD: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    uint8_t rl = ((bit_sub<25,1>(instr)));
                    uint8_t aq = ((bit_sub<26,1>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "scd"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        uint8_t res_112 = super::template read_mem<uint8_t>(traits::RES, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint64_t res = res_112;
                                        if(res != 0) {
                                            super::template write_mem<uint64_t>(traits::MEM, offs, *(X+rs2));
                                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        }
                                        if(rd != 0) {
                                            *(X+rd) = res? 0 : 1;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOSWAPD: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    uint8_t rl = ((bit_sub<25,1>(instr)));
                    uint8_t aq = ((bit_sub<26,1>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoswapd"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        uint64_t res = *(X+rs2);
                                        if(rd != 0) {
                                            int64_t res_113 = super::template read_mem<int64_t>(traits::MEM, offs);
                                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                            *(X+rd) = (uint64_t)((int64_t)res_113);
                                        }
                                        super::template write_mem<uint64_t>(traits::MEM, offs, (uint64_t)res);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOADDD: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    uint8_t rl = ((bit_sub<25,1>(instr)));
                    uint8_t aq = ((bit_sub<26,1>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoaddd"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        uint64_t res_114 = super::template read_mem<uint64_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint64_t res1 = res_114;
                                        uint64_t res2 = (uint64_t)((uint128_t)(res1 ) + (uint128_t)(*(X+rs2) ));
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res1;
                                        }
                                        super::template write_mem<uint64_t>(traits::MEM, offs, res2);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOXORD: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    uint8_t rl = ((bit_sub<25,1>(instr)));
                    uint8_t aq = ((bit_sub<26,1>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoxord"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        uint64_t res_115 = super::template read_mem<uint64_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint64_t res1 = res_115;
                                        uint64_t res2 = res1 ^ *(X+rs2);
                                        if(rd != 0) {
                                            *(X+rd) = ((uint64_t)(int64_t)res1);
                                        }
                                        super::template write_mem<uint64_t>(traits::MEM, offs, res2);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOANDD: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    uint8_t rl = ((bit_sub<25,1>(instr)));
                    uint8_t aq = ((bit_sub<26,1>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoandd"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        uint64_t res_116 = super::template read_mem<uint64_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint64_t res1 = res_116;
                                        uint64_t res2 = res1 & *(X+rs2);
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res1;
                                        }
                                        super::template write_mem<uint64_t>(traits::MEM, offs, res2);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOORD: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    uint8_t rl = ((bit_sub<25,1>(instr)));
                    uint8_t aq = ((bit_sub<26,1>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amoord"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        uint64_t res_117 = super::template read_mem<uint64_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint64_t res1 = res_117;
                                        uint64_t res2 = res1 | *(X+rs2);
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res1;
                                        }
                                        super::template write_mem<uint64_t>(traits::MEM, offs, res2);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOMIND: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    uint8_t rl = ((bit_sub<25,1>(instr)));
                    uint8_t aq = ((bit_sub<26,1>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amomind"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        int64_t res_118 = super::template read_mem<int64_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        int64_t res1 = (int64_t)res_118;
                                        uint64_t res2 = res1 > (int64_t)*(X+rs2)? (uint64_t)*(X+rs2) : ((uint64_t)res1);
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res1;
                                        }
                                        super::template write_mem<uint64_t>(traits::MEM, offs, res2);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOMAXD: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    uint8_t rl = ((bit_sub<25,1>(instr)));
                    uint8_t aq = ((bit_sub<26,1>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amomaxd"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        int64_t res_119 = super::template read_mem<int64_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        int64_t res1 = (int64_t)res_119;
                                        uint64_t res2 = res1 < (int64_t)*(X+rs2)? (uint64_t)*(X+rs2) : ((uint64_t)res1);
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res1;
                                        }
                                        super::template write_mem<uint64_t>(traits::MEM, offs, res2);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOMINUD: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    uint8_t rl = ((bit_sub<25,1>(instr)));
                    uint8_t aq = ((bit_sub<26,1>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amominud"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        uint64_t res_120 = super::template read_mem<uint64_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint64_t res1 = res_120;
                                        uint64_t res2 = res1 > *(X+rs2)? *(X+rs2) : res1;
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res1;
                                        }
                                        super::template write_mem<uint64_t>(traits::MEM, offs, res2);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::AMOMAXUD: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    uint8_t rl = ((bit_sub<25,1>(instr)));
                    uint8_t aq = ((bit_sub<26,1>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2} (aqu = {aq},rel = {rl})", fmt::arg("mnemonic", "amomaxud"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("aq", aq), fmt::arg("rl", rl));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = *(X+rs1);
                                        uint64_t res_121 = super::template read_mem<uint64_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint64_t res1 = res_121;
                                        uint64_t res2 = res1 < *(X+rs2)? (uint64_t)*(X+rs2) : res1;
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res1;
                                        }
                                        super::template write_mem<uint64_t>(traits::MEM, offs, res2);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__ADDI4SPN: {
                    uint8_t rd = ((bit_sub<2,3>(instr)));
                    uint16_t imm = ((bit_sub<5,1>(instr) << 3) | (bit_sub<6,1>(instr) << 2) | (bit_sub<7,4>(instr) << 6) | (bit_sub<11,2>(instr) << 4));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c.addi4spn"),
                            fmt::arg("rd", name(8+rd)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    if(imm) {
                                        *(X+rd + 8) = (uint64_t)((uint128_t)(*(X+2) ) + (uint128_t)(imm ));
                                    }
                                    else {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__LW: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        uint64_t offs = (uint64_t)((uint128_t)(*(X+rs1 + 8) ) + (uint128_t)(uimm ));
                        int32_t res_122 = super::template read_mem<int32_t>(traits::MEM, offs);
                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                        *(X+rd + 8) = (uint64_t)(int32_t)res_122;
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__LD: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        uint64_t offs = (uint64_t)((uint128_t)(*(X+rs1 + 8) ) + (uint128_t)(uimm ));
                        uint64_t res_123 = super::template read_mem<uint64_t>(traits::MEM, offs);
                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                        *(X+rd + 8) = (uint64_t)res_123;
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__SW: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        uint64_t offs = (uint64_t)((uint128_t)(*(X+rs1 + 8) ) + (uint128_t)(uimm ));
                        super::template write_mem<uint32_t>(traits::MEM, offs, (uint32_t)*(X+rs2 + 8));
                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__SD: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        uint64_t offs = (uint64_t)((uint128_t)(*(X+rs1 + 8) ) + (uint128_t)(uimm ));
                        super::template write_mem<uint64_t>(traits::MEM, offs, *(X+rs2 + 8));
                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__ADDI: {
                    uint8_t imm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
                    uint8_t rs1 = ((bit_sub<7,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.addi"),
                            fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rs1 != 0) {
                                            *(X+rs1) = (uint64_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int8_t)sext<6>(imm) ));
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__NOP: {
                    uint8_t nzimm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        //No disass specified, using instruction name
                        std::string mnemonic = "c.nop";
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__ADDIW: {
                    uint8_t imm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
                    uint8_t rs1 = ((bit_sub<7,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.addiw"),
                            fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS || rs1 == 0) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rs1 != 0) {
                                            int32_t res = (int32_t)((int64_t)((int32_t)*(X+rs1) ) + (int64_t)((int8_t)sext<6>(imm) ));
                                            *(X+rs1) = (uint64_t)res;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__LI: {
                    uint8_t imm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c.li"),
                            fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)((int8_t)sext<6>(imm));
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__LUI: {
                    uint32_t imm = ((bit_sub<2,5>(instr) << 12) | (bit_sub<12,1>(instr) << 17));
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "c.lui"),
                            fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        if(imm == 0 || rd >= traits::RFS) {
                            raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                        }
                        if(rd != 0) {
                            *(X+rd) = (uint64_t)((int32_t)sext<18>(imm));
                        }
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__ADDI16SP: {
                    uint16_t nzimm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 7) | (bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 4) | (bit_sub<12,1>(instr) << 9));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {nzimm:#05x}", fmt::arg("mnemonic", "c.addi16sp"),
                            fmt::arg("nzimm", nzimm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    if(nzimm) {
                                        *(X+2) = (uint64_t)((uint128_t)(*(X+2) ) + (uint128_t)((int16_t)sext<10>(nzimm) ));
                                    }
                                    else {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::__reserved_clui: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        //No disass specified, using instruction name
                        std::string mnemonic = ".reserved_clui";
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__SRLI: {
                    uint8_t nzuimm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
                    uint8_t rs1 = ((bit_sub<7,3>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs1}, {nzuimm}", fmt::arg("mnemonic", "c.srli"),
                            fmt::arg("rs1", name(8+rs1)), fmt::arg("nzuimm", nzuimm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        *(X+rs1 + 8) = *(X+rs1 + 8) >> nzuimm;
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__SRAI: {
                    uint8_t shamt = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
                    uint8_t rs1 = ((bit_sub<7,3>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs1}, {shamt}", fmt::arg("mnemonic", "c.srai"),
                            fmt::arg("rs1", name(8+rs1)), fmt::arg("shamt", shamt));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        *(X+rs1 + 8) = (uint64_t)(((int64_t)*(X+rs1 + 8)) >> shamt);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__ANDI: {
                    uint8_t imm = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
                    uint8_t rs1 = ((bit_sub<7,3>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.andi"),
                            fmt::arg("rs1", name(8+rs1)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        *(X+rs1 + 8) = (uint64_t)(*(X+rs1 + 8) & (uint64_t)((int8_t)sext<6>(imm) ));
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__SUB: {
                    uint8_t rs2 = ((bit_sub<2,3>(instr)));
                    uint8_t rd = ((bit_sub<7,3>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.sub"),
                            fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        *(X+rd + 8) = (uint64_t)((uint128_t)(*(X+rd + 8) ) - (uint128_t)(*(X+rs2 + 8) ));
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__XOR: {
                    uint8_t rs2 = ((bit_sub<2,3>(instr)));
                    uint8_t rd = ((bit_sub<7,3>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.xor"),
                            fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        *(X+rd + 8) = *(X+rd + 8) ^ *(X+rs2 + 8);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__OR: {
                    uint8_t rs2 = ((bit_sub<2,3>(instr)));
                    uint8_t rd = ((bit_sub<7,3>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.or"),
                            fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        *(X+rd + 8) = *(X+rd + 8) | *(X+rs2 + 8);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__AND: {
                    uint8_t rs2 = ((bit_sub<2,3>(instr)));
                    uint8_t rd = ((bit_sub<7,3>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.and"),
                            fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        *(X+rd + 8) = *(X+rd + 8) & *(X+rs2 + 8);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__SUBW: {
                    uint8_t rs2 = ((bit_sub<2,3>(instr)));
                    uint8_t rd = ((bit_sub<7,3>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rd}, {rs2}", fmt::arg("mnemonic", "c.subw"),
                            fmt::arg("rd", name(8+rd)), fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        int32_t res = (int32_t)((int64_t)((int32_t)*(X+rd + 8) ) - (int64_t)((int32_t)*(X+rs2 + 8) ));
                        *(X+rd + 8) = (uint64_t)res;
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__ADDW: {
                    uint8_t rs2 = ((bit_sub<2,3>(instr)));
                    uint8_t rd = ((bit_sub<7,3>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rd}, {rs2}", fmt::arg("mnemonic", "c.addw"),
                            fmt::arg("rd", name(8+rd)), fmt::arg("rd", name(8+rd)), fmt::arg("rs2", name(8+rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        int32_t res = (int32_t)((int64_t)((int32_t)*(X+rd + 8) ) + (int64_t)((int32_t)*(X+rs2 + 8) ));
                        *(X+rd + 8) = (uint64_t)res;
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__J: {
                    uint16_t imm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,3>(instr) << 1) | (bit_sub<6,1>(instr) << 7) | (bit_sub<7,1>(instr) << 6) | (bit_sub<8,1>(instr) << 10) | (bit_sub<9,2>(instr) << 8) | (bit_sub<11,1>(instr) << 4) | (bit_sub<12,1>(instr) << 11));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {imm:#05x}", fmt::arg("mnemonic", "c.j"),
                            fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    *NEXT_PC = (uint64_t)((uint128_t)(*PC ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                    this->core.reg.last_branch = 1;
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__BEQZ: {
                    uint16_t imm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 1) | (bit_sub<5,2>(instr) << 6) | (bit_sub<10,2>(instr) << 3) | (bit_sub<12,1>(instr) << 8));
                    uint8_t rs1 = ((bit_sub<7,3>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.beqz"),
                            fmt::arg("rs1", name(8+rs1)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    if(*(X+rs1 + 8) == 0) {
                                        *NEXT_PC = (uint64_t)((uint128_t)(*PC ) + (uint128_t)((int16_t)sext<9>(imm) ));
                                        this->core.reg.last_branch = 1;
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__BNEZ: {
                    uint16_t imm = ((bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 1) | (bit_sub<5,2>(instr) << 6) | (bit_sub<10,2>(instr) << 3) | (bit_sub<12,1>(instr) << 8));
                    uint8_t rs1 = ((bit_sub<7,3>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs1}, {imm:#05x}", fmt::arg("mnemonic", "c.bnez"),
                            fmt::arg("rs1", name(8+rs1)), fmt::arg("imm", imm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    if(*(X+rs1 + 8) != 0) {
                                        *NEXT_PC = (uint64_t)((uint128_t)(*PC ) + (uint128_t)((int16_t)sext<9>(imm) ));
                                        this->core.reg.last_branch = 1;
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__SLLI: {
                    uint8_t shamt = ((bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5));
                    uint8_t rs1 = ((bit_sub<7,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs1}, {shamt}", fmt::arg("mnemonic", "c.slli"),
                            fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rs1 != 0) {
                                            *(X+rs1) = *(X+rs1) << shamt;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__LWSP: {
                    uint8_t uimm = ((bit_sub<2,2>(instr) << 6) | (bit_sub<4,3>(instr) << 2) | (bit_sub<12,1>(instr) << 5));
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, sp, {uimm:#05x}", fmt::arg("mnemonic", "c.lwsp"),
                            fmt::arg("rd", name(rd)), fmt::arg("uimm", uimm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        if(rd >= traits::RFS || rd == 0) {
                            raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                        }
                        else {
                            uint64_t offs = (uint64_t)((uint128_t)(*(X+2) ) + (uint128_t)(uimm ));
                            int32_t res_124 = super::template read_mem<int32_t>(traits::MEM, offs);
                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                            *(X+rd) = (uint64_t)(int32_t)res_124;
                        }
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__LDSP: {
                    uint16_t uimm = ((bit_sub<2,3>(instr) << 6) | (bit_sub<5,2>(instr) << 3) | (bit_sub<12,1>(instr) << 5));
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {uimm}(sp)", fmt::arg("mnemonic", "c.ldsp"),
                            fmt::arg("rd", name(rd)), fmt::arg("uimm", uimm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rd == 0) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = (uint64_t)((uint128_t)(*(X+2) ) + (uint128_t)(uimm ));
                                        uint64_t res_125 = super::template read_mem<uint64_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint64_t res = (uint64_t)res_125;
                                        *(X+rd) = res;
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__MV: {
                    uint8_t rs2 = ((bit_sub<2,5>(instr)));
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.mv"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs2", name(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = *(X+rs2);
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__JR: {
                    uint8_t rs1 = ((bit_sub<7,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs1}", fmt::arg("mnemonic", "c.jr"),
                            fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    if(rs1 && rs1 < traits::RFS) {
                                        uint64_t addr_mask = (uint64_t)- 2;
                                        *NEXT_PC = *(X+(uint32_t)(rs1 ) % traits::RFS) & addr_mask;
                                        this->core.reg.last_branch = 1;
                                    }
                                    else {
                                        raise(0, 2);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::__reserved_cmv: {
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        //No disass specified, using instruction name
                        std::string mnemonic = ".reserved_cmv";
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    raise(0, 2);
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__ADD: {
                    uint8_t rs2 = ((bit_sub<2,5>(instr)));
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs2}", fmt::arg("mnemonic", "c.add"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs2", name(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)((uint128_t)(*(X+rd) ) + (uint128_t)(*(X+rs2) ));
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__JALR: {
                    uint8_t rs1 = ((bit_sub<7,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs1}", fmt::arg("mnemonic", "c.jalr"),
                            fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t addr_mask = (uint64_t)- 2;
                                        uint64_t new_pc = *(X+rs1);
                                        *(X+1) = (uint64_t)((uint128_t)(*PC ) + (uint128_t)(2 ));
                                        *NEXT_PC = new_pc & addr_mask;
                                        this->core.reg.last_branch = 1;
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__EBREAK: {
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        //No disass specified, using instruction name
                        std::string mnemonic = "c.ebreak";
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    raise(0, 3);
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__SWSP: {
                    uint8_t rs2 = ((bit_sub<2,5>(instr)));
                    uint8_t uimm = ((bit_sub<7,2>(instr) << 6) | (bit_sub<9,4>(instr) << 2));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs2}, {uimm:#05x}(sp)", fmt::arg("mnemonic", "c.swsp"),
                            fmt::arg("rs2", name(rs2)), fmt::arg("uimm", uimm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = (uint64_t)((uint128_t)(*(X+2) ) + (uint128_t)(uimm ));
                                        super::template write_mem<uint32_t>(traits::MEM, offs, (uint32_t)*(X+rs2));
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__SDSP: {
                    uint8_t rs2 = ((bit_sub<2,5>(instr)));
                    uint16_t uimm = ((bit_sub<7,3>(instr) << 6) | (bit_sub<10,3>(instr) << 3));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs2}, {uimm}(sp)", fmt::arg("mnemonic", "c.sdsp"),
                            fmt::arg("rs2", name(rs2)), fmt::arg("uimm", uimm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = (uint64_t)((uint128_t)(*(X+2) ) + (uint128_t)(uimm ));
                                        super::template write_mem<uint64_t>(traits::MEM, offs, (uint64_t)*(X+rs2));
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::DII: {
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        //No disass specified, using instruction name
                        std::string mnemonic = "dii";
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                                    raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FLW: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "flw"),
                            fmt::arg("rd", fname(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = (uint32_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                        uint32_t res_126 = super::template read_mem<uint32_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint32_t res = (uint32_t)res_126;
                                        *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)res;
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FSW: {
                    uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "fsw"),
                            fmt::arg("rs2", fname(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = (uint32_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                        super::template write_mem<uint32_t>(traits::MEM, offs, (uint32_t)*(F+rs2));
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FMADD__S: {
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
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        uint32_t res = fmadd_s(unbox_s(*(F+rs1)), unbox_s(*(F+rs2)), unbox_s(*(F+rs3)), 0, get_rm((uint8_t)rm));
                        *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)res;
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FMSUB__S: {
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
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        uint32_t res = fmadd_s(unbox_s(*(F+rs1)), unbox_s(*(F+rs2)), unbox_s(*(F+rs3)), 1, get_rm((uint8_t)rm));
                        *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)res;
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FNMADD__S: {
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
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        uint32_t frs1 = unbox_s(*(F+rs1));
                        uint32_t frs2 = unbox_s(*(F+rs2));
                        uint32_t frs3 = unbox_s(*(F+rs3));
                        uint32_t res = fmadd_s(frs1, frs2, frs3, 2, get_rm((uint8_t)rm));
                        *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)res;
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FNMSUB__S: {
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
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        uint32_t frs1 = unbox_s(*(F+rs1));
                        uint32_t frs2 = unbox_s(*(F+rs2));
                        uint32_t frs3 = unbox_s(*(F+rs3));
                        uint32_t res = fmadd_s(frs1, frs2, frs3, 3, get_rm((uint8_t)rm));
                        *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)res;
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FADD__S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fadd.s"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        uint32_t frs1 = unbox_s(*(F+rs1));
                        uint32_t frs2 = unbox_s(*(F+rs2));
                        uint32_t res = fadd_s(frs1, frs2, get_rm((uint8_t)rm));
                        *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)res;
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FSUB__S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsub.s"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        uint32_t frs1 = unbox_s(*(F+rs1));
                        uint32_t frs2 = unbox_s(*(F+rs2));
                        uint32_t res = fsub_s(frs1, frs2, get_rm((uint8_t)rm));
                        *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)res;
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FMUL__S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fmul.s"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        uint32_t frs1 = unbox_s(*(F+rs1));
                        uint32_t frs2 = unbox_s(*(F+rs2));
                        uint32_t res = fmul_s(frs1, frs2, get_rm((uint8_t)rm));
                        *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)res;
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FDIV__S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fdiv.s"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        uint32_t frs1 = unbox_s(*(F+rs1));
                        uint32_t frs2 = unbox_s(*(F+rs2));
                        uint32_t res = fdiv_s(frs1, frs2, get_rm((uint8_t)rm));
                        *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)res;
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FSQRT__S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fsqrt.s"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        uint32_t frs1 = unbox_s(*(F+rs1));
                        uint32_t res = fsqrt_s(frs1, get_rm((uint8_t)rm));
                        *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)res;
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FSGNJ__S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnj.s"),
                            fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        uint32_t frs1 = unbox_s(*(F+rs1));
                        uint32_t frs2 = unbox_s(*(F+rs2));
                        uint32_t res = (bit_sub<31, 31-31+1>(frs2)<<31)|bit_sub<0, 30-0+1>(frs1);
                        *(F+rd) = (uint64_t)(- 1 << 32) | (uint64_t)res;
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FSGNJN__S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnjn.s"),
                            fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        uint32_t frs1 = unbox_s(*(F+rs1));
                        uint32_t frs2 = unbox_s(*(F+rs2));
                        uint32_t res = (~ bit_sub<31, 31-31+1>(frs2)<<31)|bit_sub<0, 30-0+1>(frs1);
                        *(F+rd) = (uint64_t)(- 1 << 32) | (uint64_t)res;
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FSGNJX__S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnjx.s"),
                            fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        uint32_t frs1 = unbox_s(*(F+rs1));
                        uint32_t frs2 = unbox_s(*(F+rs2));
                        uint32_t res = frs1 ^ (frs2 & (uint32_t)2147483648);
                        *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)res;
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FMIN__S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fmin.s"),
                            fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        uint32_t frs1 = unbox_s(*(F+rs1));
                        uint32_t frs2 = unbox_s(*(F+rs2));
                        uint32_t res = fsel_s(frs1, frs2, 0);
                        *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)res;
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FMAX__S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fmax.s"),
                            fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        uint32_t frs1 = unbox_s(*(F+rs1));
                        uint32_t frs2 = unbox_s(*(F+rs2));
                        uint32_t res = fsel_s(frs1, frs2, 1);
                        *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)res;
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT__W__S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt.w.s"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t frs1 = unbox_s(*(F+rs1));
                                        uint32_t res = fcvt_s(frs1, 0, get_rm((uint8_t)rm));
                                        if((rd) != 0) {
                                            *(X+rd) = res;
                                        }
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT__WU__S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt.wu.s"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t frs1 = unbox_s(*(F+rs1));
                                        uint32_t res = fcvt_s(frs1, 1, get_rm((uint8_t)rm));
                                        if((rd) != 0) {
                                            *(X+rd) = res;
                                        }
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FEQ__S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "feq.s"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t frs1 = unbox_s(*(F+rs1));
                                        uint32_t frs2 = unbox_s(*(F+rs2));
                                        uint32_t res = fcmp_s(frs1, frs2, 0);
                                        if((rd) != 0) {
                                            *(X+rd) = res;
                                        }
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FLT__S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "flt.s"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t frs1 = unbox_s(*(F+rs1));
                                        uint32_t frs2 = unbox_s(*(F+rs2));
                                        uint32_t res = fcmp_s(frs1, frs2, 2);
                                        if((rd) != 0) {
                                            *(X+rd) = res;
                                        }
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FLE__S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fle.s"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t frs1 = unbox_s(*(F+rs1));
                                        uint32_t frs2 = unbox_s(*(F+rs2));
                                        uint32_t res = fcmp_s(frs1, frs2, 1);
                                        if((rd) != 0) {
                                            *(X+rd) = res;
                                        }
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCLASS__S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "fclass.s"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t res = fclass_s(unbox_s(*(F+rs1)));
                                        if((rd) != 0) {
                                            *(X+rd) = res;
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT__S__W: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt.s.w"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t res = fcvt_s((uint32_t)*(X+rs1), 2, get_rm((uint8_t)rm));
                                        *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)res;
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT__S__WU: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt.s.wu"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t res = fcvt_s((uint32_t)*(X+rs1), 3, get_rm((uint8_t)rm));
                                        *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)res;
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FMV__X__W: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "fmv.x.w"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if((rd) != 0) {
                                            *(X+rd) = (uint64_t)*(F+rs1);
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FMV__W__X: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "fmv.w.x"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)*(X+rs1);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT_L_S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_l_s"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t res = fcvt_32_64(unbox_s(*(F+rs1)), 0, get_rm((uint8_t)rm));
                                        if((rd) != 0) {
                                            *(X+rd) = (uint64_t)res;
                                        }
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT_LU_S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_lu_s"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t res = fcvt_32_64(unbox_s(*(F+rs1)), 1, get_rm((uint8_t)rm));
                                        if((rd) != 0) {
                                            *(X+rd) = (uint64_t)res;
                                        }
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT_S_L: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_s_l"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t res = fcvt_64_32(*(X+rs1), 2, get_rm((uint8_t)rm));
                                        if(traits::FLEN == 32) {
                                            *(F+rd) = res;
                                        }
                                        else {
                                            *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)res;
                                        }
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT_S_LU: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_s_lu"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t res = fcvt_64_32(*(X+rs1), 3, get_rm((uint8_t)rm));
                                        if(traits::FLEN == 32) {
                                            *(F+rd) = res;
                                        }
                                        else {
                                            *(F+rd) = (uint64_t)((int64_t)- 1 << 32) | (uint64_t)res;
                                        }
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FLD: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint16_t imm = ((bit_sub<20,12>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "fld"),
                            fmt::arg("rd", fname(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = (uint64_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                        uint64_t res_127 = super::template read_mem<uint64_t>(traits::MEM, offs);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        *(F+rd) = res_127;
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FSD: {
                    uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "fsd"),
                            fmt::arg("rs2", fname(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t offs = (uint64_t)((uint128_t)(*(X+rs1) ) + (uint128_t)((int16_t)sext<12>(imm) ));
                                        super::template write_mem<uint64_t>(traits::MEM, offs, *(F+rs2));
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FMADD_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    uint8_t rs3 = ((bit_sub<27,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}, {rs3}", fmt::arg("mnemonic", "fmadd_d"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        *(F+rd) = fmadd_d(*(F+rs1), *(F+rs2), *(F+rs3), 0, get_rm(rm));
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FMSUB_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    uint8_t rs3 = ((bit_sub<27,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}, {rs3}", fmt::arg("mnemonic", "fmsub_d"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        *(F+rd) = fmadd_d(*(F+rs1), *(F+rs2), *(F+rs3), 1, get_rm(rm));
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FNMADD_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    uint8_t rs3 = ((bit_sub<27,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}, {rs3}", fmt::arg("mnemonic", "fnmadd_d"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        *(F+rd) = fmadd_d(*(F+rs1), *(F+rs2), *(F+rs3), 2, get_rm(rm));
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FNMSUB_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    uint8_t rs3 = ((bit_sub<27,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}, {rs3}", fmt::arg("mnemonic", "fnmsub_d"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)), fmt::arg("rs3", fname(rs3)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        *(F+rd) = fmadd_d(*(F+rs1), *(F+rs2), *(F+rs3), 3, get_rm(rm));
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FADD_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fadd_d"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        *(F+rd) = fadd_d(*(F+rs1), *(F+rs2), get_rm(rm));
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FSUB_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsub_d"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        *(F+rd) = fsub_d(*(F+rs1), *(F+rs2), get_rm(rm));
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FMUL_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fmul_d"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        *(F+rd) = fmul_d(*(F+rs1), *(F+rs2), get_rm(rm));
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FDIV_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fdiv_d"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        *(F+rd) = fdiv_d(*(F+rs1), *(F+rs2), get_rm(rm));
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FSQRT_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fsqrt_d"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        *(F+rd) = fsqrt_d(*(F+rs1), get_rm(rm));
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FSGNJ_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnj_d"),
                            fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        *(F+rd) = (bit_sub<63, 63-63+1>(*(F+rs2))<<63)|bit_sub<0, 62-0+1>(*(F+rs1));
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FSGNJN_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnjn_d"),
                            fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        *(F+rd) = (~ bit_sub<63, 63-63+1>(*(F+rs2))<<63)|bit_sub<0, 62-0+1>(*(F+rs1));
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FSGNJX_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fsgnjx_d"),
                            fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        *(F+rd) = *(F+rs1) ^ (*(F+rs2) & ((uint64_t)1 << 63));
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FMIN_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fmin_d"),
                            fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        *(F+rd) = fsel_d(*(F+rs1), *(F+rs2), 0);
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FMAX_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fmax_d"),
                            fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        *(F+rd) = fsel_d(*(F+rs1), *(F+rs2), 1);
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT_S_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_s_d"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        uint32_t res = fconv_d2f(*(F+rs1), get_rm(rm));
                        *(F+rd) = (uint64_t)((int128_t)(((int64_t)- 1 << 32) ) + (int128_t)(res ));
                        uint32_t flags = fget_flags();
                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT_D_S: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_d_s"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", fname(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                        *(F+rd) = fconv_f2d((uint32_t)*(F+rs1), get_rm(rm));
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FEQ_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "feq_d"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t res = fcmp_d(*(F+rs1), *(F+rs2), 0);
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res;
                                        }
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FLT_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "flt_d"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t res = fcmp_d(*(F+rs1), *(F+rs2), 2);
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res;
                                        }
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FLE_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    uint8_t rs2 = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "fle_d"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)), fmt::arg("rs2", fname(rs2)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t res = fcmp_d(*(F+rs1), *(F+rs2), 1);
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res;
                                        }
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCLASS_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "fclass_d"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)fclass_d(*(F+rs1));
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT_W_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_w_d"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t res = fcvt_64_32(*(F+rs1), 0, get_rm(rm));
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res;
                                        }
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT_WU_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_wu_d"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t res = fcvt_64_32(*(F+rs1), 1, get_rm(rm));
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res;
                                        }
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT_D_W: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_d_w"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        *(F+rd) = fcvt_32_64((uint32_t)*(X+rs1), 2, get_rm(rm));
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT_D_WU: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_d_wu"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        *(F+rd) = fcvt_32_64((uint32_t)*(X+rs1), 3, get_rm(rm));
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT_L_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_l_d"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t res = (uint64_t)fcvt_d(*(F+rs1), 0, get_rm(rm));
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res;
                                        }
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT_LU_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_lu_d"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint64_t res = (uint64_t)fcvt_d(*(F+rs1), 1, get_rm(rm));
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)res;
                                        }
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT_D_L: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_d_l"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        *(F+rd) = fcvt_d(*(X+rs1), 2, get_rm(rm));
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FCVT_D_LU: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rm = ((bit_sub<12,3>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rm}, {rd}, {rs1}", fmt::arg("mnemonic", "fcvt_d_lu"),
                            fmt::arg("rm", name(rm)), fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]); 
                    auto* FCSR = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::FCSR]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        *(F+rd) = fcvt_d(*(X+rs1), 3, get_rm(rm));
                                        uint32_t flags = fget_flags();
                                        *FCSR = (*FCSR & ~ traits::FFLAG_MASK) | (flags & traits::FFLAG_MASK);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FMV_X_D: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "fmv_x_d"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", fname(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (uint64_t)*(F+rs1);
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::FMV_D_X: {
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    uint8_t rs1 = ((bit_sub<15,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}", fmt::arg("mnemonic", "fmv_d_x"),
                            fmt::arg("rd", fname(rd)), fmt::arg("rs1", name(rs1)));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);// calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        *(F+rd) = *(X+rs1);
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__FLD: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        uint64_t offs = (uint64_t)((uint128_t)(*(X+rs1 + 8) ) + (uint128_t)(uimm ));
                        uint64_t res_128 = super::template read_mem<uint64_t>(traits::MEM, offs);
                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                        uint64_t res = (uint64_t)res_128;
                        if(traits::FLEN == 64) {
                            *(F+rd + 8) = res;
                        }
                        else {
                            *(F+rd + 8) = (uint64_t)(((uint8_t)(- 1 << 64)) ) | res;
                        }
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__FSD: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        uint64_t offs = (uint64_t)((uint128_t)(*(X+rs1 + 8) ) + (uint128_t)(uimm ));
                        super::template write_mem<uint64_t>(traits::MEM, offs, (uint64_t)*(F+rs2 + 8));
                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__FLDSP: {
                    uint16_t uimm = ((bit_sub<2,3>(instr) << 6) | (bit_sub<5,2>(instr) << 3) | (bit_sub<12,1>(instr) << 5));
                    uint8_t rd = ((bit_sub<7,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} f {rd}, {uimm}(x2)", fmt::arg("mnemonic", "c.fldsp"),
                            fmt::arg("rd", rd), fmt::arg("uimm", uimm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        uint64_t offs = (uint64_t)((uint128_t)(*(X+2) ) + (uint128_t)(uimm ));
                        uint64_t res_129 = super::template read_mem<uint64_t>(traits::MEM, offs);
                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                        uint64_t res = (uint64_t)res_129;
                        if(traits::FLEN == 64) {
                            *(F+rd) = res;
                        }
                        else {
                            *(F+rd) = (uint64_t)(((uint8_t)(- 1 << 64)) ) | res;
                        }
                    }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::C__FSDSP: {
                    uint8_t rs2 = ((bit_sub<2,5>(instr)));
                    uint16_t uimm = ((bit_sub<7,3>(instr) << 6) | (bit_sub<10,3>(instr) << 3));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} f {rs2}, {uimm}(x2), ", fmt::arg("mnemonic", "c.fsdsp"),
                            fmt::arg("rs2", rs2), fmt::arg("uimm", uimm));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    auto* F = reinterpret_cast<uint64_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::F0]);// calculate next pc value
                    *NEXT_PC = *PC + 2;
                    // execute instruction
                    {
                        uint64_t offs = (uint64_t)((uint128_t)(*(X+2) ) + (uint128_t)(uimm ));
                        super::template write_mem<uint64_t>(traits::MEM, offs, (uint64_t)*(F+rs2));
                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                    }
                    break;
                }// @suppress("No break at end of case")
                default: {
                    *NEXT_PC = *PC + ((instr & 3) == 3 ? 4 : 2);
                    raise(0,  2);
                }
                }
            }catch(memory_access_exception& e){}
            // post execution stuff
            process_spawn_blocks();
            if(this->sync_exec && POST_SYNC) this->do_sync(POST_SYNC, static_cast<unsigned>(inst_id));
            // if(!this->core.reg.trap_state) // update trap state if there is a pending interrupt
            //    this->core.reg.trap_state =  this->core.reg.pending_trap;
            // trap check
            if(trap_state!=0){
                //In case of Instruction address misaligned (cause = 0 and trapid = 0) need the targeted addr (in tval)
                auto mcause = (trap_state>>16) & 0xff; 
                super::core.enter_trap(trap_state, pc.val, mcause ? instr:tval);
            } else {
                icount++;
                instret++;
            }
            *PC = *NEXT_PC;
            this->core.reg.trap_state =  this->core.reg.pending_trap;
        }
        fetch_count++;
        cycle++;
    }
    return pc;
}

} // namespace rv64gc

template <>
std::unique_ptr<vm_if> create<arch::rv64gc>(arch::rv64gc *core, unsigned short port, bool dump) {
    auto ret = new rv64gc::vm_impl<arch::rv64gc>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret, port);
    return std::unique_ptr<vm_if>(ret);
}
} // namespace interp
} // namespace iss

#include <iss/arch/riscv_hart_m_p.h>
#include <iss/arch/riscv_hart_mu_p.h>
#include <iss/factory.h>
namespace iss {
namespace {
volatile std::array<bool, 2> dummy = {
        core_factory::instance().register_creator("rv64gc|m_p|interp", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_m_p<iss::arch::rv64gc>();
		    auto vm = new interp::rv64gc::vm_impl<arch::rv64gc>(*cpu, false);
		    if (port != 0) debugger::server<debugger::gdb_session>::run_server(vm, port);
            if(init_data){
                auto* cb = reinterpret_cast<semihosting_cb_t<arch::traits<arch::rv64gc>::reg_t>*>(init_data);
                cpu->set_semihosting_callback(*cb);
            }
            return {cpu_ptr{cpu}, vm_ptr{vm}};
        }),
        core_factory::instance().register_creator("rv64gc|mu_p|interp", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_mu_p<iss::arch::rv64gc>();
		    auto vm = new interp::rv64gc::vm_impl<arch::rv64gc>(*cpu, false);
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
