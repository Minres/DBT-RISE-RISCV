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
#include <iss/arch/rv32i.h>
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/server.h>
#include <iss/iss.h>
#include <iss/interp/vm_base.h>


#include <util/logging.h>
#include <boost/coroutine2/all.hpp>
#include <functional>
#include <exception>
#include <vector>
#include <iss/instruction_decoder.h>


#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#include <array>
#include <iss/debugger/riscv_target_adapter.h>

namespace iss {
namespace interp {
namespace rv32i {

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

    const std::array<instruction_descriptor, 49> instr_descr = {{
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
        {32, 0b00000000000000000001000000010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::SLLI},
        {32, 0b00000000000000000101000000010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::SRLI},
        {32, 0b01000000000000000101000000010011, 0b11111110000000000111000001111111, arch::traits<ARCH>::opcode_e::SRAI},
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
        {32, 0b00000000000000000001000001110011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::CSRRW},
        {32, 0b00000000000000000010000001110011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::CSRRS},
        {32, 0b00000000000000000011000001110011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::CSRRC},
        {32, 0b00000000000000000101000001110011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::CSRRWI},
        {32, 0b00000000000000000110000001110011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::CSRRSI},
        {32, 0b00000000000000000111000001110011, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::CSRRCI},
        {32, 0b00000000000000000001000000001111, 0b00000000000000000111000001111111, arch::traits<ARCH>::opcode_e::FENCE_I},
    }};

    //needs to be declared after instr_descr
    decoder instr_decoder;

    iss::status fetch_ins(virt_addr_t pc, uint8_t * data){
        if (this->core.read({iss::address_type::LOGICAL, pc.access, arch::traits<ARCH>::IMEM, pc.val}, 4, data) != iss::Ok)
                    return iss::Err;
        return iss::Ok;
    }
};

template <typename CODE_WORD> void debug_fn(CODE_WORD insn) {
    volatile CODE_WORD x = insn;
    insn = 2 * x;
}
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
    auto* PC = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::PC]);
    auto* NEXT_PC = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::NEXT_PC]);
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
            *PC = super::core.enter_trap(trap_state, pc.val, instr);
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (uint32_t)((int32_t)imm);
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (uint32_t)((uint64_t)(*PC) + (int64_t)((int32_t)imm));
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t new_pc = (uint32_t)((uint64_t)(*PC) + (int64_t)((int32_t)sext<21>(imm)));
                                        if(new_pc % traits::INSTR_ALIGNMENT) {
                                            set_tval(new_pc);
                                            raise(0, 0);
                                        }
                                        else {
                                            if(rd != 0) {
                                                *(X+rd) = (uint32_t)((uint64_t)(*PC) + (uint64_t)(4));
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t addr_mask = (uint32_t)- 2;
                                        uint32_t new_pc = (uint32_t)(((uint64_t)(*(X+rs1)) + (int64_t)((int16_t)sext<12>(imm))) & (uint64_t)(addr_mask));
                                        if(new_pc % traits::INSTR_ALIGNMENT) {
                                            set_tval(new_pc);
                                            raise(0, 0);
                                        }
                                        else {
                                            if(rd != 0) {
                                                *(X+rd) = (uint32_t)((uint64_t)(*PC) + (uint64_t)(4));
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(*(X+rs1) == *(X+rs2)) {
                                            uint32_t new_pc = (uint32_t)((uint64_t)(*PC) + (int64_t)((int16_t)sext<13>(imm)));
                                            if(new_pc % traits::INSTR_ALIGNMENT) {
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(*(X+rs1) != *(X+rs2)) {
                                            uint32_t new_pc = (uint32_t)((uint64_t)(*PC) + (int64_t)((int16_t)sext<13>(imm)));
                                            if(new_pc % traits::INSTR_ALIGNMENT) {
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if((int32_t)*(X+rs1) < (int32_t)*(X+rs2)) {
                                            uint32_t new_pc = (uint32_t)((uint64_t)(*PC) + (int64_t)((int16_t)sext<13>(imm)));
                                            if(new_pc % traits::INSTR_ALIGNMENT) {
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if((int32_t)*(X+rs1) >= (int32_t)*(X+rs2)) {
                                            uint32_t new_pc = (uint32_t)((uint64_t)(*PC) + (int64_t)((int16_t)sext<13>(imm)));
                                            if(new_pc % traits::INSTR_ALIGNMENT) {
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(*(X+rs1) < *(X+rs2)) {
                                            uint32_t new_pc = (uint32_t)((uint64_t)(*PC) + (int64_t)((int16_t)sext<13>(imm)));
                                            if(new_pc % traits::INSTR_ALIGNMENT) {
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(*(X+rs1) >= *(X+rs2)) {
                                            uint32_t new_pc = (uint32_t)((uint64_t)(*PC) + (int64_t)((int16_t)sext<13>(imm)));
                                            if(new_pc % traits::INSTR_ALIGNMENT) {
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t load_address = (uint32_t)((uint64_t)(*(X+rs1)) + (int64_t)((int16_t)sext<12>(imm)));
                                        int8_t res_1 = super::template read_mem<int8_t>(traits::MEM, load_address);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        int8_t res = (int8_t)res_1;
                                        if(rd != 0) {
                                            *(X+rd) = (uint32_t)res;
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t load_address = (uint32_t)((uint64_t)(*(X+rs1)) + (int64_t)((int16_t)sext<12>(imm)));
                                        int16_t res_2 = super::template read_mem<int16_t>(traits::MEM, load_address);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        int16_t res = (int16_t)res_2;
                                        if(rd != 0) {
                                            *(X+rd) = (uint32_t)res;
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t load_address = (uint32_t)((uint64_t)(*(X+rs1)) + (int64_t)((int16_t)sext<12>(imm)));
                                        int32_t res_3 = super::template read_mem<int32_t>(traits::MEM, load_address);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        int32_t res = (int32_t)res_3;
                                        if(rd != 0) {
                                            *(X+rd) = (uint32_t)res;
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t load_address = (uint32_t)((uint64_t)(*(X+rs1)) + (int64_t)((int16_t)sext<12>(imm)));
                                        uint8_t res_4 = super::template read_mem<uint8_t>(traits::MEM, load_address);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint8_t res = res_4;
                                        if(rd != 0) {
                                            *(X+rd) = (uint32_t)res;
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t load_address = (uint32_t)((uint64_t)(*(X+rs1)) + (int64_t)((int16_t)sext<12>(imm)));
                                        uint16_t res_5 = super::template read_mem<uint16_t>(traits::MEM, load_address);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint16_t res = res_5;
                                        if(rd != 0) {
                                            *(X+rd) = (uint32_t)res;
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t store_address = (uint32_t)((uint64_t)(*(X+rs1)) + (int64_t)((int16_t)sext<12>(imm)));
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t store_address = (uint32_t)((uint64_t)(*(X+rs1)) + (int64_t)((int16_t)sext<12>(imm)));
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rs2 >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t store_address = (uint32_t)((uint64_t)(*(X+rs1)) + (int64_t)((int16_t)sext<12>(imm)));
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (uint32_t)((uint64_t)(*(X+rs1)) + (int64_t)((int16_t)sext<12>(imm)));
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = ((int32_t)*(X+rs1) < (int16_t)sext<12>(imm))? 1 : 0;
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (*(X+rs1) < (uint32_t)((int16_t)sext<12>(imm)))? 1 : 0;
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = *(X+rs1) ^ (uint32_t)((int16_t)sext<12>(imm));
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = *(X+rs1) | (uint32_t)((int16_t)sext<12>(imm));
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = *(X+rs1) & (uint32_t)((int16_t)sext<12>(imm));
                                        }
                                    }
                                }
                    break;
                }// @suppress("No break at end of case")
                case arch::traits<ARCH>::opcode_e::SLLI: {
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
                    // used registers
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
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
                    uint8_t shamt = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srli"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
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
                    uint8_t shamt = ((bit_sub<20,5>(instr)));
                    if(this->disass_enabled){
                        /* generate console output when executing the command */
                        auto mnemonic = fmt::format(
                            "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srai"),
                            fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    // used registers
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = ((uint32_t)((int32_t)*(X+rs1) >> shamt));
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (uint32_t)((uint64_t)(*(X+rs1)) + (uint64_t)(*(X+rs2)));
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (uint32_t)((uint64_t)(*(X+rs1)) - (uint64_t)(*(X+rs2)));
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = *(X+rs1) << ((uint64_t)(*(X+rs2)) & ((uint64_t)(traits::XLEN) - (uint64_t)(1)));
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (int32_t)*(X+rs1) < (int32_t)*(X+rs2)? 1 : 0;
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = *(X+rs1) >> ((uint64_t)(*(X+rs2)) & ((uint64_t)(traits::XLEN) - (uint64_t)(1)));
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS || rs2 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        if(rd != 0) {
                                            *(X+rd) = (uint32_t)((int32_t)*(X+rs1) >> ((uint64_t)(*(X+rs2)) & ((uint64_t)(traits::XLEN) - (uint64_t)(1))));
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
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
                    // used registers
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    super::template write_mem<uint32_t>(traits::FENCE, traits::fence, (uint8_t)pred << 4 | succ);
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
                    // used registers
                    // calculate next pc value
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
                    // used registers
                    // calculate next pc value
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
                    // used registers
                    // calculate next pc value
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
                    // used registers
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    wait(1);
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t xrs1 = *(X+rs1);
                                        if(rd != 0) {
                                            uint32_t res_6 = super::template read_mem<uint32_t>(traits::CSR, csr);
                                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                            uint32_t xrd = res_6;
                                            super::template write_mem<uint32_t>(traits::CSR, csr, xrs1);
                                            if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                            *(X+rd) = xrd;
                                        }
                                        else {
                                            super::template write_mem<uint32_t>(traits::CSR, csr, xrs1);
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t res_7 = super::template read_mem<uint32_t>(traits::CSR, csr);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint32_t xrd = res_7;
                                        uint32_t xrs1 = *(X+rs1);
                                        if(rs1 != 0) {
                                            super::template write_mem<uint32_t>(traits::CSR, csr, xrd | xrs1);
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS || rs1 >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t res_8 = super::template read_mem<uint32_t>(traits::CSR, csr);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint32_t xrd = res_8;
                                        uint32_t xrs1 = *(X+rs1);
                                        if(rs1 != 0) {
                                            super::template write_mem<uint32_t>(traits::CSR, csr, xrd & ~xrs1);
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t res_9 = super::template read_mem<uint32_t>(traits::CSR, csr);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint32_t xrd = res_9;
                                        super::template write_mem<uint32_t>(traits::CSR, csr, (uint32_t)zimm);
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t res_10 = super::template read_mem<uint32_t>(traits::CSR, csr);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint32_t xrd = res_10;
                                        if(zimm != 0) {
                                            super::template write_mem<uint32_t>(traits::CSR, csr, xrd | (uint32_t)zimm);
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
                    auto* X = reinterpret_cast<uint32_t*>(this->regs_base_ptr+arch::traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::X0]);
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    if(rd >= traits::RFS) {
                                        raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
                                    }
                                    else {
                                        uint32_t res_11 = super::template read_mem<uint32_t>(traits::CSR, csr);
                                        if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                        uint32_t xrd = res_11;
                                        if(zimm != 0) {
                                            super::template write_mem<uint32_t>(traits::CSR, csr, xrd & ~((uint32_t)zimm));
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
                    // used registers
                    // calculate next pc value
                    *NEXT_PC = *PC + 4;
                    // execute instruction
                    {
                                    super::template write_mem<uint32_t>(traits::FENCE, traits::fencei, imm);
                                    if(this->core.reg.trap_state>=0x80000000UL) throw memory_access_exception();
                                }
                    break;
                }// @suppress("No break at end of case")
                default: {
                    if(this->core.can_handle_unknown_instruction()) {
                        auto res = this->core.handle_unknown_instruction(pc.val, sizeof(instr), reinterpret_cast<uint8_t*>(&instr));
                        if(std::get<0>(res)) {
                            *NEXT_PC = std::get<1>(res);
                            break;
                        }
                    }
                    if(this->disass_enabled){
                        std::string mnemonic = "Illegal Instruction";
                        this->core.disass_output(pc.val, mnemonic);
                    }
                    *NEXT_PC = *PC + ((instr & 3) == 3 ? 4 : 2);
                    raise(0, traits::RV_CAUSE_ILLEGAL_INSTRUCTION);
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

} // namespace rv32i

template <>
std::unique_ptr<vm_if> create<arch::rv32i>(arch::rv32i *core, unsigned short port, bool dump) {
    auto ret = new rv32i::vm_impl<arch::rv32i>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret, port);
    return std::unique_ptr<vm_if>(ret);
}
} // namespace interp
} // namespace iss

#include <iss/arch/riscv_hart_m_p.h>
#include <iss/arch/riscv_hart_msu_vp.h>
#include <iss/arch/riscv_hart_mu_p.h>
#include <iss/factory.h>


namespace iss {
namespace {

volatile std::array<bool, 2> dummy = {
        core_factory::instance().register_creator("rv32i_m:interp", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_m_p<iss::arch::rv32i>();
		    auto vm = new interp::rv32i::vm_impl<arch::rv32i>(*cpu, false);
		    if (port != 0) debugger::server<debugger::gdb_session>::run_server(vm, port);
            if(init_data){
                auto* cb = reinterpret_cast<semihosting_cb_t<arch::traits<arch::rv32i>::reg_t>*>(init_data);
                cpu->set_semihosting_callback(*cb);
            }
            return {cpu_ptr{cpu}, vm_ptr{vm}};
        }),
        core_factory::instance().register_creator("rv32i_mu:interp", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_mu_p<iss::arch::rv32i>();
		    auto vm = new interp::rv32i::vm_impl<arch::rv32i>(*cpu, false);
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
