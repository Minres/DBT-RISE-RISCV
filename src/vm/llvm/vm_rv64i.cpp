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
// vm_base needs to be included before gdb_session as termios.h (via boost and gdb_server) has a define which clashes with a variable
// name in ConstantRange.h
#include <iss/llvm/vm_base.h>
#include <iss/iss.h>
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/server.h>
#include <iss/instruction_decoder.h>
#include <util/logging.h>

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#include <array>
#include <iss/debugger/riscv_target_adapter.h>

namespace iss {
namespace llvm {
namespace fp_impl {
void add_fp_functions_2_module(::llvm::Module *, unsigned, unsigned);
}

namespace rv64i {
using namespace ::llvm;
using namespace iss::arch;
using namespace iss::debugger;

template <typename ARCH> class vm_impl : public iss::llvm::vm_base<ARCH> {
public:
    using traits = arch::traits<ARCH>;
    using super = typename iss::llvm::vm_base<ARCH>;
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

    inline const char *name(size_t index){return traits::reg_aliases.at(index);}

    template <typename T> inline ConstantInt *size(T type) {
        return ConstantInt::get(getContext(), APInt(32, type->getType()->getScalarSizeInBits()));
    }

    void setup_module(Module* m) override {
        super::setup_module(m);
        iss::llvm::fp_impl::add_fp_functions_2_module(m, traits::FP_REGS_SIZE, traits::XLEN);
    }

    inline Value *gen_choose(Value *cond, Value *trueVal, Value *falseVal, unsigned size) {
        return super::gen_cond_assign(cond, this->gen_ext(trueVal, size), this->gen_ext(falseVal, size));
    }

    std::tuple<continuation_e, BasicBlock *> gen_single_inst_behavior(virt_addr_t &, BasicBlock *) override;

    void gen_leave_behavior(BasicBlock *leave_blk) override;
    void gen_raise_trap(uint16_t trap_id, uint16_t cause);
    void gen_leave_trap(unsigned lvl);
    void gen_wait(unsigned type);
    void set_tval(uint64_t new_tval);
    void set_tval(Value* new_tval);
    void gen_trap_behavior(BasicBlock *) override;
    void gen_instr_prologue();
    void gen_instr_epilogue(BasicBlock *bb);

    inline Value *gen_reg_load(unsigned i, unsigned level = 0) {
        return this->builder.CreateLoad(this->get_typeptr(i), get_reg_ptr(i), false);
    }

    inline void gen_set_pc(virt_addr_t pc, unsigned reg_num) {
        Value *next_pc_v = this->builder.CreateSExtOrTrunc(this->gen_const(traits::XLEN, pc.val),
                                                           this->get_type(traits::XLEN));
        this->builder.CreateStore(next_pc_v, get_reg_ptr(reg_num), true);
    }

    // some compile time constants

    using this_class = vm_impl<ARCH>;
    using compile_func = std::tuple<continuation_e, BasicBlock *> (this_class::*)(virt_addr_t &pc,
                                                                                  code_word_t instr,
                                                                                  BasicBlock *bb);
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
    std::tuple<continuation_e, BasicBlock*> __lui(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint32_t imm = ((bit_sub<12,20>(instr) << 12));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#05x}", fmt::arg("mnemonic", "lui"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("LUI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,0);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_const(64,(uint64_t)((int32_t)imm)),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 0);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 1: AUIPC */
    std::tuple<continuation_e, BasicBlock*> __auipc(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint32_t imm = ((bit_sub<12,20>(instr) << 12));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#08x}", fmt::arg("mnemonic", "auipc"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("AUIPC_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,1);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_const(64,(uint64_t)(PC+(int32_t)imm)),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 1);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 2: JAL */
    std::tuple<continuation_e, BasicBlock*> __jal(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint32_t imm = ((bit_sub<12,8>(instr) << 12) | (bit_sub<20,1>(instr) << 11) | (bit_sub<21,10>(instr) << 1) | (bit_sub<31,1>(instr) << 20));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#0x}", fmt::arg("mnemonic", "jal"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("JAL_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,2);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto new_pc =(uint64_t)(PC+(int32_t)sext<21>(imm));
            if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){ this->set_tval(new_pc);
            this->gen_raise_trap(0, 0);
            }
            else{
                if(rd!=0) {
                    this->builder.CreateStore(
                    this->gen_const(64,(uint64_t)(PC+4)),
                    get_reg_ptr(rd + traits::X0), false);
                }
                auto PC_val_v = new_pc;
                this->builder.CreateStore(this->gen_const(64,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
                this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
            }
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 2);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 3: JALR */
    std::tuple<continuation_e, BasicBlock*> __jalr(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm:#0x}", fmt::arg("mnemonic", "jalr"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("JALR_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,3);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto addr_mask =(uint64_t)- 2;
            auto new_pc =this->gen_ext(
                (this->builder.CreateAnd(
                   (this->builder.CreateAdd(
                      this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 128,false),
                      this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 128,true))
                   ),
                   this->gen_ext(this->gen_const(64,addr_mask), 128,false))
                ),
                64, true);
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            auto bb_else = BasicBlock::Create(this->mod->getContext(), "bb_else", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateURem(
               new_pc,
               this->gen_ext(this->gen_const(32,static_cast<uint32_t>(traits::INSTR_ALIGNMENT)), 64,false))
            ), bb_then, bb_else);
            this->builder.SetInsertPoint(bb_then);
            {
                this->set_tval(new_pc);
                this->gen_raise_trap(0, 0);
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_else);
            {
                if(rd!=0) {
                    this->builder.CreateStore(
                    this->gen_const(64,(uint64_t)(PC+4)),
                    get_reg_ptr(rd + traits::X0), false);
                }
                auto PC_val_v = new_pc;
                this->builder.CreateStore(PC_val_v, get_reg_ptr(traits::NEXT_PC), false);                            
                this->builder.CreateStore(this->gen_const(32, static_cast<int>(UNKNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 3);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 4: BEQ */
    std::tuple<continuation_e, BasicBlock*> __beq(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "beq"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("BEQ_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,4);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_EQ,
               this->gen_reg_load(traits::X0+ rs1),
               this->gen_reg_load(traits::X0+ rs2))
            ), bb_then,  bb_merge);
            this->builder.SetInsertPoint(bb_then);
            {
                auto new_pc =(uint64_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){ this->set_tval(new_pc);
                this->gen_raise_trap(0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    this->builder.CreateStore(this->gen_const(64,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
                    this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 4);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 5: BNE */
    std::tuple<continuation_e, BasicBlock*> __bne(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bne"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("BNE_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,5);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_NE,
               this->gen_reg_load(traits::X0+ rs1),
               this->gen_reg_load(traits::X0+ rs2))
            ), bb_then,  bb_merge);
            this->builder.SetInsertPoint(bb_then);
            {
                auto new_pc =(uint64_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){ this->set_tval(new_pc);
                this->gen_raise_trap(0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    this->builder.CreateStore(this->gen_const(64,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
                    this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 5);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 6: BLT */
    std::tuple<continuation_e, BasicBlock*> __blt(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "blt"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("BLT_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,6);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_SLT,
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs1),
                   64, false),
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs2),
                   64, false))
            ), bb_then,  bb_merge);
            this->builder.SetInsertPoint(bb_then);
            {
                auto new_pc =(uint64_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){ this->set_tval(new_pc);
                this->gen_raise_trap(0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    this->builder.CreateStore(this->gen_const(64,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
                    this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 6);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 7: BGE */
    std::tuple<continuation_e, BasicBlock*> __bge(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bge"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("BGE_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,7);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_SGE,
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs1),
                   64, false),
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs2),
                   64, false))
            ), bb_then,  bb_merge);
            this->builder.SetInsertPoint(bb_then);
            {
                auto new_pc =(uint64_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){ this->set_tval(new_pc);
                this->gen_raise_trap(0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    this->builder.CreateStore(this->gen_const(64,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
                    this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 7);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 8: BLTU */
    std::tuple<continuation_e, BasicBlock*> __bltu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bltu"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("BLTU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,8);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_ULT,
               this->gen_reg_load(traits::X0+ rs1),
               this->gen_reg_load(traits::X0+ rs2))
            ), bb_then,  bb_merge);
            this->builder.SetInsertPoint(bb_then);
            {
                auto new_pc =(uint64_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){ this->set_tval(new_pc);
                this->gen_raise_trap(0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    this->builder.CreateStore(this->gen_const(64,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
                    this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 8);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 9: BGEU */
    std::tuple<continuation_e, BasicBlock*> __bgeu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rs2}, {imm:#0x}", fmt::arg("mnemonic", "bgeu"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("BGEU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,9);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            {
            auto bb_merge = BasicBlock::Create(this->mod->getContext(), "bb_merge", this->func, this->leave_blk);
            auto bb_then = BasicBlock::Create(this->mod->getContext(), "bb_then", this->func, bb_merge);
            this->builder.CreateCondBr(this->gen_bool(this->builder.CreateICmp(ICmpInst::ICMP_UGE,
               this->gen_reg_load(traits::X0+ rs1),
               this->gen_reg_load(traits::X0+ rs2))
            ), bb_then,  bb_merge);
            this->builder.SetInsertPoint(bb_then);
            {
                auto new_pc =(uint64_t)(PC+(int16_t)sext<13>(imm));
                if(new_pc%static_cast<uint32_t>(traits::INSTR_ALIGNMENT)){ this->set_tval(new_pc);
                this->gen_raise_trap(0, 0);
                }
                else{
                    auto PC_val_v = new_pc;
                    this->builder.CreateStore(this->gen_const(64,PC_val_v), get_reg_ptr(traits::NEXT_PC), false);
                    this->builder.CreateStore(this->gen_const(32, static_cast<int>(KNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
                }
            }
            this->builder.CreateBr(bb_merge);
            this->builder.SetInsertPoint(bb_merge);
            }
        }
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(BRANCH,nullptr);
        
        this->gen_sync(POST_SYNC, 9);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 10: LB */
    std::tuple<continuation_e, BasicBlock*> __lb(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lb"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("LB_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,10);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto load_address =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 128,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 128,true))
                ),
                64, true);
            auto res =this->gen_ext(
                this->gen_read_mem(traits::MEM, load_address, 1),
                8, false);
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    res,
                    64, true),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 10);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 11: LH */
    std::tuple<continuation_e, BasicBlock*> __lh(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lh"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("LH_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,11);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto load_address =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 128,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 128,true))
                ),
                64, true);
            auto res =this->gen_ext(
                this->gen_read_mem(traits::MEM, load_address, 2),
                16, false);
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    res,
                    64, true),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 11);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 12: LW */
    std::tuple<continuation_e, BasicBlock*> __lw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lw"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("LW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,12);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto load_address =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 128,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 128,true))
                ),
                64, true);
            auto res =this->gen_ext(
                this->gen_read_mem(traits::MEM, load_address, 4),
                32, false);
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    res,
                    64, true),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 12);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 13: LBU */
    std::tuple<continuation_e, BasicBlock*> __lbu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lbu"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("LBU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,13);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto load_address =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 128,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 128,true))
                ),
                64, true);
            auto res =this->gen_read_mem(traits::MEM, load_address, 1);
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    res,
                    64, false),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 13);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 14: LHU */
    std::tuple<continuation_e, BasicBlock*> __lhu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lhu"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("LHU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,14);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto load_address =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 128,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 128,true))
                ),
                64, true);
            auto res =this->gen_read_mem(traits::MEM, load_address, 2);
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    res,
                    64, false),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 14);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 15: SB */
    std::tuple<continuation_e, BasicBlock*> __sb(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sb"),
                fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SB_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,15);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto store_address =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 128,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 128,true))
                ),
                64, true);
            this->gen_write_mem(traits::MEM,
            store_address,
            this->gen_ext(
                this->gen_reg_load(traits::X0+ rs2),
                8, false));
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 15);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 16: SH */
    std::tuple<continuation_e, BasicBlock*> __sh(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sh"),
                fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SH_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,16);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto store_address =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 128,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 128,true))
                ),
                64, true);
            this->gen_write_mem(traits::MEM,
            store_address,
            this->gen_ext(
                this->gen_reg_load(traits::X0+ rs2),
                16, false));
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 16);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 17: SW */
    std::tuple<continuation_e, BasicBlock*> __sw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sw"),
                fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,17);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto store_address =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 128,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 128,true))
                ),
                64, true);
            this->gen_write_mem(traits::MEM,
            store_address,
            this->gen_ext(
                this->gen_reg_load(traits::X0+ rs2),
                32, false));
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 17);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 18: ADDI */
    std::tuple<continuation_e, BasicBlock*> __addi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "addi"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("ADDI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,18);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    (this->builder.CreateAdd(
                       this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 128,false),
                       this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 128,true))
                    ),
                    64, true),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 18);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 19: SLTI */
    std::tuple<continuation_e, BasicBlock*> __slti(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "slti"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SLTI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,19);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(this->gen_choose((this->builder.CreateICmp(ICmpInst::ICMP_SLT,
                   this->gen_ext(
                       this->gen_reg_load(traits::X0+ rs1), 64,true),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 64,true))
                ),
                this->gen_const(8,1),
                this->gen_const(8,0),
                8), 64),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 19);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 20: SLTIU */
    std::tuple<continuation_e, BasicBlock*> __sltiu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "sltiu"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SLTIU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,20);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(this->gen_choose((this->builder.CreateICmp(ICmpInst::ICMP_ULT,
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_const(64,(uint64_t)((int16_t)sext<12>(imm))))
                ),
                this->gen_const(8,1),
                this->gen_const(8,0),
                8), 64),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 20);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 21: XORI */
    std::tuple<continuation_e, BasicBlock*> __xori(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "xori"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("XORI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,21);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateXor(
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_const(64,(uint64_t)((int16_t)sext<12>(imm))))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 21);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 22: ORI */
    std::tuple<continuation_e, BasicBlock*> __ori(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "ori"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("ORI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,22);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateOr(
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_const(64,(uint64_t)((int16_t)sext<12>(imm))))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 22);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 23: ANDI */
    std::tuple<continuation_e, BasicBlock*> __andi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "andi"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("ANDI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,23);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateAnd(
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_const(64,(uint64_t)((int16_t)sext<12>(imm))))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 23);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 24: SLLI */
    std::tuple<continuation_e, BasicBlock*> __slli(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,6>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "slli"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SLLI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,24);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateShl(
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_ext(this->gen_const(8,shamt), 64,false))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 24);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 25: SRLI */
    std::tuple<continuation_e, BasicBlock*> __srli(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,6>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srli"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SRLI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,25);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateLShr(
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_ext(this->gen_const(8,shamt), 64,false))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 25);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 26: SRAI */
    std::tuple<continuation_e, BasicBlock*> __srai(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,6>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srai"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SRAI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,26);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    (this->builder.CreateAShr(
                       (this->gen_ext(
                           this->gen_reg_load(traits::X0+ rs1),
                           64, false)),
                       this->gen_ext(this->gen_const(8,shamt), 64,false))
                    ),
                    64, true),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 26);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 27: ADD */
    std::tuple<continuation_e, BasicBlock*> __add(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "add"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("ADD_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,27);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    (this->builder.CreateAdd(
                       this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 128,false),
                       this->gen_ext(this->gen_reg_load(traits::X0+ rs2), 128,false))
                    ),
                    64, false),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 27);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 28: SUB */
    std::tuple<continuation_e, BasicBlock*> __sub(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sub"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SUB_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,28);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    (this->builder.CreateSub(
                       this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 128,false),
                       this->gen_ext(this->gen_reg_load(traits::X0+ rs2), 128,false))
                    ),
                    64, true),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 28);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 29: SLL */
    std::tuple<continuation_e, BasicBlock*> __sll(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sll"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SLL_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,29);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateShl(
                   this->gen_reg_load(traits::X0+ rs1),
                   (this->builder.CreateAnd(
                      this->gen_reg_load(traits::X0+ rs2),
                      this->gen_const(64,(static_cast<uint32_t>(traits::XLEN)-1)))
                   ))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 29);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 30: SLT */
    std::tuple<continuation_e, BasicBlock*> __slt(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "slt"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SLT_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,30);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(this->gen_choose(this->builder.CreateICmp(ICmpInst::ICMP_SLT,
                   this->gen_ext(
                       this->gen_reg_load(traits::X0+ rs1), 64,true),
                   this->gen_ext(
                       this->gen_reg_load(traits::X0+ rs2), 64,true))
                ,
                this->gen_const(8,1),
                this->gen_const(8,0),
                8), 64),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 30);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 31: SLTU */
    std::tuple<continuation_e, BasicBlock*> __sltu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sltu"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SLTU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,31);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(this->gen_choose(this->builder.CreateICmp(ICmpInst::ICMP_ULT,
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_reg_load(traits::X0+ rs2))
                ,
                this->gen_const(8,1),
                this->gen_const(8,0),
                8), 64),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 31);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 32: XOR */
    std::tuple<continuation_e, BasicBlock*> __xor(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "xor"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("XOR_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,32);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateXor(
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_reg_load(traits::X0+ rs2))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 32);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 33: SRL */
    std::tuple<continuation_e, BasicBlock*> __srl(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "srl"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SRL_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,33);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateLShr(
                   this->gen_reg_load(traits::X0+ rs1),
                   (this->builder.CreateAnd(
                      this->gen_reg_load(traits::X0+ rs2),
                      this->gen_const(64,(static_cast<uint32_t>(traits::XLEN)-1)))
                   ))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 33);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 34: SRA */
    std::tuple<continuation_e, BasicBlock*> __sra(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sra"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SRA_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,34);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    (this->builder.CreateAShr(
                       this->gen_ext(
                           this->gen_reg_load(traits::X0+ rs1), 64,true),
                       (this->builder.CreateAnd(
                          this->gen_reg_load(traits::X0+ rs2),
                          this->gen_const(64,(static_cast<uint32_t>(traits::XLEN)-1)))
                       ))
                    ),
                    64, true),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 34);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 35: OR */
    std::tuple<continuation_e, BasicBlock*> __or(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "or"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("OR_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,35);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateOr(
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_reg_load(traits::X0+ rs2))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 35);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 36: AND */
    std::tuple<continuation_e, BasicBlock*> __and(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "and"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("AND_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,36);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0) {
                this->builder.CreateStore(
                this->builder.CreateAnd(
                   this->gen_reg_load(traits::X0+ rs1),
                   this->gen_reg_load(traits::X0+ rs2))
                ,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 36);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 37: FENCE */
    std::tuple<continuation_e, BasicBlock*> __fence(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
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
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FENCE_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,37);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->gen_write_mem(traits::FENCE,
        static_cast<uint32_t>(traits::fence),
        this->gen_const(8,(uint8_t)pred<<4|succ));
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 37);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 38: ECALL */
    std::tuple<continuation_e, BasicBlock*> __ecall(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "ecall";
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("ECALL_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,38);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        this->gen_raise_trap(0, 11);
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(TRAP,nullptr);
        
        this->gen_sync(POST_SYNC, 38);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 39: EBREAK */
    std::tuple<continuation_e, BasicBlock*> __ebreak(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "ebreak";
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("EBREAK_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,39);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        this->gen_raise_trap(0, 3);
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(TRAP,nullptr);
        
        this->gen_sync(POST_SYNC, 39);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 40: MRET */
    std::tuple<continuation_e, BasicBlock*> __mret(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "mret";
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("MRET_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,40);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->builder.CreateStore(this->gen_const(32U, static_cast<int>(NO_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
        this->gen_leave_trap(3);
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(TRAP,nullptr);
        
        this->gen_sync(POST_SYNC, 40);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 41: WFI */
    std::tuple<continuation_e, BasicBlock*> __wfi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "wfi";
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("WFI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,41);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        std::vector<Value*> wait_57_args{
            this->gen_ext(this->gen_const(8,1), 32)
        };
        this->builder.CreateCall(this->mod->getFunction("wait"), wait_57_args);
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 41);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 42: LWU */
    std::tuple<continuation_e, BasicBlock*> __lwu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "lwu"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("LWU_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,42);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 128,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 128,true))
                ),
                64, true);
            auto res =this->gen_ext(
                this->gen_read_mem(traits::MEM, offs, 4),
                32, false);
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    res,
                    64, false),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 42);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 43: LD */
    std::tuple<continuation_e, BasicBlock*> __ld(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm}({rs1})", fmt::arg("mnemonic", "ld"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("LD_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,43);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 128,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 128,true))
                ),
                64, true);
            auto res =this->gen_ext(
                this->gen_read_mem(traits::MEM, offs, 8),
                64, false);
            if(rd!=0) {
                this->builder.CreateStore(
                this->gen_ext(
                    res,
                    64, true),
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 43);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 44: SD */
    std::tuple<continuation_e, BasicBlock*> __sd(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint16_t imm = ((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs2}, {imm}({rs1})", fmt::arg("mnemonic", "sd"),
                fmt::arg("rs2", name(rs2)), fmt::arg("imm", imm), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SD_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,44);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rs2>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto offs =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 128,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 128,true))
                ),
                64, true);
            this->gen_write_mem(traits::MEM,
            offs,
            this->gen_ext(
                this->gen_reg_load(traits::X0+ rs2),
                64, false));
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 44);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 45: ADDIW */
    std::tuple<continuation_e, BasicBlock*> __addiw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm}", fmt::arg("mnemonic", "addiw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("ADDIW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,45);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){ auto res =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_reg_load(traits::X0+ rs1), 128,false),
                   this->gen_ext(this->gen_const(16,(int16_t)sext<12>(imm)), 128,true))
                ),
                32, true);
            this->builder.CreateStore(
            this->gen_ext(
                res,
                64, true),
            get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 45);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 46: SLLIW */
    std::tuple<continuation_e, BasicBlock*> __slliw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "slliw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SLLIW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,46);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){ auto sh_val =this->builder.CreateShl(
               (this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs1),
                   32, false)),
               this->gen_ext(this->gen_const(8,shamt), 32,false))
            ;
            this->builder.CreateStore(
            this->gen_ext(
                this->gen_ext(
                    sh_val, 32,true),
                64, true),
            get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 46);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 47: SRLIW */
    std::tuple<continuation_e, BasicBlock*> __srliw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "srliw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SRLIW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,47);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){ auto sh_val =this->builder.CreateLShr(
               (this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs1),
                   32, false)),
               this->gen_ext(this->gen_const(8,shamt), 32,false))
            ;
            this->builder.CreateStore(
            this->gen_ext(
                this->gen_ext(
                    sh_val, 32,true),
                64, true),
            get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 47);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 48: SRAIW */
    std::tuple<continuation_e, BasicBlock*> __sraiw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t shamt = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {shamt}", fmt::arg("mnemonic", "sraiw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("shamt", shamt));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SRAIW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,48);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){ auto sh_val =this->builder.CreateAShr(
               (this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs1),
                   32, false)),
               this->gen_ext(this->gen_const(8,shamt), 32,false))
            ;
            this->builder.CreateStore(
            this->gen_ext(
                sh_val,
                64, true),
            get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 48);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 49: ADDW */
    std::tuple<continuation_e, BasicBlock*> __addw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "addw";
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("ADDW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,49);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){ auto res =this->gen_ext(
                (this->builder.CreateAdd(
                   this->gen_ext(this->gen_ext(
                       this->gen_reg_load(traits::X0+ rs1),
                       32, false), 64,true),
                   this->gen_ext(this->gen_ext(
                       this->gen_reg_load(traits::X0+ rs2),
                       32, false), 64,true))
                ),
                32, true);
            this->builder.CreateStore(
            this->gen_ext(
                res,
                64, true),
            get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 49);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 50: SUBW */
    std::tuple<continuation_e, BasicBlock*> __subw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            //No disass specified, using instruction name
            std::string mnemonic = "subw";
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SUBW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,50);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){ auto res =this->gen_ext(
                (this->builder.CreateSub(
                   this->gen_ext(this->gen_ext(
                       this->gen_reg_load(traits::X0+ rs1),
                       32, false), 64,true),
                   this->gen_ext(this->gen_ext(
                       this->gen_reg_load(traits::X0+ rs2),
                       32, false), 64,true))
                ),
                32, true);
            this->builder.CreateStore(
            this->gen_ext(
                res,
                64, true),
            get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 50);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 51: SLLW */
    std::tuple<continuation_e, BasicBlock*> __sllw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sllw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SLLW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,51);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){ auto count =this->builder.CreateAnd(
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs2),
                   32, false),
               this->gen_ext(this->gen_const(8,31), 32,false))
            ;
            auto sh_val =this->builder.CreateShl(
               (this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs1),
                   32, false)),
               count)
            ;
            this->builder.CreateStore(
            this->gen_ext(
                this->gen_ext(
                    sh_val, 32,true),
                64, true),
            get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 51);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 52: SRLW */
    std::tuple<continuation_e, BasicBlock*> __srlw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "srlw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SRLW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,52);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){ auto count =this->builder.CreateAnd(
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs2),
                   32, false),
               this->gen_ext(this->gen_const(8,31), 32,false))
            ;
            auto sh_val =this->builder.CreateLShr(
               (this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs1),
                   32, false)),
               count)
            ;
            this->builder.CreateStore(
            this->gen_ext(
                this->gen_ext(
                    sh_val, 32,true),
                64, true),
            get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 52);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 53: SRAW */
    std::tuple<continuation_e, BasicBlock*> __sraw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint8_t rs2 = ((bit_sub<20,5>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {rs2}", fmt::arg("mnemonic", "sraw"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("rs2", name(rs2)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("SRAW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,53);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)||rs2>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            if(rd!=0){ auto count =this->builder.CreateAnd(
               this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs2),
                   32, false),
               this->gen_ext(this->gen_const(8,31), 32,false))
            ;
            auto sh_val =this->builder.CreateAShr(
               (this->gen_ext(
                   this->gen_reg_load(traits::X0+ rs1),
                   32, false)),
               count)
            ;
            this->builder.CreateStore(
            this->gen_ext(
                sh_val,
                64, true),
            get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 53);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 54: CSRRW */
    std::tuple<continuation_e, BasicBlock*> __csrrw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {rs1}", fmt::arg("mnemonic", "csrrw"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("CSRRW_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,54);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrs1 =this->gen_reg_load(traits::X0+ rs1);
            if(rd!=0){ auto xrd =this->gen_read_mem(traits::CSR, csr, 8);
            this->gen_write_mem(traits::CSR,
            csr,
            xrs1);
            this->builder.CreateStore(
            xrd,
            get_reg_ptr(rd + traits::X0), false);
            }
            else{
                this->gen_write_mem(traits::CSR,
                csr,
                xrs1);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 54);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 55: CSRRS */
    std::tuple<continuation_e, BasicBlock*> __csrrs(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {rs1}", fmt::arg("mnemonic", "csrrs"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("CSRRS_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,55);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrd =this->gen_read_mem(traits::CSR, csr, 8);
            auto xrs1 =this->gen_reg_load(traits::X0+ rs1);
            if(rs1!=0) {
                this->gen_write_mem(traits::CSR,
                csr,
                this->builder.CreateOr(
                   xrd,
                   xrs1)
                );
            }
            if(rd!=0) {
                this->builder.CreateStore(
                xrd,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 55);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 56: CSRRC */
    std::tuple<continuation_e, BasicBlock*> __csrrc(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {rs1}", fmt::arg("mnemonic", "csrrc"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("rs1", name(rs1)));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("CSRRC_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,56);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)||rs1>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrd =this->gen_read_mem(traits::CSR, csr, 8);
            auto xrs1 =this->gen_reg_load(traits::X0+ rs1);
            if(rs1!=0) {
                this->gen_write_mem(traits::CSR,
                csr,
                this->builder.CreateAnd(
                   xrd,
                   this->builder.CreateNot(xrs1))
                );
            }
            if(rd!=0) {
                this->builder.CreateStore(
                xrd,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 56);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 57: CSRRWI */
    std::tuple<continuation_e, BasicBlock*> __csrrwi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t zimm = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {zimm:#0x}", fmt::arg("mnemonic", "csrrwi"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("zimm", zimm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("CSRRWI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,57);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrd =this->gen_read_mem(traits::CSR, csr, 8);
            this->gen_write_mem(traits::CSR,
            csr,
            this->gen_const(64,(uint64_t)zimm));
            if(rd!=0) {
                this->builder.CreateStore(
                xrd,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 57);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 58: CSRRSI */
    std::tuple<continuation_e, BasicBlock*> __csrrsi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t zimm = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {zimm:#0x}", fmt::arg("mnemonic", "csrrsi"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("zimm", zimm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("CSRRSI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,58);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrd =this->gen_read_mem(traits::CSR, csr, 8);
            if(zimm!=0) {
                this->gen_write_mem(traits::CSR,
                csr,
                this->builder.CreateOr(
                   xrd,
                   this->gen_const(64,(uint64_t)zimm))
                );
            }
            if(rd!=0) {
                this->builder.CreateStore(
                xrd,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 58);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 59: CSRRCI */
    std::tuple<continuation_e, BasicBlock*> __csrrci(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t zimm = ((bit_sub<15,5>(instr)));
        uint16_t csr = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {csr}, {zimm:#0x}", fmt::arg("mnemonic", "csrrci"),
                fmt::arg("rd", name(rd)), fmt::arg("csr", csr), fmt::arg("zimm", zimm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("CSRRCI_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,59);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        if(rd>=static_cast<uint32_t>(traits::RFS)) {
            this->gen_raise_trap(0, static_cast<uint32_t>(traits::RV_CAUSE_ILLEGAL_INSTRUCTION));
        }
        else{
            auto xrd =this->gen_read_mem(traits::CSR, csr, 8);
            if(zimm!=0) {
                this->gen_write_mem(traits::CSR,
                csr,
                this->builder.CreateAnd(
                   xrd,
                   this->gen_const(64,~ ((uint64_t)zimm)))
                );
            }
            if(rd!=0) {
                this->builder.CreateStore(
                xrd,
                get_reg_ptr(rd + traits::X0), false);
            }
        }
        bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk);
        auto returnValue = std::make_tuple(CONT,bb);
        
        this->gen_sync(POST_SYNC, 59);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /* instruction 60: FENCE_I */
    std::tuple<continuation_e, BasicBlock*> __fence_i(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
        uint64_t PC = pc.val;
        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        uint16_t imm = ((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rs1}, {rd}, {imm}", fmt::arg("mnemonic", "fence_i"),
                fmt::arg("rs1", name(rs1)), fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        bb->setName(fmt::format("FENCE_I_0x{:X}",pc.val));
        this->gen_sync(PRE_SYNC,60);
        
        this->gen_set_pc(pc, traits::PC);
        this->set_tval(instr);
        pc=pc+ 4;
        this->gen_set_pc(pc, traits::NEXT_PC);
        
        this->gen_instr_prologue();
        /*generate behavior*/
        this->gen_write_mem(traits::FENCE,
        static_cast<uint32_t>(traits::fencei),
        this->gen_const(16,imm));
        bb = this->leave_blk;
        auto returnValue = std::make_tuple(FLUSH,nullptr);
        
        this->gen_sync(POST_SYNC, 60);
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
    	return returnValue;        
    }
    
    /****************************************************************************
     * end opcode definitions
     ****************************************************************************/
    std::tuple<continuation_e, BasicBlock *> illegal_instruction(virt_addr_t &pc, code_word_t instr, BasicBlock *bb) {
        if(this->disass_enabled){
            auto mnemonic = std::string("illegal_instruction");
            std::vector<Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder.CreateGlobalStringPtr(mnemonic),
            };
            this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
        }
        this->gen_sync(iss::PRE_SYNC, instr_descr.size());
        this->builder.CreateStore(this->builder.CreateLoad(this->get_typeptr(traits::NEXT_PC), get_reg_ptr(traits::NEXT_PC), true),
                                   get_reg_ptr(traits::PC), true);
        this->builder.CreateStore(
            this->builder.CreateAdd(this->builder.CreateLoad(this->get_typeptr(traits::ICOUNT), get_reg_ptr(traits::ICOUNT), true),
                                     this->gen_const(64U, 1)),
            get_reg_ptr(traits::ICOUNT), true);
        pc = pc + ((instr & 3) == 3 ? 4 : 2);
        this->set_tval(instr);
        this->gen_raise_trap(0, 2);     // illegal instruction trap
		this->gen_sync(iss::POST_SYNC, instr_descr.size());
        bb = this->leave_blk;
        this->gen_instr_epilogue(bb);
        this->builder.CreateBr(bb);
        return std::make_tuple(ILLEGAL_INSTR, nullptr);
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
std::tuple<continuation_e, BasicBlock *>
vm_impl<ARCH>::gen_single_inst_behavior(virt_addr_t &pc, BasicBlock *this_block) {
    // we fetch at max 4 byte, alignment is 2
    enum {TRAP_ID=1<<16};
    code_word_t instr = 0;
    // const typename traits::addr_t upper_bits = ~traits::PGMASK;
    phys_addr_t paddr(pc);
    auto *const data = (uint8_t *)&instr;
    auto res = this->core.read(paddr, 4, data);
    if (res != iss::Ok) 
        return std::make_tuple(ILLEGAL_FETCH, nullptr);
    if (instr == 0x0000006f || (instr&0xffff)==0xa001){
        this->builder.CreateBr(this->leave_blk);
        return std::make_tuple(JUMP_TO_SELF, nullptr);
        }
    uint32_t inst_index = instr_decoder.decode_instr(instr);
    compile_func f = nullptr;
    if(inst_index < instr_descr.size())
        f = instr_descr[inst_index].op;
    if (f == nullptr) {
        f = &this_class::illegal_instruction;
    }
    return (this->*f)(pc, instr, this_block);
}

template <typename ARCH>
void vm_impl<ARCH>::gen_leave_behavior(BasicBlock *leave_blk) {
    this->builder.SetInsertPoint(leave_blk);
    this->builder.CreateRet(this->builder.CreateLoad(this->get_typeptr(traits::NEXT_PC),get_reg_ptr(traits::NEXT_PC), false));
}

template <typename ARCH>
void vm_impl<ARCH>::gen_raise_trap(uint16_t trap_id, uint16_t cause) {
    auto *TRAP_val = this->gen_const(32, 0x80 << 24 | (cause << 16) | trap_id);
    this->builder.CreateStore(TRAP_val, get_reg_ptr(traits::TRAP_STATE), true);
    this->builder.CreateBr(this->trap_blk);
}

template <typename ARCH>
void vm_impl<ARCH>::gen_leave_trap(unsigned lvl) {
    std::vector<Value *> args{ this->core_ptr, ConstantInt::get(getContext(), APInt(64, lvl)) };
    this->builder.CreateCall(this->mod->getFunction("leave_trap"), args);
    this->builder.CreateStore(this->gen_const(32U, static_cast<int>(UNKNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);
}

template <typename ARCH>
void vm_impl<ARCH>::gen_wait(unsigned type) {
    std::vector<Value *> args{ this->core_ptr, ConstantInt::get(getContext(), APInt(64, type)) };
    this->builder.CreateCall(this->mod->getFunction("wait"), args);
}

template <typename ARCH>
inline void vm_impl<ARCH>::set_tval(uint64_t tval) {
    auto tmp_tval = this->gen_const(64, tval);
    this->set_tval(tmp_tval);
}
template <typename ARCH>
inline void vm_impl<ARCH>::set_tval(Value* new_tval) {
    this->builder.CreateStore(this->gen_ext(new_tval, 64, false), this->tval);
}
template <typename ARCH> 
void vm_impl<ARCH>::gen_trap_behavior(BasicBlock *trap_blk) {
    this->builder.SetInsertPoint(trap_blk);
    auto *trap_state_val = this->builder.CreateLoad(this->get_typeptr(traits::TRAP_STATE), get_reg_ptr(traits::TRAP_STATE), true);
    auto *cur_pc_val = this->builder.CreateLoad(this->get_typeptr(traits::PC), get_reg_ptr(traits::PC), true);
    std::vector<Value *> args{this->core_ptr,
                                this->adj_to64(trap_state_val),
                                this->adj_to64(cur_pc_val),
                              this->adj_to64(this->builder.CreateLoad(this->get_type(64),this->tval))};
    this->builder.CreateCall(this->mod->getFunction("enter_trap"), args);
    this->builder.CreateStore(this->gen_const(32U, static_cast<int>(UNKNOWN_JUMP)), get_reg_ptr(traits::LAST_BRANCH), false);

    auto *trap_addr_val = this->builder.CreateLoad(this->get_typeptr(traits::NEXT_PC), get_reg_ptr(traits::NEXT_PC), false);
    this->builder.CreateRet(trap_addr_val);
}
template <typename ARCH>
void vm_impl<ARCH>::gen_instr_prologue() {
    auto* trap_val =
        this->builder.CreateLoad(this->get_typeptr(arch::traits<ARCH>::PENDING_TRAP), get_reg_ptr(arch::traits<ARCH>::PENDING_TRAP));
    this->builder.CreateStore(trap_val, get_reg_ptr(arch::traits<ARCH>::TRAP_STATE), false);
}
            

template <typename ARCH>
void vm_impl<ARCH>::gen_instr_epilogue(BasicBlock *bb) {
    auto* target_bb = BasicBlock::Create(this->mod->getContext(), "", this->func, bb);
    auto *v = this->builder.CreateLoad(this->get_typeptr(traits::TRAP_STATE), get_reg_ptr(traits::TRAP_STATE), true);
    this->gen_cond_branch(this->builder.CreateICmp(
                              ICmpInst::ICMP_EQ, v,
                              ConstantInt::get(getContext(), APInt(v->getType()->getIntegerBitWidth(), 0))),
                          target_bb, this->trap_blk, 1);
    this->builder.SetInsertPoint(target_bb);
    // update icount
    auto* icount_val = this->builder.CreateAdd(
        this->builder.CreateLoad(this->get_typeptr(arch::traits<ARCH>::ICOUNT), get_reg_ptr(arch::traits<ARCH>::ICOUNT)), this->gen_const(64U, 1));
    this->builder.CreateStore(icount_val, get_reg_ptr(arch::traits<ARCH>::ICOUNT), false);
    //increment cyclecount
    auto* cycle_val = this->builder.CreateAdd(
        this->builder.CreateLoad(this->get_typeptr(arch::traits<ARCH>::CYCLE), get_reg_ptr(arch::traits<ARCH>::CYCLE)), this->gen_const(64U, 1));
    this->builder.CreateStore(cycle_val, get_reg_ptr(arch::traits<ARCH>::CYCLE), false);
}

} // namespace rv64i

template <>
std::unique_ptr<vm_if> create<arch::rv64i>(arch::rv64i *core, unsigned short port, bool dump) {
    auto ret = new rv64i::vm_impl<arch::rv64i>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret, port);
    return std::unique_ptr<vm_if>(ret);
}
} // namespace llvm
} // namespace iss

#include <iss/arch/riscv_hart_m_p.h>
#include <iss/arch/riscv_hart_mu_p.h>
#include <iss/factory.h>
namespace iss {
namespace {
volatile std::array<bool, 2> dummy = {
        core_factory::instance().register_creator("rv64i|m_p|llvm", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_m_p<iss::arch::rv64i>();
		    auto vm = new llvm::rv64i::vm_impl<arch::rv64i>(*cpu, false);
		    if (port != 0) debugger::server<debugger::gdb_session>::run_server(vm, port);
            if(init_data){
                auto* cb = reinterpret_cast<std::function<void(arch_if*, arch::traits<arch::rv64i>::reg_t*, arch::traits<arch::rv64i>::reg_t*)>*>(init_data);
                cpu->set_semihosting_callback(*cb);
            }
            return {cpu_ptr{cpu}, vm_ptr{vm}};
        }),
        core_factory::instance().register_creator("rv64i|mu_p|llvm", [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr>{
            auto* cpu = new iss::arch::riscv_hart_mu_p<iss::arch::rv64i>();
		    auto vm = new llvm::rv64i::vm_impl<arch::rv64i>(*cpu, false);
		    if (port != 0) debugger::server<debugger::gdb_session>::run_server(vm, port);
            if(init_data){
                auto* cb = reinterpret_cast<std::function<void(arch_if*, arch::traits<arch::rv64i>::reg_t*, arch::traits<arch::rv64i>::reg_t*)>*>(init_data);
                cpu->set_semihosting_callback(*cb);
            }
            return {cpu_ptr{cpu}, vm_ptr{vm}};
        })
};
}
}
// clang-format on
