/*******************************************************************************
 * Copyright (C) 2017, 2018 MINRES Technologies GmbH
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

#include <iss/arch/rv64i.h>
#include <iss/arch/riscv_hart_msu_vp.h>
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/server.h>
#include <iss/iss.h>
#include <iss/llvm/vm_base.h>
#include <util/logging.h>

#define FMT_HEADER_ONLY
#include <fmt/format.h>

#include <array>
#include <iss/debugger/riscv_target_adapter.h>

namespace iss {
namespace vm {
namespace fp_impl {
void add_fp_functions_2_module(llvm::Module *, unsigned, unsigned);
}
}

namespace rv64i {
using namespace iss::arch;
using namespace llvm;
using namespace iss::debugger;
using namespace iss::vm::llvm;

template <typename ARCH> class vm_impl : public vm_base<ARCH> {
public:
    using super = typename iss::vm::llvm::vm_base<ARCH>;
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

    inline const char *name(size_t index){return traits<ARCH>::reg_aliases.at(index);}

    template <typename T> inline ConstantInt *size(T type) {
        return ConstantInt::get(getContext(), APInt(32, type->getType()->getScalarSizeInBits()));
    }

    void setup_module(Module* m) override {
        super::setup_module(m);
        iss::vm::fp_impl::add_fp_functions_2_module(m, traits<ARCH>::FP_REGS_SIZE, traits<ARCH>::XLEN);
    }

    inline Value *gen_choose(Value *cond, Value *trueVal, Value *falseVal, unsigned size) {
        return super::gen_cond_assign(cond, this->gen_ext(trueVal, size), this->gen_ext(falseVal, size));
    }

    std::tuple<continuation_e, BasicBlock *> gen_single_inst_behavior(virt_addr_t &, unsigned int &, BasicBlock *) override;

    void gen_leave_behavior(BasicBlock *leave_blk) override;

    void gen_raise_trap(uint16_t trap_id, uint16_t cause);

    void gen_leave_trap(unsigned lvl);

    void gen_wait(unsigned type);

    void gen_trap_behavior(BasicBlock *) override;

    void gen_trap_check(BasicBlock *bb);

    inline Value *gen_reg_load(unsigned i, unsigned level = 0) {
        return this->builder.CreateLoad(get_reg_ptr(i), false);
    }

    inline void gen_set_pc(virt_addr_t pc, unsigned reg_num) {
        Value *next_pc_v = this->builder.CreateSExtOrTrunc(this->gen_const(traits<ARCH>::XLEN, pc.val),
                                                           this->get_type(traits<ARCH>::XLEN));
        this->builder.CreateStore(next_pc_v, get_reg_ptr(reg_num), true);
    }

    // some compile time constants
    // enum { MASK16 = 0b1111110001100011, MASK32 = 0b11111111111100000111000001111111 };
    enum { MASK16 = 0b1111111111111111, MASK32 = 0b11111111111100000111000001111111 };
    enum { EXTR_MASK16 = MASK16 >> 2, EXTR_MASK32 = MASK32 >> 2 };
    enum { LUT_SIZE = 1 << util::bit_count(EXTR_MASK32), LUT_SIZE_C = 1 << util::bit_count(EXTR_MASK16) };

    using this_class = vm_impl<ARCH>;
    using compile_func = std::tuple<continuation_e, BasicBlock *> (this_class::*)(virt_addr_t &pc,
                                                                                  code_word_t instr,
                                                                                  BasicBlock *bb);
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

    const std::array<InstructionDesriptor, 64> instr_descr = {{
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
        {32, 0b00000000000000000001000000010011, 0b11111100000000000111000001111111, &this_class::__slli},
        /* instruction SRLI */
        {32, 0b00000000000000000101000000010011, 0b11111100000000000111000001111111, &this_class::__srli},
        /* instruction SRAI */
        {32, 0b01000000000000000101000000010011, 0b11111100000000000111000001111111, &this_class::__srai},
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
    }};
 
    /* instruction definitions */
    /* instruction 0: LUI */
    std::tuple<continuation_e, BasicBlock*> __lui(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("LUI");
    	
    	this->gen_sync(PRE_SYNC, 0);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	int32_t imm = signextend<int32_t,32>((bit_sub<12,20>(instr) << 12));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_const(64U, imm);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 0);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 1: AUIPC */
    std::tuple<continuation_e, BasicBlock*> __auipc(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("AUIPC");
    	
    	this->gen_sync(PRE_SYNC, 1);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	int32_t imm = signextend<int32_t,32>((bit_sub<12,20>(instr) << 12));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            64, true),
    	        this->gen_const(64U, imm));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 1);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 2: JAL */
    std::tuple<continuation_e, BasicBlock*> __jal(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("JAL");
    	
    	this->gen_sync(PRE_SYNC, 2);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	int32_t imm = signextend<int32_t,21>((bit_sub<12,8>(instr) << 12) | (bit_sub<20,1>(instr) << 11) | (bit_sub<21,10>(instr) << 1) | (bit_sub<31,1>(instr) << 20));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(64U, 4));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	Value* PC_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        cur_pc_val,
    	        64, true),
    	    this->gen_const(64U, imm));
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(64U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 2);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 3: JALR */
    std::tuple<continuation_e, BasicBlock*> __jalr(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("JALR");
    	
    	this->gen_sync(PRE_SYNC, 3);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* new_pc_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        64, true),
    	    this->gen_const(64U, imm));
    	Value* align_val = this->builder.CreateAnd(
    	    new_pc_val,
    	    this->gen_const(64U, 0x2));
    	{
    	    BasicBlock* bbnext = BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	    BasicBlock* bb_then = BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	    BasicBlock* bb_else = BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	    // this->builder.SetInsertPoint(bb);
    	    this->gen_cond_branch(this->builder.CreateICmp(
    	        ICmpInst::ICMP_NE,
    	        align_val,
    	        this->gen_const(64U, 0)),
    	        bb_then,
    	        bb_else);
    	    this->builder.SetInsertPoint(bb_then);
    	    {
    	        this->gen_raise_trap(0, 0);
    	    }
    	    this->builder.CreateBr(bbnext);
    	    this->builder.SetInsertPoint(bb_else);
    	    {
    	        if(rd != 0){
    	            Value* Xtmp0_val = this->builder.CreateAdd(
    	                cur_pc_val,
    	                this->gen_const(64U, 4));
    	            this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	        }
    	        Value* PC_val = this->builder.CreateAnd(
    	            new_pc_val,
    	            this->builder.CreateNot(this->gen_const(64U, 0x1)));
    	        this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	        this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()), get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	    }
    	    this->builder.CreateBr(bbnext);
    	    bb=bbnext;
    	}
    	this->builder.SetInsertPoint(bb);
    	this->gen_sync(POST_SYNC, 3);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 4: BEQ */
    std::tuple<continuation_e, BasicBlock*> __beq(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("BEQ");
    	
    	this->gen_sync(PRE_SYNC, 4);
    	
    	int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_EQ,
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            64, true),
    	        this->gen_const(64U, imm)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(64U, 4)),
    	    64);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(64U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 4);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 5: BNE */
    std::tuple<continuation_e, BasicBlock*> __bne(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("BNE");
    	
    	this->gen_sync(PRE_SYNC, 5);
    	
    	int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_NE,
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            64, true),
    	        this->gen_const(64U, imm)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(64U, 4)),
    	    64);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(64U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 5);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 6: BLT */
    std::tuple<continuation_e, BasicBlock*> __blt(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("BLT");
    	
    	this->gen_sync(PRE_SYNC, 6);
    	
    	int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_SLT,
    	        this->gen_ext(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            64, true),
    	        this->gen_ext(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            64, true)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            64, true),
    	        this->gen_const(64U, imm)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(64U, 4)),
    	    64);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(64U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 6);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 7: BGE */
    std::tuple<continuation_e, BasicBlock*> __bge(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("BGE");
    	
    	this->gen_sync(PRE_SYNC, 7);
    	
    	int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_SGE,
    	        this->gen_ext(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            64, true),
    	        this->gen_ext(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            64, true)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            64, true),
    	        this->gen_const(64U, imm)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(64U, 4)),
    	    64);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(64U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 7);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 8: BLTU */
    std::tuple<continuation_e, BasicBlock*> __bltu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("BLTU");
    	
    	this->gen_sync(PRE_SYNC, 8);
    	
    	int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_ULT,
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            64, true),
    	        this->gen_const(64U, imm)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(64U, 4)),
    	    64);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(64U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 8);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 9: BGEU */
    std::tuple<continuation_e, BasicBlock*> __bgeu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("BGEU");
    	
    	this->gen_sync(PRE_SYNC, 9);
    	
    	int16_t imm = signextend<int16_t,13>((bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (bit_sub<31,1>(instr) << 12));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_UGE,
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            64, true),
    	        this->gen_const(64U, imm)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(64U, 4)),
    	    64);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(64U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(POST_SYNC, 9);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 10: LB */
    std::tuple<continuation_e, BasicBlock*> __lb(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("LB");
    	
    	this->gen_sync(PRE_SYNC, 10);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        64, true),
    	    this->gen_const(64U, imm));
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 8/8),
    	        64,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 10);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 11: LH */
    std::tuple<continuation_e, BasicBlock*> __lh(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("LH");
    	
    	this->gen_sync(PRE_SYNC, 11);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        64, true),
    	    this->gen_const(64U, imm));
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 16/8),
    	        64,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 11);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 12: LW */
    std::tuple<continuation_e, BasicBlock*> __lw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("LW");
    	
    	this->gen_sync(PRE_SYNC, 12);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        64, true),
    	    this->gen_const(64U, imm));
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	        64,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 12);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 13: LBU */
    std::tuple<continuation_e, BasicBlock*> __lbu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("LBU");
    	
    	this->gen_sync(PRE_SYNC, 13);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        64, true),
    	    this->gen_const(64U, imm));
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 8/8),
    	        64,
    	        false);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 13);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 14: LHU */
    std::tuple<continuation_e, BasicBlock*> __lhu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("LHU");
    	
    	this->gen_sync(PRE_SYNC, 14);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        64, true),
    	    this->gen_const(64U, imm));
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 16/8),
    	        64,
    	        false);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 14);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 15: SB */
    std::tuple<continuation_e, BasicBlock*> __sb(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SB");
    	
    	this->gen_sync(PRE_SYNC, 15);
    	
    	int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        64, true),
    	    this->gen_const(64U, imm));
    	Value* MEMtmp0_val = this->gen_reg_load(rs2 + traits<ARCH>::X0, 0);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(8)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 15);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 16: SH */
    std::tuple<continuation_e, BasicBlock*> __sh(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SH");
    	
    	this->gen_sync(PRE_SYNC, 16);
    	
    	int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        64, true),
    	    this->gen_const(64U, imm));
    	Value* MEMtmp0_val = this->gen_reg_load(rs2 + traits<ARCH>::X0, 0);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(16)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 16);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 17: SW */
    std::tuple<continuation_e, BasicBlock*> __sw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SW");
    	
    	this->gen_sync(PRE_SYNC, 17);
    	
    	int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        64, true),
    	    this->gen_const(64U, imm));
    	Value* MEMtmp0_val = this->gen_reg_load(rs2 + traits<ARCH>::X0, 0);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 17);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 18: ADDI */
    std::tuple<continuation_e, BasicBlock*> __addi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("ADDI");
    	
    	this->gen_sync(PRE_SYNC, 18);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateAdd(
    	        this->gen_ext(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            64, true),
    	        this->gen_const(64U, imm));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 18);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 19: SLTI */
    std::tuple<continuation_e, BasicBlock*> __slti(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SLTI");
    	
    	this->gen_sync(PRE_SYNC, 19);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_SLT,
    	            this->gen_ext(
    	                this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	                64, true),
    	            this->gen_const(64U, imm)),
    	        this->gen_const(64U, 1),
    	        this->gen_const(64U, 0),
    	        64);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 19);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 20: SLTIU */
    std::tuple<continuation_e, BasicBlock*> __sltiu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SLTIU");
    	
    	this->gen_sync(PRE_SYNC, 20);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	int64_t full_imm_val = imm;
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            this->gen_const(64U, full_imm_val)),
    	        this->gen_const(64U, 1),
    	        this->gen_const(64U, 0),
    	        64);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 20);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 21: XORI */
    std::tuple<continuation_e, BasicBlock*> __xori(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("XORI");
    	
    	this->gen_sync(PRE_SYNC, 21);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateXor(
    	        this->gen_ext(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            64, true),
    	        this->gen_const(64U, imm));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 21);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 22: ORI */
    std::tuple<continuation_e, BasicBlock*> __ori(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("ORI");
    	
    	this->gen_sync(PRE_SYNC, 22);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateOr(
    	        this->gen_ext(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            64, true),
    	        this->gen_const(64U, imm));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 22);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 23: ANDI */
    std::tuple<continuation_e, BasicBlock*> __andi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("ANDI");
    	
    	this->gen_sync(PRE_SYNC, 23);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateAnd(
    	        this->gen_ext(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            64, true),
    	        this->gen_const(64U, imm));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 23);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 24: SLLI */
    std::tuple<continuation_e, BasicBlock*> __slli(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SLLI");
    	
    	this->gen_sync(PRE_SYNC, 24);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateShl(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_const(64U, shamt));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 24);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 25: SRLI */
    std::tuple<continuation_e, BasicBlock*> __srli(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SRLI");
    	
    	this->gen_sync(PRE_SYNC, 25);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateLShr(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_const(64U, shamt));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 25);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 26: SRAI */
    std::tuple<continuation_e, BasicBlock*> __srai(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SRAI");
    	
    	this->gen_sync(PRE_SYNC, 26);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateAShr(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_const(64U, shamt));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 26);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 27: ADD */
    std::tuple<continuation_e, BasicBlock*> __add(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("ADD");
    	
    	this->gen_sync(PRE_SYNC, 27);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateAdd(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 27);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 28: SUB */
    std::tuple<continuation_e, BasicBlock*> __sub(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SUB");
    	
    	this->gen_sync(PRE_SYNC, 28);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateSub(
    	         this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	         this->gen_reg_load(rs2 + traits<ARCH>::X0, 0));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 28);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 29: SLL */
    std::tuple<continuation_e, BasicBlock*> __sll(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SLL");
    	
    	this->gen_sync(PRE_SYNC, 29);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateShl(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->builder.CreateAnd(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            this->builder.CreateSub(
    	                 this->gen_const(64U, 64),
    	                 this->gen_const(64U, 1))));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 29);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 30: SLT */
    std::tuple<continuation_e, BasicBlock*> __slt(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SLT");
    	
    	this->gen_sync(PRE_SYNC, 30);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_SLT,
    	            this->gen_ext(
    	                this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	                64, true),
    	            this->gen_ext(
    	                this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	                64, true)),
    	        this->gen_const(64U, 1),
    	        this->gen_const(64U, 0),
    	        64);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 30);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 31: SLTU */
    std::tuple<continuation_e, BasicBlock*> __sltu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SLTU");
    	
    	this->gen_sync(PRE_SYNC, 31);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_ext(
    	                this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	                64,
    	                false),
    	            this->gen_ext(
    	                this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	                64,
    	                false)),
    	        this->gen_const(64U, 1),
    	        this->gen_const(64U, 0),
    	        64);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 31);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 32: XOR */
    std::tuple<continuation_e, BasicBlock*> __xor(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("XOR");
    	
    	this->gen_sync(PRE_SYNC, 32);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateXor(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 32);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 33: SRL */
    std::tuple<continuation_e, BasicBlock*> __srl(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SRL");
    	
    	this->gen_sync(PRE_SYNC, 33);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateLShr(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->builder.CreateAnd(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            this->builder.CreateSub(
    	                 this->gen_const(64U, 64),
    	                 this->gen_const(64U, 1))));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 33);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 34: SRA */
    std::tuple<continuation_e, BasicBlock*> __sra(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SRA");
    	
    	this->gen_sync(PRE_SYNC, 34);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateAShr(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->builder.CreateAnd(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            this->builder.CreateSub(
    	                 this->gen_const(64U, 64),
    	                 this->gen_const(64U, 1))));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 34);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 35: OR */
    std::tuple<continuation_e, BasicBlock*> __or(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("OR");
    	
    	this->gen_sync(PRE_SYNC, 35);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateOr(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 35);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 36: AND */
    std::tuple<continuation_e, BasicBlock*> __and(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("AND");
    	
    	this->gen_sync(PRE_SYNC, 36);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->builder.CreateAnd(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(rs2 + traits<ARCH>::X0, 0));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 36);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 37: FENCE */
    std::tuple<continuation_e, BasicBlock*> __fence(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FENCE");
    	
    	this->gen_sync(PRE_SYNC, 37);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t succ = ((bit_sub<20,4>(instr)));
    	uint8_t pred = ((bit_sub<24,4>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("fence"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* FENCEtmp0_val = this->builder.CreateOr(
    	    this->builder.CreateShl(
    	        this->gen_const(64U, pred),
    	        this->gen_const(64U, 4)),
    	    this->gen_const(64U, succ));
    	this->gen_write_mem(
    	    traits<ARCH>::FENCE,
    	    this->gen_const(64U, 0),
    	    this->builder.CreateZExtOrTrunc(FENCEtmp0_val,this->get_type(64)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 37);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 38: FENCE_I */
    std::tuple<continuation_e, BasicBlock*> __fence_i(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("FENCE_I");
    	
    	this->gen_sync(PRE_SYNC, 38);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint16_t imm = ((bit_sub<20,12>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("fence_i"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* FENCEtmp0_val = this->gen_const(64U, imm);
    	this->gen_write_mem(
    	    traits<ARCH>::FENCE,
    	    this->gen_const(64U, 1),
    	    this->builder.CreateZExtOrTrunc(FENCEtmp0_val,this->get_type(64)));
    	this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()), get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 38);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(FLUSH, nullptr);
    }
    
    /* instruction 39: ECALL */
    std::tuple<continuation_e, BasicBlock*> __ecall(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("ECALL");
    	
    	this->gen_sync(PRE_SYNC, 39);
    	
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("ecall"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	this->gen_raise_trap(0, 11);
    	this->gen_sync(POST_SYNC, 39);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 40: EBREAK */
    std::tuple<continuation_e, BasicBlock*> __ebreak(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("EBREAK");
    	
    	this->gen_sync(PRE_SYNC, 40);
    	
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("ebreak"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	this->gen_raise_trap(0, 3);
    	this->gen_sync(POST_SYNC, 40);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 41: URET */
    std::tuple<continuation_e, BasicBlock*> __uret(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("URET");
    	
    	this->gen_sync(PRE_SYNC, 41);
    	
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("uret"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	this->gen_leave_trap(0);
    	this->gen_sync(POST_SYNC, 41);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 42: SRET */
    std::tuple<continuation_e, BasicBlock*> __sret(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SRET");
    	
    	this->gen_sync(PRE_SYNC, 42);
    	
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("sret"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	this->gen_leave_trap(1);
    	this->gen_sync(POST_SYNC, 42);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 43: MRET */
    std::tuple<continuation_e, BasicBlock*> __mret(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("MRET");
    	
    	this->gen_sync(PRE_SYNC, 43);
    	
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("mret"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	this->gen_leave_trap(3);
    	this->gen_sync(POST_SYNC, 43);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(BRANCH, nullptr);
    }
    
    /* instruction 44: WFI */
    std::tuple<continuation_e, BasicBlock*> __wfi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("WFI");
    	
    	this->gen_sync(PRE_SYNC, 44);
    	
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("wfi"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	this->gen_wait(1);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 44);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 45: SFENCE.VMA */
    std::tuple<continuation_e, BasicBlock*> __sfence_vma(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SFENCE.VMA");
    	
    	this->gen_sync(PRE_SYNC, 45);
    	
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("sfence.vma"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* FENCEtmp0_val = this->gen_const(64U, rs1);
    	this->gen_write_mem(
    	    traits<ARCH>::FENCE,
    	    this->gen_const(64U, 2),
    	    this->builder.CreateZExtOrTrunc(FENCEtmp0_val,this->get_type(64)));
    	Value* FENCEtmp1_val = this->gen_const(64U, rs2);
    	this->gen_write_mem(
    	    traits<ARCH>::FENCE,
    	    this->gen_const(64U, 3),
    	    this->builder.CreateZExtOrTrunc(FENCEtmp1_val,this->get_type(64)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 45);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 46: CSRRW */
    std::tuple<continuation_e, BasicBlock*> __csrrw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("CSRRW");
    	
    	this->gen_sync(PRE_SYNC, 46);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* rs_val_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	if(rd != 0){
    	    Value* csr_val_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, csr), 64/8);
    	    Value* CSRtmp0_val = rs_val_val;
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, csr),
    	        this->builder.CreateZExtOrTrunc(CSRtmp0_val,this->get_type(64)));
    	    Value* Xtmp1_val = csr_val_val;
    	    this->builder.CreateStore(Xtmp1_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	} else {
    	    Value* CSRtmp2_val = rs_val_val;
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, csr),
    	        this->builder.CreateZExtOrTrunc(CSRtmp2_val,this->get_type(64)));
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 46);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 47: CSRRS */
    std::tuple<continuation_e, BasicBlock*> __csrrs(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("CSRRS");
    	
    	this->gen_sync(PRE_SYNC, 47);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* xrd_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, csr), 64/8);
    	Value* xrs1_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	if(rd != 0){
    	    Value* Xtmp0_val = xrd_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	if(rs1 != 0){
    	    Value* CSRtmp1_val = this->builder.CreateOr(
    	        xrd_val,
    	        xrs1_val);
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, csr),
    	        this->builder.CreateZExtOrTrunc(CSRtmp1_val,this->get_type(64)));
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 47);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 48: CSRRC */
    std::tuple<continuation_e, BasicBlock*> __csrrc(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("CSRRC");
    	
    	this->gen_sync(PRE_SYNC, 48);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* xrd_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, csr), 64/8);
    	Value* xrs1_val = this->gen_reg_load(rs1 + traits<ARCH>::X0, 0);
    	if(rd != 0){
    	    Value* Xtmp0_val = xrd_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	if(rs1 != 0){
    	    Value* CSRtmp1_val = this->builder.CreateAnd(
    	        xrd_val,
    	        this->builder.CreateNot(xrs1_val));
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, csr),
    	        this->builder.CreateZExtOrTrunc(CSRtmp1_val,this->get_type(64)));
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 48);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 49: CSRRWI */
    std::tuple<continuation_e, BasicBlock*> __csrrwi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("CSRRWI");
    	
    	this->gen_sync(PRE_SYNC, 49);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, csr), 64/8);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	Value* CSRtmp1_val = this->gen_ext(
    	    this->gen_const(64U, zimm),
    	    64,
    	    false);
    	this->gen_write_mem(
    	    traits<ARCH>::CSR,
    	    this->gen_const(16U, csr),
    	    this->builder.CreateZExtOrTrunc(CSRtmp1_val,this->get_type(64)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 49);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 50: CSRRSI */
    std::tuple<continuation_e, BasicBlock*> __csrrsi(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("CSRRSI");
    	
    	this->gen_sync(PRE_SYNC, 50);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, csr), 64/8);
    	if(zimm != 0){
    	    Value* CSRtmp0_val = this->builder.CreateOr(
    	        res_val,
    	        this->gen_ext(
    	            this->gen_const(64U, zimm),
    	            64,
    	            false));
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, csr),
    	        this->builder.CreateZExtOrTrunc(CSRtmp0_val,this->get_type(64)));
    	}
    	if(rd != 0){
    	    Value* Xtmp1_val = res_val;
    	    this->builder.CreateStore(Xtmp1_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 50);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 51: CSRRCI */
    std::tuple<continuation_e, BasicBlock*> __csrrci(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("CSRRCI");
    	
    	this->gen_sync(PRE_SYNC, 51);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, csr), 64/8);
    	if(rd != 0){
    	    Value* Xtmp0_val = res_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	if(zimm != 0){
    	    Value* CSRtmp1_val = this->builder.CreateAnd(
    	        res_val,
    	        this->builder.CreateNot(this->gen_ext(
    	            this->gen_const(64U, zimm),
    	            64,
    	            false)));
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, csr),
    	        this->builder.CreateZExtOrTrunc(CSRtmp1_val,this->get_type(64)));
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 51);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 52: LWU */
    std::tuple<continuation_e, BasicBlock*> __lwu(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("LWU");
    	
    	this->gen_sync(PRE_SYNC, 52);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        64, true),
    	    this->gen_const(64U, imm));
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	        64,
    	        false);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 52);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 53: LD */
    std::tuple<continuation_e, BasicBlock*> __ld(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("LD");
    	
    	this->gen_sync(PRE_SYNC, 53);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        64, true),
    	    this->gen_const(64U, imm));
    	if(rd != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 64/8),
    	        64,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 53);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 54: SD */
    std::tuple<continuation_e, BasicBlock*> __sd(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SD");
    	
    	this->gen_sync(PRE_SYNC, 54);
    	
    	int16_t imm = signextend<int16_t,12>((bit_sub<7,5>(instr)) | (bit_sub<25,7>(instr) << 5));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	        64, true),
    	    this->gen_const(64U, imm));
    	Value* MEMtmp0_val = this->gen_reg_load(rs2 + traits<ARCH>::X0, 0);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(64)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 54);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 55: ADDIW */
    std::tuple<continuation_e, BasicBlock*> __addiw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("ADDIW");
    	
    	this->gen_sync(PRE_SYNC, 55);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* res_val = this->builder.CreateAdd(
    	        this->gen_ext(
    	            this->builder.CreateTrunc(
    	                this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	                this-> get_type(32) 
    	            ),
    	            32, true),
    	        this->gen_const(32U, imm));
    	    Value* Xtmp0_val = this->gen_ext(
    	        res_val,
    	        64,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 55);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 56: SLLIW */
    std::tuple<continuation_e, BasicBlock*> __slliw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SLLIW");
    	
    	this->gen_sync(PRE_SYNC, 56);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* sh_val_val = this->builder.CreateShl(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            this-> get_type(32) 
    	        ),
    	        this->gen_const(32U, shamt));
    	    Value* Xtmp0_val = this->gen_ext(
    	        sh_val_val,
    	        64,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 56);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 57: SRLIW */
    std::tuple<continuation_e, BasicBlock*> __srliw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SRLIW");
    	
    	this->gen_sync(PRE_SYNC, 57);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* sh_val_val = this->builder.CreateLShr(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            this-> get_type(32) 
    	        ),
    	        this->gen_const(32U, shamt));
    	    Value* Xtmp0_val = this->gen_ext(
    	        sh_val_val,
    	        64,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 57);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 58: SRAIW */
    std::tuple<continuation_e, BasicBlock*> __sraiw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SRAIW");
    	
    	this->gen_sync(PRE_SYNC, 58);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* sh_val_val = this->builder.CreateAShr(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            this-> get_type(32) 
    	        ),
    	        this->gen_const(32U, shamt));
    	    Value* Xtmp0_val = this->gen_ext(
    	        sh_val_val,
    	        64,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 58);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 59: ADDW */
    std::tuple<continuation_e, BasicBlock*> __addw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("ADDW");
    	
    	this->gen_sync(PRE_SYNC, 59);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("addw"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* res_val = this->builder.CreateAdd(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            this-> get_type(32) 
    	        ),
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            this-> get_type(32) 
    	        ));
    	    Value* Xtmp0_val = this->gen_ext(
    	        res_val,
    	        64,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 59);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 60: SUBW */
    std::tuple<continuation_e, BasicBlock*> __subw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SUBW");
    	
    	this->gen_sync(PRE_SYNC, 60);
    	
    	uint8_t rd = ((bit_sub<7,5>(instr)));
    	uint8_t rs1 = ((bit_sub<15,5>(instr)));
    	uint8_t rs2 = ((bit_sub<20,5>(instr)));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("subw"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    Value* res_val = this->builder.CreateSub(
    	         this->builder.CreateTrunc(
    	             this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	             this-> get_type(32) 
    	         ),
    	         this->builder.CreateTrunc(
    	             this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	             this-> get_type(32) 
    	         ));
    	    Value* Xtmp0_val = this->gen_ext(
    	        res_val,
    	        64,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 60);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 61: SLLW */
    std::tuple<continuation_e, BasicBlock*> __sllw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SLLW");
    	
    	this->gen_sync(PRE_SYNC, 61);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    uint32_t mask_val = 0x1f;
    	    Value* count_val = this->builder.CreateAnd(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            this-> get_type(32) 
    	        ),
    	        this->gen_const(32U, mask_val));
    	    Value* sh_val_val = this->builder.CreateShl(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            this-> get_type(32) 
    	        ),
    	        count_val);
    	    Value* Xtmp0_val = this->gen_ext(
    	        sh_val_val,
    	        64,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 61);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 62: SRLW */
    std::tuple<continuation_e, BasicBlock*> __srlw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SRLW");
    	
    	this->gen_sync(PRE_SYNC, 62);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    uint32_t mask_val = 0x1f;
    	    Value* count_val = this->builder.CreateAnd(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            this-> get_type(32) 
    	        ),
    	        this->gen_const(32U, mask_val));
    	    Value* sh_val_val = this->builder.CreateLShr(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            this-> get_type(32) 
    	        ),
    	        count_val);
    	    Value* Xtmp0_val = this->gen_ext(
    	        sh_val_val,
    	        64,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 62);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /* instruction 63: SRAW */
    std::tuple<continuation_e, BasicBlock*> __sraw(virt_addr_t& pc, code_word_t instr, BasicBlock* bb){
    	bb->setName("SRAW");
    	
    	this->gen_sync(PRE_SYNC, 63);
    	
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
    	
    	Value* cur_pc_val = this->gen_const(64, pc.val);
    	pc=pc+4;
    	
    	if(rd != 0){
    	    uint32_t mask_val = 0x1f;
    	    Value* count_val = this->builder.CreateAnd(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs2 + traits<ARCH>::X0, 0),
    	            this-> get_type(32) 
    	        ),
    	        this->gen_const(32U, mask_val));
    	    Value* sh_val_val = this->builder.CreateAShr(
    	        this->builder.CreateTrunc(
    	            this->gen_reg_load(rs1 + traits<ARCH>::X0, 0),
    	            this-> get_type(32) 
    	        ),
    	        count_val);
    	    Value* Xtmp0_val = this->gen_ext(
    	        sh_val_val,
    	        64,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(POST_SYNC, 63);
    	bb = BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(CONT, bb);
    }
    
    /****************************************************************************
     * end opcode definitions
     ****************************************************************************/
    std::tuple<continuation_e, BasicBlock *> illegal_intruction(virt_addr_t &pc, code_word_t instr, BasicBlock *bb) {
		this->gen_sync(iss::PRE_SYNC, instr_descr.size());
        this->builder.CreateStore(this->builder.CreateLoad(get_reg_ptr(traits<ARCH>::NEXT_PC), true),
                                   get_reg_ptr(traits<ARCH>::PC), true);
        this->builder.CreateStore(
            this->builder.CreateAdd(this->builder.CreateLoad(get_reg_ptr(traits<ARCH>::ICOUNT), true),
                                     this->gen_const(64U, 1)),
            get_reg_ptr(traits<ARCH>::ICOUNT), true);
        pc = pc + ((instr & 3) == 3 ? 4 : 2);
        this->gen_raise_trap(0, 2);     // illegal instruction trap
		this->gen_sync(iss::POST_SYNC, instr_descr.size());
        this->gen_trap_check(this->leave_blk);
        return std::make_tuple(BRANCH, nullptr);
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
std::tuple<continuation_e, BasicBlock *>
vm_impl<ARCH>::gen_single_inst_behavior(virt_addr_t &pc, unsigned int &inst_cnt, BasicBlock *this_block) {
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
    return (this->*f)(pc, insn, this_block);
}

template <typename ARCH> void vm_impl<ARCH>::gen_leave_behavior(BasicBlock *leave_blk) {
    this->builder.SetInsertPoint(leave_blk);
    this->builder.CreateRet(this->builder.CreateLoad(get_reg_ptr(arch::traits<ARCH>::NEXT_PC), false));
}

template <typename ARCH> void vm_impl<ARCH>::gen_raise_trap(uint16_t trap_id, uint16_t cause) {
    auto *TRAP_val = this->gen_const(32, 0x80 << 24 | (cause << 16) | trap_id);
    this->builder.CreateStore(TRAP_val, get_reg_ptr(traits<ARCH>::TRAP_STATE), true);
    this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()), get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
}

template <typename ARCH> void vm_impl<ARCH>::gen_leave_trap(unsigned lvl) {
    std::vector<Value *> args{ this->core_ptr, ConstantInt::get(getContext(), APInt(64, lvl)) };
    this->builder.CreateCall(this->mod->getFunction("leave_trap"), args);
    auto *PC_val = this->gen_read_mem(traits<ARCH>::CSR, (lvl << 8) + 0x41, traits<ARCH>::XLEN / 8);
    this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()), get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
}

template <typename ARCH> void vm_impl<ARCH>::gen_wait(unsigned type) {
    std::vector<Value *> args{ this->core_ptr, ConstantInt::get(getContext(), APInt(64, type)) };
    this->builder.CreateCall(this->mod->getFunction("wait"), args);
}

template <typename ARCH> void vm_impl<ARCH>::gen_trap_behavior(BasicBlock *trap_blk) {
    this->builder.SetInsertPoint(trap_blk);
    auto *trap_state_val = this->builder.CreateLoad(get_reg_ptr(traits<ARCH>::TRAP_STATE), true);
    this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()),
                              get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    std::vector<Value *> args{this->core_ptr, this->adj_to64(trap_state_val),
                              this->adj_to64(this->builder.CreateLoad(get_reg_ptr(traits<ARCH>::PC), false))};
    this->builder.CreateCall(this->mod->getFunction("enter_trap"), args);
    auto *trap_addr_val = this->builder.CreateLoad(get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    this->builder.CreateRet(trap_addr_val);
}

template <typename ARCH> inline void vm_impl<ARCH>::gen_trap_check(BasicBlock *bb) {
    auto *v = this->builder.CreateLoad(get_reg_ptr(arch::traits<ARCH>::TRAP_STATE), true);
    this->gen_cond_branch(this->builder.CreateICmp(
                              ICmpInst::ICMP_EQ, v,
                              ConstantInt::get(getContext(), APInt(v->getType()->getIntegerBitWidth(), 0))),
                          bb, this->trap_blk, 1);
}

} // namespace rv64i

template <>
std::unique_ptr<vm_if> create<arch::rv64i>(arch::rv64i *core, unsigned short port, bool dump) {
    auto ret = new rv64i::vm_impl<arch::rv64i>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret, port);
    return std::unique_ptr<vm_if>(ret);
}

} // namespace iss
