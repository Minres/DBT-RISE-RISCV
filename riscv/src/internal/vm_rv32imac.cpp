////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, MINRES Technologies GmbH
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Contributors:
//       eyck@minres.com - initial API and implementation
//
//
////////////////////////////////////////////////////////////////////////////////

#include <iss/arch/rv32imac.h>
#include <iss/arch/riscv_hart_msu_vp.h>
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/server.h>
#include <iss/iss.h>
#include <iss/vm_base.h>
#include <util/logging.h>

#include <boost/format.hpp>

#include <iss/debugger/riscv_target_adapter.h>
#include <array>

namespace iss {
namespace vm {
namespace fp_impl{
void add_fp_functions_2_module(llvm::Module *, unsigned);
}
}

namespace rv32imac {
using namespace iss::arch;
using namespace llvm;
using namespace iss::debugger;

template <typename ARCH> class vm_impl : public vm::vm_base<ARCH> {
public:
    using super = typename vm::vm_base<ARCH>;
    using virt_addr_t = typename super::virt_addr_t;
    using phys_addr_t = typename super::phys_addr_t;
    using code_word_t = typename super::code_word_t;
    using addr_t = typename super::addr_t;

    vm_impl();

    vm_impl(ARCH &core, unsigned core_id = 0, unsigned cluster_id = 0);

    void enableDebug(bool enable) { super::sync_exec = super::ALL_SYNC; }

    target_adapter_if *accquire_target_adapter(server_if *srv) {
        debugger_if::dbg_enabled = true;
        if (vm::vm_base<ARCH>::tgt_adapter == nullptr)
            vm::vm_base<ARCH>::tgt_adapter = new riscv_target_adapter<ARCH>(srv, this->get_arch());
        return vm::vm_base<ARCH>::tgt_adapter;
    }

protected:
    using vm::vm_base<ARCH>::get_reg_ptr;

    template <typename T> inline llvm::ConstantInt *size(T type) {
        return llvm::ConstantInt::get(getContext(), llvm::APInt(32, type->getType()->getScalarSizeInBits()));
    }

    void setup_module(llvm::Module* m) override {
        super::setup_module(m);
        vm::fp_impl::add_fp_functions_2_module(m, traits<ARCH>::FP_REGS_SIZE);
    }

    inline llvm::Value *gen_choose(llvm::Value *cond, llvm::Value *trueVal, llvm::Value *falseVal, unsigned size) {
        return super::gen_cond_assign(cond, this->gen_ext(trueVal, size), this->gen_ext(falseVal, size));
    }

    std::tuple<vm::continuation_e, llvm::BasicBlock *> gen_single_inst_behavior(virt_addr_t &, unsigned int &,
                                                                                llvm::BasicBlock *) override;

    void gen_leave_behavior(llvm::BasicBlock *leave_blk) override;

    void gen_raise_trap(uint16_t trap_id, uint16_t cause);

    void gen_leave_trap(unsigned lvl);

    void gen_wait(unsigned type);

    void gen_trap_behavior(llvm::BasicBlock *) override;

    void gen_trap_check(llvm::BasicBlock *bb);

    inline llvm::Value *gen_reg_load(unsigned i, unsigned level = 0) {
        return this->builder.CreateLoad(get_reg_ptr(i), false);
    }

    inline void gen_set_pc(virt_addr_t pc, unsigned reg_num) {
        llvm::Value *next_pc_v = this->builder.CreateSExtOrTrunc(this->gen_const(traits<ARCH>::XLEN, pc.val),
                                                                  this->get_type(traits<ARCH>::XLEN));
        this->builder.CreateStore(next_pc_v, get_reg_ptr(reg_num), true);
    }

    // some compile time constants
    // enum { MASK16 = 0b1111110001100011, MASK32 = 0b11111111111100000111000001111111 };
    enum { MASK16 = 0b1111111111111111, MASK32 = 0b11111111111100000111000001111111 };
    enum { EXTR_MASK16 = MASK16 >> 2, EXTR_MASK32 = MASK32 >> 2 };
    enum { LUT_SIZE = 1 << util::bit_count(EXTR_MASK32), LUT_SIZE_C = 1 << util::bit_count(EXTR_MASK16) };

    using this_class = vm_impl<ARCH>;
    using compile_func = std::tuple<vm::continuation_e, llvm::BasicBlock *> (this_class::*)(virt_addr_t &pc,
                                                                                            code_word_t instr,
                                                                                            llvm::BasicBlock *bb);
    std::array<compile_func, LUT_SIZE> lut;

    std::array<compile_func, LUT_SIZE_C> lut_00, lut_01, lut_10;
    std::array<compile_func, LUT_SIZE> lut_11;

	std::array<compile_func*, 4> qlut;

	std::array<const uint32_t, 4> lutmasks = { { EXTR_MASK16, EXTR_MASK16, EXTR_MASK16, EXTR_MASK32 } };

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
        {32, 0b00000000000000000001000000010011, 0b11111110000000000111000001111111, &this_class::__slli},
        /* instruction SRLI */
        {32, 0b00000000000000000101000000010011, 0b11111110000000000111000001111111, &this_class::__srli},
        /* instruction SRAI */
        {32, 0b01000000000000000101000000010011, 0b11111110000000000111000001111111, &this_class::__srai},
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
        /* instruction MUL */
        {32, 0b00000010000000000000000000110011, 0b11111110000000000111000001111111, &this_class::__mul},
        /* instruction MULH */
        {32, 0b00000010000000000001000000110011, 0b11111110000000000111000001111111, &this_class::__mulh},
        /* instruction MULHSU */
        {32, 0b00000010000000000010000000110011, 0b11111110000000000111000001111111, &this_class::__mulhsu},
        /* instruction MULHU */
        {32, 0b00000010000000000011000000110011, 0b11111110000000000111000001111111, &this_class::__mulhu},
        /* instruction DIV */
        {32, 0b00000010000000000100000000110011, 0b11111110000000000111000001111111, &this_class::__div},
        /* instruction DIVU */
        {32, 0b00000010000000000101000000110011, 0b11111110000000000111000001111111, &this_class::__divu},
        /* instruction REM */
        {32, 0b00000010000000000110000000110011, 0b11111110000000000111000001111111, &this_class::__rem},
        /* instruction REMU */
        {32, 0b00000010000000000111000000110011, 0b11111110000000000111000001111111, &this_class::__remu},
        /* instruction LR.W */
        {32, 0b00010000000000000010000000101111, 0b11111001111100000111000001111111, &this_class::__lr_w},
        /* instruction SC.W */
        {32, 0b00011000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__sc_w},
        /* instruction AMOSWAP.W */
        {32, 0b00001000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoswap_w},
        /* instruction AMOADD.W */
        {32, 0b00000000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoadd_w},
        /* instruction AMOXOR.W */
        {32, 0b00100000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoxor_w},
        /* instruction AMOAND.W */
        {32, 0b01100000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoand_w},
        /* instruction AMOOR.W */
        {32, 0b01000000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amoor_w},
        /* instruction AMOMIN.W */
        {32, 0b10000000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amomin_w},
        /* instruction AMOMAX.W */
        {32, 0b10100000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amomax_w},
        /* instruction AMOMINU.W */
        {32, 0b11000000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amominu_w},
        /* instruction AMOMAXU.W */
        {32, 0b11100000000000000010000000101111, 0b11111000000000000111000001111111, &this_class::__amomaxu_w},
        /* instruction C.ADDI4SPN */
        {16, 0b0000000000000000, 0b1110000000000011, &this_class::__c_addi4spn},
        /* instruction C.LW */
        {16, 0b0100000000000000, 0b1110000000000011, &this_class::__c_lw},
        /* instruction C.SW */
        {16, 0b1100000000000000, 0b1110000000000011, &this_class::__c_sw},
        /* instruction C.ADDI */
        {16, 0b0000000000000001, 0b1110000000000011, &this_class::__c_addi},
        /* instruction C.NOP */
        {16, 0b0000000000000001, 0b1111111111111111, &this_class::__c_nop},
        /* instruction C.JAL */
        {16, 0b0010000000000001, 0b1110000000000011, &this_class::__c_jal},
        /* instruction C.LI */
        {16, 0b0100000000000001, 0b1110000000000011, &this_class::__c_li},
        /* instruction C.LUI */
        {16, 0b0110000000000001, 0b1110000000000011, &this_class::__c_lui},
        /* instruction C.ADDI16SP */
        {16, 0b0110000100000001, 0b1110111110000011, &this_class::__c_addi16sp},
        /* instruction C.SRLI */
        {16, 0b1000000000000001, 0b1111110000000011, &this_class::__c_srli},
        /* instruction C.SRAI */
        {16, 0b1000010000000001, 0b1111110000000011, &this_class::__c_srai},
        /* instruction C.ANDI */
        {16, 0b1000100000000001, 0b1110110000000011, &this_class::__c_andi},
        /* instruction C.SUB */
        {16, 0b1000110000000001, 0b1111110001100011, &this_class::__c_sub},
        /* instruction C.XOR */
        {16, 0b1000110000100001, 0b1111110001100011, &this_class::__c_xor},
        /* instruction C.OR */
        {16, 0b1000110001000001, 0b1111110001100011, &this_class::__c_or},
        /* instruction C.AND */
        {16, 0b1000110001100001, 0b1111110001100011, &this_class::__c_and},
        /* instruction C.J */
        {16, 0b1010000000000001, 0b1110000000000011, &this_class::__c_j},
        /* instruction C.BEQZ */
        {16, 0b1100000000000001, 0b1110000000000011, &this_class::__c_beqz},
        /* instruction C.BNEZ */
        {16, 0b1110000000000001, 0b1110000000000011, &this_class::__c_bnez},
        /* instruction C.SLLI */
        {16, 0b0000000000000010, 0b1111000000000011, &this_class::__c_slli},
        /* instruction C.LWSP */
        {16, 0b0100000000000010, 0b1110000000000011, &this_class::__c_lwsp},
        /* instruction C.MV */
        {16, 0b1000000000000010, 0b1111000000000011, &this_class::__c_mv},
        /* instruction C.JR */
        {16, 0b1000000000000010, 0b1111000001111111, &this_class::__c_jr},
        /* instruction C.ADD */
        {16, 0b1001000000000010, 0b1111000000000011, &this_class::__c_add},
        /* instruction C.JALR */
        {16, 0b1001000000000010, 0b1111000001111111, &this_class::__c_jalr},
        /* instruction C.EBREAK */
        {16, 0b1001000000000010, 0b1111111111111111, &this_class::__c_ebreak},
        /* instruction C.SWSP */
        {16, 0b1100000000000010, 0b1110000000000011, &this_class::__c_swsp},
        /* instruction DII */
        {16, 0b0000000000000000, 0b1111111111111111, &this_class::__dii},
    }};
 
    /* instruction definitions */
    /* instruction 0: LUI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __lui(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("LUI");
    	
    	this->gen_sync(iss::PRE_SYNC, 0);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	int32_t fld_imm_val = 0 | (signed_bit_sub<12,20>(instr) << 12);
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("LUI x%1$d, 0x%2$05x");
    	    ins_fmter % (uint64_t)fld_rd_val % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->gen_const(32U, fld_imm_val);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 0);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 1: AUIPC */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __auipc(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("AUIPC");
    	
    	this->gen_sync(iss::PRE_SYNC, 1);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	int32_t fld_imm_val = 0 | (signed_bit_sub<12,20>(instr) << 12);
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("AUIPC x%1%, 0x%2$08x");
    	    ins_fmter % (uint64_t)fld_rd_val % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, fld_imm_val));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 1);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 2: JAL */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __jal(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("JAL");
    	
    	this->gen_sync(iss::PRE_SYNC, 2);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	int32_t fld_imm_val = 0 | (bit_sub<12,8>(instr) << 12) | (bit_sub<20,1>(instr) << 11) | (bit_sub<21,10>(instr) << 1) | (signed_bit_sub<31,1>(instr) << 20);
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("JAL x%1$d, 0x%2$x");
    	    ins_fmter % (uint64_t)fld_rd_val % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 4));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	Value* PC_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        cur_pc_val,
    	        32, true),
    	    this->gen_const(32U, fld_imm_val));
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(iss::POST_SYNC, 2);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 3: JALR */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __jalr(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("JALR");
    	
    	this->gen_sync(iss::PRE_SYNC, 3);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	int16_t fld_imm_val = 0 | (signed_bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("JALR x%1$d, x%2$d, 0x%3$x");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* new_pc_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, fld_imm_val));
    	Value* align_val = this->builder.CreateAnd(
    	    new_pc_val,
    	    this->gen_const(32U, 0x1));
    	{
    	    llvm::BasicBlock* bbnext = llvm::BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	    llvm::BasicBlock* bb_then = llvm::BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	    llvm::BasicBlock* bb_else = llvm::BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	    // this->builder.SetInsertPoint(bb);
    	    this->gen_cond_branch(this->builder.CreateICmp(
    	        ICmpInst::ICMP_NE,
    	        align_val,
    	        this->gen_const(32U, 0)),
    	        bb_then,
    	        bb_else);
    	    this->builder.SetInsertPoint(bb_then);
    	    {
    	        this->gen_raise_trap(0, 0);
    	    }
    	    this->builder.CreateBr(bbnext);
    	    this->builder.SetInsertPoint(bb_else);
    	    {
    	        if(fld_rd_val != 0){
    	            Value* Xtmp0_val = this->builder.CreateAdd(
    	                cur_pc_val,
    	                this->gen_const(32U, 4));
    	            this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	        }
    	        Value* PC_val = this->builder.CreateAnd(
    	            new_pc_val,
    	            this->builder.CreateNot(this->gen_const(32U, 0x1)));
    	        this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	        this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()), get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	    }
    	    this->builder.CreateBr(bbnext);
    	    bb=bbnext;
    	}
    	this->builder.SetInsertPoint(bb);
    	this->gen_sync(iss::POST_SYNC, 3);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 4: BEQ */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __beq(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("BEQ");
    	
    	this->gen_sync(iss::PRE_SYNC, 4);
    	
    	int16_t fld_imm_val = 0 | (bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (signed_bit_sub<31,1>(instr) << 12);
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("BEQ x%1$d, x%2$d, 0x%3$x");
    	    ins_fmter % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_EQ,
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, fld_imm_val)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 4)),
    	    32);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(iss::POST_SYNC, 4);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 5: BNE */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __bne(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("BNE");
    	
    	this->gen_sync(iss::PRE_SYNC, 5);
    	
    	int16_t fld_imm_val = 0 | (bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (signed_bit_sub<31,1>(instr) << 12);
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("BNE x%1$d, x%2$d, 0x%3$x");
    	    ins_fmter % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_NE,
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, fld_imm_val)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 4)),
    	    32);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(iss::POST_SYNC, 5);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 6: BLT */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __blt(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("BLT");
    	
    	this->gen_sync(iss::PRE_SYNC, 6);
    	
    	int16_t fld_imm_val = 0 | (bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (signed_bit_sub<31,1>(instr) << 12);
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("BLT x%1$d, x%2$d, 0x%3$x");
    	    ins_fmter % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_SLT,
    	        this->gen_ext(
    	            this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	            32, true),
    	        this->gen_ext(
    	            this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	            32, true)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, fld_imm_val)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 4)),
    	    32);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(iss::POST_SYNC, 6);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 7: BGE */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __bge(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("BGE");
    	
    	this->gen_sync(iss::PRE_SYNC, 7);
    	
    	int16_t fld_imm_val = 0 | (bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (signed_bit_sub<31,1>(instr) << 12);
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("BGE x%1$d, x%2$d, 0x%3$x");
    	    ins_fmter % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_SGE,
    	        this->gen_ext(
    	            this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	            32, true),
    	        this->gen_ext(
    	            this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	            32, true)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, fld_imm_val)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 4)),
    	    32);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(iss::POST_SYNC, 7);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 8: BLTU */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __bltu(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("BLTU");
    	
    	this->gen_sync(iss::PRE_SYNC, 8);
    	
    	int16_t fld_imm_val = 0 | (bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (signed_bit_sub<31,1>(instr) << 12);
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("BLTU x%1$d, x%2$d, 0x%3$x");
    	    ins_fmter % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_ULT,
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, fld_imm_val)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 4)),
    	    32);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(iss::POST_SYNC, 8);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 9: BGEU */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __bgeu(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("BGEU");
    	
    	this->gen_sync(iss::PRE_SYNC, 9);
    	
    	int16_t fld_imm_val = 0 | (bit_sub<7,1>(instr) << 11) | (bit_sub<8,4>(instr) << 1) | (bit_sub<25,6>(instr) << 5) | (signed_bit_sub<31,1>(instr) << 12);
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("BGEU x%1$d, x%2$d, 0x%3$x");
    	    ins_fmter % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_UGE,
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, fld_imm_val)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 4)),
    	    32);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(iss::POST_SYNC, 9);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 10: LB */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __lb(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("LB");
    	
    	this->gen_sync(iss::PRE_SYNC, 10);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	int16_t fld_imm_val = 0 | (signed_bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("LB x%1$d, %2%(x%3$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (int64_t)fld_imm_val % (uint64_t)fld_rs1_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, fld_imm_val));
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 8/8),
    	        32,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 10);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 11: LH */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __lh(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("LH");
    	
    	this->gen_sync(iss::PRE_SYNC, 11);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	int16_t fld_imm_val = 0 | (signed_bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("LH x%1$d, %2%(x%3$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (int64_t)fld_imm_val % (uint64_t)fld_rs1_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, fld_imm_val));
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 16/8),
    	        32,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 11);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 12: LW */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __lw(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("LW");
    	
    	this->gen_sync(iss::PRE_SYNC, 12);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	int16_t fld_imm_val = 0 | (signed_bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("LW x%1$d, %2%(x%3$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (int64_t)fld_imm_val % (uint64_t)fld_rs1_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, fld_imm_val));
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	        32,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 12);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 13: LBU */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __lbu(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("LBU");
    	
    	this->gen_sync(iss::PRE_SYNC, 13);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	int16_t fld_imm_val = 0 | (signed_bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("LBU x%1$d, %2%(x%3$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (int64_t)fld_imm_val % (uint64_t)fld_rs1_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, fld_imm_val));
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 8/8),
    	        32,
    	        false);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 13);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 14: LHU */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __lhu(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("LHU");
    	
    	this->gen_sync(iss::PRE_SYNC, 14);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	int16_t fld_imm_val = 0 | (signed_bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("LHU x%1$d, %2%(x%3$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (int64_t)fld_imm_val % (uint64_t)fld_rs1_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, fld_imm_val));
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 16/8),
    	        32,
    	        false);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 14);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 15: SB */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sb(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("SB");
    	
    	this->gen_sync(iss::PRE_SYNC, 15);
    	
    	int16_t fld_imm_val = 0 | (bit_sub<7,5>(instr)) | (signed_bit_sub<25,7>(instr) << 5);
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("SB x%1$d, %2%(x%3$d)");
    	    ins_fmter % (uint64_t)fld_rs2_val % (int64_t)fld_imm_val % (uint64_t)fld_rs1_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, fld_imm_val));
    	Value* MEMtmp0_val = this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(8)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 15);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 16: SH */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sh(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("SH");
    	
    	this->gen_sync(iss::PRE_SYNC, 16);
    	
    	int16_t fld_imm_val = 0 | (bit_sub<7,5>(instr)) | (signed_bit_sub<25,7>(instr) << 5);
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("SH x%1$d, %2%(x%3$d)");
    	    ins_fmter % (uint64_t)fld_rs2_val % (int64_t)fld_imm_val % (uint64_t)fld_rs1_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, fld_imm_val));
    	Value* MEMtmp0_val = this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(16)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 16);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 17: SW */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sw(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("SW");
    	
    	this->gen_sync(iss::PRE_SYNC, 17);
    	
    	int16_t fld_imm_val = 0 | (bit_sub<7,5>(instr)) | (signed_bit_sub<25,7>(instr) << 5);
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("SW x%1$d, %2%(x%3$d)");
    	    ins_fmter % (uint64_t)fld_rs2_val % (int64_t)fld_imm_val % (uint64_t)fld_rs1_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, fld_imm_val));
    	Value* MEMtmp0_val = this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 17);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 18: ADDI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __addi(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("ADDI");
    	
    	this->gen_sync(iss::PRE_SYNC, 18);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	int16_t fld_imm_val = 0 | (signed_bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("ADDI x%1$d, x%2$d, %3%");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->builder.CreateAdd(
    	        this->gen_ext(
    	            this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	            32, true),
    	        this->gen_const(32U, fld_imm_val));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 18);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 19: SLTI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __slti(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("SLTI");
    	
    	this->gen_sync(iss::PRE_SYNC, 19);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	int16_t fld_imm_val = 0 | (signed_bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("SLTI x%1$d, x%2$d, %3%");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_SLT,
    	            this->gen_ext(
    	                this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	                32, true),
    	            this->gen_const(32U, fld_imm_val)),
    	        this->gen_const(32U, 1),
    	        this->gen_const(32U, 0),
    	        32);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 19);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 20: SLTIU */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sltiu(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("SLTIU");
    	
    	this->gen_sync(iss::PRE_SYNC, 20);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	int16_t fld_imm_val = 0 | (signed_bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("SLTIU x%1$d, x%2$d, %3%");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	int32_t full_imm_val = fld_imm_val;
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, full_imm_val)),
    	        this->gen_const(32U, 1),
    	        this->gen_const(32U, 0),
    	        32);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 20);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 21: XORI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __xori(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("XORI");
    	
    	this->gen_sync(iss::PRE_SYNC, 21);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint16_t fld_imm_val = 0 | (bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("XORI x%1$d, x%2$d, %3%");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->builder.CreateXor(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        this->gen_const(32U, fld_imm_val));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 21);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 22: ORI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __ori(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("ORI");
    	
    	this->gen_sync(iss::PRE_SYNC, 22);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint16_t fld_imm_val = 0 | (bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("ORI x%1$d, x%2$d, %3%");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->builder.CreateOr(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        this->gen_const(32U, fld_imm_val));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 22);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 23: ANDI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __andi(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("ANDI");
    	
    	this->gen_sync(iss::PRE_SYNC, 23);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint16_t fld_imm_val = 0 | (bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("ANDI x%1$d, x%2$d, %3%");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->builder.CreateAnd(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        this->gen_const(32U, fld_imm_val));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 23);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 24: SLLI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __slli(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("SLLI");
    	
    	this->gen_sync(iss::PRE_SYNC, 24);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_shamt_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("SLLI x%1$d, x%2$d, %3%");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_shamt_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_shamt_val > 31){
    	    this->gen_raise_trap(0, 0);
    	} else {
    	    if(fld_rd_val != 0){
    	        Value* Xtmp0_val = this->builder.CreateShl(
    	            this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, fld_shamt_val));
    	        this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	    }
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 24);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 25: SRLI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __srli(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("SRLI");
    	
    	this->gen_sync(iss::PRE_SYNC, 25);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_shamt_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("SRLI x%1$d, x%2$d, %3%");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_shamt_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_shamt_val > 31){
    	    this->gen_raise_trap(0, 0);
    	} else {
    	    if(fld_rd_val != 0){
    	        Value* Xtmp0_val = this->builder.CreateLShr(
    	            this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, fld_shamt_val));
    	        this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	    }
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 25);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 26: SRAI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __srai(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("SRAI");
    	
    	this->gen_sync(iss::PRE_SYNC, 26);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_shamt_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("SRAI x%1$d, x%2$d, %3%");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_shamt_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_shamt_val > 31){
    	    this->gen_raise_trap(0, 0);
    	} else {
    	    if(fld_rd_val != 0){
    	        Value* Xtmp0_val = this->builder.CreateAShr(
    	            this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, fld_shamt_val));
    	        this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	    }
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 26);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 27: ADD */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __add(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("ADD");
    	
    	this->gen_sync(iss::PRE_SYNC, 27);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("ADD x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->builder.CreateAdd(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 27);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 28: SUB */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sub(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("SUB");
    	
    	this->gen_sync(iss::PRE_SYNC, 28);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("SUB x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->builder.CreateSub(
    	         this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	         this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 28);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 29: SLL */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sll(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("SLL");
    	
    	this->gen_sync(iss::PRE_SYNC, 29);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("SLL x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->builder.CreateShl(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        this->builder.CreateAnd(
    	            this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, 0x1f)));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 29);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 30: SLT */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __slt(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("SLT");
    	
    	this->gen_sync(iss::PRE_SYNC, 30);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("SLT x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_SLT,
    	            this->gen_ext(
    	                this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	                32, true),
    	            this->gen_ext(
    	                this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	                32, true)),
    	        this->gen_const(32U, 1),
    	        this->gen_const(32U, 0),
    	        32);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 30);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 31: SLTU */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sltu(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("SLTU");
    	
    	this->gen_sync(iss::PRE_SYNC, 31);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("SLTU x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_ULT,
    	            this->gen_ext(
    	                this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	                32,
    	                false),
    	            this->gen_ext(
    	                this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	                32,
    	                false)),
    	        this->gen_const(32U, 1),
    	        this->gen_const(32U, 0),
    	        32);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 31);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 32: XOR */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __xor(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("XOR");
    	
    	this->gen_sync(iss::PRE_SYNC, 32);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("XOR x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->builder.CreateXor(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 32);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 33: SRL */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __srl(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("SRL");
    	
    	this->gen_sync(iss::PRE_SYNC, 33);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("SRL x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->builder.CreateLShr(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        this->builder.CreateAnd(
    	            this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, 0x1f)));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 33);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 34: SRA */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sra(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("SRA");
    	
    	this->gen_sync(iss::PRE_SYNC, 34);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("SRA x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->builder.CreateAShr(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        this->builder.CreateAnd(
    	            this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, 0x1f)));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 34);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 35: OR */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __or(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("OR");
    	
    	this->gen_sync(iss::PRE_SYNC, 35);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("OR x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->builder.CreateOr(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 35);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 36: AND */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __and(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("AND");
    	
    	this->gen_sync(iss::PRE_SYNC, 36);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("AND x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->builder.CreateAnd(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0));
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 36);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 37: FENCE */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __fence(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("FENCE");
    	
    	this->gen_sync(iss::PRE_SYNC, 37);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_succ_val = 0 | (bit_sub<20,4>(instr));
    	uint8_t fld_pred_val = 0 | (bit_sub<24,4>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("FENCE"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* FENCEtmp0_val = this->builder.CreateOr(
    	    this->builder.CreateShl(
    	        this->gen_const(32U, fld_pred_val),
    	        this->gen_const(32U, 4)),
    	    this->gen_const(32U, fld_succ_val));
    	this->gen_write_mem(
    	    traits<ARCH>::FENCE,
    	    this->gen_const(64U, 0),
    	    this->builder.CreateZExtOrTrunc(FENCEtmp0_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 37);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 38: FENCE_I */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __fence_i(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("FENCE_I");
    	
    	this->gen_sync(iss::PRE_SYNC, 38);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint16_t fld_imm_val = 0 | (bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("FENCE_I"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* FENCEtmp0_val = this->gen_const(32U, fld_imm_val);
    	this->gen_write_mem(
    	    traits<ARCH>::FENCE,
    	    this->gen_const(64U, 1),
    	    this->builder.CreateZExtOrTrunc(FENCEtmp0_val,this->get_type(32)));
    	this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()), get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 38);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::FLUSH, nullptr);
    }
    
    /* instruction 39: ECALL */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __ecall(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("ECALL");
    	
    	this->gen_sync(iss::PRE_SYNC, 39);
    	
    	;
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("ECALL"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	this->gen_raise_trap(0, 11);
    	this->gen_sync(iss::POST_SYNC, 39);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 40: EBREAK */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __ebreak(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("EBREAK");
    	
    	this->gen_sync(iss::PRE_SYNC, 40);
    	
    	;
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("EBREAK"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	this->gen_raise_trap(0, 3);
    	this->gen_sync(iss::POST_SYNC, 40);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 41: URET */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __uret(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("URET");
    	
    	this->gen_sync(iss::PRE_SYNC, 41);
    	
    	;
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("URET"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	this->gen_leave_trap(0);
    	this->gen_sync(iss::POST_SYNC, 41);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 42: SRET */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sret(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("SRET");
    	
    	this->gen_sync(iss::PRE_SYNC, 42);
    	
    	;
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("SRET"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	this->gen_leave_trap(1);
    	this->gen_sync(iss::POST_SYNC, 42);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 43: MRET */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __mret(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("MRET");
    	
    	this->gen_sync(iss::PRE_SYNC, 43);
    	
    	;
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("MRET"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	this->gen_leave_trap(3);
    	this->gen_sync(iss::POST_SYNC, 43);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 44: WFI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __wfi(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("WFI");
    	
    	this->gen_sync(iss::PRE_SYNC, 44);
    	
    	;
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("WFI"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	this->gen_wait(1);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 44);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 45: SFENCE.VMA */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sfence_vma(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("SFENCE.VMA");
    	
    	this->gen_sync(iss::PRE_SYNC, 45);
    	
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("SFENCE.VMA"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* FENCEtmp0_val = this->gen_const(32U, fld_rs1_val);
    	this->gen_write_mem(
    	    traits<ARCH>::FENCE,
    	    this->gen_const(64U, 2),
    	    this->builder.CreateZExtOrTrunc(FENCEtmp0_val,this->get_type(32)));
    	Value* FENCEtmp1_val = this->gen_const(32U, fld_rs2_val);
    	this->gen_write_mem(
    	    traits<ARCH>::FENCE,
    	    this->gen_const(64U, 3),
    	    this->builder.CreateZExtOrTrunc(FENCEtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 45);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 46: CSRRW */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __csrrw(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("CSRRW");
    	
    	this->gen_sync(iss::PRE_SYNC, 46);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint16_t fld_csr_val = 0 | (bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("CSRRW x%1$d, %2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_csr_val % (uint64_t)fld_rs1_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* rs_val_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0);
    	if(fld_rd_val != 0){
    	    Value* csr_val_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, fld_csr_val), 32/8);
    	    Value* CSRtmp0_val = rs_val_val;
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, fld_csr_val),
    	        this->builder.CreateZExtOrTrunc(CSRtmp0_val,this->get_type(32)));
    	    Value* Xtmp1_val = csr_val_val;
    	    this->builder.CreateStore(Xtmp1_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	} else {
    	    Value* CSRtmp2_val = rs_val_val;
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, fld_csr_val),
    	        this->builder.CreateZExtOrTrunc(CSRtmp2_val,this->get_type(32)));
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 46);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 47: CSRRS */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __csrrs(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("CSRRS");
    	
    	this->gen_sync(iss::PRE_SYNC, 47);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint16_t fld_csr_val = 0 | (bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("CSRRS x%1$d, %2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_csr_val % (uint64_t)fld_rs1_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* xrd_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, fld_csr_val), 32/8);
    	Value* xrs1_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0);
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = xrd_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	if(fld_rs1_val != 0){
    	    Value* CSRtmp1_val = this->builder.CreateOr(
    	        xrd_val,
    	        xrs1_val);
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, fld_csr_val),
    	        this->builder.CreateZExtOrTrunc(CSRtmp1_val,this->get_type(32)));
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 47);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 48: CSRRC */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __csrrc(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("CSRRC");
    	
    	this->gen_sync(iss::PRE_SYNC, 48);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint16_t fld_csr_val = 0 | (bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("CSRRC x%1$d, %2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_csr_val % (uint64_t)fld_rs1_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* xrd_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, fld_csr_val), 32/8);
    	Value* xrs1_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0);
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = xrd_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	if(fld_rs1_val != 0){
    	    Value* CSRtmp1_val = this->builder.CreateAnd(
    	        xrd_val,
    	        this->builder.CreateNot(xrs1_val));
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, fld_csr_val),
    	        this->builder.CreateZExtOrTrunc(CSRtmp1_val,this->get_type(32)));
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 48);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 49: CSRRWI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __csrrwi(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("CSRRWI");
    	
    	this->gen_sync(iss::PRE_SYNC, 49);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_zimm_val = 0 | (bit_sub<15,5>(instr));
    	uint16_t fld_csr_val = 0 | (bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("CSRRWI x%1$d, %2$d, 0x%3$x");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_csr_val % (uint64_t)fld_zimm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, fld_csr_val), 32/8);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	Value* CSRtmp1_val = this->gen_ext(
    	    this->gen_const(32U, fld_zimm_val),
    	    32,
    	    false);
    	this->gen_write_mem(
    	    traits<ARCH>::CSR,
    	    this->gen_const(16U, fld_csr_val),
    	    this->builder.CreateZExtOrTrunc(CSRtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 49);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 50: CSRRSI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __csrrsi(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("CSRRSI");
    	
    	this->gen_sync(iss::PRE_SYNC, 50);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_zimm_val = 0 | (bit_sub<15,5>(instr));
    	uint16_t fld_csr_val = 0 | (bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("CSRRSI x%1$d, %2$d, 0x%3$x");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_csr_val % (uint64_t)fld_zimm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, fld_csr_val), 32/8);
    	if(fld_zimm_val != 0){
    	    Value* CSRtmp0_val = this->builder.CreateOr(
    	        res_val,
    	        this->gen_ext(
    	            this->gen_const(32U, fld_zimm_val),
    	            32,
    	            false));
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, fld_csr_val),
    	        this->builder.CreateZExtOrTrunc(CSRtmp0_val,this->get_type(32)));
    	}
    	if(fld_rd_val != 0){
    	    Value* Xtmp1_val = res_val;
    	    this->builder.CreateStore(Xtmp1_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 50);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 51: CSRRCI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __csrrci(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("CSRRCI");
    	
    	this->gen_sync(iss::PRE_SYNC, 51);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_zimm_val = 0 | (bit_sub<15,5>(instr));
    	uint16_t fld_csr_val = 0 | (bit_sub<20,12>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("CSRRCI x%1$d, %2$d, 0x%3$x");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_csr_val % (uint64_t)fld_zimm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* res_val = this->gen_read_mem(traits<ARCH>::CSR, this->gen_const(16U, fld_csr_val), 32/8);
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = res_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	if(fld_zimm_val != 0){
    	    Value* CSRtmp1_val = this->builder.CreateAnd(
    	        res_val,
    	        this->builder.CreateNot(this->gen_ext(
    	            this->gen_const(32U, fld_zimm_val),
    	            32,
    	            false)));
    	    this->gen_write_mem(
    	        traits<ARCH>::CSR,
    	        this->gen_const(16U, fld_csr_val),
    	        this->builder.CreateZExtOrTrunc(CSRtmp1_val,this->get_type(32)));
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 51);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 52: MUL */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __mul(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("MUL");
    	
    	this->gen_sync(iss::PRE_SYNC, 52);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("MUL x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* res_val = this->builder.CreateMul(
    	        this->gen_ext(
    	            this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	            128,
    	            false),
    	        this->gen_ext(
    	            this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	            128,
    	            false));
    	    Value* Xtmp0_val = this->gen_ext(
    	        res_val,
    	        32,
    	        false);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 52);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 53: MULH */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __mulh(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("MULH");
    	
    	this->gen_sync(iss::PRE_SYNC, 53);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("MULH x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* res_val = this->builder.CreateMul(
    	        this->gen_ext(
    	            this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	            128,
    	            true),
    	        this->gen_ext(
    	            this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	            128,
    	            true));
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->builder.CreateLShr(
    	            res_val,
    	            this->gen_const(32U, 32)),
    	        32,
    	        false);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 53);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 54: MULHSU */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __mulhsu(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("MULHSU");
    	
    	this->gen_sync(iss::PRE_SYNC, 54);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("MULHSU x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* res_val = this->builder.CreateMul(
    	        this->gen_ext(
    	            this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	            128,
    	            true),
    	        this->gen_ext(
    	            this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	            128,
    	            false));
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->builder.CreateLShr(
    	            res_val,
    	            this->gen_const(32U, 32)),
    	        32,
    	        false);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 54);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 55: MULHU */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __mulhu(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("MULHU");
    	
    	this->gen_sync(iss::PRE_SYNC, 55);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("MULHU x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* res_val = this->builder.CreateMul(
    	        this->gen_ext(
    	            this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	            128,
    	            false),
    	        this->gen_ext(
    	            this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	            128,
    	            false));
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->builder.CreateLShr(
    	            res_val,
    	            this->gen_const(32U, 32)),
    	        32,
    	        false);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 55);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 56: DIV */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __div(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("DIV");
    	
    	this->gen_sync(iss::PRE_SYNC, 56);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("DIV x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    {
    	        llvm::BasicBlock* bbnext = llvm::BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	        llvm::BasicBlock* bb_then = llvm::BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	        llvm::BasicBlock* bb_else = llvm::BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	        // this->builder.SetInsertPoint(bb);
    	        this->gen_cond_branch(this->builder.CreateICmp(
    	            ICmpInst::ICMP_NE,
    	            this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, 0)),
    	            bb_then,
    	            bb_else);
    	        this->builder.SetInsertPoint(bb_then);
    	        {
    	            uint32_t M1_val = - 1;
    	            uint32_t MMIN_val = - 1 << 32 - 1;
    	            {
    	                llvm::BasicBlock* bbnext = llvm::BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	                llvm::BasicBlock* bb_then = llvm::BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	                llvm::BasicBlock* bb_else = llvm::BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	                // this->builder.SetInsertPoint(bb);
    	                this->gen_cond_branch(this->builder.CreateICmp(
    	                    ICmpInst::ICMP_EQ,
    	                    this->gen_ext(
    	                        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 1),
    	                        32, true),
    	                    this->gen_ext(
    	                        this->gen_const(32U, MMIN_val),
    	                        32, true)),
    	                    bb_then,
    	                    bb_else);
    	                this->builder.SetInsertPoint(bb_then);
    	                {
    	                    {
    	                        llvm::BasicBlock* bbnext = llvm::BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	                        llvm::BasicBlock* bb_then = llvm::BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	                        llvm::BasicBlock* bb_else = llvm::BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	                        // this->builder.SetInsertPoint(bb);
    	                        this->gen_cond_branch(this->builder.CreateICmp(
    	                            ICmpInst::ICMP_EQ,
    	                            this->gen_ext(
    	                                this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 2),
    	                                32, true),
    	                            this->gen_ext(
    	                                this->gen_const(32U, M1_val),
    	                                32, true)),
    	                            bb_then,
    	                            bb_else);
    	                        this->builder.SetInsertPoint(bb_then);
    	                        {
    	                            Value* Xtmp0_val = this->gen_const(32U, MMIN_val);
    	                            this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	                        }
    	                        this->builder.CreateBr(bbnext);
    	                        this->builder.SetInsertPoint(bb_else);
    	                        {
    	                            Value* Xtmp1_val = this->builder.CreateSDiv(
    	                                this->gen_ext(
    	                                    this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 3),
    	                                    32, true),
    	                                this->gen_ext(
    	                                    this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 3),
    	                                    32, true));
    	                            this->builder.CreateStore(Xtmp1_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	                        }
    	                        this->builder.CreateBr(bbnext);
    	                        bb=bbnext;
    	                    }
    	                    this->builder.SetInsertPoint(bb);
    	                }
    	                this->builder.CreateBr(bbnext);
    	                this->builder.SetInsertPoint(bb_else);
    	                {
    	                    Value* Xtmp2_val = this->builder.CreateSDiv(
    	                        this->gen_ext(
    	                            this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 2),
    	                            32, true),
    	                        this->gen_ext(
    	                            this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 2),
    	                            32, true));
    	                    this->builder.CreateStore(Xtmp2_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	                }
    	                this->builder.CreateBr(bbnext);
    	                bb=bbnext;
    	            }
    	            this->builder.SetInsertPoint(bb);
    	        }
    	        this->builder.CreateBr(bbnext);
    	        this->builder.SetInsertPoint(bb_else);
    	        {
    	            Value* Xtmp3_val = this->builder.CreateNeg(this->gen_const(32U, 1));
    	            this->builder.CreateStore(Xtmp3_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	        }
    	        this->builder.CreateBr(bbnext);
    	        bb=bbnext;
    	    }
    	    this->builder.SetInsertPoint(bb);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 56);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 57: DIVU */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __divu(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("DIVU");
    	
    	this->gen_sync(iss::PRE_SYNC, 57);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("DIVU x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    {
    	        llvm::BasicBlock* bbnext = llvm::BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	        llvm::BasicBlock* bb_then = llvm::BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	        llvm::BasicBlock* bb_else = llvm::BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	        // this->builder.SetInsertPoint(bb);
    	        this->gen_cond_branch(this->builder.CreateICmp(
    	            ICmpInst::ICMP_NE,
    	            this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, 0)),
    	            bb_then,
    	            bb_else);
    	        this->builder.SetInsertPoint(bb_then);
    	        {
    	            Value* Xtmp0_val = this->builder.CreateUDiv(
    	                this->gen_ext(
    	                    this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 1),
    	                    32,
    	                    false),
    	                this->gen_ext(
    	                    this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 1),
    	                    32,
    	                    false));
    	            this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	        }
    	        this->builder.CreateBr(bbnext);
    	        this->builder.SetInsertPoint(bb_else);
    	        {
    	            Value* Xtmp1_val = this->builder.CreateNeg(this->gen_const(32U, 1));
    	            this->builder.CreateStore(Xtmp1_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	        }
    	        this->builder.CreateBr(bbnext);
    	        bb=bbnext;
    	    }
    	    this->builder.SetInsertPoint(bb);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 57);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 58: REM */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __rem(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("REM");
    	
    	this->gen_sync(iss::PRE_SYNC, 58);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("REM x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    {
    	        llvm::BasicBlock* bbnext = llvm::BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	        llvm::BasicBlock* bb_then = llvm::BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	        llvm::BasicBlock* bb_else = llvm::BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	        // this->builder.SetInsertPoint(bb);
    	        this->gen_cond_branch(this->builder.CreateICmp(
    	            ICmpInst::ICMP_NE,
    	            this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, 0)),
    	            bb_then,
    	            bb_else);
    	        this->builder.SetInsertPoint(bb_then);
    	        {
    	            uint32_t M1_val = - 1;
    	            uint32_t MMIN_val = - 1 << 32 - 1;
    	            {
    	                llvm::BasicBlock* bbnext = llvm::BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	                llvm::BasicBlock* bb_then = llvm::BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	                llvm::BasicBlock* bb_else = llvm::BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	                // this->builder.SetInsertPoint(bb);
    	                this->gen_cond_branch(this->builder.CreateICmp(
    	                    ICmpInst::ICMP_EQ,
    	                    this->gen_ext(
    	                        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 1),
    	                        32, true),
    	                    this->gen_ext(
    	                        this->gen_const(32U, MMIN_val),
    	                        32, true)),
    	                    bb_then,
    	                    bb_else);
    	                this->builder.SetInsertPoint(bb_then);
    	                {
    	                    {
    	                        llvm::BasicBlock* bbnext = llvm::BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	                        llvm::BasicBlock* bb_then = llvm::BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	                        llvm::BasicBlock* bb_else = llvm::BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	                        // this->builder.SetInsertPoint(bb);
    	                        this->gen_cond_branch(this->builder.CreateICmp(
    	                            ICmpInst::ICMP_EQ,
    	                            this->gen_ext(
    	                                this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 2),
    	                                32, true),
    	                            this->gen_ext(
    	                                this->gen_const(32U, M1_val),
    	                                32, true)),
    	                            bb_then,
    	                            bb_else);
    	                        this->builder.SetInsertPoint(bb_then);
    	                        {
    	                            Value* Xtmp0_val = this->gen_const(32U, 0);
    	                            this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	                        }
    	                        this->builder.CreateBr(bbnext);
    	                        this->builder.SetInsertPoint(bb_else);
    	                        {
    	                            Value* Xtmp1_val = this->builder.CreateURem(
    	                                this->gen_ext(
    	                                    this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 3),
    	                                    32,
    	                                    true),
    	                                this->gen_ext(
    	                                    this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 3),
    	                                    32,
    	                                    true));
    	                            this->builder.CreateStore(Xtmp1_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	                        }
    	                        this->builder.CreateBr(bbnext);
    	                        bb=bbnext;
    	                    }
    	                    this->builder.SetInsertPoint(bb);
    	                }
    	                this->builder.CreateBr(bbnext);
    	                this->builder.SetInsertPoint(bb_else);
    	                {
    	                    Value* Xtmp2_val = this->builder.CreateURem(
    	                        this->gen_ext(
    	                            this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 2),
    	                            32,
    	                            true),
    	                        this->gen_ext(
    	                            this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 2),
    	                            32,
    	                            true));
    	                    this->builder.CreateStore(Xtmp2_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	                }
    	                this->builder.CreateBr(bbnext);
    	                bb=bbnext;
    	            }
    	            this->builder.SetInsertPoint(bb);
    	        }
    	        this->builder.CreateBr(bbnext);
    	        this->builder.SetInsertPoint(bb_else);
    	        {
    	            Value* Xtmp3_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 1);
    	            this->builder.CreateStore(Xtmp3_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	        }
    	        this->builder.CreateBr(bbnext);
    	        bb=bbnext;
    	    }
    	    this->builder.SetInsertPoint(bb);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 58);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 59: REMU */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __remu(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("REMU");
    	
    	this->gen_sync(iss::PRE_SYNC, 59);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("REMU x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    {
    	        llvm::BasicBlock* bbnext = llvm::BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	        llvm::BasicBlock* bb_then = llvm::BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	        llvm::BasicBlock* bb_else = llvm::BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
    	        // this->builder.SetInsertPoint(bb);
    	        this->gen_cond_branch(this->builder.CreateICmp(
    	            ICmpInst::ICMP_NE,
    	            this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	            this->gen_const(32U, 0)),
    	            bb_then,
    	            bb_else);
    	        this->builder.SetInsertPoint(bb_then);
    	        {
    	            Value* Xtmp0_val = this->builder.CreateURem(
    	                this->gen_ext(
    	                    this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 1),
    	                    32,
    	                    false),
    	                this->gen_ext(
    	                    this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 1),
    	                    32,
    	                    false));
    	            this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	        }
    	        this->builder.CreateBr(bbnext);
    	        this->builder.SetInsertPoint(bb_else);
    	        {
    	            Value* Xtmp1_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 1);
    	            this->builder.CreateStore(Xtmp1_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	        }
    	        this->builder.CreateBr(bbnext);
    	        bb=bbnext;
    	    }
    	    this->builder.SetInsertPoint(bb);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 59);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 60: LR.W */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __lr_w(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("LR.W");
    	
    	this->gen_sync(iss::PRE_SYNC, 60);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
    	uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("LR.W x%1$d, x%2$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	if(fld_rd_val != 0){
    	    Value* offs_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0);
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	        32,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	    Value* REStmp1_val = this->gen_ext(
    	        this->builder.CreateNeg(this->gen_const(8U, 1)),
    	        32,
    	        true);
    	    this->gen_write_mem(
    	        traits<ARCH>::RES,
    	        offs_val,
    	        this->builder.CreateZExtOrTrunc(REStmp1_val,this->get_type(32)));
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 60);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 61: SC.W */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sc_w(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("SC.W");
    	
    	this->gen_sync(iss::PRE_SYNC, 61);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
    	uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("SC.W x%1$d, x%2$d, x%3$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_read_mem(traits<ARCH>::RES, offs_val, 32/8);
    	{
    	    llvm::BasicBlock* bbnext = llvm::BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
    	    llvm::BasicBlock* bb_then = llvm::BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
    	    // this->builder.SetInsertPoint(bb);
    	    this->gen_cond_branch(this->builder.CreateICmp(
    	        ICmpInst::ICMP_NE,
    	        res1_val,
    	        this->gen_const(32U, 0)),
    	        bb_then,
    	        bbnext);
    	    this->builder.SetInsertPoint(bb_then);
    	    {
    	        Value* MEMtmp0_val = this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 1);
    	        this->gen_write_mem(
    	            traits<ARCH>::MEM,
    	            offs_val,
    	            this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(32)));
    	    }
    	    this->builder.CreateBr(bbnext);
    	    bb=bbnext;
    	}
    	this->builder.SetInsertPoint(bb);
    	if(fld_rd_val != 0){
    	    Value* Xtmp1_val = this->gen_choose(
    	        this->builder.CreateICmp(
    	            ICmpInst::ICMP_NE,
    	            res1_val,
    	            this->gen_const(32U, 0)),
    	        this->gen_const(32U, 0),
    	        this->gen_const(32U, 1),
    	        32);
    	    this->builder.CreateStore(Xtmp1_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 61);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 62: AMOSWAP.W */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amoswap_w(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("AMOSWAP.W");
    	
    	this->gen_sync(iss::PRE_SYNC, 62);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
    	uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("AMOSWAP.W x%1$d, x%2$d, x%3$d (aqu=%4$d,rel=%5$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val % (uint64_t)fld_aq_val % (uint64_t)fld_rl_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0);
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = this->gen_ext(
    	        this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	        32,
    	        true);
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	Value* MEMtmp1_val = this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 62);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 63: AMOADD.W */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amoadd_w(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("AMOADD.W");
    	
    	this->gen_sync(iss::PRE_SYNC, 63);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
    	uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("AMOADD.W x%1$d, x%2$d, x%3$d (aqu=%4$d,rel=%5$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val % (uint64_t)fld_aq_val % (uint64_t)fld_rl_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_ext(
    	    this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	    32,
    	    true);
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = res1_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	Value* res2_val = this->builder.CreateAdd(
    	    res1_val,
    	    this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0));
    	Value* MEMtmp1_val = res2_val;
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 63);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 64: AMOXOR.W */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amoxor_w(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("AMOXOR.W");
    	
    	this->gen_sync(iss::PRE_SYNC, 64);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
    	uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("AMOXOR.W x%1$d, x%2$d, x%3$d (aqu=%4$d,rel=%5$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val % (uint64_t)fld_aq_val % (uint64_t)fld_rl_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_ext(
    	    this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	    32,
    	    true);
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = res1_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	Value* res2_val = this->builder.CreateXor(
    	    res1_val,
    	    this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0));
    	Value* MEMtmp1_val = res2_val;
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 64);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 65: AMOAND.W */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amoand_w(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("AMOAND.W");
    	
    	this->gen_sync(iss::PRE_SYNC, 65);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
    	uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("AMOAND.W x%1$d, x%2$d, x%3$d (aqu=%4$d,rel=%5$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val % (uint64_t)fld_aq_val % (uint64_t)fld_rl_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_ext(
    	    this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	    32,
    	    true);
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = res1_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	Value* res2_val = this->builder.CreateAnd(
    	    res1_val,
    	    this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0));
    	Value* MEMtmp1_val = res2_val;
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 65);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 66: AMOOR.W */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amoor_w(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("AMOOR.W");
    	
    	this->gen_sync(iss::PRE_SYNC, 66);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
    	uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("AMOOR.W x%1$d, x%2$d, x%3$d (aqu=%4$d,rel=%5$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val % (uint64_t)fld_aq_val % (uint64_t)fld_rl_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_ext(
    	    this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	    32,
    	    true);
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = res1_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	Value* res2_val = this->builder.CreateOr(
    	    res1_val,
    	    this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0));
    	Value* MEMtmp1_val = res2_val;
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 66);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 67: AMOMIN.W */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amomin_w(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("AMOMIN.W");
    	
    	this->gen_sync(iss::PRE_SYNC, 67);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
    	uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("AMOMIN.W x%1$d, x%2$d, x%3$d (aqu=%4$d,rel=%5$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val % (uint64_t)fld_aq_val % (uint64_t)fld_rl_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_ext(
    	    this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	    32,
    	    true);
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = res1_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	Value* res2_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_SGT,
    	        this->gen_ext(
    	            res1_val,
    	            32, true),
    	        this->gen_ext(
    	            this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	            32, true)),
    	    this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	    res1_val,
    	    32);
    	Value* MEMtmp1_val = res2_val;
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 67);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 68: AMOMAX.W */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amomax_w(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("AMOMAX.W");
    	
    	this->gen_sync(iss::PRE_SYNC, 68);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
    	uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("AMOMAX.W x%1$d, x%2$d, x%3$d (aqu=%4$d,rel=%5$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val % (uint64_t)fld_aq_val % (uint64_t)fld_rl_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_ext(
    	    this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	    32,
    	    true);
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = res1_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	Value* res2_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_SLT,
    	        this->gen_ext(
    	            res1_val,
    	            32, true),
    	        this->gen_ext(
    	            this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	            32, true)),
    	    this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	    res1_val,
    	    32);
    	Value* MEMtmp1_val = res2_val;
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 68);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 69: AMOMINU.W */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amominu_w(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("AMOMINU.W");
    	
    	this->gen_sync(iss::PRE_SYNC, 69);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
    	uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("AMOMINU.W x%1$d, x%2$d, x%3$d (aqu=%4$d,rel=%5$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val % (uint64_t)fld_aq_val % (uint64_t)fld_rl_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_ext(
    	    this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	    32,
    	    false);
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = res1_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	Value* res2_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_UGT,
    	        res1_val,
    	        this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0)),
    	    this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	    res1_val,
    	    32);
    	Value* MEMtmp1_val = res2_val;
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 69);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 70: AMOMAXU.W */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amomaxu_w(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("AMOMAXU.W");
    	
    	this->gen_sync(iss::PRE_SYNC, 70);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
    	uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
    	uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
    	uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("AMOMAXU.W x%1$d, x%2$d, x%3$d (aqu=%4$d,rel=%5$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val % (uint64_t)fld_aq_val % (uint64_t)fld_rl_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+4;
    	
    	Value* offs_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0);
    	Value* res1_val = this->gen_ext(
    	    this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
    	    32,
    	    false);
    	if(fld_rd_val != 0){
    	    Value* Xtmp0_val = res1_val;
    	    this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	}
    	Value* res2_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_ULT,
    	        res1_val,
    	        this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0)),
    	    this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0),
    	    res1_val,
    	    32);
    	Value* MEMtmp1_val = res2_val;
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp1_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 70);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 71: C.ADDI4SPN */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_addi4spn(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.ADDI4SPN");
    	
    	this->gen_sync(iss::PRE_SYNC, 71);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<2,3>(instr));
    	uint16_t fld_imm_val = 0 | (bit_sub<5,1>(instr) << 3) | (bit_sub<6,1>(instr) << 2) | (bit_sub<7,4>(instr) << 6) | (bit_sub<11,2>(instr) << 4);
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.ADDI4SPN x%1$d, 0x%2$05x");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	if(fld_imm_val == 0){
    	    this->gen_raise_trap(0, 2);
    	}
    	Value* Xtmp0_val = this->builder.CreateAdd(
    	    this->gen_reg_load(2 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, fld_imm_val));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + 8 + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 71);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 72: C.LW */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_lw(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.LW");
    	
    	this->gen_sync(iss::PRE_SYNC, 72);
    	
    	uint8_t fld_rd_val = 0 | (bit_sub<2,3>(instr));
    	uint8_t fld_uimm_val = 0 | (bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3);
    	uint8_t fld_rs1_val = 0 | (bit_sub<7,3>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.LW x(8+%1$d), x(8+%2$d), 0x%3$05x");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_uimm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_reg_load(fld_rs1_val + 8 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, fld_uimm_val));
    	Value* Xtmp0_val = this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8);
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + 8 + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 72);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 73: C.SW */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_sw(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.SW");
    	
    	this->gen_sync(iss::PRE_SYNC, 73);
    	
    	uint8_t fld_rs2_val = 0 | (bit_sub<2,3>(instr));
    	uint8_t fld_uimm_val = 0 | (bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 2) | (bit_sub<10,3>(instr) << 3);
    	uint8_t fld_rs1_val = 0 | (bit_sub<7,3>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.SW x(8+%1$d), x(8+%2$d), 0x%3$05x");
    	    ins_fmter % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val % (uint64_t)fld_uimm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_reg_load(fld_rs1_val + 8 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, fld_uimm_val));
    	Value* MEMtmp0_val = this->gen_reg_load(fld_rs2_val + 8 + traits<ARCH>::X0, 0);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 73);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 74: C.ADDI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_addi(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.ADDI");
    	
    	this->gen_sync(iss::PRE_SYNC, 74);
    	
    	int8_t fld_imm_val = 0 | (bit_sub<2,5>(instr)) | (signed_bit_sub<12,1>(instr) << 5);
    	uint8_t fld_rs1_val = 0 | (bit_sub<7,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.ADDI x%1$d, 0x%2$05x");
    	    ins_fmter % (uint64_t)fld_rs1_val % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* Xtmp0_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, fld_imm_val));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rs1_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 74);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 75: C.NOP */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_nop(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.NOP");
    	
    	this->gen_sync(iss::PRE_SYNC, 75);
    	
    	;
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("C.NOP"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	/* TODO: describe operations for C.NOP ! */
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 75);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 76: C.JAL */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_jal(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.JAL");
    	
    	this->gen_sync(iss::PRE_SYNC, 76);
    	
    	int16_t fld_imm_val = 0 | (bit_sub<2,1>(instr) << 5) | (bit_sub<3,3>(instr) << 1) | (bit_sub<6,1>(instr) << 7) | (bit_sub<7,1>(instr) << 6) | (bit_sub<8,1>(instr) << 10) | (bit_sub<9,2>(instr) << 8) | (bit_sub<11,1>(instr) << 4) | (signed_bit_sub<12,1>(instr) << 11);
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.JAL 0x%1$05x");
    	    ins_fmter % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* Xtmp0_val = this->builder.CreateAdd(
    	    cur_pc_val,
    	    this->gen_const(32U, 2));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(1 + traits<ARCH>::X0), false);
    	Value* PC_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        cur_pc_val,
    	        32, true),
    	    this->gen_const(32U, fld_imm_val));
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(iss::POST_SYNC, 76);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 77: C.LI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_li(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.LI");
    	
    	this->gen_sync(iss::PRE_SYNC, 77);
    	
    	int8_t fld_imm_val = 0 | (bit_sub<2,5>(instr)) | (signed_bit_sub<12,1>(instr) << 5);
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.LI x%1$d, 0x%2$05x");
    	    ins_fmter % (uint64_t)fld_rd_val % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	if(fld_rd_val == 0){
    	    this->gen_raise_trap(0, 2);
    	}
    	Value* Xtmp0_val = this->gen_const(32U, fld_imm_val);
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 77);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 78: C.LUI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_lui(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.LUI");
    	
    	this->gen_sync(iss::PRE_SYNC, 78);
    	
    	int32_t fld_imm_val = 0 | (bit_sub<2,5>(instr) << 12) | (bit_sub<12,1>(instr) << 17);
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.LUI x%1$d, 0x%2$05x");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	if(fld_rd_val == 0){
    	    this->gen_raise_trap(0, 2);
    	}
    	if(fld_imm_val == 0){
    	    this->gen_raise_trap(0, 2);
    	}
    	Value* Xtmp0_val = this->gen_const(32U, fld_imm_val);
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 78);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 79: C.ADDI16SP */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_addi16sp(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.ADDI16SP");
    	
    	this->gen_sync(iss::PRE_SYNC, 79);
    	
    	int16_t fld_imm_val = 0 | (bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 7) | (bit_sub<5,1>(instr) << 6) | (bit_sub<6,1>(instr) << 4) | (signed_bit_sub<12,1>(instr) << 9);
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.ADDI16SP 0x%1$05x");
    	    ins_fmter % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* Xtmp0_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        this->gen_reg_load(2 + traits<ARCH>::X0, 0),
    	        32, true),
    	    this->gen_const(32U, fld_imm_val));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(2 + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 79);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 80: C.SRLI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_srli(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.SRLI");
    	
    	this->gen_sync(iss::PRE_SYNC, 80);
    	
    	uint8_t fld_shamt_val = 0 | (bit_sub<2,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<7,3>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.SRLI x(8+%1$d), %2$d");
    	    ins_fmter % (uint64_t)fld_rs1_val % (uint64_t)fld_shamt_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	uint8_t rs1_idx_val = fld_rs1_val + 8;
    	Value* Xtmp0_val = this->builder.CreateLShr(
    	    this->gen_reg_load(rs1_idx_val + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, fld_shamt_val));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rs1_idx_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 80);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 81: C.SRAI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_srai(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.SRAI");
    	
    	this->gen_sync(iss::PRE_SYNC, 81);
    	
    	uint8_t fld_shamt_val = 0 | (bit_sub<2,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<7,3>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.SRAI x(8+%1$d), %2$d");
    	    ins_fmter % (uint64_t)fld_rs1_val % (uint64_t)fld_shamt_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	uint8_t rs1_idx_val = fld_rs1_val + 8;
    	Value* Xtmp0_val = this->builder.CreateAShr(
    	    this->gen_reg_load(rs1_idx_val + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, fld_shamt_val));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rs1_idx_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 81);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 82: C.ANDI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_andi(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.ANDI");
    	
    	this->gen_sync(iss::PRE_SYNC, 82);
    	
    	uint8_t fld_imm_val = 0 | (bit_sub<2,5>(instr)) | (bit_sub<12,1>(instr) << 5);
    	uint8_t fld_rs1_val = 0 | (bit_sub<7,3>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.ANDI x(8+%1$d), 0x%2$05x");
    	    ins_fmter % (uint64_t)fld_rs1_val % (uint64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	uint8_t rs1_idx_val = fld_rs1_val + 8;
    	Value* Xtmp0_val = this->builder.CreateAnd(
    	    this->gen_reg_load(rs1_idx_val + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, fld_imm_val));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rs1_idx_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 82);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 83: C.SUB */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_sub(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.SUB");
    	
    	this->gen_sync(iss::PRE_SYNC, 83);
    	
    	uint8_t fld_rs2_val = 0 | (bit_sub<2,3>(instr));
    	uint8_t fld_rd_val = 0 | (bit_sub<7,3>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.SUB x(8+%1$d), x(8+%2$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	uint8_t rd_idx_val = fld_rd_val + 8;
    	Value* Xtmp0_val = this->builder.CreateSub(
    	     this->gen_reg_load(rd_idx_val + traits<ARCH>::X0, 0),
    	     this->gen_reg_load(fld_rs2_val + 8 + traits<ARCH>::X0, 0));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd_idx_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 83);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 84: C.XOR */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_xor(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.XOR");
    	
    	this->gen_sync(iss::PRE_SYNC, 84);
    	
    	uint8_t fld_rs2_val = 0 | (bit_sub<2,3>(instr));
    	uint8_t fld_rd_val = 0 | (bit_sub<7,3>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.XOR x(8+%1$d), x(8+%2$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	uint8_t rd_idx_val = fld_rd_val + 8;
    	Value* Xtmp0_val = this->builder.CreateXor(
    	    this->gen_reg_load(rd_idx_val + traits<ARCH>::X0, 0),
    	    this->gen_reg_load(fld_rs2_val + 8 + traits<ARCH>::X0, 0));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd_idx_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 84);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 85: C.OR */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_or(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.OR");
    	
    	this->gen_sync(iss::PRE_SYNC, 85);
    	
    	uint8_t fld_rs2_val = 0 | (bit_sub<2,3>(instr));
    	uint8_t fld_rd_val = 0 | (bit_sub<7,3>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.OR x(8+%1$d), x(8+%2$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	uint8_t rd_idx_val = fld_rd_val + 8;
    	Value* Xtmp0_val = this->builder.CreateOr(
    	    this->gen_reg_load(rd_idx_val + traits<ARCH>::X0, 0),
    	    this->gen_reg_load(fld_rs2_val + 8 + traits<ARCH>::X0, 0));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd_idx_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 85);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 86: C.AND */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_and(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.AND");
    	
    	this->gen_sync(iss::PRE_SYNC, 86);
    	
    	uint8_t fld_rs2_val = 0 | (bit_sub<2,3>(instr));
    	uint8_t fld_rd_val = 0 | (bit_sub<7,3>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.AND x(8+%1$d), x(8+%2$d)");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	uint8_t rd_idx_val = fld_rd_val + 8;
    	Value* Xtmp0_val = this->builder.CreateAnd(
    	    this->gen_reg_load(rd_idx_val + traits<ARCH>::X0, 0),
    	    this->gen_reg_load(fld_rs2_val + 8 + traits<ARCH>::X0, 0));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(rd_idx_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 86);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 87: C.J */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_j(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.J");
    	
    	this->gen_sync(iss::PRE_SYNC, 87);
    	
    	int16_t fld_imm_val = 0 | (bit_sub<2,1>(instr) << 5) | (bit_sub<3,3>(instr) << 1) | (bit_sub<6,1>(instr) << 7) | (bit_sub<7,1>(instr) << 6) | (bit_sub<8,1>(instr) << 10) | (bit_sub<9,2>(instr) << 8) | (bit_sub<11,1>(instr) << 4) | (signed_bit_sub<12,1>(instr) << 11);
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.J 0x%1$05x");
    	    ins_fmter % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* PC_val = this->builder.CreateAdd(
    	    this->gen_ext(
    	        cur_pc_val,
    	        32, true),
    	    this->gen_const(32U, fld_imm_val));
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(iss::POST_SYNC, 87);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 88: C.BEQZ */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_beqz(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.BEQZ");
    	
    	this->gen_sync(iss::PRE_SYNC, 88);
    	
    	int16_t fld_imm_val = 0 | (bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 1) | (bit_sub<5,2>(instr) << 6) | (bit_sub<10,2>(instr) << 3) | (signed_bit_sub<12,1>(instr) << 8);
    	uint8_t fld_rs1_val = 0 | (bit_sub<7,3>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.BEQZ x(8+%1$d), 0x%2$05x");
    	    ins_fmter % (uint64_t)fld_rs1_val % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_EQ,
    	        this->gen_reg_load(fld_rs1_val + 8 + traits<ARCH>::X0, 0),
    	        this->gen_const(32U, 0)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, fld_imm_val)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 2)),
    	    32);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(iss::POST_SYNC, 88);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 89: C.BNEZ */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_bnez(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.BNEZ");
    	
    	this->gen_sync(iss::PRE_SYNC, 89);
    	
    	int16_t fld_imm_val = 0 | (bit_sub<2,1>(instr) << 5) | (bit_sub<3,2>(instr) << 1) | (bit_sub<5,2>(instr) << 6) | (bit_sub<10,2>(instr) << 3) | (signed_bit_sub<12,1>(instr) << 8);
    	uint8_t fld_rs1_val = 0 | (bit_sub<7,3>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.BNEZ x(8+%1$d), 0x%2$05x");
    	    ins_fmter % (uint64_t)fld_rs1_val % (int64_t)fld_imm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* PC_val = this->gen_choose(
    	    this->builder.CreateICmp(
    	        ICmpInst::ICMP_NE,
    	        this->gen_reg_load(fld_rs1_val + 8 + traits<ARCH>::X0, 0),
    	        this->gen_const(32U, 0)),
    	    this->builder.CreateAdd(
    	        this->gen_ext(
    	            cur_pc_val,
    	            32, true),
    	        this->gen_const(32U, fld_imm_val)),
    	    this->builder.CreateAdd(
    	        cur_pc_val,
    	        this->gen_const(32U, 2)),
    	    32);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	Value* is_cont_v = this->builder.CreateICmp(ICmpInst::ICMP_NE, PC_val, this->gen_const(32U, pc.val), "is_cont_v");
    	this->builder.CreateStore(this->gen_ext(is_cont_v, 32U, false),	get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(iss::POST_SYNC, 89);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 90: C.SLLI */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_slli(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.SLLI");
    	
    	this->gen_sync(iss::PRE_SYNC, 90);
    	
    	uint8_t fld_shamt_val = 0 | (bit_sub<2,5>(instr));
    	uint8_t fld_rs1_val = 0 | (bit_sub<7,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.SLLI x%1$d, %2$d");
    	    ins_fmter % (uint64_t)fld_rs1_val % (uint64_t)fld_shamt_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	if(fld_rs1_val == 0){
    	    this->gen_raise_trap(0, 2);
    	}
    	Value* Xtmp0_val = this->builder.CreateShl(
    	    this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, fld_shamt_val));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rs1_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 90);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 91: C.LWSP */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_lwsp(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.LWSP");
    	
    	this->gen_sync(iss::PRE_SYNC, 91);
    	
    	uint8_t fld_uimm_val = 0 | (bit_sub<2,2>(instr) << 6) | (bit_sub<4,3>(instr) << 2) | (bit_sub<12,1>(instr) << 5);
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.LWSP x%1$d, sp, 0x%2$05x");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_uimm_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_reg_load(2 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, fld_uimm_val));
    	Value* Xtmp0_val = this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8);
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 91);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 92: C.MV */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_mv(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.MV");
    	
    	this->gen_sync(iss::PRE_SYNC, 92);
    	
    	uint8_t fld_rs2_val = 0 | (bit_sub<2,5>(instr));
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.MV x%1$d, x%2$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* Xtmp0_val = this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0);
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 92);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 93: C.JR */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_jr(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.JR");
    	
    	this->gen_sync(iss::PRE_SYNC, 93);
    	
    	uint8_t fld_rs1_val = 0 | (bit_sub<7,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.JR x%1$d");
    	    ins_fmter % (uint64_t)fld_rs1_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* PC_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()), get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(iss::POST_SYNC, 93);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 94: C.ADD */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_add(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.ADD");
    	
    	this->gen_sync(iss::PRE_SYNC, 94);
    	
    	uint8_t fld_rs2_val = 0 | (bit_sub<2,5>(instr));
    	uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.ADD x%1$d, x%2$d");
    	    ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* Xtmp0_val = this->builder.CreateAdd(
    	    this->gen_reg_load(fld_rd_val + traits<ARCH>::X0, 0),
    	    this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(fld_rd_val + traits<ARCH>::X0), false);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 94);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 95: C.JALR */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_jalr(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.JALR");
    	
    	this->gen_sync(iss::PRE_SYNC, 95);
    	
    	uint8_t fld_rs1_val = 0 | (bit_sub<7,5>(instr));
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.JALR x%1$d");
    	    ins_fmter % (uint64_t)fld_rs1_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* Xtmp0_val = this->builder.CreateAdd(
    	    cur_pc_val,
    	    this->gen_const(32U, 2));
    	this->builder.CreateStore(Xtmp0_val, get_reg_ptr(1 + traits<ARCH>::X0), false);
    	Value* PC_val = this->gen_reg_load(fld_rs1_val + traits<ARCH>::X0, 0);
    	this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    	this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()), get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    	this->gen_sync(iss::POST_SYNC, 95);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 96: C.EBREAK */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_ebreak(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.EBREAK");
    	
    	this->gen_sync(iss::PRE_SYNC, 96);
    	
    	;
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("C.EBREAK"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	this->gen_raise_trap(0, 3);
    	this->gen_sync(iss::POST_SYNC, 96);
    	this->gen_trap_check(this->leave_blk);
    	return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    /* instruction 97: C.SWSP */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __c_swsp(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("C.SWSP");
    	
    	this->gen_sync(iss::PRE_SYNC, 97);
    	
    	uint8_t fld_rs2_val = 0 | (bit_sub<2,5>(instr));
    	uint8_t fld_uimm_val = 0 | (bit_sub<7,2>(instr) << 6) | (bit_sub<9,4>(instr) << 2);
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    boost::format ins_fmter("C.SWSP x2+0x%1$05x, x%2$d");
    	    ins_fmter % (uint64_t)fld_uimm_val % (uint64_t)fld_rs2_val;
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr(ins_fmter.str()),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	Value* offs_val = this->builder.CreateAdd(
    	    this->gen_reg_load(2 + traits<ARCH>::X0, 0),
    	    this->gen_const(32U, fld_uimm_val));
    	Value* MEMtmp0_val = this->gen_reg_load(fld_rs2_val + traits<ARCH>::X0, 0);
    	this->gen_write_mem(
    	    traits<ARCH>::MEM,
    	    offs_val,
    	    this->builder.CreateZExtOrTrunc(MEMtmp0_val,this->get_type(32)));
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 97);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /* instruction 98: DII */
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __dii(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
    	bb->setName("DII");
    	
    	this->gen_sync(iss::PRE_SYNC, 98);
    	
    	;
    	if(this->disass_enabled){
    	    /* generate console output when executing the command */
    	    std::vector<llvm::Value*> args {
    	        this->core_ptr,
    	        this->gen_const(64, pc.val),
    	        this->builder.CreateGlobalStringPtr("DII"),
    	    };
    	    this->builder.CreateCall(this->mod->getFunction("print_disass"), args);
    	}
    	
    	Value* cur_pc_val = this->gen_const(32, pc.val);
    	pc=pc+2;
    	
    	this->gen_raise_trap(0, 2);
    	this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
    	this->gen_sync(iss::POST_SYNC, 98);
    	bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
    	this->gen_trap_check(bb);
    	return std::make_tuple(vm::CONT, bb);
    }
    
    /****************************************************************************
     * end opcode definitions
     ****************************************************************************/
    std::tuple<vm::continuation_e, llvm::BasicBlock *> illegal_intruction(virt_addr_t &pc, code_word_t instr,
                                                                          llvm::BasicBlock *bb) {
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
        return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
};

template <typename CODE_WORD> void debug_fn(CODE_WORD insn) {
    volatile CODE_WORD x = insn;
    insn = 2 * x;
}

template <typename ARCH> vm_impl<ARCH>::vm_impl() { this(new ARCH()); }

template <typename ARCH>
vm_impl<ARCH>::vm_impl(ARCH &core, unsigned core_id, unsigned cluster_id)
: vm::vm_base<ARCH>(core, core_id, cluster_id) {
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
std::tuple<vm::continuation_e, llvm::BasicBlock *>
vm_impl<ARCH>::gen_single_inst_behavior(virt_addr_t &pc, unsigned int &inst_cnt, llvm::BasicBlock *this_block) {
    // we fetch at max 4 byte, alignment is 2
    code_word_t insn = 0;
    const typename traits<ARCH>::addr_t upper_bits = ~traits<ARCH>::PGMASK;
    phys_addr_t paddr(pc);
    try {
        uint8_t *const data = (uint8_t *)&insn;
        paddr = this->core.v2p(pc);
        if ((pc.val & upper_bits) != ((pc.val + 2) & upper_bits)) { // we may cross a page boundary
            auto res = this->core.read(paddr, 2, data);
            if (res != iss::Ok) throw trap_access(1, pc.val);
            if ((insn & 0x3) == 0x3) { // this is a 32bit instruction
                res = this->core.read(this->core.v2p(pc + 2), 2, data + 2);
            }
        } else {
            auto res = this->core.read(paddr, 4, data);
            if (res != iss::Ok) throw trap_access(1, pc.val);
        }
    } catch (trap_access &ta) {
        throw trap_access(ta.id, pc.val);
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

template <typename ARCH> void vm_impl<ARCH>::gen_leave_behavior(llvm::BasicBlock *leave_blk) {
    this->builder.SetInsertPoint(leave_blk);
    this->builder.CreateRet(this->builder.CreateLoad(get_reg_ptr(arch::traits<ARCH>::NEXT_PC), false));
}

template <typename ARCH> void vm_impl<ARCH>::gen_raise_trap(uint16_t trap_id, uint16_t cause) {
    auto *TRAP_val = this->gen_const(32, 0x80 << 24 | (cause << 16) | trap_id);
    this->builder.CreateStore(TRAP_val, get_reg_ptr(traits<ARCH>::TRAP_STATE), true);
    this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()), get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
}

template <typename ARCH> void vm_impl<ARCH>::gen_leave_trap(unsigned lvl) {
    std::vector<llvm::Value *> args{
        this->core_ptr, llvm::ConstantInt::get(getContext(), llvm::APInt(64, lvl)),
    };
    this->builder.CreateCall(this->mod->getFunction("leave_trap"), args);
    auto *PC_val = this->gen_read_mem(traits<ARCH>::CSR, (lvl << 8) + 0x41, traits<ARCH>::XLEN / 8);
    this->builder.CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()), get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
}

template <typename ARCH> void vm_impl<ARCH>::gen_wait(unsigned type) {
    std::vector<llvm::Value *> args{
        this->core_ptr, llvm::ConstantInt::get(getContext(), llvm::APInt(64, type)),
    };
    this->builder.CreateCall(this->mod->getFunction("wait"), args);
}

template <typename ARCH> void vm_impl<ARCH>::gen_trap_behavior(llvm::BasicBlock *trap_blk) {
    this->builder.SetInsertPoint(trap_blk);
    auto *trap_state_val = this->builder.CreateLoad(get_reg_ptr(traits<ARCH>::TRAP_STATE), true);
    this->builder.CreateStore(this->gen_const(32U, std::numeric_limits<uint32_t>::max()), get_reg_ptr(traits<ARCH>::LAST_BRANCH), false);
    std::vector<llvm::Value *> args{
    	this->core_ptr,
    	this->adj_to64(trap_state_val),
        this->adj_to64(this->builder.CreateLoad(get_reg_ptr(traits<ARCH>::PC), false))};
    this->builder.CreateCall(this->mod->getFunction("enter_trap"), args);
    auto *trap_addr_val = this->builder.CreateLoad(get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    this->builder.CreateRet(trap_addr_val);
}

template <typename ARCH> inline void vm_impl<ARCH>::gen_trap_check(llvm::BasicBlock *bb) {
    auto *v = this->builder.CreateLoad(get_reg_ptr(arch::traits<ARCH>::TRAP_STATE), true);
    this->gen_cond_branch(this->builder.CreateICmp(
                              ICmpInst::ICMP_EQ, v,
                              llvm::ConstantInt::get(getContext(), llvm::APInt(v->getType()->getIntegerBitWidth(), 0))),
                          bb, this->trap_blk, 1);
}

} // namespace rv32imac

template <>
std::unique_ptr<vm_if> create<arch::rv32imac>(arch::rv32imac *core, unsigned short port, bool dump) {
    std::unique_ptr<rv32imac::vm_impl<arch::rv32imac>> ret =
        std::make_unique<rv32imac::vm_impl<arch::rv32imac>>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret.get(), port);
    return ret;
}

} // namespace iss
