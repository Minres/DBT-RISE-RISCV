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

#include <iss/arch/rv64ia.h>
#include <iss/arch/riscv_hart_msu_vp.h>
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/server.h>
#include <iss/iss.h>
#include <iss/vm_base.h>
#include <util/logging.h>

#include <boost/format.hpp>

#include <iss/debugger/riscv_target_adapter.h>

namespace iss {
namespace rv64ia {
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

    vm_impl(ARCH &core, bool dump = false);

    void enableDebug(bool enable) { super::sync_exec = super::ALL_SYNC; }

    target_adapter_if *accquire_target_adapter(server_if *srv) {
        debugger_if::dbg_enabled = true;
        if (vm::vm_base<ARCH>::tgt_adapter == nullptr)
            vm::vm_base<ARCH>::tgt_adapter = new riscv_target_adapter<ARCH>(srv, this->get_arch());
        return vm::vm_base<ARCH>::tgt_adapter;
    }

protected:
    template <typename T> inline llvm::ConstantInt *size(T type) {
        return llvm::ConstantInt::get(getContext(), llvm::APInt(32, type->getType()->getScalarSizeInBits()));
    }

    inline llvm::Value *gen_choose(llvm::Value *cond, llvm::Value *trueVal, llvm::Value *falseVal,
                                   unsigned size) const {
        return this->gen_cond_assign(cond, this->gen_ext(trueVal, size), this->gen_ext(falseVal, size));
    }

    std::tuple<vm::continuation_e, llvm::BasicBlock *> gen_single_inst_behavior(virt_addr_t &, unsigned int &,
                                                                                llvm::BasicBlock *) override;

    void gen_leave_behavior(llvm::BasicBlock *leave_blk) override;

    void gen_raise_trap(uint16_t trap_id, uint16_t cause);

    void gen_leave_trap(unsigned lvl);

    void gen_wait(unsigned type);

    void gen_trap_behavior(llvm::BasicBlock *) override;

    void gen_trap_check(llvm::BasicBlock *bb);

    inline void gen_set_pc(virt_addr_t pc, unsigned reg_num) {
        llvm::Value *next_pc_v = this->builder->CreateSExtOrTrunc(this->gen_const(traits<ARCH>::XLEN, pc.val),
                                                                  this->get_type(traits<ARCH>::XLEN));
        this->builder->CreateStore(next_pc_v, get_reg_ptr(reg_num), true);
    }

    inline llvm::Value *get_reg_ptr(unsigned i) {
        void *ptr = this->core.get_regs_base_ptr() + traits<ARCH>::reg_byte_offset(i);
        llvm::PointerType *ptrType = nullptr;
        switch (traits<ARCH>::reg_bit_width(i) >> 3) {
        case 8:
            ptrType = llvm::Type::getInt64PtrTy(this->mod->getContext());
            break;
        case 4:
            ptrType = llvm::Type::getInt32PtrTy(this->mod->getContext());
            break;
        case 2:
            ptrType = llvm::Type::getInt16PtrTy(this->mod->getContext());
            break;
        case 1:
            ptrType = llvm::Type::getInt8PtrTy(this->mod->getContext());
            break;
        default:
            throw std::runtime_error("unsupported access with");
            break;
        }
        return llvm::ConstantExpr::getIntToPtr(
            llvm::ConstantInt::get(this->mod->getContext(),
                                   llvm::APInt(8 /*bits*/ * sizeof(uint8_t *), reinterpret_cast<uint64_t>(ptr))),
            ptrType);
    }

    inline llvm::Value *gen_reg_load(unsigned i, unsigned level = 0) {
        //        if(level){
        return this->builder->CreateLoad(get_reg_ptr(i), false);
        //        } else {
        //            if(!this->loaded_regs[i])
        //                this->loaded_regs[i]=this->builder->CreateLoad(get_reg_ptr(i),
        //                false);
        //            return this->loaded_regs[i];
        //        }
    }

    inline void gen_set_pc(virt_addr_t pc) {
        llvm::Value *pc_l = this->builder->CreateSExt(this->gen_const(traits<ARCH>::caddr_bit_width, (unsigned)pc),
                                                      this->get_type(traits<ARCH>::caddr_bit_width));
        super::gen_set_reg(traits<ARCH>::PC, pc_l);
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
    compile_func lut[LUT_SIZE];

    std::array<compile_func, LUT_SIZE_C> lut_00, lut_01, lut_10;
    std::array<compile_func, LUT_SIZE> lut_11;

    compile_func *qlut[4]; // = {lut_00, lut_01, lut_10, lut_11};

    const uint32_t lutmasks[4] = {EXTR_MASK16, EXTR_MASK16, EXTR_MASK16, EXTR_MASK32};

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

    /* start generated code */
    const InstructionDesriptor instr_descr[75] = {
        /* entries are: valid value, valid mask, function ptr */
        /* instruction LWU */
        {32, 0b00000000000000000110000000000011, 0b00000000000000000111000001111111, &this_class::__lwu},
        /* instruction LD */
        {32, 0b00000000000000000011000000000011, 0b00000000000000000111000001111111, &this_class::__ld},
        /* instruction SD */
        {32, 0b00000000000000000011000000100011, 0b00000000000000000111000001111111, &this_class::__sd},
        /* instruction SLLI */
        {32, 0b00000000000000000001000000010011, 0b11111100000000000111000001111111, &this_class::__slli},
        /* instruction SRLI */
        {32, 0b00000000000000000101000000010011, 0b11111100000000000111000001111111, &this_class::__srli},
        /* instruction SRAI */
        {32, 0b01000000000000000101000000010011, 0b11111100000000000111000001111111, &this_class::__srai},
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
        /* instruction LR.D */
        {32, 0b00010000000000000011000000101111, 0b11111001111100000111000001111111, &this_class::__lr_d},
        /* instruction SC.D */
        {32, 0b00011000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__sc_d},
        /* instruction AMOSWAP.D */
        {32, 0b00001000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amoswap_d},
        /* instruction AMOADD.D */
        {32, 0b00000000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amoadd_d},
        /* instruction AMOXOR.D */
        {32, 0b00100000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amoxor_d},
        /* instruction AMOAND.D */
        {32, 0b01100000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amoand_d},
        /* instruction AMOOR.D */
        {32, 0b01000000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amoor_d},
        /* instruction AMOMIN.D */
        {32, 0b10000000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amomin_d},
        /* instruction AMOMAX.D */
        {32, 0b10100000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amomax_d},
        /* instruction AMOMINU.D */
        {32, 0b11000000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amominu_d},
        /* instruction AMOMAXU.D */
        {32, 0b11100000000000000011000000101111, 0b11111000000000000111000001111111, &this_class::__amomaxu_d},
    };
    // instruction LWU
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __lwu(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("LWU");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        int16_t fld_imm_val = 0 | (signed_bit_sub<20,12>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("LWU x%1$d, %2%(x%3$d)");
            ins_fmter % (uint64_t)fld_rd_val % (int64_t)fld_imm_val % (uint64_t)fld_rs1_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->builder->CreateAdd(
            this->gen_reg_load(fld_rs1_val, 0),
            this->gen_const(64U, fld_imm_val));
        if(fld_rd_val != 0){
            Value* X_rd_val = this->gen_ext(
                this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
                64,
                false);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction LD
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __ld(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("LD");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        int16_t fld_imm_val = 0 | (signed_bit_sub<20,12>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("LD x%1$d, %2%(x%3$d)");
            ins_fmter % (uint64_t)fld_rd_val % (int64_t)fld_imm_val % (uint64_t)fld_rs1_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->builder->CreateAdd(
            this->gen_reg_load(fld_rs1_val, 0),
            this->gen_const(64U, fld_imm_val));
        if(fld_rd_val != 0){
            Value* X_rd_val = this->gen_ext(
                this->gen_read_mem(traits<ARCH>::MEM, offs_val, 64/8),
                64,
                true);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SD
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sd(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SD");
    
        this->gen_sync(iss::PRE_SYNC);
    
        int16_t fld_imm_val = 0 | (bit_sub<7,5>(instr)) | (signed_bit_sub<25,7>(instr) << 5);
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("SD x%1$d, %2%(x%3$d)");
            ins_fmter % (uint64_t)fld_rs2_val % (int64_t)fld_imm_val % (uint64_t)fld_rs1_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->builder->CreateAdd(
            this->gen_reg_load(fld_rs1_val, 0),
            this->gen_const(64U, fld_imm_val));
        Value* MEM_offs_val = this->gen_reg_load(fld_rs2_val, 0);
        this->gen_write_mem(
            traits<ARCH>::MEM,
            offs_val,
            this->builder->CreateZExtOrTrunc(MEM_offs_val,this->get_type(64)));
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SLLI
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __slli(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SLLI");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_shamt_val = 0 | (bit_sub<20,6>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("SLLI x%1$d, x%2$d, %3%");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_shamt_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateShl(
                this->gen_reg_load(fld_rs1_val, 0),
                this->gen_const(64U, fld_shamt_val));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SRLI
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __srli(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SRLI");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_shamt_val = 0 | (bit_sub<20,6>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("SRLI x%1$d, x%2$d, %3%");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_shamt_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateLShr(
                this->gen_reg_load(fld_rs1_val, 0),
                this->gen_const(64U, fld_shamt_val));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SRAI
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __srai(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SRAI");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_shamt_val = 0 | (bit_sub<20,6>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("SRAI x%1$d, x%2$d, %3%");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_shamt_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateAShr(
                this->gen_reg_load(fld_rs1_val, 0),
                this->gen_const(64U, fld_shamt_val));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction ADDIW
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __addiw(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("ADDIW");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        int16_t fld_imm_val = 0 | (signed_bit_sub<20,12>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("ADDIW x%1$d, x%2$d, %3%");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (int64_t)fld_imm_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* res_val = this->builder->CreateAdd(
                this->builder->CreateTrunc(
                    this->gen_reg_load(fld_rs1_val, 0),
                    this-> get_type(32) 
                ),
                this->gen_const(32U, fld_imm_val));
            Value* X_rd_val = this->gen_ext(
                res_val,
                64,
                true);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SLLIW
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __slliw(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SLLIW");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_shamt_val = 0 | (bit_sub<20,5>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("SLLIW x%1$d, x%2$d, %3%");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_shamt_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* sh_val_val = this->builder->CreateShl(
                this->builder->CreateTrunc(
                    this->gen_reg_load(fld_rs1_val, 0),
                    this-> get_type(32) 
                ),
                this->gen_const(32U, fld_shamt_val));
            Value* X_rd_val = this->gen_ext(
                sh_val_val,
                64,
                true);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SRLIW
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __srliw(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SRLIW");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_shamt_val = 0 | (bit_sub<20,5>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("SRLIW x%1$d, x%2$d, %3%");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_shamt_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* sh_val_val = this->builder->CreateLShr(
                this->builder->CreateTrunc(
                    this->gen_reg_load(fld_rs1_val, 0),
                    this-> get_type(32) 
                ),
                this->gen_const(32U, fld_shamt_val));
            Value* X_rd_val = this->gen_ext(
                sh_val_val,
                64,
                true);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SRAIW
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sraiw(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SRAIW");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_shamt_val = 0 | (bit_sub<20,5>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("SRAIW x%1$d, x%2$d, %3%");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_shamt_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* sh_val_val = this->builder->CreateAShr(
                this->builder->CreateTrunc(
                    this->gen_reg_load(fld_rs1_val, 0),
                    this-> get_type(32) 
                ),
                this->gen_const(32U, fld_shamt_val));
            Value* X_rd_val = this->gen_ext(
                sh_val_val,
                64,
                true);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction ADDW
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __addw(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("ADDW");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr("ADDW"),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* res_val = this->builder->CreateAdd(
                this->builder->CreateTrunc(
                    this->gen_reg_load(fld_rs1_val, 0),
                    this-> get_type(32) 
                ),
                this->builder->CreateTrunc(
                    this->gen_reg_load(fld_rs2_val, 0),
                    this-> get_type(32) 
                ));
            Value* X_rd_val = this->gen_ext(
                res_val,
                64,
                true);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SUBW
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __subw(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SUBW");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr("SUBW"),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* res_val = this->builder->CreateSub(
                this->builder->CreateTrunc(
                    this->gen_reg_load(fld_rs1_val, 0),
                    this-> get_type(32) 
                ),
                this->builder->CreateTrunc(
                    this->gen_reg_load(fld_rs2_val, 0),
                    this-> get_type(32) 
                ));
            Value* X_rd_val = this->gen_ext(
                res_val,
                64,
                true);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SLLW
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sllw(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SLLW");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("SLLW x%1$d, x%2$d, x%3$d");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* mask_val = this->gen_const(32U, 31);
            Value* count_val = this->builder->CreateAnd(
                this->builder->CreateTrunc(
                    this->gen_reg_load(fld_rs2_val, 0),
                    this-> get_type(32) 
                ),
                mask_val);
            Value* sh_val_val = this->builder->CreateShl(
                this->builder->CreateTrunc(
                    this->gen_reg_load(fld_rs1_val, 0),
                    this-> get_type(32) 
                ),
                count_val);
            Value* X_rd_val = this->gen_ext(
                sh_val_val,
                64,
                true);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SRLW
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __srlw(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SRLW");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("SRLW x%1$d, x%2$d, x%3$d");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* mask_val = this->gen_const(32U, 31);
            Value* count_val = this->builder->CreateAnd(
                this->builder->CreateTrunc(
                    this->gen_reg_load(fld_rs2_val, 0),
                    this-> get_type(32) 
                ),
                mask_val);
            Value* sh_val_val = this->builder->CreateLShr(
                this->builder->CreateTrunc(
                    this->gen_reg_load(fld_rs1_val, 0),
                    this-> get_type(32) 
                ),
                count_val);
            Value* X_rd_val = this->gen_ext(
                sh_val_val,
                64,
                true);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SRAW
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sraw(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SRAW");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("SRAW x%1$d, x%2$d, x%3$d");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* mask_val = this->gen_const(32U, 31);
            Value* count_val = this->builder->CreateAnd(
                this->builder->CreateTrunc(
                    this->gen_reg_load(fld_rs2_val, 0),
                    this-> get_type(32) 
                ),
                mask_val);
            Value* sh_val_val = this->builder->CreateAShr(
                this->builder->CreateTrunc(
                    this->gen_reg_load(fld_rs1_val, 0),
                    this-> get_type(32) 
                ),
                count_val);
            Value* X_rd_val = this->gen_ext(
                sh_val_val,
                64,
                true);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction LUI
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __lui(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("LUI");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        int32_t fld_imm_val = 0 | (signed_bit_sub<12,20>(instr) << 12);
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("LUI x%1$d, 0x%2$05x");
            ins_fmter % (uint64_t)fld_rd_val % (int64_t)fld_imm_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->gen_const(64U, fld_imm_val);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction AUIPC
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __auipc(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("AUIPC");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        int32_t fld_imm_val = 0 | (signed_bit_sub<12,20>(instr) << 12);
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("AUIPC x%1%, 0x%2$08x");
            ins_fmter % (uint64_t)fld_rd_val % (int64_t)fld_imm_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateAdd(
                this->gen_reg_load(traits<ARCH>::PC, 0),
                this->gen_const(64U, fld_imm_val));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction JAL
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __jal(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("JAL");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        int32_t fld_imm_val = 0 | (bit_sub<12,8>(instr) << 12) | (bit_sub<20,1>(instr) << 11) | (bit_sub<21,10>(instr) << 1) | (signed_bit_sub<31,1>(instr) << 20);
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("JAL x%1$d, 0x%2$x");
            ins_fmter % (uint64_t)fld_rd_val % (int64_t)fld_imm_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateAdd(
                this->gen_reg_load(traits<ARCH>::PC, 0),
                this->gen_const(64U, 4));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        Value* PC_val = this->builder->CreateAdd(
            this->gen_reg_load(traits<ARCH>::PC, 0),
            this->gen_const(64U, fld_imm_val));
        this->builder->CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        this->gen_trap_check(this->leave_blk);
        return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    // instruction JALR
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __jalr(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("JALR");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateAdd(
                this->gen_reg_load(traits<ARCH>::PC, 0),
                this->gen_const(64U, 4));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        Value* ret_val = this->builder->CreateAdd(
            this->gen_reg_load(fld_rs1_val, 0),
            this->gen_const(64U, fld_imm_val));
        Value* PC_val = this->builder->CreateAnd(
            ret_val,
            this->builder->CreateNot(this->gen_const(64U, 1)));
        this->builder->CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        this->gen_trap_check(this->leave_blk);
        return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    // instruction BEQ
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __beq(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("BEQ");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* PC_val = this->gen_choose(
            this->builder->CreateICmp(
                ICmpInst::ICMP_EQ,
                this->gen_reg_load(fld_rs1_val, 0),
                this->gen_reg_load(fld_rs2_val, 0)),
            this->builder->CreateAdd(
                this->gen_reg_load(traits<ARCH>::PC, 0),
                this->gen_const(64U, fld_imm_val)),
            this->builder->CreateAdd(
                this->gen_reg_load(traits<ARCH>::PC, 0),
                this->gen_const(64U, 4)),
            64);
        this->builder->CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        this->gen_trap_check(this->leave_blk);
        return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    // instruction BNE
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __bne(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("BNE");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* PC_val = this->gen_choose(
            this->builder->CreateICmp(
                ICmpInst::ICMP_NE,
                this->gen_reg_load(fld_rs1_val, 0),
                this->gen_reg_load(fld_rs2_val, 0)),
            this->builder->CreateAdd(
                this->gen_reg_load(traits<ARCH>::PC, 0),
                this->gen_const(64U, fld_imm_val)),
            this->builder->CreateAdd(
                this->gen_reg_load(traits<ARCH>::PC, 0),
                this->gen_const(64U, 4)),
            64);
        this->builder->CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        this->gen_trap_check(this->leave_blk);
        return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    // instruction BLT
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __blt(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("BLT");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* PC_val = this->gen_choose(
            this->builder->CreateICmp(
                ICmpInst::ICMP_SLT,
                this->gen_ext(
                    this->gen_reg_load(fld_rs1_val, 0),
                    64, true),
                this->gen_ext(
                    this->gen_reg_load(fld_rs2_val, 0),
                    64, true)),
            this->builder->CreateAdd(
                this->gen_reg_load(traits<ARCH>::PC, 0),
                this->gen_const(64U, fld_imm_val)),
            this->builder->CreateAdd(
                this->gen_reg_load(traits<ARCH>::PC, 0),
                this->gen_const(64U, 4)),
            64);
        this->builder->CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        this->gen_trap_check(this->leave_blk);
        return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    // instruction BGE
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __bge(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("BGE");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* PC_val = this->gen_choose(
            this->builder->CreateICmp(
                ICmpInst::ICMP_SGE,
                this->gen_ext(
                    this->gen_reg_load(fld_rs1_val, 0),
                    64, true),
                this->gen_ext(
                    this->gen_reg_load(fld_rs2_val, 0),
                    64, true)),
            this->builder->CreateAdd(
                this->gen_reg_load(traits<ARCH>::PC, 0),
                this->gen_const(64U, fld_imm_val)),
            this->builder->CreateAdd(
                this->gen_reg_load(traits<ARCH>::PC, 0),
                this->gen_const(64U, 4)),
            64);
        this->builder->CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        this->gen_trap_check(this->leave_blk);
        return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    // instruction BLTU
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __bltu(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("BLTU");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* PC_val = this->gen_choose(
            this->builder->CreateICmp(
                ICmpInst::ICMP_ULT,
                this->gen_reg_load(fld_rs1_val, 0),
                this->gen_reg_load(fld_rs2_val, 0)),
            this->builder->CreateAdd(
                this->gen_reg_load(traits<ARCH>::PC, 0),
                this->gen_const(64U, fld_imm_val)),
            this->builder->CreateAdd(
                this->gen_reg_load(traits<ARCH>::PC, 0),
                this->gen_const(64U, 4)),
            64);
        this->builder->CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        this->gen_trap_check(this->leave_blk);
        return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    // instruction BGEU
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __bgeu(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("BGEU");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* PC_val = this->gen_choose(
            this->builder->CreateICmp(
                ICmpInst::ICMP_UGE,
                this->gen_reg_load(fld_rs1_val, 0),
                this->gen_reg_load(fld_rs2_val, 0)),
            this->builder->CreateAdd(
                this->gen_reg_load(traits<ARCH>::PC, 0),
                this->gen_const(64U, fld_imm_val)),
            this->builder->CreateAdd(
                this->gen_reg_load(traits<ARCH>::PC, 0),
                this->gen_const(64U, 4)),
            64);
        this->builder->CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        this->gen_trap_check(this->leave_blk);
        return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    // instruction LB
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __lb(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("LB");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->builder->CreateAdd(
            this->gen_reg_load(fld_rs1_val, 0),
            this->gen_const(64U, fld_imm_val));
        if(fld_rd_val != 0){
            Value* X_rd_val = this->gen_ext(
                this->gen_read_mem(traits<ARCH>::MEM, offs_val, 8/8),
                64,
                true);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction LH
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __lh(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("LH");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->builder->CreateAdd(
            this->gen_reg_load(fld_rs1_val, 0),
            this->gen_const(64U, fld_imm_val));
        if(fld_rd_val != 0){
            Value* X_rd_val = this->gen_ext(
                this->gen_read_mem(traits<ARCH>::MEM, offs_val, 16/8),
                64,
                true);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction LW
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __lw(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("LW");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->builder->CreateAdd(
            this->gen_reg_load(fld_rs1_val, 0),
            this->gen_const(64U, fld_imm_val));
        if(fld_rd_val != 0){
            Value* X_rd_val = this->gen_ext(
                this->gen_read_mem(traits<ARCH>::MEM, offs_val, 32/8),
                64,
                true);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction LBU
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __lbu(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("LBU");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->builder->CreateAdd(
            this->gen_reg_load(fld_rs1_val, 0),
            this->gen_const(64U, fld_imm_val));
        if(fld_rd_val != 0){
            Value* X_rd_val = this->gen_ext(
                this->gen_read_mem(traits<ARCH>::MEM, offs_val, 8/8),
                64,
                false);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction LHU
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __lhu(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("LHU");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->builder->CreateAdd(
            this->gen_reg_load(fld_rs1_val, 0),
            this->gen_const(64U, fld_imm_val));
        if(fld_rd_val != 0){
            Value* X_rd_val = this->gen_ext(
                this->gen_read_mem(traits<ARCH>::MEM, offs_val, 16/8),
                64,
                false);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SB
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sb(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SB");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->builder->CreateAdd(
            this->gen_reg_load(fld_rs1_val, 0),
            this->gen_const(64U, fld_imm_val));
        Value* MEM_offs_val = this->gen_reg_load(fld_rs2_val, 0);
        this->gen_write_mem(
            traits<ARCH>::MEM,
            offs_val,
            this->builder->CreateZExtOrTrunc(MEM_offs_val,this->get_type(8)));
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SH
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sh(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SH");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->builder->CreateAdd(
            this->gen_reg_load(fld_rs1_val, 0),
            this->gen_const(64U, fld_imm_val));
        Value* MEM_offs_val = this->gen_reg_load(fld_rs2_val, 0);
        this->gen_write_mem(
            traits<ARCH>::MEM,
            offs_val,
            this->builder->CreateZExtOrTrunc(MEM_offs_val,this->get_type(16)));
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SW
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sw(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SW");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->builder->CreateAdd(
            this->gen_reg_load(fld_rs1_val, 0),
            this->gen_const(64U, fld_imm_val));
        Value* MEM_offs_val = this->gen_reg_load(fld_rs2_val, 0);
        this->gen_write_mem(
            traits<ARCH>::MEM,
            offs_val,
            this->builder->CreateZExtOrTrunc(MEM_offs_val,this->get_type(32)));
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction ADDI
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __addi(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("ADDI");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateAdd(
                this->gen_reg_load(fld_rs1_val, 0),
                this->gen_const(64U, fld_imm_val));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SLTI
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __slti(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SLTI");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->gen_choose(
                this->builder->CreateICmp(
                    ICmpInst::ICMP_SLT,
                    this->gen_ext(
                        this->gen_reg_load(fld_rs1_val, 0),
                        64, true),
                    this->gen_ext(
                        this->gen_const(64U, fld_imm_val),
                        64, true)),
                this->gen_const(64U, 1),
                this->gen_const(64U, 0),
                64);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SLTIU
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sltiu(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SLTIU");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        int64_t full_imm_val = fld_imm_val;
        if(fld_rd_val != 0){
            Value* X_rd_val = this->gen_choose(
                this->builder->CreateICmp(
                    ICmpInst::ICMP_ULT,
                    this->gen_ext(
                        this->gen_reg_load(fld_rs1_val, 0),
                        64, false),
                    this->gen_ext(
                        full_imm_val,
                        64, false)),
                this->gen_const(64U, 1),
                this->gen_const(64U, 0),
                64);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction XORI
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __xori(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("XORI");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        int16_t fld_imm_val = 0 | (signed_bit_sub<20,12>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("XORI x%1$d, x%2$d, %3%");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (int64_t)fld_imm_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateXor(
                this->gen_reg_load(fld_rs1_val, 0),
                this->gen_const(64U, fld_imm_val));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction ORI
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __ori(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("ORI");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        int16_t fld_imm_val = 0 | (signed_bit_sub<20,12>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("ORI x%1$d, x%2$d, %3%");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (int64_t)fld_imm_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateOr(
                this->gen_reg_load(fld_rs1_val, 0),
                this->gen_const(64U, fld_imm_val));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction ANDI
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __andi(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("ANDI");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        int16_t fld_imm_val = 0 | (signed_bit_sub<20,12>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("ANDI x%1$d, x%2$d, %3%");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (int64_t)fld_imm_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateAnd(
                this->gen_reg_load(fld_rs1_val, 0),
                this->gen_const(64U, fld_imm_val));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction ADD
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __add(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("ADD");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateAdd(
                this->gen_reg_load(fld_rs1_val, 0),
                this->gen_reg_load(fld_rs2_val, 0));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SUB
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sub(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SUB");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateSub(
                this->gen_reg_load(fld_rs1_val, 0),
                this->gen_reg_load(fld_rs2_val, 0));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SLL
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sll(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SLL");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateShl(
                this->gen_reg_load(fld_rs1_val, 0),
                this->builder->CreateAnd(
                    this->gen_reg_load(fld_rs2_val, 0),
                    63));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SLT
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __slt(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SLT");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->gen_choose(
                this->builder->CreateICmp(
                    ICmpInst::ICMP_SLT,
                    this->gen_ext(
                        this->gen_reg_load(fld_rs1_val, 0),
                        64, true),
                    this->gen_ext(
                        this->gen_reg_load(fld_rs2_val, 0),
                        64, true)),
                this->gen_const(64U, 1),
                this->gen_const(64U, 0),
                64);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SLTU
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sltu(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SLTU");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->gen_choose(
                this->builder->CreateICmp(
                    ICmpInst::ICMP_ULT,
                    this->gen_ext(
                        this->gen_reg_load(fld_rs1_val, 0),
                        64,
                        false),
                    this->gen_ext(
                        this->gen_reg_load(fld_rs2_val, 0),
                        64,
                        false)),
                this->gen_const(64U, 1),
                this->gen_const(64U, 0),
                64);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction XOR
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __xor(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("XOR");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateXor(
                this->gen_reg_load(fld_rs1_val, 0),
                this->gen_reg_load(fld_rs2_val, 0));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SRL
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __srl(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SRL");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateLShr(
                this->gen_reg_load(fld_rs1_val, 0),
                this->builder->CreateAnd(
                    this->gen_reg_load(fld_rs2_val, 0),
                    63));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SRA
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sra(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SRA");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateAShr(
                this->gen_reg_load(fld_rs1_val, 0),
                this->builder->CreateAnd(
                    this->gen_reg_load(fld_rs2_val, 0),
                    63));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction OR
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __or(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("OR");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateOr(
                this->gen_reg_load(fld_rs1_val, 0),
                this->gen_reg_load(fld_rs2_val, 0));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction AND
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __and(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("AND");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->builder->CreateAnd(
                this->gen_reg_load(fld_rs1_val, 0),
                this->gen_reg_load(fld_rs2_val, 0));
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction FENCE
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __fence(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("FENCE");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_succ_val = 0 | (bit_sub<20,4>(instr));
        uint8_t fld_pred_val = 0 | (bit_sub<24,4>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr("FENCE"),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* FENCE_fence_val = this->builder->CreateOr(
            this->builder->CreateShl(
                this->gen_const(64U, fld_pred_val),
                this->gen_const(64U, 4)),
            this->gen_const(64U, fld_succ_val));
        this->gen_write_mem(
            traits<ARCH>::FENCE,
            (uint64_t)0,
            this->builder->CreateZExtOrTrunc(FENCE_fence_val,this->get_type(64)));
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction FENCE_I
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __fence_i(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("FENCE_I");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint16_t fld_imm_val = 0 | (bit_sub<20,12>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr("FENCE_I"),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* FENCE_fencei_val = this->gen_const(64U, fld_imm_val);
        this->gen_write_mem(
            traits<ARCH>::FENCE,
            (uint64_t)1,
            this->builder->CreateZExtOrTrunc(FENCE_fencei_val,this->get_type(64)));
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        this->gen_trap_check(this->leave_blk);
        return std::make_tuple(iss::vm::FLUSH, nullptr);
    }
    
    // instruction ECALL
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __ecall(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("ECALL");
    
        this->gen_sync(iss::PRE_SYNC);
    
        ;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr("ECALL"),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        this->gen_raise_trap(0, 11);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        this->gen_trap_check(this->leave_blk);
        return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    // instruction EBREAK
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __ebreak(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("EBREAK");
    
        this->gen_sync(iss::PRE_SYNC);
    
        ;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr("EBREAK"),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        this->gen_raise_trap(0, 3);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        this->gen_trap_check(this->leave_blk);
        return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    // instruction URET
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __uret(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("URET");
    
        this->gen_sync(iss::PRE_SYNC);
    
        ;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr("URET"),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        this->gen_leave_trap(0);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        this->gen_trap_check(this->leave_blk);
        return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    // instruction SRET
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sret(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SRET");
    
        this->gen_sync(iss::PRE_SYNC);
    
        ;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr("SRET"),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        this->gen_leave_trap(1);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        this->gen_trap_check(this->leave_blk);
        return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    // instruction MRET
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __mret(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("MRET");
    
        this->gen_sync(iss::PRE_SYNC);
    
        ;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr("MRET"),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        this->gen_leave_trap(3);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        this->gen_trap_check(this->leave_blk);
        return std::make_tuple(iss::vm::BRANCH, nullptr);
    }
    
    // instruction WFI
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __wfi(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("WFI");
    
        this->gen_sync(iss::PRE_SYNC);
    
        ;
        if(this->disass_enabled){
            /* generate console output when executing the command */
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr("WFI"),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        this->gen_wait(1);
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SFENCE.VMA
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sfence_vma(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SFENCE.VMA");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr("SFENCE.VMA"),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* FENCE_fencevmal_val = this->gen_const(64U, fld_rs1_val);
        this->gen_write_mem(
            traits<ARCH>::FENCE,
            (uint64_t)2,
            this->builder->CreateZExtOrTrunc(FENCE_fencevmal_val,this->get_type(64)));
        Value* FENCE_fencevmau_val = this->gen_const(64U, fld_rs2_val);
        this->gen_write_mem(
            traits<ARCH>::FENCE,
            (uint64_t)3,
            this->builder->CreateZExtOrTrunc(FENCE_fencevmau_val,this->get_type(64)));
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction CSRRW
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __csrrw(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("CSRRW");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* rs_val_val = this->gen_reg_load(fld_rs1_val, 0);
        if(fld_rd_val != 0){
            Value* csr_val_val = this->gen_read_mem(traits<ARCH>::CSR, fld_csr_val, 64/8);
            Value* CSR_csr_val = rs_val_val;
            this->gen_write_mem(
                traits<ARCH>::CSR,
                fld_csr_val,
                this->builder->CreateZExtOrTrunc(CSR_csr_val,this->get_type(64)));
            Value* X_rd_val = csr_val_val;
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        } else {
            Value* CSR_csr_val = rs_val_val;
            this->gen_write_mem(
                traits<ARCH>::CSR,
                fld_csr_val,
                this->builder->CreateZExtOrTrunc(CSR_csr_val,this->get_type(64)));
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction CSRRS
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __csrrs(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("CSRRS");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* xrd_val = this->gen_read_mem(traits<ARCH>::CSR, fld_csr_val, 64/8);
        Value* xrs1_val = this->gen_reg_load(fld_rs1_val, 0);
        if(fld_rd_val != 0){
            Value* X_rd_val = xrd_val;
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        if(fld_rs1_val != 0){
            Value* CSR_csr_val = this->builder->CreateOr(
                xrd_val,
                xrs1_val);
            this->gen_write_mem(
                traits<ARCH>::CSR,
                fld_csr_val,
                this->builder->CreateZExtOrTrunc(CSR_csr_val,this->get_type(64)));
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction CSRRC
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __csrrc(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("CSRRC");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* xrd_val = this->gen_read_mem(traits<ARCH>::CSR, fld_csr_val, 64/8);
        Value* xrs1_val = this->gen_reg_load(fld_rs1_val, 0);
        if(fld_rd_val != 0){
            Value* X_rd_val = xrd_val;
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        if(fld_rs1_val != 0){
            Value* CSR_csr_val = this->builder->CreateAnd(
                xrd_val,
                this->builder->CreateNot(xrs1_val));
            this->gen_write_mem(
                traits<ARCH>::CSR,
                fld_csr_val,
                this->builder->CreateZExtOrTrunc(CSR_csr_val,this->get_type(64)));
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction CSRRWI
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __csrrwi(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("CSRRWI");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* X_rd_val = this->gen_read_mem(traits<ARCH>::CSR, fld_csr_val, 64/8);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        Value* CSR_csr_val = this->gen_ext(
            this->gen_const(64U, fld_zimm_val),
            64,
            false);
        this->gen_write_mem(
            traits<ARCH>::CSR,
            fld_csr_val,
            this->builder->CreateZExtOrTrunc(CSR_csr_val,this->get_type(64)));
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction CSRRSI
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __csrrsi(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("CSRRSI");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* res_val = this->gen_read_mem(traits<ARCH>::CSR, fld_csr_val, 64/8);
        if(fld_zimm_val != 0){
            Value* CSR_csr_val = this->builder->CreateOr(
                res_val,
                this->gen_ext(
                    this->gen_const(64U, fld_zimm_val),
                    64,
                    false));
            this->gen_write_mem(
                traits<ARCH>::CSR,
                fld_csr_val,
                this->builder->CreateZExtOrTrunc(CSR_csr_val,this->get_type(64)));
        }
        if(fld_rd_val != 0){
            Value* X_rd_val = res_val;
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction CSRRCI
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __csrrci(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("CSRRCI");
    
        this->gen_sync(iss::PRE_SYNC);
    
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
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* res_val = this->gen_read_mem(traits<ARCH>::CSR, fld_csr_val, 64/8);
        if(fld_rd_val != 0){
            Value* X_rd_val = res_val;
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        if(fld_zimm_val != 0){
            Value* CSR_csr_val = this->builder->CreateAnd(
                res_val,
                this->builder->CreateNot(this->gen_ext(
                    this->gen_const(64U, fld_zimm_val),
                    64,
                    false)));
            this->gen_write_mem(
                traits<ARCH>::CSR,
                fld_csr_val,
                this->builder->CreateZExtOrTrunc(CSR_csr_val,this->get_type(64)));
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction LR.D
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __lr_d(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("LR.D");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
        uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("LR.D x%1$d, x%2$d");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        if(fld_rd_val != 0){
            Value* offs_val = this->gen_reg_load(fld_rs1_val, 0);
            Value* X_rd_val = this->gen_ext(
                this->gen_read_mem(traits<ARCH>::MEM, offs_val, 64/8),
                64,
                true);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
            Value* RES_offs_val = this->gen_ext(
                this->builder->CreateNeg(this->gen_const(8U, 1)),
                64,
                true);
            this->gen_write_mem(
                traits<ARCH>::RES,
                offs_val,
                this->builder->CreateZExtOrTrunc(RES_offs_val,this->get_type(64)));
        }
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction SC.D
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __sc_d(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("SC.D");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
        uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
        uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("SC.D x%1$d, x%2$d, x%3$d");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->gen_reg_load(fld_rs1_val, 0);
        Value* res_val = this->gen_read_mem(traits<ARCH>::RES, offs_val, 8/8);
        llvm::BasicBlock* bbnext = llvm::BasicBlock::Create(this->mod->getContext(), "endif", this->func, this->leave_blk);
        llvm::BasicBlock* bb_then = llvm::BasicBlock::Create(this->mod->getContext(), "thenbr", this->func, bbnext);
        llvm::BasicBlock* bb_else = llvm::BasicBlock::Create(this->mod->getContext(), "elsebr", this->func, bbnext);
        // this->builder->SetInsertPoint(bb);
        this->gen_cond_branch(this->builder->CreateICmp(
            ICmpInst::ICMP_NE,
            res_val,
            this->gen_const(64U, 0)),
            bb_then,
            bb_else);
        this->builder->SetInsertPoint(bb_then);
        {
            Value* MEM_offs_val = this->gen_reg_load(fld_rs2_val, 1);
            this->gen_write_mem(
                traits<ARCH>::MEM,
                offs_val,
                this->builder->CreateZExtOrTrunc(MEM_offs_val,this->get_type(64)));if(fld_rd_val != 0){
                Value* X_rd_val = this->gen_const(64U, 0);
                this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
            }
        }
        this->builder->CreateBr(bbnext);
        this->builder->SetInsertPoint(bb_else);
        {
            if(fld_rd_val != 0){
                Value* X_rd_val = this->gen_const(64U, 1);
                this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
            }
        }
        this->builder->CreateBr(bbnext);
        bb=bbnext;
        this->builder->SetInsertPoint(bb);
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction AMOSWAP.D
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amoswap_d(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("AMOSWAP.D");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
        uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
        uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("AMOSWAP.D x%1$d, x%2$d, x%3$d (aqu=%a,rel=%rl)");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->gen_reg_load(fld_rs1_val, 0);
        if(fld_rd_val != 0){
            Value* X_rd_val = this->gen_ext(
                this->gen_read_mem(traits<ARCH>::MEM, offs_val, 64/8),
                64,
                true);
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        Value* MEM_offs_val = this->gen_reg_load(fld_rs2_val, 0);
        this->gen_write_mem(
            traits<ARCH>::MEM,
            offs_val,
            this->builder->CreateZExtOrTrunc(MEM_offs_val,this->get_type(64)));
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction AMOADD.D
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amoadd_d(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("AMOADD.D");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
        uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
        uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("AMOADD.D x%1$d, x%2$d, x%3$d (aqu=%a,rel=%rl)");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->gen_reg_load(fld_rs1_val, 0);
        Value* res_val = this->gen_ext(
            this->gen_read_mem(traits<ARCH>::MEM, offs_val, 64/8),
            64,
            true);
        if(fld_rd_val != 0){
            Value* X_rd_val = res_val;
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        Value* res2_val = this->builder->CreateAdd(
            res_val,
            this->gen_reg_load(fld_rs2_val, 0));
        Value* MEM_offs_val = res2_val;
        this->gen_write_mem(
            traits<ARCH>::MEM,
            offs_val,
            this->builder->CreateZExtOrTrunc(MEM_offs_val,this->get_type(64)));
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction AMOXOR.D
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amoxor_d(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("AMOXOR.D");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
        uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
        uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("AMOXOR.D x%1$d, x%2$d, x%3$d (aqu=%a,rel=%rl)");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->gen_reg_load(fld_rs1_val, 0);
        Value* res_val = this->gen_ext(
            this->gen_read_mem(traits<ARCH>::MEM, offs_val, 64/8),
            64,
            true);
        if(fld_rd_val != 0){
            Value* X_rd_val = res_val;
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        Value* res2_val = this->builder->CreateXor(
            res_val,
            this->gen_reg_load(fld_rs2_val, 0));
        Value* MEM_offs_val = res2_val;
        this->gen_write_mem(
            traits<ARCH>::MEM,
            offs_val,
            this->builder->CreateZExtOrTrunc(MEM_offs_val,this->get_type(64)));
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction AMOAND.D
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amoand_d(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("AMOAND.D");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
        uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
        uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("AMOAND.D x%1$d, x%2$d, x%3$d (aqu=%a,rel=%rl)");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->gen_reg_load(fld_rs1_val, 0);
        Value* res_val = this->gen_ext(
            this->gen_read_mem(traits<ARCH>::MEM, offs_val, 64/8),
            64,
            true);
        if(fld_rd_val != 0){
            Value* X_rd_val = res_val;
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        Value* res2_val = this->builder->CreateAnd(
            res_val,
            this->gen_reg_load(fld_rs2_val, 0));
        Value* MEM_offs_val = res2_val;
        this->gen_write_mem(
            traits<ARCH>::MEM,
            offs_val,
            this->builder->CreateZExtOrTrunc(MEM_offs_val,this->get_type(64)));
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction AMOOR.D
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amoor_d(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("AMOOR.D");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
        uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
        uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("AMOOR.D x%1$d, x%2$d, x%3$d (aqu=%a,rel=%rl)");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->gen_reg_load(fld_rs1_val, 0);
        Value* res_val = this->gen_ext(
            this->gen_read_mem(traits<ARCH>::MEM, offs_val, 64/8),
            64,
            true);
        if(fld_rd_val != 0){
            Value* X_rd_val = res_val;
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        Value* res2_val = this->builder->CreateOr(
            res_val,
            this->gen_reg_load(fld_rs2_val, 0));
        Value* MEM_offs_val = res2_val;
        this->gen_write_mem(
            traits<ARCH>::MEM,
            offs_val,
            this->builder->CreateZExtOrTrunc(MEM_offs_val,this->get_type(64)));
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction AMOMIN.D
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amomin_d(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("AMOMIN.D");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
        uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
        uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("AMOMIN.D x%1$d, x%2$d, x%3$d (aqu=%a,rel=%rl)");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->gen_reg_load(fld_rs1_val, 0);
        Value* res_val = this->gen_ext(
            this->gen_read_mem(traits<ARCH>::MEM, offs_val, 64/8),
            64,
            true);
        if(fld_rd_val != 0){
            Value* X_rd_val = res_val;
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        Value* res2_val = this->gen_choose(
            this->builder->CreateICmp(
                ICmpInst::ICMP_SGT,
                this->gen_ext(
                    res_val,
                    64, true),
                this->gen_ext(
                    this->gen_reg_load(fld_rs2_val, 0),
                    64, true)),
            this->gen_reg_load(fld_rs2_val, 0),
            res_val,
            64);
        Value* MEM_offs_val = res_val;
        this->gen_write_mem(
            traits<ARCH>::MEM,
            offs_val,
            this->builder->CreateZExtOrTrunc(MEM_offs_val,this->get_type(64)));
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction AMOMAX.D
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amomax_d(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("AMOMAX.D");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
        uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
        uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("AMOMAX.D x%1$d, x%2$d, x%3$d (aqu=%a,rel=%rl)");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->gen_reg_load(fld_rs1_val, 0);
        Value* res_val = this->gen_ext(
            this->gen_read_mem(traits<ARCH>::MEM, offs_val, 64/8),
            64,
            true);
        if(fld_rd_val != 0){
            Value* X_rd_val = res_val;
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        Value* res2_val = this->gen_choose(
            this->builder->CreateICmp(
                ICmpInst::ICMP_SLT,
                this->gen_ext(
                    res_val,
                    64, true),
                this->gen_ext(
                    this->gen_reg_load(fld_rs2_val, 0),
                    64, true)),
            this->gen_reg_load(fld_rs2_val, 0),
            res_val,
            64);
        Value* MEM_offs_val = res2_val;
        this->gen_write_mem(
            traits<ARCH>::MEM,
            offs_val,
            this->builder->CreateZExtOrTrunc(MEM_offs_val,this->get_type(64)));
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction AMOMINU.D
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amominu_d(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("AMOMINU.D");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
        uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
        uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("AMOMINU.D x%1$d, x%2$d, x%3$d (aqu=%a,rel=%rl)");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->gen_reg_load(fld_rs1_val, 0);
        Value* res_val = this->gen_ext(
            this->gen_read_mem(traits<ARCH>::MEM, offs_val, 64/8),
            64,
            false);
        if(fld_rd_val != 0){
            Value* X_rd_val = res_val;
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        Value* res2_val = this->gen_choose(
            this->builder->CreateICmp(
                ICmpInst::ICMP_UGT,
                res_val,
                this->gen_reg_load(fld_rs2_val, 0)),
            this->gen_reg_load(fld_rs2_val, 0),
            res_val,
            64);
        Value* MEM_offs_val = res2_val;
        this->gen_write_mem(
            traits<ARCH>::MEM,
            offs_val,
            this->builder->CreateZExtOrTrunc(MEM_offs_val,this->get_type(64)));
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
    // instruction AMOMAXU.D
    std::tuple<vm::continuation_e, llvm::BasicBlock*> __amomaxu_d(virt_addr_t& pc, code_word_t instr, llvm::BasicBlock* bb){
        bb->setName("AMOMAXU.D");
    
        this->gen_sync(iss::PRE_SYNC);
    
        uint8_t fld_rd_val = 0 | (bit_sub<7,5>(instr));
        uint8_t fld_rs1_val = 0 | (bit_sub<15,5>(instr));
        uint8_t fld_rs2_val = 0 | (bit_sub<20,5>(instr));
        uint8_t fld_rl_val = 0 | (bit_sub<25,1>(instr));
        uint8_t fld_aq_val = 0 | (bit_sub<26,1>(instr));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            boost::format ins_fmter("AMOMAXU.D x%1$d, x%2$d, x%3$d (aqu=%a,rel=%rl)");
            ins_fmter % (uint64_t)fld_rd_val % (uint64_t)fld_rs1_val % (uint64_t)fld_rs2_val;
            std::vector<llvm::Value*> args {
                this->core_ptr,
                this->gen_const(64, pc.val),
                this->builder->CreateGlobalStringPtr(ins_fmter.str()),
            };
            this->builder->CreateCall(this->mod->getFunction("print_disass"), args);
        }
        pc=pc+4;
    
        Value* offs_val = this->gen_reg_load(fld_rs1_val, 0);
        Value* res_val = this->gen_ext(
            this->gen_read_mem(traits<ARCH>::MEM, offs_val, 64/8),
            64,
            false);
        if(fld_rd_val != 0){
            Value* X_rd_val = res_val;
            this->builder->CreateStore(X_rd_val, get_reg_ptr(fld_rd_val), false);
        }
        Value* res2_val = this->gen_choose(
            this->builder->CreateICmp(
                ICmpInst::ICMP_ULT,
                res_val,
                this->gen_reg_load(fld_rs2_val, 0)),
            this->gen_reg_load(fld_rs2_val, 0),
            res_val,
            64);
        Value* MEM_offs_val = res2_val;
        this->gen_write_mem(
            traits<ARCH>::MEM,
            offs_val,
            this->builder->CreateZExtOrTrunc(MEM_offs_val,this->get_type(64)));
        this->gen_set_pc(pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
        bb = llvm::BasicBlock::Create(this->mod->getContext(), "entry", this->func, this->leave_blk); /* create next BasicBlock in chain */
        this->gen_trap_check(bb);
        return std::make_tuple(vm::CONT, bb);
    }
    
/* end generated code  */
    /****************************************************************************
     * end opcode definitions
     ****************************************************************************/
    std::tuple<vm::continuation_e, llvm::BasicBlock *> illegal_intruction(virt_addr_t &pc, code_word_t instr,
                                                                          llvm::BasicBlock *bb) {
        // this->gen_sync(iss::PRE_SYNC);
        this->builder->CreateStore(this->builder->CreateLoad(get_reg_ptr(traits<ARCH>::NEXT_PC), true),
                                   get_reg_ptr(traits<ARCH>::PC), true);
        this->builder->CreateStore(
            this->builder->CreateAdd(this->builder->CreateLoad(get_reg_ptr(traits<ARCH>::ICOUNT), true),
                                     this->gen_const(64U, 1)),
            get_reg_ptr(traits<ARCH>::ICOUNT), true);
        if (this->debugging_enabled()) this->gen_sync(iss::PRE_SYNC);
        pc = pc + ((instr & 3) == 3 ? 4 : 2);
        this->gen_raise_trap(0, 2);     // illegal instruction trap
        this->gen_sync(iss::POST_SYNC); /* call post-sync if needed */
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
vm_impl<ARCH>::vm_impl(ARCH &core, bool dump)
: vm::vm_base<ARCH>(core, dump) {
    qlut[0] = lut_00.data();
    qlut[1] = lut_01.data();
    qlut[2] = lut_10.data();
    qlut[3] = lut_11.data();
    for (auto instr : instr_descr) {
        auto quantrant = instr.value & 0x3;
        expand_bit_mask(29, lutmasks[quantrant], instr.value >> 2, instr.mask >> 2, 0, qlut[quantrant], instr.op);
    }
    this->sync_exec = static_cast<sync_type>(this->sync_exec | core.needed_sync());
}

template <typename ARCH>
std::tuple<vm::continuation_e, llvm::BasicBlock *>
vm_impl<ARCH>::gen_single_inst_behavior(virt_addr_t &pc, unsigned int &inst_cnt, llvm::BasicBlock *this_block) {
    // we fetch at max 4 byte, alignment is 2
    code_word_t insn = 0;
    iss::addr_t paddr;
    const typename traits<ARCH>::addr_t upper_bits = ~traits<ARCH>::PGMASK;
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
    typename vm_impl<ARCH>::processing_pc_entry addr(*this, pc, paddr);
    ++inst_cnt;
    auto lut_val = extract_fields(insn);
    auto f = qlut[insn & 0x3][lut_val];
    if (f == nullptr) {
        f = &this_class::illegal_intruction;
    }
    return (this->*f)(pc, insn, this_block);
}

template <typename ARCH> void vm_impl<ARCH>::gen_leave_behavior(llvm::BasicBlock *leave_blk) {
    this->builder->SetInsertPoint(leave_blk);
    this->builder->CreateRet(this->builder->CreateLoad(get_reg_ptr(arch::traits<ARCH>::NEXT_PC), false));
}

template <typename ARCH> void vm_impl<ARCH>::gen_raise_trap(uint16_t trap_id, uint16_t cause) {
    auto *TRAP_val = this->gen_const(32, 0x80 << 24 | (cause << 16) | trap_id);
    this->builder->CreateStore(TRAP_val, get_reg_ptr(traits<ARCH>::TRAP_STATE), true);
}

template <typename ARCH> void vm_impl<ARCH>::gen_leave_trap(unsigned lvl) {
    std::vector<llvm::Value *> args{
        this->core_ptr, llvm::ConstantInt::get(getContext(), llvm::APInt(64, lvl)),
    };
    this->builder->CreateCall(this->mod->getFunction("leave_trap"), args);
    auto *PC_val = this->gen_read_mem(traits<ARCH>::CSR, (lvl << 8) + 0x41, traits<ARCH>::XLEN / 8);
    this->builder->CreateStore(PC_val, get_reg_ptr(traits<ARCH>::NEXT_PC), false);
}

template <typename ARCH> void vm_impl<ARCH>::gen_wait(unsigned type) {
    std::vector<llvm::Value *> args{
        this->core_ptr, llvm::ConstantInt::get(getContext(), llvm::APInt(64, type)),
    };
    this->builder->CreateCall(this->mod->getFunction("wait"), args);
}

template <typename ARCH> void vm_impl<ARCH>::gen_trap_behavior(llvm::BasicBlock *trap_blk) {
    this->builder->SetInsertPoint(trap_blk);
    auto *trap_state_val = this->builder->CreateLoad(get_reg_ptr(traits<ARCH>::TRAP_STATE), true);
    std::vector<llvm::Value *> args{this->core_ptr, this->adj_to64(trap_state_val),
                                    this->adj_to64(this->builder->CreateLoad(get_reg_ptr(traits<ARCH>::PC), false))};
    this->builder->CreateCall(this->mod->getFunction("enter_trap"), args);
    auto *trap_addr_val = this->builder->CreateLoad(get_reg_ptr(traits<ARCH>::NEXT_PC), false);
    this->builder->CreateRet(trap_addr_val);
}

template <typename ARCH> inline void vm_impl<ARCH>::gen_trap_check(llvm::BasicBlock *bb) {
    auto *v = this->builder->CreateLoad(get_reg_ptr(arch::traits<ARCH>::TRAP_STATE), true);
    this->gen_cond_branch(this->builder->CreateICmp(
                              ICmpInst::ICMP_EQ, v,
                              llvm::ConstantInt::get(getContext(), llvm::APInt(v->getType()->getIntegerBitWidth(), 0))),
                          bb, this->trap_blk, 1);
}

} // namespace rv64ia

template <>
std::unique_ptr<vm_if> create<arch::rv64ia>(arch::rv64ia *core, unsigned short port, bool dump) {
    std::unique_ptr<rv64ia::vm_impl<arch::rv64ia>> ret =
        std::make_unique<rv64ia::vm_impl<arch::rv64ia>>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret.get(), port);
    return ret;
}

} // namespace iss
