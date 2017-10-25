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

#include <iss/arch/CORE_DEF_NAME.h>
#include <iss/arch/riscv_hart_msu_vp.h>
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/server.h>
#include <iss/iss.h>
#include <iss/vm_base.h>
#include <util/logging.h>

#include <boost/format.hpp>

#include <iss/debugger/riscv_target_adapter.h>

namespace iss {
namespace CORE_DEF_NAME {
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

    /* «start generated code» */
    InstructionDesriptor instr_descr[0] = {};
    /* «end generated code»  */
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

} // namespace CORE_DEF_NAME

template <>
std::unique_ptr<vm_if> create<arch::CORE_DEF_NAME>(arch::CORE_DEF_NAME *core, unsigned short port, bool dump) {
    std::unique_ptr<CORE_DEF_NAME::vm_impl<arch::CORE_DEF_NAME>> ret =
        std::make_unique<CORE_DEF_NAME::vm_impl<arch::CORE_DEF_NAME>>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret.get(), port);
    return ret;
}

} // namespace iss
