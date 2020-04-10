/*******************************************************************************
 * Copyright (C) 2020 MINRES Technologies GmbH
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

#include <iss/arch/mnrv32.h>
#include <iss/arch/riscv_hart_msu_vp.h>
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/server.h>
#include <iss/iss.h>
#include <iss/tcc/vm_base.h>
#include <util/logging.h>
#include <sstream>

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#include <array>
#include <iss/debugger/riscv_target_adapter.h>

namespace iss {
namespace tcc {
namespace mnrv32 {
using namespace iss::arch;
using namespace iss::debugger;

template <typename ARCH> class vm_impl : public iss::tcc::vm_base<ARCH> {
public:
    using super = typename iss::tcc::vm_base<ARCH>;
    using virt_addr_t = typename super::virt_addr_t;
    using phys_addr_t = typename super::phys_addr_t;
    using code_word_t = typename super::code_word_t;
    using addr_t = typename super::addr_t;

    using Value = void;
    using ConstantInt = void;
    using Type = void;

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

    using this_class = vm_impl<ARCH>;
    using compile_ret_t = std::tuple<continuation_e>;
    using compile_func = compile_ret_t (this_class::*)(virt_addr_t &pc, code_word_t instr, std::ostringstream&);

    inline const char *name(size_t index){return traits<ARCH>::reg_aliases.at(index);}

    template <typename T> inline ConstantInt *size(T type) {
        return nullptr;
    }

    void setup_module(std::string m) override {
        super::setup_module(m);
    }

    inline Value *gen_choose(Value *cond, Value *trueVal, Value *falseVal, unsigned size) {
        return super::gen_cond_assign(cond, this->gen_ext(trueVal, size), this->gen_ext(falseVal, size));
    }

    compile_ret_t gen_single_inst_behavior(virt_addr_t &, unsigned int &, std::ostringstream&) override;

    void gen_leave_behavior(std::ostringstream& os) override;

    void gen_raise_trap(uint16_t trap_id, uint16_t cause);

    void gen_leave_trap(unsigned lvl);

    void gen_wait(unsigned type);

    void gen_trap_behavior(std::ostringstream& os) override;

    void gen_trap_check(std::ostringstream& os){
        os<< fmt::format("if(*(uint32_t){})!=0) goto trap_blk;\n", get_reg_ptr(arch::traits<ARCH>::TRAP_STATE));
    }

    inline Value *gen_reg_load(unsigned i, unsigned level = 0) {
        return this->builder.CreateLoad(get_reg_ptr(i), false);
    }

    inline void gen_set_pc(std::ostringstream& os, virt_addr_t pc, unsigned reg_num) {
        os<< fmt::format("*((uint64_t*){}) = {}\n", get_reg_ptr(reg_num), pc.val);
    }

    // some compile time constants
    // enum { MASK16 = 0b1111110001100011, MASK32 = 0b11111111111100000111000001111111 };
    enum { MASK16 = 0b1111111111111111, MASK32 = 0b11111111111100000111000001111111 };
    enum { EXTR_MASK16 = MASK16 >> 2, EXTR_MASK32 = MASK32 >> 2 };
    enum { LUT_SIZE = 1 << util::bit_count(EXTR_MASK32), LUT_SIZE_C = 1 << util::bit_count(EXTR_MASK16) };

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

    const std::array<InstructionDesriptor, 52> instr_descr = {{
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
    }};
 
    /* instruction definitions */
    /* instruction 0: LUI */
    compile_ret_t __lui(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 1: AUIPC */
    compile_ret_t __auipc(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
        os<<fmt::format("AUIPC-{:%08x}:\n", pc.val);

        this->gen_sync(os, PRE_SYNC, 1);

        uint8_t rd = ((bit_sub<7,5>(instr)));
        int32_t imm = signextend<int32_t,32>((bit_sub<12,20>(instr) << 12));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {imm:#08x}", fmt::arg("mnemonic", "auipc"),
                fmt::arg("rd", name(rd)), fmt::arg("imm", imm));
            os<<fmt::format("\tprint_disass((void*){}, {}, {});\n", this->core_ptr, pc.val, mnemonic);
        }

        Value* cur_pc_val = this->gen_const(64, pc.val);
        pc=pc+4;

        if(rd != 0){
            os<<fmt::format("uint64_t res = {} + {};\n", cur_pc_val, imm);
            os<<fmt::format("*((uint64_t*){}) = ret\n", get_reg_ptr(rd + traits<ARCH>::X0));
        }
        this->gen_set_pc(os, pc, traits<ARCH>::NEXT_PC);
        this->gen_sync(os, POST_SYNC, 1);
        this->gen_trap_check(os);
        return std::make_tuple(CONT);
   }
    
    /* instruction 2: JAL */
    compile_ret_t __jal(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
        this->gen_sync(os, PRE_SYNC, 0);

        uint8_t rd = ((bit_sub<7,5>(instr)));
        uint8_t rs1 = ((bit_sub<15,5>(instr)));
        int16_t imm = signextend<int16_t,12>((bit_sub<20,12>(instr)));
        if(this->disass_enabled){
            /* generate console output when executing the command */
            auto mnemonic = fmt::format(
                "{mnemonic:10} {rd}, {rs1}, {imm:#0x}", fmt::arg("mnemonic", "jalr"),
                fmt::arg("rd", name(rd)), fmt::arg("rs1", name(rs1)), fmt::arg("imm", imm));
            os<<"print_disass(0x"<<std::hex<<this->core_ptr<<", 0x"<<pc.val<<", \""<<mnemonic<<"\");\n";
        }

        auto cur_pc_val = pc.val;
        pc=pc+4;



    }
    
    /* instruction 3: JALR */
    compile_ret_t __jalr(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 4: BEQ */
    compile_ret_t __beq(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 5: BNE */
    compile_ret_t __bne(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 6: BLT */
    compile_ret_t __blt(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 7: BGE */
    compile_ret_t __bge(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 8: BLTU */
    compile_ret_t __bltu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 9: BGEU */
    compile_ret_t __bgeu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 10: LB */
    compile_ret_t __lb(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 11: LH */
    compile_ret_t __lh(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 12: LW */
    compile_ret_t __lw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 13: LBU */
    compile_ret_t __lbu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 14: LHU */
    compile_ret_t __lhu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 15: SB */
    compile_ret_t __sb(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 16: SH */
    compile_ret_t __sh(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 17: SW */
    compile_ret_t __sw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 18: ADDI */
    compile_ret_t __addi(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 19: SLTI */
    compile_ret_t __slti(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 20: SLTIU */
    compile_ret_t __sltiu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 21: XORI */
    compile_ret_t __xori(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 22: ORI */
    compile_ret_t __ori(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 23: ANDI */
    compile_ret_t __andi(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 24: SLLI */
    compile_ret_t __slli(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 25: SRLI */
    compile_ret_t __srli(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 26: SRAI */
    compile_ret_t __srai(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 27: ADD */
    compile_ret_t __add(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 28: SUB */
    compile_ret_t __sub(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 29: SLL */
    compile_ret_t __sll(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 30: SLT */
    compile_ret_t __slt(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 31: SLTU */
    compile_ret_t __sltu(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 32: XOR */
    compile_ret_t __xor(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 33: SRL */
    compile_ret_t __srl(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 34: SRA */
    compile_ret_t __sra(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 35: OR */
    compile_ret_t __or(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 36: AND */
    compile_ret_t __and(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 37: FENCE */
    compile_ret_t __fence(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 38: FENCE_I */
    compile_ret_t __fence_i(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 39: ECALL */
    compile_ret_t __ecall(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 40: EBREAK */
    compile_ret_t __ebreak(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 41: URET */
    compile_ret_t __uret(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 42: SRET */
    compile_ret_t __sret(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 43: MRET */
    compile_ret_t __mret(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 44: WFI */
    compile_ret_t __wfi(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 45: SFENCE.VMA */
    compile_ret_t __sfence_vma(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 46: CSRRW */
    compile_ret_t __csrrw(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 47: CSRRS */
    compile_ret_t __csrrs(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 48: CSRRC */
    compile_ret_t __csrrc(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 49: CSRRWI */
    compile_ret_t __csrrwi(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 50: CSRRSI */
    compile_ret_t __csrrsi(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /* instruction 51: CSRRCI */
    compile_ret_t __csrrci(virt_addr_t& pc, code_word_t instr, std::ostringstream& os){
    }
    
    /****************************************************************************
     * end opcode definitions
     ****************************************************************************/
    compile_ret_t illegal_intruction(virt_addr_t &pc, code_word_t instr, std::ostringstream& os) {
        this->gen_sync(os, iss::PRE_SYNC, instr_descr.size());
        pc = pc + ((instr & 3) == 3 ? 4 : 2);
        gen_raise_trap(0, 2);     // illegal instruction trap
        this->gen_sync(os, iss::POST_SYNC, instr_descr.size());
        this->gen_trap_check(os);
        return BRANCH;
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
std::tuple<continuation_e>
vm_impl<ARCH>::gen_single_inst_behavior(virt_addr_t &pc, unsigned int &inst_cnt, std::ostringstream& os) {
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
    return (this->*f)(pc, insn, os);
}

template <typename ARCH> void vm_impl<ARCH>::gen_leave_behavior(std::ostringstream& os) {
}

template <typename ARCH> void vm_impl<ARCH>::gen_raise_trap(uint16_t trap_id, uint16_t cause) {
}

template <typename ARCH> void vm_impl<ARCH>::gen_leave_trap(unsigned lvl) {
}

template <typename ARCH> void vm_impl<ARCH>::gen_wait(unsigned type) {
}

template <typename ARCH> void vm_impl<ARCH>::gen_trap_behavior(std::ostringstream& os) {
}

} // namespace mnrv32

template <>
std::unique_ptr<vm_if> create<arch::mnrv32>(arch::mnrv32 *core, unsigned short port, bool dump) {
    auto ret = new mnrv32::vm_impl<arch::mnrv32>(*core, dump);
    if (port != 0) debugger::server<debugger::gdb_session>::run_server(ret, port);
    return std::unique_ptr<vm_if>(ret);
}
}
} // namespace iss
