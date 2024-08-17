/*******************************************************************************
 * Copyright (C) 2017 - 2023 MINRES Technologies GmbH
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
 * Contributors:
 *       eyck@minres.com - initial implementation
 ******************************************************************************/

#ifndef _RISCV_HART_MSU_VP_H
#define _RISCV_HART_MSU_VP_H

#include "iss/arch/traits.h"
#include "iss/instrumentation_if.h"
#include "iss/log_categories.h"
#include "iss/vm_if.h"
#include "iss/vm_types.h"
#include "riscv_hart_common.h"
#include <stdexcept>
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <array>
#include <elfio/elfio.hpp>
#include <fmt/format.h>
#include <functional>
#include <iomanip>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <util/bit_field.h>
#include <util/ities.h>
#include <util/sparse_array.h>

#include <iss/semihosting/semihosting.h>

namespace iss {
namespace arch {

template <typename BASE> class riscv_hart_msu_vp : public BASE, public riscv_hart_common {
protected:
    const std::array<const char, 4> lvl = {{'U', 'S', 'H', 'M'}};
    const std::array<const char*, 16> trap_str = {{""
                                                   "Instruction address misaligned", // 0
                                                   "Instruction access fault",       // 1
                                                   "Illegal instruction",            // 2
                                                   "Breakpoint",                     // 3
                                                   "Load address misaligned",        // 4
                                                   "Load access fault",              // 5
                                                   "Store/AMO address misaligned",   // 6
                                                   "Store/AMO access fault",         // 7
                                                   "Environment call from U-mode",   // 8
                                                   "Environment call from S-mode",   // 9
                                                   "Reserved",                       // a
                                                   "Environment call from M-mode",   // b
                                                   "Instruction page fault",         // c
                                                   "Load page fault",                // d
                                                   "Reserved",                       // e
                                                   "Store/AMO page fault"}};
    const std::array<const char*, 12> irq_str = {{"User software interrupt", "Supervisor software interrupt", "Reserved",
                                                  "Machine software interrupt", "User timer interrupt", "Supervisor timer interrupt",
                                                  "Reserved", "Machine timer interrupt", "User external interrupt",
                                                  "Supervisor external interrupt", "Reserved", "Machine external interrupt"}};

public:
    using core = BASE;
    using this_class = riscv_hart_msu_vp<BASE>;
    using virt_addr_t = typename core::virt_addr_t;
    using phys_addr_t = typename core::phys_addr_t;
    using reg_t = typename core::reg_t;
    using addr_t = typename core::addr_t;

    using rd_csr_f = iss::status (this_class::*)(unsigned addr, reg_t&);
    using wr_csr_f = iss::status (this_class::*)(unsigned addr, reg_t);

    // primary template
    template <class T, class Enable = void> struct hart_state {};
    // specialization 32bit
    template <typename T> class hart_state<T, typename std::enable_if<std::is_same<T, uint32_t>::value>::type> {
    public:
        BEGIN_BF_DECL(mstatus_t, T);
        // SD bit is read-only and is set when either the FS or XS bits encode a Dirty state (i.e., SD=((FS==11) OR
        // XS==11)))
        BF_FIELD(SD, 31, 1);
        // Trap SRET
        BF_FIELD(TSR, 22, 1);
        // Timeout Wait
        BF_FIELD(TW, 21, 1);
        // Trap Virtual Memory
        BF_FIELD(TVM, 20, 1);
        // Make eXecutable Readable
        BF_FIELD(MXR, 19, 1);
        // permit Supervisor User Memory access
        BF_FIELD(SUM, 18, 1);
        // Modify PRiVilege
        BF_FIELD(MPRV, 17, 1);
        // status of additional user-mode extensions and associated state, All off/None dirty or clean, some on/None
        // dirty, some clean/Some dirty
        BF_FIELD(XS, 15, 2);
        // floating-point unit status Off/Initial/Clean/Dirty
        BF_FIELD(FS, 13, 2);
        // machine previous privilege
        BF_FIELD(MPP, 11, 2);
        // supervisor previous privilege
        BF_FIELD(SPP, 8, 1);
        // previous machine interrupt-enable
        BF_FIELD(MPIE, 7, 1);
        // previous supervisor interrupt-enable
        BF_FIELD(SPIE, 5, 1);
        // previous user interrupt-enable
        BF_FIELD(UPIE, 4, 1);
        // machine interrupt-enable
        BF_FIELD(MIE, 3, 1);
        // supervisor interrupt-enable
        BF_FIELD(SIE, 1, 1);
        // user interrupt-enable
        BF_FIELD(UIE, 0, 1);
        END_BF_DECL();

        mstatus_t mstatus;

        static const reg_t mstatus_reset_val = 0x1800;

        void write_mstatus(T val, unsigned priv_lvl) {
            auto mask = get_mask(priv_lvl);
            auto new_val = (mstatus.st.value & ~mask) | (val & mask);
            mstatus = new_val;
        }

        T satp;

        static constexpr T get_misa() { return (1UL << 30) | ISA_I | ISA_M | ISA_A | ISA_U | ISA_S | ISA_M; }

        static constexpr uint32_t get_mask(unsigned priv_lvl) {
#if __cplusplus < 201402L
            return priv_lvl == PRIV_U ? 0x80000011UL : priv_lvl == PRIV_S ? 0x800de133UL : 0x807ff9ddUL;
#else
            switch(priv_lvl) {
            case PRIV_U:
                return 0x80000011UL; // 0b1000 0000 0000 0000 0000 0000 0001 0001
            case PRIV_S:
                return 0x800de133UL; // 0b1000 0000 0000 1101 1110 0001 0011 0011
            default:
                return 0x807ff9ddUL; // 0b1000 0000 0111 1111 1111 1001 1011 1011
            }
#endif
        }

        static inline vm_info decode_vm_info(uint32_t state, T sptbr) {
            if(state == PRIV_M)
                return {0, 0, 0, 0};
            if(state <= PRIV_S)
                switch(bit_sub<31, 1>(sptbr)) {
                case 0:
                    return {0, 0, 0, 0}; // off
                case 1:
                    return {2, 10, 4, bit_sub<0, 22>(sptbr) << PGSHIFT}; // SV32
                default:
                    abort();
                }
            abort();
            return {0, 0, 0, 0}; // dummy
        }
    };
    // specialization 64bit
    template <typename T> class hart_state<T, typename std::enable_if<std::is_same<T, uint64_t>::value>::type> {
    public:
        BEGIN_BF_DECL(mstatus_t, T);
        // SD bit is read-only and is set when either the FS or XS bits encode a Dirty state (i.e., SD=((FS==11) OR
        // XS==11)))
        BF_FIELD(SD, 63, 1);
        // value of XLEN for S-mode
        BF_FIELD(SXL, 34, 2);
        // value of XLEN for U-mode
        BF_FIELD(UXL, 32, 2);
        // Trap SRET
        BF_FIELD(TSR, 22, 1);
        // Timeout Wait
        BF_FIELD(TW, 21, 1);
        // Trap Virtual Memory
        BF_FIELD(TVM, 20, 1);
        // Make eXecutable Readable
        BF_FIELD(MXR, 19, 1);
        // permit Supervisor User Memory access
        BF_FIELD(SUM, 18, 1);
        // Modify PRiVilege
        BF_FIELD(MPRV, 17, 1);
        // status of additional user-mode extensions and associated state, All off/None dirty or clean, some on/None
        // dirty, some clean/Some dirty
        BF_FIELD(XS, 15, 2);
        // floating-point unit status Off/Initial/Clean/Dirty
        BF_FIELD(FS, 13, 2);
        // machine previous privilege
        BF_FIELD(MPP, 11, 2);
        // supervisor previous privilege
        BF_FIELD(SPP, 8, 1);
        // previous machine interrupt-enable
        BF_FIELD(MPIE, 7, 1);
        // previous supervisor interrupt-enable
        BF_FIELD(SPIE, 5, 1);
        // previous user interrupt-enable
        BF_FIELD(UPIE, 4, 1);
        // machine interrupt-enable
        BF_FIELD(MIE, 3, 1);
        // supervisor interrupt-enable
        BF_FIELD(SIE, 1, 1);
        // user interrupt-enable
        BF_FIELD(UIE, 0, 1);
        END_BF_DECL();

        mstatus_t mstatus;

        static const reg_t mstatus_reset_val = 0xa00000000;

        void write_mstatus(T val, unsigned priv_lvl) {
            T old_val = mstatus;
            auto mask = get_mask(priv_lvl);
            auto new_val = (old_val & ~mask) | (val & mask);
            if((new_val & mstatus.SXL.Mask) == 0) {
                new_val |= old_val & mstatus.SXL.Mask;
            }
            if((new_val & mstatus.UXL.Mask) == 0) {
                new_val |= old_val & mstatus.UXL.Mask;
            }
            mstatus = new_val;
        }

        T satp;

        static constexpr T get_misa() { return (2ULL << 62) | ISA_I | ISA_M | ISA_A | ISA_U | ISA_S | ISA_M; }

        static constexpr T get_mask(unsigned priv_lvl) {
            uint64_t ret;
            switch(priv_lvl) {
            case PRIV_U:
                ret = 0x8000000f00000011ULL;
                break; // 0b1...0 1111 0000 0000 0111 1111 1111 1001 1011 1011
            case PRIV_S:
                ret = 0x8000000f000de133ULL;
                break; // 0b1...0 0011 0000 0000 0000 1101 1110 0001 0011 0011
            default:
                ret = 0x8000000f007ff9ddULL;
                break; // 0b1...0 1111 0000 0000 0111 1111 1111 1001 1011 1011
            }
            return ret;
        }

        static inline vm_info decode_vm_info(uint32_t state, T sptbr) {
            if(state == PRIV_M)
                return {0, 0, 0, 0};
            if(state <= PRIV_S)
                switch(bit_sub<60, 4>(sptbr)) {
                case 0:
                    return {0, 0, 0, 0}; // off
                case 8:
                    return {3, 9, 8, bit_sub<0, 44>(sptbr) << PGSHIFT}; // SV39
                case 9:
                    return {4, 9, 8, bit_sub<0, 44>(sptbr) << PGSHIFT}; // SV48
                case 10:
                    return {5, 9, 8, bit_sub<0, 44>(sptbr) << PGSHIFT}; // SV57
                case 11:
                    return {6, 9, 8, bit_sub<0, 44>(sptbr) << PGSHIFT}; // SV64
                default:
                    abort();
                }
            abort();
            return {0, 0, 0, 0}; // dummy
        }
    };
    using hart_state_type = hart_state<reg_t>;

    const typename core::reg_t PGSIZE = 1 << PGSHIFT;
    const typename core::reg_t PGMASK = PGSIZE - 1;

    constexpr reg_t get_irq_mask(size_t mode) {
        std::array<const reg_t, 4> m = {{
            0b000100010001, // U mode
            0b001100110011, // S mode
            0,
            0b101110111011 // M mode
        }};
        return m[mode];
    }

    riscv_hart_msu_vp(feature_config cfg = feature_config{});
    virtual ~riscv_hart_msu_vp() = default;

    void reset(uint64_t address) override;

    std::pair<uint64_t, bool> load_file(std::string name, int type = -1) override;

    phys_addr_t virt2phys(const iss::addr_t& addr) override;

    iss::status read(const address_type type, const access_type access, const uint32_t space, const uint64_t addr, const unsigned length,
                     uint8_t* const data) override;
    iss::status write(const address_type type, const access_type access, const uint32_t space, const uint64_t addr, const unsigned length,
                      const uint8_t* const data) override;

    uint64_t enter_trap(uint64_t flags) override { return riscv_hart_msu_vp::enter_trap(flags, fault_data, fault_data); }
    uint64_t enter_trap(uint64_t flags, uint64_t addr, uint64_t instr) override;
    uint64_t leave_trap(uint64_t flags) override;
    void wait_until(uint64_t flags) override;

    void disass_output(uint64_t pc, const std::string instr) override {
        CLOG(INFO, disass) << fmt::format("0x{:016x}    {:40} [p:{};s:0x{:x};c:{}]", pc, instr, lvl[this->reg.PRIV], (reg_t)state.mstatus,
                                          this->reg.cycle + cycle_offset);
    };

    iss::instrumentation_if* get_instrumentation_if() override { return &instr_if; }

    void set_csr(unsigned addr, reg_t val) { csr[addr & csr.page_addr_mask] = val; }

    void set_irq_num(unsigned i) { mcause_max_irq = 1 << util::ilog2(i); }

    void set_semihosting_callback(std::function<void(arch_if*, reg_t, reg_t)>& cb) { semihosting_cb = cb; };

protected:
    struct riscv_instrumentation_if : public iss::instrumentation_if {

        riscv_instrumentation_if(riscv_hart_msu_vp<BASE>& arch)
        : arch(arch) {}
        /**
         * get the name of this architecture
         *
         * @return the name of this architecture
         */
        const std::string core_type_name() const override { return traits<BASE>::core_type; }

        uint64_t get_pc() override { return arch.reg.PC; }

        uint64_t get_next_pc() override { return arch.reg.NEXT_PC; }

        uint64_t get_instr_word() override { return arch.reg.instruction; }

        uint64_t get_instr_count() override { return arch.reg.icount; }

        uint64_t get_pendig_traps() override { return arch.reg.trap_state; }

        uint64_t get_total_cycles() override { return arch.reg.cycle + arch.cycle_offset; }

        void update_last_instr_cycles(unsigned cycles) override { arch.cycle_offset += cycles - 1; }

        bool is_branch_taken() override { return arch.reg.last_branch; }

        unsigned get_reg_num() override { return traits<BASE>::NUM_REGS; }

        unsigned get_reg_size(unsigned num) override { return traits<BASE>::reg_bit_widths[num]; }

        std::unordered_map<std::string, uint64_t> get_symbol_table(std::string name) override { return arch.get_sym_table(name); }

        riscv_hart_msu_vp<BASE>& arch;
    };

    friend struct riscv_instrumentation_if;
    addr_t get_pc() { return this->reg.PC; }
    addr_t get_next_pc() { return this->reg.NEXT_PC; }

    virtual iss::status read_mem(phys_addr_t addr, unsigned length, uint8_t* const data);
    virtual iss::status write_mem(phys_addr_t addr, unsigned length, const uint8_t* const data);

    virtual iss::status read_csr(unsigned addr, reg_t& val);
    virtual iss::status write_csr(unsigned addr, reg_t val);

    hart_state_type state;
    int64_t cycle_offset{0};
    uint64_t mcycle_csr{0};
    int64_t instret_offset{0};
    uint64_t minstret_csr{0};
    reg_t fault_data;
    std::array<vm_info, 2> vm;
    uint64_t tohost = tohost_dflt;
    uint64_t fromhost = fromhost_dflt;
    bool tohost_lower_written = false;
    riscv_instrumentation_if instr_if;

    std::function<void(arch_if*, reg_t, reg_t)> semihosting_cb;

    using mem_type = util::sparse_array<uint8_t, 1ULL << 32>;
    using csr_type = util::sparse_array<typename traits<BASE>::reg_t, 1ULL << 12, 12>;
    using csr_page_type = typename csr_type::page_type;
    mem_type mem;
    csr_type csr;
    void update_vm_info();
    std::stringstream uart_buf;
    std::unordered_map<reg_t, uint64_t> ptw;
    std::unordered_map<uint64_t, uint8_t> atomic_reservation;
    std::unordered_map<unsigned, rd_csr_f> csr_rd_cb;
    std::unordered_map<unsigned, wr_csr_f> csr_wr_cb;

    std::vector<uint8_t> tcm;

    iss::status read_csr_reg(unsigned addr, reg_t& val);
    iss::status write_csr_reg(unsigned addr, reg_t val);
    iss::status read_null(unsigned addr, reg_t& val);
    iss::status write_null(unsigned addr, reg_t val) { return iss::status::Ok; }
    iss::status read_cycle(unsigned addr, reg_t& val);
    iss::status write_cycle(unsigned addr, reg_t val);
    iss::status read_instret(unsigned addr, reg_t& val);
    iss::status write_instret(unsigned addr, reg_t val);
    iss::status read_tvec(unsigned addr, reg_t& val);
    iss::status read_time(unsigned addr, reg_t& val);
    iss::status read_status(unsigned addr, reg_t& val);
    iss::status write_status(unsigned addr, reg_t val);
    iss::status write_cause(unsigned addr, reg_t val);
    iss::status read_ie(unsigned addr, reg_t& val);
    iss::status write_ie(unsigned addr, reg_t val);
    iss::status read_ip(unsigned addr, reg_t& val);
    iss::status write_ideleg(unsigned addr, reg_t val);
    iss::status write_edeleg(unsigned addr, reg_t val);
    iss::status read_hartid(unsigned addr, reg_t& val);
    iss::status write_epc(unsigned addr, reg_t val);
    iss::status read_satp(unsigned addr, reg_t& val);
    iss::status write_satp(unsigned addr, reg_t val);
    iss::status read_fcsr(unsigned addr, reg_t& val);
    iss::status write_fcsr(unsigned addr, reg_t val);

    virtual iss::status read_custom_csr_reg(unsigned addr, reg_t& val) { return iss::status::Err; };
    virtual iss::status write_custom_csr_reg(unsigned addr, reg_t val) { return iss::status::Err; };

    void register_custom_csr_rd(unsigned addr) { csr_rd_cb[addr] = &this_class::read_custom_csr_reg; }
    void register_custom_csr_wr(unsigned addr) { csr_wr_cb[addr] = &this_class::write_custom_csr_reg; }

    reg_t mhartid_reg{0x0};

    void check_interrupt();
};

template <typename BASE>
riscv_hart_msu_vp<BASE>::riscv_hart_msu_vp()
: state()
, instr_if(*this) {
    this->_has_mmu = true;
    // reset values
    csr[misa] = traits<BASE>::MISA_VAL;
    csr[mvendorid] = 0x669;
    csr[marchid] = traits<BASE>::MARCHID_VAL;
    csr[mimpid] = 1;

    uart_buf.str("");
    for(unsigned addr = mhpmcounter3; addr <= mhpmcounter31; ++addr) {
        csr_rd_cb[addr] = &this_class::read_null;
        csr_wr_cb[addr] = &this_class::write_csr_reg;
    }
    for(unsigned addr = mhpmcounter3h; addr <= mhpmcounter31h; ++addr) {
        csr_rd_cb[addr] = &this_class::read_null;
        csr_wr_cb[addr] = &this_class::write_csr_reg;
    }
    for(unsigned addr = mhpmevent3; addr <= mhpmevent31; ++addr) {
        csr_rd_cb[addr] = &this_class::read_null;
        csr_wr_cb[addr] = &this_class::write_csr_reg;
    }
    for(unsigned addr = hpmcounter3; addr <= hpmcounter31; ++addr) {
        csr_rd_cb[addr] = &this_class::read_null;
    }
    for(unsigned addr = cycleh; addr <= hpmcounter31h; ++addr) {
        csr_rd_cb[addr] = &this_class::read_null;
        // csr_wr_cb[addr] = &this_class::write_csr_reg;
    }
    // common regs
    const std::array<unsigned, 22> addrs{{misa,  mvendorid, marchid,  mimpid, mepc,     mtvec,   mscratch, mcause,
                                          mtval, mscratch,  sepc,     stvec,  sscratch, scause,  stval,    sscratch,
                                          uepc,  utvec,     uscratch, ucause, utval,    uscratch}};
    for(auto addr : addrs) {
        csr_rd_cb[addr] = &this_class::read_csr_reg;
        csr_wr_cb[addr] = &this_class::write_csr_reg;
    }
    // special handling & overrides
    csr_rd_cb[time] = &this_class::read_time;
    if(traits<BASE>::XLEN == 32)
        csr_rd_cb[timeh] = &this_class::read_time;
    csr_rd_cb[cycle] = &this_class::read_cycle;
    if(traits<BASE>::XLEN == 32)
        csr_rd_cb[cycleh] = &this_class::read_cycle;
    csr_rd_cb[instret] = &this_class::read_instret;
    if(traits<BASE>::XLEN == 32)
        csr_rd_cb[instreth] = &this_class::read_instret;

    csr_rd_cb[mcycle] = &this_class::read_cycle;
    csr_wr_cb[mcycle] = &this_class::write_cycle;
    if(traits<BASE>::XLEN == 32)
        csr_rd_cb[mcycleh] = &this_class::read_cycle;
    if(traits<BASE>::XLEN == 32)
        csr_wr_cb[mcycleh] = &this_class::write_cycle;
    csr_rd_cb[minstret] = &this_class::read_instret;
    csr_wr_cb[minstret] = &this_class::write_instret;
    if(traits<BASE>::XLEN == 32)
        csr_rd_cb[minstreth] = &this_class::read_instret;
    if(traits<BASE>::XLEN == 32)
        csr_wr_cb[minstreth] = &this_class::write_instret;
    csr_rd_cb[mstatus] = &this_class::read_status;
    csr_wr_cb[mstatus] = &this_class::write_status;
    csr_wr_cb[mcause] = &this_class::write_cause;
    csr_rd_cb[sstatus] = &this_class::read_status;
    csr_wr_cb[sstatus] = &this_class::write_status;
    csr_wr_cb[scause] = &this_class::write_cause;
    csr_rd_cb[ustatus] = &this_class::read_status;
    csr_wr_cb[ustatus] = &this_class::write_status;
    csr_wr_cb[ucause] = &this_class::write_cause;
    csr_rd_cb[mtvec] = &this_class::read_tvec;
    csr_rd_cb[stvec] = &this_class::read_tvec;
    csr_rd_cb[utvec] = &this_class::read_tvec;
    csr_wr_cb[mepc] = &this_class::write_epc;
    csr_wr_cb[sepc] = &this_class::write_epc;
    csr_wr_cb[uepc] = &this_class::write_epc;
    csr_rd_cb[mip] = &this_class::read_ip;
    csr_wr_cb[mip] = &this_class::write_null;
    csr_rd_cb[sip] = &this_class::read_ip;
    csr_wr_cb[sip] = &this_class::write_null;
    csr_rd_cb[uip] = &this_class::read_ip;
    csr_wr_cb[uip] = &this_class::write_null;
    csr_rd_cb[mie] = &this_class::read_ie;
    csr_wr_cb[mie] = &this_class::write_ie;
    csr_rd_cb[sie] = &this_class::read_ie;
    csr_wr_cb[sie] = &this_class::write_ie;
    csr_rd_cb[uie] = &this_class::read_ie;
    csr_wr_cb[uie] = &this_class::write_ie;
    csr_rd_cb[mhartid] = &this_class::read_hartid;
    csr_rd_cb[mcounteren] = &this_class::read_null;
    csr_wr_cb[mcounteren] = &this_class::write_null;
    csr_wr_cb[misa] = &this_class::write_null;
    csr_wr_cb[mvendorid] = &this_class::write_null;
    csr_wr_cb[marchid] = &this_class::write_null;
    csr_wr_cb[mimpid] = &this_class::write_null;
    csr_rd_cb[satp] = &this_class::read_satp;
    csr_wr_cb[satp] = &this_class::write_satp;
    csr_rd_cb[fcsr] = &this_class::read_fcsr;
    csr_wr_cb[fcsr] = &this_class::write_fcsr;
    csr_rd_cb[fflags] = &this_class::read_fcsr;
    csr_wr_cb[fflags] = &this_class::write_fcsr;
    csr_rd_cb[frm] = &this_class::read_fcsr;
    csr_wr_cb[frm] = &this_class::write_fcsr;
}

template <typename BASE> std::pair<uint64_t, bool> riscv_hart_msu_vp<BASE>::load_file(std::string name, int type) {
    FILE* fp = fopen(name.c_str(), "r");
    if(fp) {
        std::array<char, 5> buf;
        auto n = fread(buf.data(), 1, 4, fp);
        fclose(fp);
        if(n != 4)
            throw std::runtime_error("input file has insufficient size");
        buf[4] = 0;
        if(strcmp(buf.data() + 1, "ELF") == 0) {
            // Create elfio reader
            ELFIO::elfio reader;
            // Load ELF data
            if(!reader.load(name))
                throw std::runtime_error("could not process elf file");
            // check elf properties
            if(reader.get_class() != ELFCLASS32)
                if(sizeof(reg_t) == 4)
                    throw std::runtime_error("wrong elf class in file");
            if(reader.get_type() != ET_EXEC)
                throw std::runtime_error("wrong elf type in file");
            if(reader.get_machine() != EM_RISCV)
                throw std::runtime_error("wrong elf machine in file");
            auto entry = reader.get_entry();
            for(const auto pseg : reader.segments) {
                const auto fsize = pseg->get_file_size(); // 0x42c/0x0
                const auto seg_data = pseg->get_data();
                const auto type = pseg->get_type();
                if(type == 1 && fsize > 0) {
                    auto res = this->write(iss::address_type::PHYSICAL, iss::access_type::DEBUG_WRITE, traits<BASE>::MEM,
                                           pseg->get_physical_address(), fsize, reinterpret_cast<const uint8_t* const>(seg_data));
                    if(res != iss::Ok)
                        CPPLOG(ERR) << "problem writing " << fsize << "bytes to 0x" << std::hex << pseg->get_physical_address();
                }
            }
            for(const auto sec : reader.sections) {
                if(sec->get_name() == ".symtab") {
                    if(SHT_SYMTAB == sec->get_type() || SHT_DYNSYM == sec->get_type()) {
                        ELFIO::symbol_section_accessor symbols(reader, sec);
                        auto sym_no = symbols.get_symbols_num();
                        std::string name;
                        ELFIO::Elf64_Addr value = 0;
                        ELFIO::Elf_Xword size = 0;
                        unsigned char bind = 0;
                        unsigned char type = 0;
                        ELFIO::Elf_Half section = 0;
                        unsigned char other = 0;
                        for(auto i = 0U; i < sym_no; ++i) {
                            symbols.get_symbol(i, name, value, size, bind, type, section, other);
                            if(name == "tohost") {
                                tohost = value;
                            } else if(name == "fromhost") {
                                fromhost = value;
                            }
                        }
                    }
                } else if(sec->get_name() == ".tohost") {
                    tohost = sec->get_address();
                    fromhost = tohost + 0x40;
                }
            }
            return std::make_pair(entry, true);
        }
        throw std::runtime_error(fmt::format("memory load file {} is not a valid elf file", name));
    }
    throw std::runtime_error(fmt::format("memory load file not found, check if {} is a valid file", name));
}

template <typename BASE>
iss::status riscv_hart_msu_vp<BASE>::read(const address_type type, const access_type access, const uint32_t space, const uint64_t addr,
                                          const unsigned length, uint8_t* const data) {
#ifndef NDEBUG
    if(access && iss::access_type::DEBUG) {
        CPPLOG(TRACEALL) << "debug read of " << length << " bytes @addr 0x" << std::hex << addr;
    } else if(access && iss::access_type::FETCH) {
        CPPLOG(TRACEALL) << "fetch of " << length << " bytes  @addr 0x" << std::hex << addr;
    } else {
        CPPLOG(TRACE) << "read of " << length << " bytes  @addr 0x" << std::hex << addr;
    }
#endif
    try {
        switch(space) {
        case traits<BASE>::MEM: {
            auto alignment = is_fetch(access) ? (traits<BASE>::MISA_VAL & 0x100 ? 2 : 4) : length;
            if(unlikely(is_fetch(access) && (addr & (alignment - 1)))) {
                fault_data = addr;
                if(access && iss::access_type::DEBUG)
                    throw trap_access(0, addr);
                this->reg.trap_state = (1 << 31); // issue trap 0
                return iss::Err;
            }
            try {
                if(!is_debug(access) && (addr & (alignment - 1))) {
                    this->reg.trap_state = 1 << 31 | 4 << 16;
                    fault_data = addr;
                    return iss::Err;
                }
                if(unlikely((addr & ~PGMASK) != ((addr + length - 1) & ~PGMASK))) { // we may cross a page boundary
                    vm_info vm = hart_state_type::decode_vm_info(this->reg.PRIV, state.satp);
                    if(vm.levels != 0) { // VM is active
                        auto split_addr = (addr + length) & ~PGMASK;
                        auto len1 = split_addr - addr;
                        auto res = read(type, access, space, addr, len1, data);
                        if(res == iss::Ok)
                            res = read(type, access, space, split_addr, length - len1, data + len1);
                        return res;
                    }
                }
                auto res = read_mem(BASE::v2p(iss::addr_t{access, type, space, addr}), length, data);
                if(unlikely(res != iss::Ok && (access & access_type::DEBUG) == 0)) {
                    this->reg.trap_state = (1 << 31) | (5 << 16); // issue trap 5 (load access fault
                    fault_data = addr;
                }
                return res;
            } catch(trap_access& ta) {
                if((access & access_type::DEBUG) == 0) {
                    this->reg.trap_state = (1UL << 31) | ta.id;
                    fault_data = ta.addr;
                }
                return iss::Err;
            }
        } break;
        case traits<BASE>::CSR: {
            if(length != sizeof(reg_t))
                return iss::Err;
            return read_csr(addr, *reinterpret_cast<reg_t* const>(data));
        } break;
        case traits<BASE>::FENCE: {
            if((addr + length) > mem.size())
                return iss::Err;
            switch(addr) {
            case 2:   // SFENCE:VMA lower
            case 3: { // SFENCE:VMA upper
                auto tvm = state.mstatus.TVM;
                if(this->reg.PRIV == PRIV_S & tvm != 0) {
                    this->reg.trap_state = (1 << 31) | (2 << 16);
                    this->fault_data = this->reg.PC;
                    return iss::Err;
                }
                return iss::Ok;
            }
            }
        } break;
        case traits<BASE>::RES: {
            auto it = atomic_reservation.find(addr);
            if(it != atomic_reservation.end() && it->second != 0) {
                memset(data, 0xff, length);
                atomic_reservation.erase(addr);
            } else
                memset(data, 0, length);
        } break;
        default:
            return iss::Err; // assert("Not supported");
        }
        return iss::Ok;
    } catch(trap_access& ta) {
        if((access & access_type::DEBUG) == 0) {
            this->reg.trap_state = (1UL << 31) | ta.id;
            fault_data = ta.addr;
        }
        return iss::Err;
    }
}

template <typename BASE>
iss::status riscv_hart_msu_vp<BASE>::write(const address_type type, const access_type access, const uint32_t space, const uint64_t addr,
                                           const unsigned length, const uint8_t* const data) {
#ifndef NDEBUG
    const char* prefix = (access && iss::access_type::DEBUG) ? "debug " : "";
    switch(length) {
    case 8:
        CPPLOG(TRACE) << prefix << "write of " << length << " bytes (0x" << std::hex << *(uint64_t*)&data[0] << std::dec << ") @addr 0x"
                      << std::hex << addr;
        break;
    case 4:
        CPPLOG(TRACE) << prefix << "write of " << length << " bytes (0x" << std::hex << *(uint32_t*)&data[0] << std::dec << ") @addr 0x"
                      << std::hex << addr;
        break;
    case 2:
        CPPLOG(TRACE) << prefix << "write of " << length << " bytes (0x" << std::hex << *(uint16_t*)&data[0] << std::dec << ") @addr 0x"
                      << std::hex << addr;
        break;
    case 1:
        CPPLOG(TRACE) << prefix << "write of " << length << " bytes (0x" << std::hex << (uint16_t)data[0] << std::dec << ") @addr 0x"
                      << std::hex << addr;
        break;
    default:
        CPPLOG(TRACE) << prefix << "write of " << length << " bytes @addr " << addr;
    }
#endif
    try {
        switch(space) {
        case traits<BASE>::MEM: {
            if(unlikely((access && iss::access_type::FETCH) && (addr & 0x1) == 1)) {
                fault_data = addr;
                if(access && iss::access_type::DEBUG)
                    throw trap_access(0, addr);
                this->reg.trap_state = (1 << 31); // issue trap 0
                return iss::Err;
            }
            phys_addr_t paddr = BASE::v2p(iss::addr_t{access, type, space, addr});
            try {
                if(unlikely((addr & ~PGMASK) != ((addr + length - 1) & ~PGMASK))) { // we may cross a page boundary
                    vm_info vm = hart_state_type::decode_vm_info(this->reg.PRIV, state.satp);
                    if(vm.levels != 0) { // VM is active
                        auto split_addr = (addr + length) & ~PGMASK;
                        auto len1 = split_addr - addr;
                        auto res = write(type, access, space, addr, len1, data);
                        if(res == iss::Ok)
                            res = write(type, access, space, split_addr, length - len1, data + len1);
                        return res;
                    }
                }
                auto res = write_mem(paddr, length, data);
                if(unlikely(res != iss::Ok && (access & access_type::DEBUG) == 0)) {
                    this->reg.trap_state = (1UL << 31) | (7UL << 16); // issue trap 7 (Store/AMO access fault)
                    fault_data = addr;
                }
                return res;
            } catch(trap_access& ta) {
                this->reg.trap_state = (1UL << 31) | ta.id;
                fault_data = ta.addr;
                return iss::Err;
            }

            if((paddr.val + length) > mem.size())
                return iss::Err;
            switch(paddr.val) {
            case 0x10013000: // UART0 base, TXFIFO reg
            case 0x10023000: // UART1 base, TXFIFO reg
                uart_buf << (char)data[0];
                if(((char)data[0]) == '\n' || data[0] == 0) {
                    // CPPLOG(INFO)<<"UART"<<((paddr.val>>16)&0x3)<<" send
                    // '"<<uart_buf.str()<<"'";
                    std::cout << uart_buf.str();
                    uart_buf.str("");
                }
                return iss::Ok;
            case 0x10008000: { // HFROSC base, hfrosccfg reg
                auto& p = mem(paddr.val / mem.page_size);
                auto offs = paddr.val & mem.page_addr_mask;
                std::copy(data, data + length, p.data() + offs);
                auto& x = *(p.data() + offs + 3);
                if(x & 0x40)
                    x |= 0x80; // hfroscrdy = 1 if hfroscen==1
                return iss::Ok;
            }
            case 0x10008008: { // HFROSC base, pllcfg reg
                auto& p = mem(paddr.val / mem.page_size);
                auto offs = paddr.val & mem.page_addr_mask;
                std::copy(data, data + length, p.data() + offs);
                auto& x = *(p.data() + offs + 3);
                x |= 0x80; // set pll lock upon writing
                return iss::Ok;
            } break;
            default: {
            }
            }
        } break;
        case traits<BASE>::CSR: {
            if(length != sizeof(reg_t))
                return iss::Err;
            return write_csr(addr, *reinterpret_cast<const reg_t*>(data));
        } break;
        case traits<BASE>::FENCE: {
            if((addr + length) > mem.size())
                return iss::Err;
            switch(addr) {
            case 2:
            case 3: {
                ptw.clear();
                auto tvm = state.mstatus.TVM;
                if(this->reg.PRIV == PRIV_S & tvm != 0) {
                    this->reg.trap_state = (1 << 31) | (2 << 16);
                    this->fault_data = this->reg.PC;
                    return iss::Err;
                }
                return iss::Ok;
            }
            }
        } break;
        case traits<BASE>::RES: {
            atomic_reservation[addr] = data[0];
        } break;
        default:
            return iss::Err;
        }
        return iss::Ok;
    } catch(trap_access& ta) {
        if((access & access_type::DEBUG) == 0) {
            this->reg.trap_state = (1UL << 31) | ta.id;
            fault_data = ta.addr;
        }
        return iss::Err;
    }
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_csr(unsigned addr, reg_t& val) {
    if(addr >= csr.size())
        return iss::Err;
    auto req_priv_lvl = (addr >> 8) & 0x3;
    if(this->reg.PRIV < req_priv_lvl) // not having required privileges
        throw illegal_instruction_fault(this->fault_data);
    auto it = csr_rd_cb.find(addr);
    if(it == csr_rd_cb.end() || !it->second) // non existent register
        throw illegal_instruction_fault(this->fault_data);
    return (this->*(it->second))(addr, val);
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::write_csr(unsigned addr, reg_t val) {
    if(addr >= csr.size())
        return iss::Err;
    auto req_priv_lvl = (addr >> 8) & 0x3;
    if(this->reg.PRIV < req_priv_lvl) // not having required privileges
        throw illegal_instruction_fault(this->fault_data);
    if((addr & 0xc00) == 0xc00) // writing to read-only region
        throw illegal_instruction_fault(this->fault_data);
    auto it = csr_wr_cb.find(addr);
    if(it == csr_wr_cb.end() || !it->second) // non existent register
        throw illegal_instruction_fault(this->fault_data);
    return (this->*(it->second))(addr, val);
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_reg(unsigned addr, reg_t& val) {
    val = csr[addr];
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_null(unsigned addr, reg_t& val) {
    val = 0;
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::write_reg(unsigned addr, reg_t val) {
    csr[addr] = val;
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_cycle(unsigned addr, reg_t& val) {
    auto cycle_val = this->reg.cycle + cycle_offset;
    if(addr == mcycle) {
        val = static_cast<reg_t>(cycle_val);
    } else if(addr == mcycleh) {
        if(sizeof(typename traits<BASE>::reg_t) != 4)
            return iss::Err;
        val = static_cast<reg_t>(cycle_val >> 32);
    }
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::write_cycle(unsigned addr, reg_t val) {
    if(sizeof(typename traits<BASE>::reg_t) != 4) {
        mcycle_csr = static_cast<uint64_t>(val);
    } else {
        if(addr == mcycle) {
            mcycle_csr = (mcycle_csr & 0xffffffff00000000) + val;
        } else {
            mcycle_csr = (static_cast<uint64_t>(val) << 32) + (mcycle_csr & 0xffffffff);
        }
    }
    cycle_offset = mcycle_csr - this->reg.cycle; // TODO: relying on wrap-around
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_instret(unsigned addr, reg_t& val) {
    if((addr & 0xff) == (minstret & 0xff)) {
        val = static_cast<reg_t>(this->reg.instret);
    } else if((addr & 0xff) == (minstreth & 0xff)) {
        val = static_cast<reg_t>(this->reg.instret >> 32);
    }
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::write_instret(unsigned addr, reg_t val) {
    if(sizeof(typename traits<BASE>::reg_t) != 4) {
        this->reg.instret = static_cast<uint64_t>(val);
    } else {
        if((addr & 0xff) == (minstret & 0xff)) {
            this->reg.instret = (this->reg.instret & 0xffffffff00000000) + val;
        } else {
            this->reg.instret = (static_cast<uint64_t>(val) << 32) + (this->reg.instret & 0xffffffff);
        }
    }
    this->reg.instret--;
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_time(unsigned addr, reg_t& val) {
    uint64_t time_val = this->reg.cycle / (100000000 / 32768 - 1); //-> ~3052;
    if(addr == time) {
        val = static_cast<reg_t>(time_val);
    } else if(addr == timeh) {
        if(sizeof(typename traits<BASE>::reg_t) != 4)
            return iss::Err;
        val = static_cast<reg_t>(time_val >> 32);
    }
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_tvec(unsigned addr, reg_t& val) {
    val = csr[addr] & ~2;
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_status(unsigned addr, reg_t& val) {
    auto req_priv_lvl = (addr >> 8) & 0x3;
    val = state.mstatus & hart_state_type::get_mask(req_priv_lvl);
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::write_status(unsigned addr, reg_t val) {
    auto req_priv_lvl = (addr >> 8) & 0x3;
    state.write_mstatus(val, req_priv_lvl);
    check_interrupt();
    update_vm_info();
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::write_cause(unsigned addr, reg_t val) {
    csr[addr] = val & ((1UL << (traits<BASE>::XLEN - 1)) | 0xf); // TODO: make exception code size configurable
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_ie(unsigned addr, reg_t& val) {
    val = csr[mie];
    if(addr < mie)
        val &= csr[mideleg];
    if(addr < sie)
        val &= csr[sideleg];
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_hartid(unsigned addr, reg_t& val) {
    val = mhartid_reg;
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::write_ie(unsigned addr, reg_t val) {
    auto req_priv_lvl = (addr >> 8) & 0x3;
    auto mask = get_irq_mask(req_priv_lvl);
    csr[mie] = (csr[mie] & ~mask) | (val & mask);
    check_interrupt();
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_ip(unsigned addr, reg_t& val) {
    val = csr[mip];
    if(addr < mip)
        val &= csr[mideleg];
    if(addr < sip)
        val &= csr[sideleg];
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::write_epc(unsigned addr, reg_t val) {
    csr[addr] = val & get_pc_mask();
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_satp(unsigned addr, reg_t& val) {
    reg_t tvm = state.mstatus.TVM;
    if(this->reg.PRIV == PRIV_S & tvm != 0) {
        this->reg.trap_state = (1 << 31) | (2 << 16);
        this->fault_data = this->reg.PC;
        return iss::Err;
    }
    val = state.satp;
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::write_satp(unsigned addr, reg_t val) {
    reg_t tvm = state.mstatus.TVM;
    if(this->reg.PRIV == PRIV_S & tvm != 0) {
        this->reg.trap_state = (1 << 31) | (2 << 16);
        this->fault_data = this->reg.PC;
        return iss::Err;
    }
    state.satp = val;
    update_vm_info();
    return iss::Ok;
}
template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_fcsr(unsigned addr, reg_t& val) {
    switch(addr) {
    case 1: // fflags, 4:0
        val = bit_sub<0, 5>(this->get_fcsr());
        break;
    case 2: // frm, 7:5
        val = bit_sub<5, 3>(this->get_fcsr());
        break;
    case 3: // fcsr
        val = this->get_fcsr();
        break;
    default:
        return iss::Err;
    }
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::write_fcsr(unsigned addr, reg_t val) {
    switch(addr) {
    case 1: // fflags, 4:0
        this->set_fcsr((this->get_fcsr() & 0xffffffe0) | (val & 0x1f));
        break;
    case 2: // frm, 7:5
        this->set_fcsr((this->get_fcsr() & 0xffffff1f) | ((val & 0x7) << 5));
        break;
    case 3: // fcsr
        this->set_fcsr(val & 0xff);
        break;
    default:
        return iss::Err;
    }
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_mem(phys_addr_t paddr, unsigned length, uint8_t* const data) {
    switch(paddr.val) {
    default: {
        for(auto offs = 0U; offs < length; ++offs) {
            *(data + offs) = mem[(paddr.val + offs) % mem.size()];
        }
    }
    }
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::write_mem(phys_addr_t paddr, unsigned length, const uint8_t* const data) {
    switch(paddr.val) {
    case 0xFFFF0000: // UART0 base, TXFIFO reg
        if(((char)data[0]) == '\n' || data[0] == 0) {
            CPPLOG(INFO) << "UART" << ((paddr.val >> 12) & 0x3) << " send '" << uart_buf.str() << "'";
            uart_buf.str("");
        } else if(((char)data[0]) != '\r')
            uart_buf << (char)data[0];
        break;
    default: {
        mem_type::page_type& p = mem(paddr.val / mem.page_size);
        std::copy(data, data + length, p.data() + (paddr.val & mem.page_addr_mask));
        // tohost handling in case of riscv-test
        if(paddr.access && iss::access_type::FUNC) {
            auto tohost_upper =
                (traits<BASE>::XLEN == 32 && paddr.val == (tohost + 4)) || (traits<BASE>::XLEN == 64 && paddr.val == tohost);
            auto tohost_lower = (traits<BASE>::XLEN == 32 && paddr.val == tohost) || (traits<BASE>::XLEN == 64 && paddr.val == tohost);
            if(tohost_lower || tohost_upper) {
                uint64_t hostvar = *reinterpret_cast<uint64_t*>(p.data() + (tohost & mem.page_addr_mask));
                // in case of 32 bit system, two writes to tohost are needed, only evaluate on the second (high) write
                if(tohost_upper && (tohost_lower || tohost_lower_written)) {
                    switch(hostvar >> 48) {
                    case 0:
                        if(hostvar != 0x1) {
                            CPPLOG(FATAL) << "tohost value is 0x" << std::hex << hostvar << std::dec << " (" << hostvar
                                          << "), stopping simulation";
                        } else {
                            CPPLOG(INFO) << "tohost value is 0x" << std::hex << hostvar << std::dec << " (" << hostvar
                                         << "), stopping simulation";
                        }
                        this->reg.trap_state = std::numeric_limits<uint32_t>::max();
                        this->interrupt_sim = hostvar;
#ifndef WITH_TCC
                        throw(iss::simulation_stopped(hostvar));
#endif
                        break;
                    case 0x0101: {
                        char c = static_cast<char>(hostvar & 0xff);
                        if(c == '\n' || c == 0) {
                            CPPLOG(INFO) << "tohost send '" << uart_buf.str() << "'";
                            uart_buf.str("");
                        } else
                            uart_buf << c;
                    } break;
                    default:
                        break;
                    }
                    tohost_lower_written = false;
                } else if(tohost_lower)
                    tohost_lower_written = true;
            } else if((traits<BASE>::XLEN == 32 && paddr.val == fromhost + 4) || (traits<BASE>::XLEN == 64 && paddr.val == fromhost)) {
                uint64_t fhostvar = *reinterpret_cast<uint64_t*>(p.data() + (fromhost & mem.page_addr_mask));
                *reinterpret_cast<uint64_t*>(p.data() + (tohost & mem.page_addr_mask)) = fhostvar;
            }
        }
    }
    }
    return iss::Ok;
}

template <typename BASE> inline void riscv_hart_msu_vp<BASE>::reset(uint64_t address) {
    BASE::reset(address);
    state.mstatus = hart_state_type::mstatus_reset_val;
    update_vm_info();
}

template <typename BASE> inline void riscv_hart_msu_vp<BASE>::update_vm_info() {
    vm[1] = hart_state_type::decode_vm_info(this->reg.PRIV, state.satp);
    BASE::addr_mode[3] = BASE::addr_mode[2] = vm[1].is_active() ? iss::address_type::VIRTUAL : iss::address_type::PHYSICAL;
    if(state.mstatus.MPRV)
        vm[0] = hart_state_type::decode_vm_info(state.mstatus.MPP, state.satp);
    else
        vm[0] = vm[1];
    BASE::addr_mode[1] = BASE::addr_mode[0] = vm[0].is_active() ? iss::address_type::VIRTUAL : iss::address_type::PHYSICAL;
    ptw.clear();
}

template <typename BASE> void riscv_hart_msu_vp<BASE>::check_interrupt() {
    auto status = state.mstatus;
    auto ip = csr[mip];
    auto ie = csr[mie];
    auto ideleg = csr[mideleg];
    // Multiple simultaneous interrupts and traps at the same privilege level are
    // handled in the following decreasing priority order:
    // external interrupts, software interrupts, timer interrupts, then finally
    // any synchronous traps.
    auto ena_irq = ip & ie;

    bool mie = state.mstatus.MIE;
    auto m_enabled = this->reg.PRIV < PRIV_M || (this->reg.PRIV == PRIV_M && mie);
    auto enabled_interrupts = m_enabled ? ena_irq & ~ideleg : 0;

    if(enabled_interrupts == 0) {
        auto sie = state.mstatus.SIE;
        auto s_enabled = this->reg.PRIV < PRIV_S || (this->reg.PRIV == PRIV_S && sie);
        enabled_interrupts = s_enabled ? ena_irq & ideleg : 0;
    }
    if(enabled_interrupts != 0) {
        int res = 0;
        while((enabled_interrupts & 1) == 0)
            enabled_interrupts >>= 1, res++;
        this->reg.pending_trap = res << 16 | 1; // 0x80 << 24 | (cause << 16) | trap_id
    }
}

template <typename BASE> typename riscv_hart_msu_vp<BASE>::phys_addr_t riscv_hart_msu_vp<BASE>::virt2phys(const iss::addr_t& addr) {
    const auto type = addr.access & iss::access_type::FUNC;
    auto it = ptw.find(addr.val >> PGSHIFT);
    if(it != ptw.end()) {
        const reg_t pte = it->second;
        const reg_t ad = PTE_A | (type == iss::access_type::WRITE) * PTE_D;
#ifdef RISCV_ENABLE_DIRTY
        // set accessed and possibly dirty bits.
        *(uint32_t*)ppte |= ad;
        return {addr.getAccessType(), addr.space, (pte & (~PGMASK)) | (addr.val & PGMASK)};
#else
        // take exception if access or possibly dirty bit is not set.
        if((pte & ad) == ad)
            return {addr.access, addr.space, (pte & (~PGMASK)) | (addr.val & PGMASK)};
        else
            ptw.erase(it); // throw an exception
#endif
    } else {
        uint32_t mode = type != iss::access_type::FETCH && state.mstatus.MPRV ? // MPRV
                            state.mstatus.MPP
                                                                              : this->reg.PRIV;

        const vm_info& vm = this->vm[static_cast<uint16_t>(type) / 2];

        const bool s_mode = mode == PRIV_S;
        const bool sum = state.mstatus.SUM;
        const bool mxr = state.mstatus.MXR;

        // verify bits xlen-1:va_bits-1 are all equal
        const int va_bits = PGSHIFT + vm.levels * vm.idxbits;
        const reg_t mask = (reg_t(1) << (traits<BASE>::XLEN > -(va_bits - 1))) - 1;
        const reg_t masked_msbs = (addr.val >> (va_bits - 1)) & mask;
        const int levels = (masked_msbs != 0 && masked_msbs != mask) ? 0 : vm.levels;

        reg_t base = vm.ptbase;
        for(int i = levels - 1; i >= 0; i--) {
            const int ptshift = i * vm.idxbits;
            const reg_t idx = (addr.val >> (PGSHIFT + ptshift)) & ((1 << vm.idxbits) - 1);

            // check that physical address of PTE is legal
            reg_t pte = 0;
            const uint8_t res = this->read(iss::address_type::PHYSICAL, addr.access, traits<BASE>::MEM, base + idx * vm.ptesize, vm.ptesize,
                                           (uint8_t*)&pte);
            if(res != 0)
                throw trap_load_access_fault(addr.val);
            const reg_t ppn = pte >> PTE_PPN_SHIFT;

            if(PTE_TABLE(pte)) { // next level of page table
                base = ppn << PGSHIFT;
            } else if((pte & PTE_U) ? s_mode && (type == iss::access_type::FETCH || !sum) : !s_mode) {
                break;
            } else if(!(pte & PTE_V) || (!(pte & PTE_R) && (pte & PTE_W))) {
                break;
            } else if(type == (iss::access_type::FETCH          ? !(pte & PTE_X)
                               : type == iss::access_type::READ ? !(pte & PTE_R) && !(mxr && (pte & PTE_X))
                                                                : !((pte & PTE_R) && (pte & PTE_W)))) {
                break;
            } else if((ppn & ((reg_t(1) << ptshift) - 1)) != 0) {
                break;
            } else {
                const reg_t ad = PTE_A | ((type == iss::access_type::WRITE) * PTE_D);
#ifdef RISCV_ENABLE_DIRTY
                // set accessed and possibly dirty bits.
                *(uint32_t*)ppte |= ad;
#else
                // take exception if access or possibly dirty bit is not set.
                if((pte & ad) != ad)
                    break;
#endif
                // for superpage mappings, make a fake leaf PTE for the TLB's benefit.
                const reg_t vpn = addr.val >> PGSHIFT;
                const reg_t value = (ppn | (vpn & ((reg_t(1) << ptshift) - 1))) << PGSHIFT;
                const reg_t offset = addr.val & PGMASK;
                ptw[vpn] = value | (pte & 0xff);
                return {addr.access, addr.space, value | offset};
            }
        }
    }
    switch(type) {
    case access_type::FETCH:
        this->fault_data = addr.val;
        throw trap_instruction_page_fault(addr.val);
    case access_type::READ:
        this->fault_data = addr.val;
        throw trap_load_page_fault(addr.val);
    case access_type::WRITE:
        this->fault_data = addr.val;
        throw trap_store_page_fault(addr.val);
    default:
        abort();
    }
}

template <typename BASE> uint64_t riscv_hart_msu_vp<BASE>::enter_trap(uint64_t flags, uint64_t addr, uint64_t instr) {
    auto cur_priv = this->reg.PRIV;
    // flags are ACTIVE[31:31], CAUSE[30:16], TRAPID[15:0]
    // calculate and write mcause val
    if(flags == std::numeric_limits<uint64_t>::max())
        flags = this->reg.trap_state;
    auto trap_id = bit_sub<0, 16>(flags);
    auto cause = bit_sub<16, 15>(flags);
    if(trap_id == 0 && cause == 11)
        cause = 0x8 + cur_priv; // adjust environment call cause
    // calculate effective privilege level
    auto new_priv = PRIV_M;
    if(trap_id == 0) { // exception
        if(cur_priv != PRIV_M && ((csr[medeleg] >> cause) & 0x1) != 0)
            new_priv = (csr[sedeleg] >> cause) & 0x1 ? PRIV_U : PRIV_S;
        // store ret addr in xepc register
        csr[uepc | (new_priv << 8)] = static_cast<reg_t>(addr); // store actual address instruction of exception
        /*
         * write mtval if new_priv=M_MODE, spec says:
         * When a hardware breakpoint is triggered, or an instruction-fetch, load,
         * or store address-misaligned,
         * access, or page-fault exception occurs, mtval is written with the
         * faulting effective address.
         */
        switch(cause) {
        case 0:
            csr[utval | (new_priv << 8)] = static_cast<reg_t>(addr);
            break;
        case 2:
            csr[utval | (new_priv << 8)] = (instr & 0x3) == 3 ? instr : instr & 0xffff;
            break;
        case 3:
            // TODO: implement debug mode behavior
            // csr[dpc] = addr;
            // csr[dcsr] = (csr[dcsr] & ~0x1c3) | (1<<6) | PRIV_M; //FIXME: cause should not be 4 (stepi)
            csr[utval | (new_priv << 8)] = addr;
            if(semihosting_cb) {
                // Check for semihosting call
                phys_addr_t p_addr(access_type::DEBUG_READ, traits<BASE>::MEM, addr - 4);
                std::array<uint8_t, 8> data;
                // check for SLLI_X0_X0_0X1F and SRAI_X0_X0_0X07
                this->read_mem(p_addr, 4, data.data());
                p_addr.val += 8;
                this->read_mem(p_addr, 4, data.data() + 4);

                const std::array<uint8_t, 8> ref_data = {0x13, 0x10, 0xf0, 0x01, 0x13, 0x50, 0x70, 0x40};
                if(data == ref_data) {
                    this->reg.NEXT_PC = addr + 8;

                    std::array<char, 32> buffer;
#if defined(_MSC_VER)
                    sprintf(buffer.data(), "0x%016llx", addr);
#else
                    sprintf(buffer.data(), "0x%016lx", addr);
#endif
                    CLOG(INFO, disass) << "Semihosting call at address " << buffer.data() << " occurred ";

                    semihosting_callback(this, this->reg.X10 /*a0*/, this->reg.X11 /*a1*/);
                    return this->reg.NEXT_PC;
                }
            }
            break;
        case 4:
        case 6:
        case 7:
            csr[utval | (new_priv << 8)] = fault_data;
            break;
        default:
            csr[utval | (new_priv << 8)] = 0;
        }
        fault_data = 0;
    } else {
        if(cur_priv != PRIV_M && ((csr[mideleg] >> cause) & 0x1) != 0)
            new_priv = (csr[sideleg] >> cause) & 0x1 ? PRIV_U : PRIV_S;
        csr[uepc | (new_priv << 8)] = this->reg.NEXT_PC; // store next address if interrupt
        this->reg.pending_trap = 0;
    }
    size_t adr = ucause | (new_priv << 8);
    csr[adr] = (trap_id << (traits<BASE>::XLEN - 1)) + cause;
    // update mstatus
    // xPP field of mstatus is written with the active privilege mode at the time
    // of the trap; the x PIE field of mstatus
    // is written with the value of the active interrupt-enable bit at the time of
    // the trap; and the x IE field of mstatus
    // is cleared
    // store the actual privilege level in yPP and store interrupt enable flags
    switch(new_priv) {
    case PRIV_M:
        state.mstatus.MPP = cur_priv;
        state.mstatus.MPIE = state.mstatus.MIE;
        state.mstatus.MIE = false;
        break;
    case PRIV_S:
        state.mstatus.SPP = cur_priv;
        state.mstatus.SPIE = state.mstatus.SIE;
        state.mstatus.SIE = false;
        break;
    case PRIV_U:
        state.mstatus.UPIE = state.mstatus.UIE;
        state.mstatus.UIE = false;
        break;
    default:
        break;
    }

    // get trap vector
    auto ivec = csr[utvec | (new_priv << 8)];
    // calculate addr// set NEXT_PC to trap addressess to jump to based on MODE
    // bits in mtvec
    this->reg.NEXT_PC = ivec & ~0x3UL;
    if((ivec & 0x1) == 1 && trap_id != 0)
        this->reg.NEXT_PC += 4 * cause;
    std::array<char, 32> buffer;
    sprintf(buffer.data(), "0x%016lx", addr);
    if((flags & 0xffffffff) != 0xffffffff)
        CLOG(INFO, disass) << (trap_id ? "Interrupt" : "Trap") << " with cause '" << (trap_id ? irq_str[cause] : trap_str[cause]) << "' ("
                           << cause << ")"
                           << " at address " << buffer.data() << " occurred, changing privilege level from " << lvl[cur_priv] << " to "
                           << lvl[new_priv];
    // reset trap state
    this->reg.PRIV = new_priv;
    this->reg.trap_state = 0;
    update_vm_info();
    return this->reg.NEXT_PC;
}

template <typename BASE> uint64_t riscv_hart_msu_vp<BASE>::leave_trap(uint64_t flags) {
    auto cur_priv = this->reg.PRIV;
    auto inst_priv = flags & 0x3;
    auto status = state.mstatus;

    auto tsr = state.mstatus.TSR;
    if(cur_priv == PRIV_S && inst_priv == PRIV_S && tsr != 0) {
        this->reg.trap_state = (1 << 31) | (2 << 16);
        this->fault_data = this->reg.PC;
        return this->reg.PC;
    }

    // pop the relevant lower-privilege interrupt enable and privilege mode stack
    // clear respective yIE
    switch(inst_priv) {
    case PRIV_M:
        this->reg.PRIV = state.mstatus.MPP;
        state.mstatus.MPP = 0; // clear mpp to U mode
        state.mstatus.MIE = state.mstatus.MPIE;
        state.mstatus.MPIE = 1;
        break;
    case PRIV_S:
        this->reg.PRIV = state.mstatus.SPP;
        state.mstatus.SPP = 0; // clear spp to U mode
        state.mstatus.SIE = state.mstatus.SPIE;
        state.mstatus.SPIE = 1;
        break;
    case PRIV_U:
        this->reg.PRIV = 0;
        state.mstatus.UIE = state.mstatus.UPIE;
        state.mstatus.UPIE = 1;
        break;
    }
    // sets the pc to the value stored in the x epc register.
    this->reg.NEXT_PC = csr[uepc | inst_priv << 8];
    CLOG(INFO, disass) << "Executing xRET , changing privilege level from " << lvl[cur_priv] << " to " << lvl[this->reg.PRIV];
    update_vm_info();
    check_interrupt();
    return this->reg.NEXT_PC;
}

template <typename BASE> void riscv_hart_msu_vp<BASE>::wait_until(uint64_t flags) {
    auto status = state.mstatus;
    auto tw = status.TW;
    if(this->reg.PRIV == PRIV_S && tw != 0) {
        this->reg.trap_state = (1 << 31) | (2 << 16);
        this->fault_data = this->reg.PC;
    }
}
} // namespace arch
} // namespace iss

#endif /* _RISCV_HART_MSU_VP_H */
