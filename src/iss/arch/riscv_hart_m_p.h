/*******************************************************************************
 * Copyright (C) 2019 - 2023 MINRES Technologies GmbH
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

#ifndef _RISCV_HART_M_P_H
#define _RISCV_HART_M_P_H

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

template <typename BASE, features_e FEAT = FEAT_NONE, typename LOGCAT = logging::disass>
class riscv_hart_m_p : public BASE, public riscv_hart_common {
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
    using this_class = riscv_hart_m_p<BASE, FEAT, LOGCAT>;
    using phys_addr_t = typename core::phys_addr_t;
    using reg_t = typename core::reg_t;
    using addr_t = typename core::addr_t;

    using rd_csr_f = iss::status (this_class::*)(unsigned addr, reg_t&);
    using wr_csr_f = iss::status (this_class::*)(unsigned addr, reg_t);
    using mem_read_f = iss::status(phys_addr_t addr, unsigned, uint8_t* const);
    using mem_write_f = iss::status(phys_addr_t addr, unsigned, uint8_t const* const);

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

        void write_mstatus(T val) {
            auto mask = get_mask() & 0xff; // MPP is hardcode as 0x3
            auto new_val = (mstatus.backing.val & ~mask) | (val & mask);
            mstatus = new_val;
        }

        static constexpr uint32_t get_mask() {
            // return 0x807ff988UL; // 0b1000 0000 0111 1111 1111 1000 1000 1000  // only machine mode is supported
            //       +-SD
            //       |        +-TSR
            //       |        |+-TW
            //       |        ||+-TVM
            //       |        |||+-MXR
            //       |        ||||+-SUM
            //       |        |||||+-MPRV
            //       |        |||||| +-XS
            //       |        |||||| | +-FS
            //       |        |||||| | | +-MPP
            //       |        |||||| | | |  +-SPP
            //       |        |||||| | | |  |+-MPIE
            //       |        ||||||/|/|/|  ||   +-MIE
            return 0b00000000000000000001100010001000;
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

        static const reg_t mstatus_reset_val = 0x1800;

        void write_mstatus(T val) {
            auto mask = get_mask() & 0xff; // MPP is hardcode as 0x3
            auto new_val = (mstatus.backing.val & ~mask) | (val & mask);
            mstatus = new_val;
        }

        static constexpr T get_mask() {
            // return 0x8000000f007ff9ddULL; // 0b1...0 1111 0000 0000 0111 1111 1111 1001 1011 1011
            //
            //                +-TSR
            //                |+-TW
            //                ||+-TVM
            //                |||+-MXR
            //                ||||+-SUM
            //                |||||+-MPRV
            //                |||||| +-XS
            //                |||||| | +-FS
            //                |||||| | | +-MPP
            //                |||||| | | |  +-SPP
            //                |||||| | | |  |+-MPIE
            //                ||||||/|/|/|  ||   +-MIE
            return 0b00000000000000000001100010001000;
        }
    };

    using hart_state_type = hart_state<reg_t>;

    constexpr reg_t get_irq_mask() {
        return 0b100010001000; // only machine mode is supported
    }

    constexpr bool has_compressed() { return traits<BASE>::MISA_VAL & 0b0100; }
    constexpr reg_t get_pc_mask() { return has_compressed() ? (reg_t)~1 : (reg_t)~3; }

    riscv_hart_m_p(feature_config cfg = feature_config{});
    virtual ~riscv_hart_m_p() = default;

    void reset(uint64_t address) override;

    std::pair<uint64_t, bool> load_file(std::string name, int type = -1) override;

    iss::status read(const address_type type, const access_type access, const uint32_t space, const uint64_t addr, const unsigned length,
                     uint8_t* const data) override;
    iss::status write(const address_type type, const access_type access, const uint32_t space, const uint64_t addr, const unsigned length,
                      const uint8_t* const data) override;

    uint64_t enter_trap(uint64_t flags) override { return riscv_hart_m_p::enter_trap(flags, fault_data, fault_data); }
    uint64_t enter_trap(uint64_t flags, uint64_t addr, uint64_t instr) override;
    uint64_t leave_trap(uint64_t flags) override;

    const reg_t& get_mhartid() const { return mhartid_reg; }
    void set_mhartid(reg_t mhartid) { mhartid_reg = mhartid; };

    void disass_output(uint64_t pc, const std::string instr) override {
        NSCLOG(INFO, LOGCAT) << fmt::format("0x{:016x}    {:40} [s:0x{:x};c:{}]", pc, instr, (reg_t)state.mstatus,
                                            this->reg.cycle + cycle_offset);
    };

    iss::instrumentation_if* get_instrumentation_if() override { return &instr_if; }

    void set_csr(unsigned addr, reg_t val) { csr[addr & csr.page_addr_mask] = val; }

    void set_irq_num(unsigned i) { mcause_max_irq = 1 << util::ilog2(i); }

    void set_semihosting_callback(semihosting_cb_t<reg_t> cb) { semihosting_cb = cb; };

protected:
    struct riscv_instrumentation_if : public iss::instrumentation_if {

        riscv_instrumentation_if(riscv_hart_m_p<BASE, FEAT, LOGCAT>& arch)
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

        riscv_hart_m_p<BASE, FEAT, LOGCAT>& arch;
    };

    friend struct riscv_instrumentation_if;

    virtual iss::status read_mem(phys_addr_t addr, unsigned length, uint8_t* const data);
    virtual iss::status write_mem(phys_addr_t addr, unsigned length, const uint8_t* const data);

    iss::status read_clic(uint64_t addr, unsigned length, uint8_t* const data);
    iss::status write_clic(uint64_t addr, unsigned length, const uint8_t* const data);

    virtual iss::status read_csr(unsigned addr, reg_t& val);
    virtual iss::status write_csr(unsigned addr, reg_t val);

    hart_state_type state;
    int64_t cycle_offset{0};
    uint64_t mcycle_csr{0};
    int64_t instret_offset{0};
    uint64_t minstret_csr{0};
    reg_t fault_data;
    uint64_t tohost = tohost_dflt;
    uint64_t fromhost = fromhost_dflt;
    bool tohost_lower_written = false;
    riscv_instrumentation_if instr_if;

    semihosting_cb_t<reg_t> semihosting_cb;

    using mem_type = util::sparse_array<uint8_t, 1ULL << 32>;
    using csr_type = util::sparse_array<typename traits<BASE>::reg_t, 1ULL << 12, 12>;
    using csr_page_type = typename csr_type::page_type;
    mem_type mem;
    csr_type csr;
    std::stringstream uart_buf;
    std::unordered_map<reg_t, uint64_t> ptw;
    std::unordered_map<uint64_t, uint8_t> atomic_reservation;
    std::unordered_map<unsigned, rd_csr_f> csr_rd_cb;
    std::unordered_map<unsigned, wr_csr_f> csr_wr_cb;
    uint8_t clic_cfg_reg{0};
    std::array<uint32_t, 32> clic_inttrig_reg;
    union clic_int_reg_t {
        struct {
            uint8_t ip;
            uint8_t ie;
            uint8_t attr;
            uint8_t ctl;
        };
        uint32_t raw;
    };
    std::vector<clic_int_reg_t> clic_int_reg;
    uint8_t clic_mprev_lvl{0};
    uint8_t clic_mact_lvl{0};

    std::vector<uint8_t> tcm;

    iss::status read_plain(unsigned addr, reg_t& val);
    iss::status write_plain(unsigned addr, reg_t val);
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
    iss::status read_cause(unsigned addr, reg_t& val);
    iss::status write_cause(unsigned addr, reg_t val);
    iss::status read_ie(unsigned addr, reg_t& val);
    iss::status write_ie(unsigned addr, reg_t val);
    iss::status read_ip(unsigned addr, reg_t& val);
    iss::status read_hartid(unsigned addr, reg_t& val);
    iss::status write_epc(unsigned addr, reg_t val);
    iss::status read_intstatus(unsigned addr, reg_t& val);
    iss::status write_intthresh(unsigned addr, reg_t val);
    iss::status write_xtvt(unsigned addr, reg_t val);
    iss::status write_dcsr(unsigned addr, reg_t val);
    iss::status read_debug(unsigned addr, reg_t& val);
    iss::status write_dscratch(unsigned addr, reg_t val);
    iss::status read_dpc(unsigned addr, reg_t& val);
    iss::status write_dpc(unsigned addr, reg_t val);
    iss::status read_fcsr(unsigned addr, reg_t& val);
    iss::status write_fcsr(unsigned addr, reg_t val);

    virtual iss::status read_custom_csr(unsigned addr, reg_t& val) { return iss::status::Err; };
    virtual iss::status write_custom_csr(unsigned addr, reg_t val) { return iss::status::Err; };

    void register_custom_csr_rd(unsigned addr) { csr_rd_cb[addr] = &this_class::read_custom_csr; }
    void register_custom_csr_wr(unsigned addr) { csr_wr_cb[addr] = &this_class::write_custom_csr; }

    reg_t mhartid_reg{0x0};

    void check_interrupt();
    bool pmp_check(const access_type type, const uint64_t addr, const unsigned len);
    std::vector<std::tuple<uint64_t, uint64_t>> memfn_range;
    std::vector<std::function<mem_read_f>> memfn_read;
    std::vector<std::function<mem_write_f>> memfn_write;
    void insert_mem_range(uint64_t, uint64_t, std::function<mem_read_f>, std::function<mem_write_f>);
    feature_config cfg;
    unsigned mcause_max_irq{(FEAT & features_e::FEAT_CLIC) ? std::max(16U, static_cast<unsigned>(traits<BASE>::CLIC_NUM_IRQ)) : 16U};
    inline bool debug_mode_active() { return this->reg.PRIV & 0x4; }

    std::pair<std::function<mem_read_f>, std::function<mem_write_f>> replace_mem_access(std::function<mem_read_f> rd,
                                                                                        std::function<mem_write_f> wr) {
        std::pair<std::function<mem_read_f>, std::function<mem_write_f>> ret{hart_mem_rd_delegate, hart_mem_wr_delegate};
        hart_mem_rd_delegate = rd;
        hart_mem_wr_delegate = wr;
        return ret;
    }
    std::function<mem_read_f> hart_mem_rd_delegate;
    std::function<mem_write_f> hart_mem_wr_delegate;
};

template <typename BASE, features_e FEAT, typename LOGCAT>
riscv_hart_m_p<BASE, FEAT, LOGCAT>::riscv_hart_m_p(feature_config cfg)
: state()
, instr_if(*this)
, cfg(cfg) {
    // reset values
    csr[misa] = traits<BASE>::MISA_VAL;
    csr[mvendorid] = 0x669;
    csr[marchid] = traits<BASE>::MARCHID_VAL;
    csr[mimpid] = 1;

    uart_buf.str("");
    if(traits<BASE>::FLEN > 0) {
        csr_rd_cb[fcsr] = &this_class::read_fcsr;
        csr_wr_cb[fcsr] = &this_class::write_fcsr;
    }
    for(unsigned addr = mhpmcounter3; addr <= mhpmcounter31; ++addr) {
        csr_rd_cb[addr] = &this_class::read_null;
        csr_wr_cb[addr] = &this_class::write_plain;
    }
    if(traits<BASE>::XLEN == 32)
        for(unsigned addr = mhpmcounter3h; addr <= mhpmcounter31h; ++addr) {
            csr_rd_cb[addr] = &this_class::read_null;
            csr_wr_cb[addr] = &this_class::write_plain;
        }
    for(unsigned addr = mhpmevent3; addr <= mhpmevent31; ++addr) {
        csr_rd_cb[addr] = &this_class::read_null;
        csr_wr_cb[addr] = &this_class::write_plain;
    }
    for(unsigned addr = hpmcounter3; addr <= hpmcounter31; ++addr) {
        csr_rd_cb[addr] = &this_class::read_null;
    }
    if(traits<BASE>::XLEN == 32)
        for(unsigned addr = hpmcounter3h; addr <= hpmcounter31h; ++addr) {
            csr_rd_cb[addr] = &this_class::read_null;
        }
    // common regs
    const std::array<unsigned, 4> roaddrs{{misa, mvendorid, marchid, mimpid}};
    for(auto addr : roaddrs) {
        csr_rd_cb[addr] = &this_class::read_plain;
        csr_wr_cb[addr] = &this_class::write_null;
    }
    const std::array<unsigned, 4> rwaddrs{{mepc, mtvec, mscratch, mtval}};
    for(auto addr : rwaddrs) {
        csr_rd_cb[addr] = &this_class::read_plain;
        csr_wr_cb[addr] = &this_class::write_plain;
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
    csr_rd_cb[mcause] = &this_class::read_cause;
    csr_wr_cb[mcause] = &this_class::write_cause;
    csr_rd_cb[mtvec] = &this_class::read_tvec;
    csr_wr_cb[mepc] = &this_class::write_epc;
    csr_rd_cb[mip] = &this_class::read_ip;
    csr_wr_cb[mip] = &this_class::write_null;
    csr_rd_cb[mie] = &this_class::read_ie;
    csr_wr_cb[mie] = &this_class::write_ie;
    csr_rd_cb[mhartid] = &this_class::read_hartid;
    csr_wr_cb[misa] = &this_class::write_null;
    csr_wr_cb[mvendorid] = &this_class::write_null;
    csr_wr_cb[marchid] = &this_class::write_null;
    csr_wr_cb[mimpid] = &this_class::write_null;
    if(FEAT & FEAT_CLIC) {
        csr_rd_cb[mtvt] = &this_class::read_plain;
        csr_wr_cb[mtvt] = &this_class::write_xtvt;
        //        csr_rd_cb[mxnti] = &this_class::read_csr_reg;
        //        csr_wr_cb[mxnti] = &this_class::write_csr_reg;
        csr_rd_cb[mintstatus] = &this_class::read_intstatus;
        csr_wr_cb[mintstatus] = &this_class::write_null;
        //        csr_rd_cb[mscratchcsw] = &this_class::read_csr_reg;
        //        csr_wr_cb[mscratchcsw] = &this_class::write_csr_reg;
        //        csr_rd_cb[mscratchcswl] = &this_class::read_csr_reg;
        //        csr_wr_cb[mscratchcswl] = &this_class::write_csr_reg;
        csr_rd_cb[mintthresh] = &this_class::read_plain;
        csr_wr_cb[mintthresh] = &this_class::write_intthresh;
        clic_int_reg.resize(cfg.clic_num_irq, clic_int_reg_t{.raw = 0});
        clic_cfg_reg = 0x20;
        clic_mact_lvl = clic_mprev_lvl = (1 << (cfg.clic_int_ctl_bits)) - 1;
        csr[mintthresh] = (1 << (cfg.clic_int_ctl_bits)) - 1;
        insert_mem_range(
            cfg.clic_base, 0x5000UL,
            [this](phys_addr_t addr, unsigned length, uint8_t* const data) { return read_clic(addr.val, length, data); },
            [this](phys_addr_t addr, unsigned length, uint8_t const* const data) { return write_clic(addr.val, length, data); });
    }
    if(FEAT & FEAT_TCM) {
        tcm.resize(cfg.tcm_size);
        std::function<mem_read_f> read_clic_cb = [this](phys_addr_t addr, unsigned length, uint8_t* const data) {
            auto offset = addr.val - this->cfg.tcm_base;
            std::copy(tcm.data() + offset, tcm.data() + offset + length, data);
            return iss::Ok;
        };
        std::function<mem_write_f> write_clic_cb = [this](phys_addr_t addr, unsigned length, uint8_t const* const data) {
            auto offset = addr.val - this->cfg.tcm_base;
            std::copy(data, data + length, tcm.data() + offset);
            return iss::Ok;
        };
        insert_mem_range(cfg.tcm_base, cfg.tcm_size, read_clic_cb, write_clic_cb);
    }
    if(FEAT & FEAT_DEBUG) {
        csr_wr_cb[dscratch0] = &this_class::write_dscratch;
        csr_rd_cb[dscratch0] = &this_class::read_debug;
        csr_wr_cb[dscratch1] = &this_class::write_dscratch;
        csr_rd_cb[dscratch1] = &this_class::read_debug;
        csr_wr_cb[dpc] = &this_class::write_dpc;
        csr_rd_cb[dpc] = &this_class::read_dpc;
        csr_wr_cb[dcsr] = &this_class::write_dcsr;
        csr_rd_cb[dcsr] = &this_class::read_debug;
    }
    hart_mem_rd_delegate = [this](phys_addr_t a, unsigned l, uint8_t* const d) -> iss::status { return this->read_mem(a, l, d); };
    hart_mem_wr_delegate = [this](phys_addr_t a, unsigned l, uint8_t const* const d) -> iss::status { return this->write_mem(a, l, d); };
}

template <typename BASE, features_e FEAT, typename LOGCAT>
std::pair<uint64_t, bool> riscv_hart_m_p<BASE, FEAT, LOGCAT>::load_file(std::string name, int type) {
    get_sym_table(name);
    try {
        tohost = symbol_table.at("tohost");
        fromhost = symbol_table.at("fromhost");
    } catch(std::out_of_range& e) {
    }
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
            if(reader.get_class() != ELFIO::ELFCLASS32)
                if(sizeof(reg_t) == 4)
                    throw std::runtime_error("wrong elf class in file");
            if(reader.get_type() != ELFIO::ET_EXEC)
                throw std::runtime_error("wrong elf type in file");
            if(reader.get_machine() != ELFIO::EM_RISCV)
                throw std::runtime_error("wrong elf machine in file");
            auto entry = reader.get_entry();
            for(const auto& pseg : reader.segments) {
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
            for(const auto& sec : reader.sections) {
                if(sec->get_name() == ".tohost") {
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

template <typename BASE, features_e FEAT, typename LOGCAT>
inline void riscv_hart_m_p<BASE, FEAT, LOGCAT>::insert_mem_range(uint64_t base, uint64_t size, std::function<mem_read_f> rd_f,
                                                                 std::function<mem_write_f> wr_fn) {
    std::tuple<uint64_t, uint64_t> entry{base, size};
    auto it = std::upper_bound(
        memfn_range.begin(), memfn_range.end(), entry,
        [](std::tuple<uint64_t, uint64_t> const& a, std::tuple<uint64_t, uint64_t> const& b) { return std::get<0>(a) < std::get<0>(b); });
    auto idx = std::distance(memfn_range.begin(), it);
    memfn_range.insert(it, entry);
    memfn_read.insert(std::begin(memfn_read) + idx, rd_f);
    memfn_write.insert(std::begin(memfn_write) + idx, wr_fn);
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read(const address_type type, const access_type access, const uint32_t space,
                                                     const uint64_t addr, const unsigned length, uint8_t* const data) {
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
            auto alignment = is_fetch(access) ? (has_compressed() ? 2 : 4) : length;
            if(unlikely(is_fetch(access) && (addr & (alignment - 1)))) {
                fault_data = addr;
                if(is_debug(access))
                    throw trap_access(0, addr);
                this->reg.trap_state = (1UL << 31); // issue trap 0
                return iss::Err;
            }
            try {
                if(!is_debug(access) && (addr & (alignment - 1))) {
                    this->reg.trap_state = (1UL << 31) | 4 << 16;
                    fault_data = addr;
                    return iss::Err;
                }
                phys_addr_t phys_addr{access, space, addr};
                auto res = iss::Err;
                if(!is_fetch(access) && memfn_range.size()) {
                    auto it =
                        std::find_if(std::begin(memfn_range), std::end(memfn_range), [phys_addr](std::tuple<uint64_t, uint64_t> const& a) {
                            return std::get<0>(a) <= phys_addr.val && (std::get<0>(a) + std::get<1>(a)) > phys_addr.val;
                        });
                    if(it != std::end(memfn_range)) {
                        auto idx = std::distance(std::begin(memfn_range), it);
                        res = memfn_read[idx](phys_addr, length, data);
                    } else
                        res = hart_mem_rd_delegate(phys_addr, length, data);
                } else {
                    res = hart_mem_rd_delegate(phys_addr, length, data);
                }
                if(unlikely(res != iss::Ok && (access & access_type::DEBUG) == 0)) {
                    this->reg.trap_state = (1UL << 31) | (5 << 16); // issue trap 5 (load access fault
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
            return iss::Ok;
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

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::write(const address_type type, const access_type access, const uint32_t space,
                                                      const uint64_t addr, const unsigned length, const uint8_t* const data) {
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
                this->reg.trap_state = (1UL << 31); // issue trap 0
                return iss::Err;
            }
            try {
                if(length > 1 && (addr & (length - 1)) && (access & access_type::DEBUG) != access_type::DEBUG) {
                    this->reg.trap_state = (1UL << 31) | 6 << 16;
                    fault_data = addr;
                    return iss::Err;
                }
                phys_addr_t phys_addr{access, space, addr};
                auto res = iss::Err;
                if(!is_fetch(access) && memfn_range.size()) {
                    auto it =
                        std::find_if(std::begin(memfn_range), std::end(memfn_range), [phys_addr](std::tuple<uint64_t, uint64_t> const& a) {
                            return std::get<0>(a) <= phys_addr.val && (std::get<0>(a) + std::get<1>(a)) > phys_addr.val;
                        });
                    if(it != std::end(memfn_range)) {
                        auto idx = std::distance(std::begin(memfn_range), it);
                        res = memfn_write[idx](phys_addr, length, data);
                    } else
                        res = write_mem(phys_addr, length, data);
                } else {
                    res = write_mem(phys_addr, length, data);
                }
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

            if((addr + length) > mem.size())
                return iss::Err;
            switch(addr) {
            case 0x10013000: // UART0 base, TXFIFO reg
            case 0x10023000: // UART1 base, TXFIFO reg
                uart_buf << (char)data[0];
                if(((char)data[0]) == '\n' || data[0] == 0) {
                    std::cout << uart_buf.str();
                    uart_buf.str("");
                }
                return iss::Ok;
            case 0x10008000: { // HFROSC base, hfrosccfg reg
                auto& p = mem(addr / mem.page_size);
                auto offs = addr & mem.page_addr_mask;
                std::copy(data, data + length, p.data() + offs);
                auto& x = *(p.data() + offs + 3);
                if(x & 0x40)
                    x |= 0x80; // hfroscrdy = 1 if hfroscen==1
                return iss::Ok;
            }
            case 0x10008008: { // HFROSC base, pllcfg reg
                auto& p = mem(addr / mem.page_size);
                auto offs = addr & mem.page_addr_mask;
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

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_csr(unsigned addr, reg_t& val) {
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

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::write_csr(unsigned addr, reg_t val) {
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

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_null(unsigned addr, reg_t& val) {
    val = 0;
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_plain(unsigned addr, reg_t& val) {
    val = csr[addr];
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::write_plain(unsigned addr, reg_t val) {
    csr[addr] = val;
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_cycle(unsigned addr, reg_t& val) {
    auto cycle_val = this->reg.cycle + cycle_offset;
    if(addr == mcycle) {
        val = static_cast<reg_t>(cycle_val);
    } else if(addr == mcycleh) {
        val = static_cast<reg_t>(cycle_val >> 32);
    }
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::write_cycle(unsigned addr, reg_t val) {
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

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_instret(unsigned addr, reg_t& val) {
    if((addr & 0xff) == (minstret & 0xff)) {
        val = static_cast<reg_t>(this->reg.instret);
    } else if((addr & 0xff) == (minstreth & 0xff)) {
        val = static_cast<reg_t>(this->reg.instret >> 32);
    }
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::write_instret(unsigned addr, reg_t val) {
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

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_time(unsigned addr, reg_t& val) {
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

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_tvec(unsigned addr, reg_t& val) {
    val = FEAT & features_e::FEAT_CLIC ? csr[addr] : csr[addr] & ~2;
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_status(unsigned addr, reg_t& val) {
    val = state.mstatus & hart_state_type::get_mask();
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::write_status(unsigned addr, reg_t val) {
    state.write_mstatus(val);
    check_interrupt();
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_cause(unsigned addr, reg_t& val) {
    if((FEAT & features_e::FEAT_CLIC) && (csr[mtvec] & 0x3) == 3) {
        val = csr[addr] & ((1UL << (traits<BASE>::XLEN - 1)) | (mcause_max_irq - 1) | (0xfUL << 16));
        val |= clic_mprev_lvl << 16;
        val |= state.mstatus.MPIE << 27;
        val |= state.mstatus.MPP << 28;
    } else
        val = csr[addr] & ((1UL << (traits<BASE>::XLEN - 1)) | (mcause_max_irq - 1));
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::write_cause(unsigned addr, reg_t val) {
    if((FEAT & features_e::FEAT_CLIC) && (csr[mtvec] & 0x3) == 3) {
        auto mask = ((1UL << (traits<BASE>::XLEN - 1)) | (mcause_max_irq - 1) | (0xfUL << 16));
        csr[addr] = (val & mask) | (csr[addr] & ~mask);
        clic_mprev_lvl = ((val >> 16) & 0xff) | (1 << (8 - cfg.clic_int_ctl_bits)) - 1;
        state.mstatus.MPIE = (val >> 27) & 0x1;
        state.mstatus.MPP = (val >> 28) & 0x3;
    } else {
        auto mask = ((1UL << (traits<BASE>::XLEN - 1)) | (mcause_max_irq - 1));
        csr[addr] = (val & mask) | (csr[addr] & ~mask);
    }
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_hartid(unsigned addr, reg_t& val) {
    val = mhartid_reg;
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_ie(unsigned addr, reg_t& val) {
    auto mask = get_irq_mask();
    val = csr[mie] & mask;
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::write_ie(unsigned addr, reg_t val) {
    auto mask = get_irq_mask();
    csr[mie] = (csr[mie] & ~mask) | (val & mask);
    check_interrupt();
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_ip(unsigned addr, reg_t& val) {
    auto mask = get_irq_mask();
    val = csr[mip] & mask;
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::write_epc(unsigned addr, reg_t val) {
    csr[addr] = val & get_pc_mask();
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::write_dcsr(unsigned addr, reg_t val) {
    if(!debug_mode_active())
        throw illegal_instruction_fault(this->fault_data);
    //                  +-------------- ebreakm
    //                  |   +---------- stepi
    //                  |   |  +++----- cause
    //                  |   |  |||   +- step
    csr[addr] = val & 0b1000100111000100U;
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_debug(unsigned addr, reg_t& val) {
    if(!debug_mode_active())
        throw illegal_instruction_fault(this->fault_data);
    val = csr[addr];
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::write_dscratch(unsigned addr, reg_t val) {
    if(!debug_mode_active())
        throw illegal_instruction_fault(this->fault_data);
    csr[addr] = val;
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_dpc(unsigned addr, reg_t& val) {
    if(!debug_mode_active())
        throw illegal_instruction_fault(this->fault_data);
    val = this->reg.DPC;
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::write_dpc(unsigned addr, reg_t val) {
    if(!debug_mode_active())
        throw illegal_instruction_fault(this->fault_data);
    this->reg.DPC = val;
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_intstatus(unsigned addr, reg_t& val) {
    val = (clic_mact_lvl & 0xff) << 24;
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_fcsr(unsigned addr, reg_t& val) {
    val = this->get_fcsr();
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::write_fcsr(unsigned addr, reg_t val) {
    this->set_fcsr(val);
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::write_intthresh(unsigned addr, reg_t val) {
    csr[addr] = (val & 0xff) | (1 << (cfg.clic_int_ctl_bits)) - 1;
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::write_xtvt(unsigned addr, reg_t val) {
    csr[addr] = val & ~0x3fULL;
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_mem(phys_addr_t paddr, unsigned length, uint8_t* const data) {
    switch(paddr.val) {
    default: {
        for(auto offs = 0U; offs < length; ++offs) {
            *(data + offs) = mem[(paddr.val + offs) % mem.size()];
        }
    }
    }
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::write_mem(phys_addr_t paddr, unsigned length, const uint8_t* const data) {
    switch(paddr.val) {
    // TODO remove UART, Peripherals should not be part of the ISS
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

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::read_clic(uint64_t addr, unsigned length, uint8_t* const data) {
    if(addr == cfg.clic_base) { // cliccfg
        *data = clic_cfg_reg;
        for(auto i = 1; i < length; ++i)
            *(data + i) = 0;
    } else if(addr >= (cfg.clic_base + 0x40) && (addr + length) <= (cfg.clic_base + 0x40 + cfg.clic_num_trigger * 4)) { // clicinttrig
        auto offset = ((addr & 0x7fff) - 0x40) / 4;
        read_reg_uint32(addr, clic_inttrig_reg[offset], data, length);
    } else if(addr >= (cfg.clic_base + 0x1000) &&
              (addr + length) <= (cfg.clic_base + 0x1000 + cfg.clic_num_irq * 4)) { // clicintip/clicintie/clicintattr/clicintctl
        auto offset = ((addr & 0x7fff) - 0x1000) / 4;
        read_reg_uint32(addr, clic_int_reg[offset].raw, data, length);
    } else {
        for(auto i = 0U; i < length; ++i)
            *(data + i) = 0;
    }
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_m_p<BASE, FEAT, LOGCAT>::write_clic(uint64_t addr, unsigned length, const uint8_t* const data) {
    if(addr == cfg.clic_base) { // cliccfg
        clic_cfg_reg = (clic_cfg_reg & ~0x1e) | (*data & 0x1e);
    } else if(addr >= (cfg.clic_base + 0x40) && (addr + length) <= (cfg.clic_base + 0x40 + cfg.clic_num_trigger * 4)) { // clicinttrig
        auto offset = ((addr & 0x7fff) - 0x40) / 4;
        write_reg_uint32(addr, clic_inttrig_reg[offset], data, length);
    } else if(addr >= (cfg.clic_base + 0x1000) &&
              (addr + length) <= (cfg.clic_base + 0x1000 + cfg.clic_num_irq * 4)) { // clicintip/clicintie/clicintattr/clicintctl
        auto offset = ((addr & 0x7fff) - 0x1000) / 4;
        write_reg_uint32(addr, clic_int_reg[offset].raw, data, length);
        clic_int_reg[offset].raw &= 0xf0c70101; // clicIntCtlBits->0xf0, clicintattr->0xc7, clicintie->0x1, clicintip->0x1
    }
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT> inline void riscv_hart_m_p<BASE, FEAT, LOGCAT>::reset(uint64_t address) {
    BASE::reset(address);
    state.mstatus = hart_state_type::mstatus_reset_val;
}

template <typename BASE, features_e FEAT, typename LOGCAT> void riscv_hart_m_p<BASE, FEAT, LOGCAT>::check_interrupt() {
    // TODO: Implement CLIC functionality
    // auto ideleg = csr[mideleg];
    // Multiple simultaneous interrupts and traps at the same privilege level are
    // handled in the following decreasing priority order:
    // external interrupts, software interrupts, timer interrupts, then finally
    // any synchronous traps.
    auto ena_irq = csr[mip] & csr[mie];

    bool mstatus_mie = state.mstatus.MIE;
    auto m_enabled = this->reg.PRIV < PRIV_M || mstatus_mie;
    auto enabled_interrupts = m_enabled ? ena_irq : 0;

    if(enabled_interrupts != 0) {
        int res = 0;
        while((enabled_interrupts & 1) == 0) {
            enabled_interrupts >>= 1;
            res++;
        }
        this->reg.pending_trap = res << 16 | 1; // 0x80 << 24 | (cause << 16) | trap_id
    }
}

template <typename BASE, features_e FEAT, typename LOGCAT>
uint64_t riscv_hart_m_p<BASE, FEAT, LOGCAT>::enter_trap(uint64_t flags, uint64_t addr, uint64_t tval) {
    // flags are ACTIVE[31:31], CAUSE[30:16], TRAPID[15:0]
    // calculate and write mcause val
    auto const trap_id = bit_sub<0, 16>(flags);
    auto cause = bit_sub<16, 15>(flags);
    // calculate effective privilege level
    unsigned new_priv = PRIV_M;
    if(trap_id == 0) { // exception
        if(cause == 11)
            cause = 0x8 + PRIV_M; // adjust environment call cause
        // store ret addr in xepc register
        csr[mepc] = static_cast<reg_t>(addr) & get_pc_mask(); // store actual address instruction of exception
        /*
         * write mtval if new_priv=M_MODE, spec says:
         * When a hardware breakpoint is triggered, or an instruction-fetch, load,
         * or store address-misaligned,
         * access, or page-fault exception occurs, mtval is written with the
         * faulting effective address.
         */
        switch(cause) {
        case 0:
            csr[mtval] = static_cast<reg_t>(tval);
            break;
        case 2:
            csr[mtval] = (!has_compressed() || (tval & 0x3) == 3) ? tval : tval & 0xffff;
            break;
        case 3:
            if((FEAT & FEAT_DEBUG) && (csr[dcsr] & 0x8000)) {
                this->reg.DPC = addr;
                csr[dcsr] = (csr[dcsr] & ~0x1c3) | (1 << 6) | PRIV_M; // FIXME: cause should not be 4 (stepi)
                new_priv = this->reg.PRIV | PRIV_D;
            } else {
                csr[mtval] = addr;
            }
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
                    NSCLOG(INFO, LOGCAT) << "Semihosting call at address " << buffer.data() << " occurred ";

                    semihosting_cb(this, &(this->reg.X10) /*a0*/, &(this->reg.X11) /*a1*/);
                    return this->reg.NEXT_PC;
                }
            }
            break;
        case 4:
        case 6:
            csr[mtval] = fault_data;
            break;
        default:
            csr[mtval] = 0;
        }
        fault_data = 0;
    } else {
        csr[mepc] = this->reg.NEXT_PC & get_pc_mask(); // store next address if interrupt
        this->reg.pending_trap = 0;
    }
    csr[mcause] = (trap_id << (traits<BASE>::XLEN - 1)) + cause;
    // update mstatus
    // xPP field of mstatus is written with the active privilege mode at the time
    // of the trap; the x PIE field of mstatus
    // is written with the value of the active interrupt-enable bit at the time of
    // the trap; and the x IE field of mstatus
    // is cleared
    // store the actual privilege level in yPP and store interrupt enable flags
    state.mstatus.MPP = PRIV_M;
    state.mstatus.MPIE = state.mstatus.MIE;
    state.mstatus.MIE = false;

    // get trap vector
    auto xtvec = csr[mtvec];
    // calculate adds// set NEXT_PC to trap addressess to jump to based on MODE
    if((FEAT & features_e::FEAT_CLIC) && trap_id != 0 && (xtvec & 0x3UL) == 3UL) {
        reg_t data;
        auto ret = read(address_type::LOGICAL, access_type::READ, 0, csr[mtvt], sizeof(reg_t), reinterpret_cast<uint8_t*>(&data));
        if(ret == iss::Err)
            return this->reg.PC;
        this->reg.NEXT_PC = data;
    } else {
        // bits in mtvec
        this->reg.NEXT_PC = xtvec & ~0x3UL;
        if((xtvec & 0x1) == 1 && trap_id != 0)
            this->reg.NEXT_PC += 4 * cause;
    }
    // reset trap state
    this->reg.PRIV = new_priv;
    this->reg.trap_state = 0;
    std::array<char, 32> buffer;
#if defined(_MSC_VER)
    sprintf(buffer.data(), "0x%016llx", addr);
#else
    sprintf(buffer.data(), "0x%016lx", addr);
#endif
    if((flags & 0xffffffff) != 0xffffffff)
        NSCLOG(INFO, LOGCAT) << (trap_id ? "Interrupt" : "Trap") << " with cause '" << (trap_id ? irq_str[cause] : trap_str[cause]) << "' ("
                             << cause << ")"
                             << " at address " << buffer.data() << " occurred";
    return this->reg.NEXT_PC;
}

template <typename BASE, features_e FEAT, typename LOGCAT> uint64_t riscv_hart_m_p<BASE, FEAT, LOGCAT>::leave_trap(uint64_t flags) {
    state.mstatus.MIE = state.mstatus.MPIE;
    state.mstatus.MPIE = 1;
    // sets the pc to the value stored in the x epc register.
    this->reg.NEXT_PC = csr[mepc] & get_pc_mask();
    NSCLOG(INFO, LOGCAT) << "Executing xRET";
    check_interrupt();
    this->reg.trap_state = this->reg.pending_trap;
    return this->reg.NEXT_PC;
}

} // namespace arch
} // namespace iss

#endif /* _RISCV_HART_M_P_H */
