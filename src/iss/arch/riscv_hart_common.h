/*******************************************************************************
 * Copyright (C) 2017 - 2025 MINRES Technologies GmbH
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

#ifndef _RISCV_HART_COMMON
#define _RISCV_HART_COMMON

#include "mstatus.h"
#include "util/delegate.h"
#include <array>
#include <cstdint>
#include <elfio/elfio.hpp>
#include <fmt/format.h>
#include <iss/arch/traits.h>
#include <iss/arch_if.h>
#include <iss/log_categories.h>
#include <iss/mem/memory_if.h>
#include <iss/semihosting/semihosting.h>
#include <iss/vm_types.h>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <util/logging.h>
#include <util/sparse_array.h>

#if defined(__GNUC__)
#define likely(x) ::__builtin_expect(!!(x), 1)
#define unlikely(x) ::__builtin_expect(!!(x), 0)
#else
#define likely(x) x
#define unlikely(x) x
#endif

namespace iss {
namespace arch {

enum features_e { FEAT_NONE, FEAT_EXT_N = 1, FEAT_DEBUG = 2 };

enum riscv_csr {
    /* user-level CSR */
    // User Trap Setup
    ustatus = 0x000,
    uie = 0x004,
    utvec = 0x005,
    utvt = 0x007, // CLIC
    // User Trap Handling
    uscratch = 0x040,
    uepc = 0x041,
    ucause = 0x042,
    utval = 0x043,
    uip = 0x044,
    uxnti = 0x045,        // CLIC
    uintstatus = 0xCB1,   // MRW Current interrupt levels (CLIC) - addr subject to change
    uintthresh = 0x047,   // MRW Interrupt-level threshold (CLIC) - addr subject to change
    uscratchcsw = 0x048,  // MRW Conditional scratch swap on priv mode change (CLIC)
    uscratchcswl = 0x049, // MRW Conditional scratch swap on level change (CLIC)
    // User Floating-Point CSRs
    fflags = 0x001,
    frm = 0x002,
    fcsr = 0x003,
    // User Counter/Timers
    cycle = 0xC00,
    time = 0xC01,
    instret = 0xC02,
    hpmcounter3 = 0xC03,
    hpmcounter4 = 0xC04,
    /*...*/
    hpmcounter31 = 0xC1F,
    cycleh = 0xC80,
    timeh = 0xC81,
    instreth = 0xC82,
    hpmcounter3h = 0xC83,
    hpmcounter4h = 0xC84,
    /*...*/
    hpmcounter31h = 0xC9F,
    /* supervisor-level CSR */
    // Supervisor Trap Setup
    sstatus = 0x100,
    sedeleg = 0x102,
    sideleg = 0x103,
    sie = 0x104,
    stvec = 0x105,
    scounteren = 0x106,
    // Supervisor Trap Handling
    sscratch = 0x140,
    sepc = 0x141,
    scause = 0x142,
    stval = 0x143,
    sip = 0x144,
    // Supervisor Protection and Translation
    satp = 0x180,
    /* machine-level CSR */
    // Machine Information Registers
    mvendorid = 0xF11,
    marchid = 0xF12,
    mimpid = 0xF13,
    mhartid = 0xF14,
    // Machine Trap Setup
    mstatus = 0x300,
    misa = 0x301,
    medeleg = 0x302,
    mideleg = 0x303,
    mie = 0x304,
    mtvec = 0x305,
    mcounteren = 0x306,
    mtvt = 0x307, // CLIC
    medelegh = 0x312,
    // Machine Trap Handling
    mscratch = 0x340,
    mepc = 0x341,
    mcause = 0x342,
    mtval = 0x343,
    mip = 0x344,
    mxnti = 0x345,        // CLIC
    mintstatus = 0xFB1,   // MRW Current interrupt levels (CLIC) - addr subject to change
    mintthresh = 0x347,   // MRW Interrupt-level threshold (CLIC) - addr subject to change
    mscratchcsw = 0x348,  // MRW Conditional scratch swap on priv mode change (CLIC)
    mscratchcswl = 0x349, // MRW Conditional scratch swap on level change (CLIC)
    // Physical Memory Protection
    pmpcfg0 = 0x3A0,
    pmpcfg1 = 0x3A1,
    pmpcfg2 = 0x3A2,
    pmpcfg3 = 0x3A3,
    pmpaddr0 = 0x3B0,
    pmpaddr1 = 0x3B1,
    pmpaddr2 = 0x3B2,
    pmpaddr3 = 0x3B3,
    pmpaddr4 = 0x3B4,
    pmpaddr5 = 0x3B5,
    pmpaddr6 = 0x3B6,
    pmpaddr7 = 0x3B7,
    pmpaddr8 = 0x3B8,
    pmpaddr9 = 0x3B9,
    pmpaddr10 = 0x3BA,
    pmpaddr11 = 0x3BB,
    pmpaddr12 = 0x3BC,
    pmpaddr13 = 0x3BD,
    pmpaddr14 = 0x3BE,
    pmpaddr15 = 0x3BF,
    // Machine Counter/Timers
    mcycle = 0xB00,
    minstret = 0xB02,
    mhpmcounter3 = 0xB03,
    mhpmcounter4 = 0xB04,
    /*...*/
    mhpmcounter31 = 0xB1F,
    mcycleh = 0xB80,
    minstreth = 0xB82,
    mhpmcounter3h = 0xB83,
    mhpmcounter4h = 0xB84,
    /*...*/
    mhpmcounter31h = 0xB9F,
    // Machine Counter Setup
    mhpmevent3 = 0x323,
    mhpmevent4 = 0x324,
    /*...*/
    mhpmevent31 = 0x33F,
    // Debug/Trace Registers (shared with Debug Mode)
    tselect = 0x7A0,
    tdata1 = 0x7A1,
    tdata2 = 0x7A2,
    tdata3 = 0x7A3,
    // Debug Mode Registers
    dcsr = 0x7B0,
    dpc = 0x7B1,
    dscratch0 = 0x7B2,
    dscratch1 = 0x7B3,
    // vector CSR
    //  URW
    vstart = 0x008,
    vxsat = 0x009,
    vxrm = 0x00A,
    vcsr = 0x00F,
    // URO
    vl = 0xC20,
    vtype = 0xC21,
    vlenb = 0xC22,
};

enum { PRIV_U = 0, PRIV_S = 1, PRIV_M = 3, PRIV_D = 4 };

enum {
    ISA_A = 1,
    ISA_B = 1 << 1,
    ISA_C = 1 << 2,
    ISA_D = 1 << 3,
    ISA_E = 1 << 4,
    ISA_F = 1 << 5,
    ISA_G = 1 << 6,
    ISA_I = 1 << 8,
    ISA_M = 1 << 12,
    ISA_N = 1 << 13,
    ISA_Q = 1 << 16,
    ISA_S = 1 << 18,
    ISA_U = 1 << 20
};

class trap_instruction_access_fault : public trap_access {
public:
    trap_instruction_access_fault(uint64_t badaddr)
    : trap_access(1 << 16, badaddr) {}
};
class trap_load_access_fault : public trap_access {
public:
    trap_load_access_fault(uint64_t badaddr)
    : trap_access(5 << 16, badaddr) {}
};
class trap_store_access_fault : public trap_access {
public:
    trap_store_access_fault(uint64_t badaddr)
    : trap_access(7 << 16, badaddr) {}
};
class trap_instruction_page_fault : public trap_access {
public:
    trap_instruction_page_fault(uint64_t badaddr)
    : trap_access(12 << 16, badaddr) {}
};
class trap_load_page_fault : public trap_access {
public:
    trap_load_page_fault(uint64_t badaddr)
    : trap_access(13 << 16, badaddr) {}
};
class trap_store_page_fault : public trap_access {
public:
    trap_store_page_fault(uint64_t badaddr)
    : trap_access(15 << 16, badaddr) {}
};

template <typename WORD_TYPE> struct priv_if {
    using rd_csr_f = std::function<iss::status(unsigned addr, WORD_TYPE&)>;
    using wr_csr_f = std::function<iss::status(unsigned addr, WORD_TYPE)>;

    std::function<iss::status(unsigned, WORD_TYPE&)> read_csr;
    std::function<iss::status(unsigned, WORD_TYPE)> write_csr;
    std::function<iss::status(uint8_t const*)> exec_htif;
    std::function<void(uint16_t, uint16_t, WORD_TYPE)> raise_trap; // trap_id, cause, fault_data
    std::unordered_map<unsigned, rd_csr_f>& csr_rd_cb;
    std::unordered_map<unsigned, wr_csr_f>& csr_wr_cb;
    hart_state<WORD_TYPE>& state;
    uint8_t& PRIV;
    WORD_TYPE& PC;
    uint64_t& tohost;
    uint64_t& fromhost;
    unsigned& max_irq;
};

template <typename BASE, typename LOGCAT = logging::disass> struct riscv_hart_common : public BASE, public mem::memory_elem {
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
    constexpr static unsigned MEM = traits<BASE>::MEM;

    using core = BASE;
    using this_class = riscv_hart_common<BASE, LOGCAT>;
    using reg_t = typename core::reg_t;

    using rd_csr_f = std::function<iss::status(unsigned addr, reg_t&)>;
    using wr_csr_f = std::function<iss::status(unsigned addr, reg_t)>;

#define MK_CSR_RD_CB(FCT) [this](unsigned a, reg_t& r) -> iss::status { return this->FCT(a, r); };
#define MK_CSR_WR_CB(FCT) [this](unsigned a, reg_t r) -> iss::status { return this->FCT(a, r); };

    riscv_hart_common()
    : state()
    , instr_if(*this) {
        // reset values
        csr[misa] = traits<BASE>::MISA_VAL;
        csr[mvendorid] = 0x669;
        csr[marchid] = traits<BASE>::MARCHID_VAL;
        csr[mimpid] = 1;

        if(traits<BASE>::FLEN > 0) {
            csr_rd_cb[fcsr] = MK_CSR_RD_CB(read_fcsr);
            csr_wr_cb[fcsr] = MK_CSR_WR_CB(write_fcsr);
            csr_rd_cb[fflags] = MK_CSR_RD_CB(read_fcsr);
            csr_wr_cb[fflags] = MK_CSR_WR_CB(write_fcsr);
            csr_rd_cb[frm] = MK_CSR_RD_CB(read_fcsr);
            csr_wr_cb[frm] = MK_CSR_WR_CB(write_fcsr);
        }
        if(traits<BASE>::V_REGS_SIZE > 0) {
            csr_rd_cb[vstart] = MK_CSR_RD_CB(read_vstart);
            csr_wr_cb[vstart] = MK_CSR_WR_CB(write_vstart);
            csr_rd_cb[vxsat] = MK_CSR_RD_CB(read_vxsat);
            csr_wr_cb[vxsat] = MK_CSR_WR_CB(write_vxsat);
            csr_rd_cb[vxrm] = MK_CSR_RD_CB(read_vxrm);
            csr_wr_cb[vxrm] = MK_CSR_WR_CB(write_vxrm);
            csr_rd_cb[vcsr] = MK_CSR_RD_CB(read_vcsr);
            csr_wr_cb[vcsr] = MK_CSR_WR_CB(write_vcsr);
            csr_rd_cb[vl] = MK_CSR_RD_CB(read_vl);
            csr_rd_cb[vtype] = MK_CSR_RD_CB(read_vtype);
            csr_rd_cb[vlenb] = MK_CSR_RD_CB(read_vlenb);
        }
        for(unsigned addr = mhpmcounter3; addr <= mhpmcounter31; ++addr) {
            csr_rd_cb[addr] = MK_CSR_RD_CB(read_null);
            csr_wr_cb[addr] = MK_CSR_WR_CB(write_plain);
        }
        if(traits<BASE>::XLEN == 32)
            for(unsigned addr = mhpmcounter3h; addr <= mhpmcounter31h; ++addr) {
                csr_rd_cb[addr] = MK_CSR_RD_CB(read_null);
                csr_wr_cb[addr] = MK_CSR_WR_CB(write_plain);
            }
        for(unsigned addr = mhpmevent3; addr <= mhpmevent31; ++addr) {
            csr_rd_cb[addr] = MK_CSR_RD_CB(read_null);
            csr_wr_cb[addr] = MK_CSR_WR_CB(write_plain);
        }
        for(unsigned addr = hpmcounter3; addr <= hpmcounter31; ++addr) {
            csr_rd_cb[addr] = MK_CSR_RD_CB(read_null);
        }
        if(traits<BASE>::XLEN == 32)
            for(unsigned addr = hpmcounter3h; addr <= hpmcounter31h; ++addr) {
                csr_rd_cb[addr] = MK_CSR_RD_CB(read_null);
            }
        // common regs
        const std::array<unsigned, 4> roaddrs{{misa, mvendorid, marchid, mimpid}};
        for(auto addr : roaddrs) {
            csr_rd_cb[addr] = MK_CSR_RD_CB(read_plain);
            csr_wr_cb[addr] = MK_CSR_WR_CB(write_null);
        }
        // special handling & overrides
        csr_rd_cb[time] = MK_CSR_RD_CB(read_time);
        if(traits<BASE>::XLEN == 32)
            csr_rd_cb[timeh] = MK_CSR_RD_CB(read_time);
        csr_rd_cb[cycle] = MK_CSR_RD_CB(read_cycle);
        if(traits<BASE>::XLEN == 32)
            csr_rd_cb[cycleh] = MK_CSR_RD_CB(read_cycle);
        csr_rd_cb[instret] = MK_CSR_RD_CB(read_instret);
        if(traits<BASE>::XLEN == 32)
            csr_rd_cb[instreth] = MK_CSR_RD_CB(read_instret);

        csr_rd_cb[mcycle] = MK_CSR_RD_CB(read_cycle);
        csr_wr_cb[mcycle] = MK_CSR_WR_CB(write_cycle);
        if(traits<BASE>::XLEN == 32)
            csr_rd_cb[mcycleh] = MK_CSR_RD_CB(read_cycle);
        if(traits<BASE>::XLEN == 32)
            csr_wr_cb[mcycleh] = MK_CSR_WR_CB(write_cycle);
        csr_rd_cb[minstret] = MK_CSR_RD_CB(read_instret);
        csr_wr_cb[minstret] = MK_CSR_WR_CB(write_instret);
        if(traits<BASE>::XLEN == 32)
            csr_rd_cb[minstreth] = MK_CSR_RD_CB(read_instret);
        if(traits<BASE>::XLEN == 32)
            csr_wr_cb[minstreth] = MK_CSR_WR_CB(write_instret);
        csr_rd_cb[mhartid] = MK_CSR_RD_CB(read_hartid);
    };

    ~riscv_hart_common() {
        if(io_buf.str().length()) {
            CPPLOG(INFO) << "tohost send '" << io_buf.str() << "'";
        }
    }

    std::unordered_map<std::string, uint64_t> symbol_table;
    uint64_t entry_address{0};
    uint64_t tohost = std::numeric_limits<uint64_t>::max();
    uint64_t fromhost = std::numeric_limits<uint64_t>::max();
    std::stringstream io_buf;

    void set_semihosting_callback(semihosting_cb_t<reg_t> cb) { semihosting_cb = cb; };

    std::pair<uint64_t, bool> load_file(std::string name, int type) {
        return std::make_pair(entry_address, read_elf_file(name, sizeof(reg_t) == 4 ? ELFIO::ELFCLASS32 : ELFIO::ELFCLASS64));
    }

    bool read_elf_file(std::string name, uint8_t expected_elf_class) {
        // Create elfio reader
        ELFIO::elfio reader;
        // Load ELF data
        if(reader.load(name)) {
            // check elf properties
            if(reader.get_class() != expected_elf_class) {
                CPPLOG(ERR) << "ISA missmatch, selected XLEN does not match supplied file ";
                return false;
            }
            if(reader.get_type() != ELFIO::ET_EXEC)
                return false;
            if(reader.get_machine() != ELFIO::EM_RISCV)
                return false;
            entry_address = reader.get_entry();
            for(const auto& pseg : reader.segments) {
                const auto fsize = pseg->get_file_size(); // 0x42c/0x0
                const auto seg_data = pseg->get_data();
                const auto type = pseg->get_type();
                if(type == ELFIO::PT_LOAD && fsize > 0) {
                    auto res = this->write(iss::address_type::PHYSICAL, iss::access_type::DEBUG_WRITE, traits<BASE>::MEM,
                                           pseg->get_physical_address(), fsize, reinterpret_cast<const uint8_t* const>(seg_data));
                    if(res != iss::Ok)
                        CPPLOG(ERR) << "problem writing " << fsize << " bytes to 0x" << std::hex << pseg->get_physical_address();
                }
            }
            const auto sym_sec = reader.sections[".symtab"];
            if(ELFIO::SHT_SYMTAB == sym_sec->get_type() || ELFIO::SHT_DYNSYM == sym_sec->get_type()) {
                ELFIO::symbol_section_accessor symbols(reader, sym_sec);
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
                    if(name != "") {
                        this->symbol_table[name] = value;
#ifndef NDEBUG
                        CPPLOG(TRACE) << "Found Symbol " << name;
#endif
                    }
                }
                auto to_it = symbol_table.find("tohost");
                if(to_it != std::end(symbol_table))
                    tohost = to_it->second;
                auto from_it = symbol_table.find("tohost");
                if(from_it != std::end(symbol_table))
                    tohost = from_it->second;
            }
            return true;
        }
        return false;
    };

    iss::status execute_sys_write(arch_if* aif, const std::array<uint64_t, 8>& loaded_payload, unsigned mem_type) {
        uint64_t fd = loaded_payload[1];
        uint64_t buf_ptr = loaded_payload[2];
        uint64_t len = loaded_payload[3];
        std::vector<char> buf(len);
        if(aif->read(address_type::PHYSICAL, access_type::DEBUG_READ, mem_type, buf_ptr, len, reinterpret_cast<uint8_t*>(buf.data()))) {
            CPPLOG(ERR) << "SYS_WRITE buffer read went wrong";
            return iss::Err;
        }
        // we disregard the fd and just log to stdout
        for(size_t i = 0; i < len; i++) {
            if(buf[i] == '\n' || buf[i] == '\0') {
                CPPLOG(INFO) << "tohost send '" << io_buf.str() << "'";
                io_buf.str("");
            } else
                io_buf << buf[i];
        }

        // Not sure what the correct return value should be
        uint8_t ret_val = 1;
        if(fromhost != std::numeric_limits<uint64_t>::max())
            if(aif->write(address_type::PHYSICAL, access_type::DEBUG_WRITE, mem_type, fromhost, 1, &ret_val)) {
                CPPLOG(ERR) << "Fromhost write went wrong";
                return iss::Err;
            }
        return iss::Ok;
    }

    constexpr bool has_compressed() { return traits<BASE>::MISA_VAL & 0b0100; }

    constexpr reg_t get_pc_mask() { return has_compressed() ? (reg_t)~1 : (reg_t)~3; }

    void disass_output(uint64_t pc, const std::string instr) override {
        // NSCLOG(INFO, LOGCAT) << fmt::format("0x{:016x}    {:40} [p:{};s:0x{:x};c:{}]", pc, instr, lvl[this->reg.PRIV],
        // (reg_t)state.mstatus,
        //                                     this->reg.cycle + cycle_offset);
        NSCLOG(INFO, LOGCAT) << fmt::format("0x{:016x}    {:40} [p:{};c:{}]", pc, instr, lvl[this->reg.PRIV],
                                            this->reg.cycle + cycle_offset);
    };

    void register_csr(unsigned addr, rd_csr_f f) { csr_rd_cb[addr] = f; }
    void register_csr(unsigned addr, wr_csr_f f) { csr_wr_cb[addr] = f; }
    void register_csr(unsigned addr, rd_csr_f rdf, wr_csr_f wrf) {
        csr_rd_cb[addr] = rdf;
        csr_wr_cb[addr] = wrf;
    }
    void unregister_csr_rd(unsigned addr) { csr_rd_cb.erase(addr); }
    void unregister_csr_wr(unsigned addr) { csr_wr_cb.erase(addr); }

    bool debug_mode_active() { return this->reg.PRIV & 0x4; }

    const reg_t& get_mhartid() const { return mhartid_reg; }
    void set_mhartid(reg_t mhartid) { mhartid_reg = mhartid; };

    iss::status read_csr(unsigned addr, reg_t& val) {
        if(addr >= csr.size()) {
            this->reg.trap_state = (1U << 31) | traits<BASE>::RV_CAUSE_ILLEGAL_INSTRUCTION << 16;
            return iss::Err;
        }
        auto req_priv_lvl = (addr >> 8) & 0x3;
        if(this->reg.PRIV < req_priv_lvl) { // not having required privileges
            this->reg.trap_state = (1U << 31) | traits<BASE>::RV_CAUSE_ILLEGAL_INSTRUCTION << 16;
            return iss::Err;
        }
        auto it = csr_rd_cb.find(addr);
        if(it == csr_rd_cb.end() || !it->second) { // non existent register
            this->reg.trap_state = (1U << 31) | traits<BASE>::RV_CAUSE_ILLEGAL_INSTRUCTION << 16;
            return iss::Err;
        }
        return it->second(addr, val);
    }

    iss::status write_csr(unsigned addr, reg_t val) {
        if(addr >= csr.size()) {
            this->reg.trap_state = (1U << 31) | traits<BASE>::RV_CAUSE_ILLEGAL_INSTRUCTION << 16;
            return iss::Err;
        }
        auto req_priv_lvl = (addr >> 8) & 0x3;
        if(this->reg.PRIV < req_priv_lvl) { // not having required privileges
            this->reg.trap_state = (1U << 31) | traits<BASE>::RV_CAUSE_ILLEGAL_INSTRUCTION << 16;
            return iss::Err;
        }
        if((addr & 0xc00) == 0xc00) { // writing to read-only region
            this->reg.trap_state = (1U << 31) | traits<BASE>::RV_CAUSE_ILLEGAL_INSTRUCTION << 16;
            return iss::Err;
        }
        auto it = csr_wr_cb.find(addr);
        if(it == csr_wr_cb.end() || !it->second) { // non existent register
            this->reg.trap_state = (1U << 31) | traits<BASE>::RV_CAUSE_ILLEGAL_INSTRUCTION << 16;
            return iss::Err;
        }
        return it->second(addr, val);
    }

    iss::status read_null(unsigned addr, reg_t& val) {
        val = 0;
        return iss::Ok;
    }

    iss::status write_null(unsigned addr, reg_t val) { return iss::status::Ok; }

    iss::status read_plain(unsigned addr, reg_t& val) {
        val = csr[addr];
        return iss::Ok;
    }

    iss::status write_plain(unsigned addr, reg_t val) {
        csr[addr] = val;
        return iss::Ok;
    }

    iss::status read_cycle(unsigned addr, reg_t& val) {
        auto cycle_val = this->reg.cycle + cycle_offset;
        if(addr == mcycle) {
            val = static_cast<reg_t>(cycle_val);
        } else if(addr == mcycleh) {
            val = static_cast<reg_t>(cycle_val >> 32);
        }
        return iss::Ok;
    }

    iss::status write_cycle(unsigned addr, reg_t val) {
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

    iss::status read_instret(unsigned addr, reg_t& val) {
        if((addr & 0xff) == (minstret & 0xff)) {
            val = static_cast<reg_t>(this->reg.instret);
        } else if((addr & 0xff) == (minstreth & 0xff)) {
            val = static_cast<reg_t>(this->reg.instret >> 32);
        }
        return iss::Ok;
    }

    iss::status write_instret(unsigned addr, reg_t val) {
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

    iss::status read_time(unsigned addr, reg_t& val) {
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

    iss::status read_tvec(unsigned addr, reg_t& val) {
        val = csr[addr] & ~2;
        return iss::Ok;
    }

    iss::status read_hartid(unsigned addr, reg_t& val) {
        val = mhartid_reg;
        return iss::Ok;
    }

    iss::status write_epc(unsigned addr, reg_t val) {
        csr[addr] = val & get_pc_mask();
        return iss::Ok;
    }

    iss::status write_dcsr(unsigned addr, reg_t val) {
        if(!debug_mode_active())
            return iss::Err;
        //                  +-------------- ebreakm
        //                  |   +---------- stepi
        //                  |   |  +++----- cause
        //                  |   |  |||   +- step
        csr[addr] = val & 0b1000100111000100U;
        return iss::Ok;
    }

    iss::status read_debug(unsigned addr, reg_t& val) {
        if(!debug_mode_active())
            return iss::Err;
        val = csr[addr];
        return iss::Ok;
    }

    iss::status write_dscratch(unsigned addr, reg_t val) {
        if(!debug_mode_active())
            return iss::Err;
        csr[addr] = val;
        return iss::Ok;
    }

    iss::status read_dpc(unsigned addr, reg_t& val) {
        if(!debug_mode_active())
            return iss::Err;
        val = this->reg.DPC;
        return iss::Ok;
    }

    iss::status write_dpc(unsigned addr, reg_t val) {
        if(!debug_mode_active())
            return iss::Err;
        this->reg.DPC = val;
        return iss::Ok;
    }

    iss::status read_fcsr(unsigned addr, reg_t& val) {
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

    iss::status write_fcsr(unsigned addr, reg_t val) {
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

    iss::status read_vstart(unsigned addr, reg_t& val) {
        val = this->get_vstart();
        return iss::Ok;
    }

    iss::status write_vstart(unsigned addr, reg_t val) {
        this->set_vstart(val);
        return iss::Ok;
    }

    iss::status read_vxsat(unsigned addr, reg_t& val) {
        val = this->get_vxsat();
        return iss::Ok;
    }

    iss::status write_vxsat(unsigned addr, reg_t val) {
        this->set_vxsat(val & 1);
        csr[vcsr] = (~1ULL & csr[vcsr]) | (val & 1);
        return iss::Ok;
    }

    iss::status read_vxrm(unsigned addr, reg_t& val) {
        val = this->get_vxrm();
        return iss::Ok;
    }

    iss::status write_vxrm(unsigned addr, reg_t val) {
        this->set_vxrm(val & 0b11);
        csr[vcsr] = (~0b110ULL & csr[vcsr]) | ((val & 0b11) << 1);
        return iss::Ok;
    }

    iss::status read_vcsr(unsigned addr, reg_t& val) {
        val = csr[vcsr];
        return iss::Ok;
    }

    iss::status write_vcsr(unsigned addr, reg_t val) {
        csr[vcsr] = val;
        return iss::Ok;
    }

    iss::status read_vl(unsigned addr, reg_t& val) {
        val = this->get_vl();
        return iss::Ok;
    }

    iss::status read_vtype(unsigned addr, reg_t& val) {
        val = this->get_vtype();
        return iss::Ok;
    }

    iss::status read_vlenb(unsigned addr, reg_t& val) {
        val = csr[vlenb];
        return iss::Ok;
    }

    priv_if<reg_t> get_priv_if() {
        return priv_if<reg_t>{.read_csr = [this](unsigned addr, reg_t& val) -> iss::status { return read_csr(addr, val); },
                              .write_csr = [this](unsigned addr, reg_t val) -> iss::status { return write_csr(addr, val); },
                              .exec_htif = [this](uint8_t const* data) -> iss::status { return execute_htif(data); },
                              .raise_trap =
                                  [this](uint16_t trap_id, uint16_t cause, reg_t fault_data) {
                                      this->reg.trap_state = 0x80ULL << 24 | (cause << 16) | trap_id;
                                      this->fault_data = fault_data;
                                  },
                              .csr_rd_cb{this->csr_rd_cb},
                              .csr_wr_cb{csr_wr_cb},
                              .state{this->state},
                              .PRIV{this->reg.PRIV},
                              .PC{this->reg.PC},
                              .tohost{this->tohost},
                              .fromhost{this->fromhost},
                              .max_irq{mcause_max_irq}};
    }

    iss::status execute_htif(uint8_t const* data) {
        reg_t cur_data = *reinterpret_cast<const reg_t*>(data);
        // Extract Device (bits 63:56)
        uint8_t device = traits<BASE>::XLEN == 32 ? 0 : (cur_data >> 56) & 0xFF;
        // Extract Command (bits 55:48)
        uint8_t command = traits<BASE>::XLEN == 32 ? 0 : (cur_data >> 48) & 0xFF;
        // Extract payload (bits 47:0)
        uint64_t payload_addr = cur_data & 0xFFFFFFFFFFFFULL;
        if(payload_addr & 1) {
            CPPLOG(FATAL) << "this->tohost value is 0x" << std::hex << payload_addr << std::dec << " (" << payload_addr
                          << "), stopping simulation";
            this->reg.trap_state = std::numeric_limits<uint32_t>::max();
            this->interrupt_sim = payload_addr;
            return iss::Ok;
        } else if(device == 0 && command == 0) {
            std::array<uint64_t, 8> loaded_payload;
            if(memory.rd_mem(access_type::DEBUG_READ, payload_addr, 8 * sizeof(uint64_t),
                             reinterpret_cast<uint8_t*>(loaded_payload.data())) == iss::Err)
                CPPLOG(ERR) << "Syscall read went wrong";
            uint64_t syscall_num = loaded_payload.at(0);
            if(syscall_num == 64) { // SYS_WRITE
                return this->execute_sys_write(this, loaded_payload, traits<BASE>::MEM);
            } else {
                CPPLOG(ERR) << "this->tohost syscall with number 0x" << std::hex << syscall_num << std::dec << " (" << syscall_num
                            << ") not implemented";
                this->reg.trap_state = std::numeric_limits<uint32_t>::max();
                this->interrupt_sim = payload_addr;
                return iss::Ok;
            }
        } else {
            CPPLOG(ERR) << "this->tohost functionality not implemented for device " << device << " and command " << command;
            this->reg.trap_state = std::numeric_limits<uint32_t>::max();
            this->interrupt_sim = payload_addr;
            return iss::Ok;
        }
    }

    mem::memory_hierarchy memories;

    mem::memory_if get_mem_if() override {
        assert(false || "This function should never be called");
        return mem::memory_if{};
    }

    void set_next(mem::memory_if mem_if) override { memory = mem_if; };

    void set_irq_num(unsigned i) { mcause_max_irq = 1 << util::ilog2(i); }

protected:
    hart_state<reg_t> state;

    static constexpr reg_t get_mstatus_mask_t(unsigned priv_lvl = PRIV_M) {
        if(sizeof(reg_t) == 4) {
            return priv_lvl == PRIV_U ? 0x80000011UL :   // 0b1...0 0001 0001
                       priv_lvl == PRIV_S ? 0x800de133UL // 0b0...0 0001 1000 1001 1001;
                                          : 0x807ff9ddUL;
        } else {
            return priv_lvl == PRIV_U ? 0x011ULL : // 0b1...0 0001 0001
                       priv_lvl == PRIV_S ? 0x000de133ULL
                                          : 0x007ff9ddULL;
        }
    }

    mem::memory_if memory;
    struct riscv_instrumentation_if : public iss::instrumentation_if {

        riscv_instrumentation_if(riscv_hart_common<BASE, LOGCAT>& arch)
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

        std::unordered_map<std::string, uint64_t> const& get_symbol_table(std::string name) override { return arch.symbol_table; }

        riscv_hart_common<BASE, LOGCAT>& arch;
    };

    friend struct riscv_instrumentation_if;
    riscv_instrumentation_if instr_if;

    instrumentation_if* get_instrumentation_if() override { return &instr_if; };

    using csr_type = std::array<typename traits<BASE>::reg_t, 1ULL << 12>;
    csr_type csr;

    std::unordered_map<unsigned, rd_csr_f> csr_rd_cb;
    std::unordered_map<unsigned, wr_csr_f> csr_wr_cb;

    reg_t mhartid_reg{0x0};
    uint64_t mcycle_csr{0};
    uint64_t minstret_csr{0};
    reg_t fault_data;

    int64_t cycle_offset{0};
    int64_t instret_offset{0};
    semihosting_cb_t<reg_t> semihosting_cb;
    unsigned mcause_max_irq{16U};
};

} // namespace arch
} // namespace iss

#endif
