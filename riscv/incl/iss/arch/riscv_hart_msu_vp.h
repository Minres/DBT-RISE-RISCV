/*******************************************************************************
 * Copyright (C) 2017, MINRES Technologies GmbH
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
 *       eyck@minres.com - initial API and implementation
 ******************************************************************************/

#ifndef _RISCV_CORE_H_
#define _RISCV_CORE_H_

#include <elfio/elfio.hpp>
#include <iomanip>
#include <iss/arch/traits.h>
#include <iss/arch_if.h>
#include <iss/log_categories.h>
#include <iss/vm_if.h>
#include <sstream>
#include <unordered_map>
#include <util/ities.h>
#include <util/sparse_array.h>
#include <util/bit_field.h>

namespace iss {
namespace arch {

enum { tohost_dflt = 0xF0001000, fromhost_dflt = 0xF0001040 };

enum csr_name {
    /* user-level CSR */
    // User Trap Setup
    ustatus = 0x000,
    uie = 0x004,
    utvec = 0x005,
    // User Trap Handling
    uscratch = 0x040,
    uepc = 0x041,
    ucause = 0x042,
    utval = 0x043,
    uip = 0x044,
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
    // Machine Trap Handling
    mscratch = 0x340,
    mepc = 0x341,
    mcause = 0x342,
    mtval = 0x343,
    mip = 0x344,
    // Machine Protection and Translation
    pmpcfg0 = 0x3A0,
    pmpcfg1 = 0x3A1,
    pmpcfg2 = 0x3A2,
    pmpcfg3 = 0x3A3,
    pmpaddr0 = 0x3B0,
    pmpaddr1 = 0x3B1,
    /*...*/
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
    dscratch = 0x7B2
};

namespace {

const char lvl[] = {'U', 'S', 'H', 'M'};

const char *trap_str[] = {"Instruction address misaligned",
                          "Instruction access fault",
                          "Illegal instruction",
                          "Breakpoint",
                          "Load address misaligned",
                          "Load access fault",
                          "Store/AMO address misaligned",
                          "Store/AMO access fault",
                          "Environment call from U-mode",
                          "Environment call from S-mode",
                          "Reserved",
                          "Environment call from M-mode",
                          "Instruction page fault",
                          "Load page fault",
                          "Reserved",
                          "Store/AMO page fault"};
const char *irq_str[] = {
    "User software interrupt", "Supervisor software interrupt", "Reserved", "Machine software interrupt",
    "User timer interrupt",    "Supervisor timer interrupt",    "Reserved", "Machine timer interrupt",
    "User external interrupt", "Supervisor external interrupt", "Reserved", "Machine external interrupt"};

enum {
    PGSHIFT = 12,
    PTE_PPN_SHIFT = 10,
    // page table entry (PTE) fields
    PTE_V = 0x001,   // Valid
    PTE_R = 0x002,   // Read
    PTE_W = 0x004,   // Write
    PTE_X = 0x008,   // Execute
    PTE_U = 0x010,   // User
    PTE_G = 0x020,   // Global
    PTE_A = 0x040,   // Accessed
    PTE_D = 0x080,   // Dirty
    PTE_SOFT = 0x300 // Reserved for Software
};

template <typename T> inline bool PTE_TABLE(T PTE) { return (((PTE) & (PTE_V | PTE_R | PTE_W | PTE_X)) == PTE_V); }

enum { PRIV_U = 0, PRIV_S = 1, PRIV_M = 3 };

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

struct vm_info {
    int levels;
    int idxbits;
    int ptesize;
    uint64_t ptbase;
};

class trap_load_access_fault : public trap_access {
public:
    trap_load_access_fault(uint64_t badaddr)
    : trap_access(5 << 16, badaddr) {}
};
class illegal_instruction_fault : public trap_access {
public:
    illegal_instruction_fault(uint64_t badaddr)
    : trap_access(2 << 16, badaddr) {}
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
}

template <typename BASE> class riscv_hart_msu_vp : public BASE {
public:
    using super = BASE;
    using this_class = riscv_hart_msu_vp<BASE>;
    using virt_addr_t = typename super::virt_addr_t;
    using phys_addr_t = typename super::phys_addr_t;
    using reg_t = typename super::reg_t;
    using addr_t = typename super::addr_t;

    using rd_csr_f = iss::status (this_class::*)(unsigned addr, reg_t &);
    using wr_csr_f = iss::status (this_class::*)(unsigned addr, reg_t);

    // primary template
    template<class T, class Enable = void> struct hart_state { };
    // specialization 32bit
    template <typename T>
    class hart_state<T, typename std::enable_if<std::is_same<T, uint32_t>::value>::type> {
    public:
        BEGIN_BF_DECL(mstatus_t, T);
        // SD bit is read-only and is set when either the FS or XS bits encode a Dirty state (i.e., SD=((FS==11) OR XS==11)))
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
        // status of additional user-mode extensions and associated state, All off/None dirty or clean, some on/None dirty, some clean/Some dirty
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

        T satp;

        static constexpr T get_misa() { return (1UL << 30) | ISA_I | ISA_M | ISA_A | ISA_U | ISA_S | ISA_M; }

        static constexpr uint32_t get_mask(unsigned priv_lvl) {
            switch (priv_lvl) {
            case PRIV_U: return 0x80000011UL; // 0b1000 0000 0000 0000 0000 0000 0001 0001
            case PRIV_S: return 0x800de133UL; // 0b1000 0000 0000 1101 1110 0001 0011 0011
            default:     return 0x807ff9ddUL; // 0b1000 0000 0111 1111 1111 1001 1011 1011
            }
        }

        static inline vm_info decode_vm_info(uint32_t state, T sptbr) {
            if (state == PRIV_M) return {0, 0, 0, 0};
            if (state <= PRIV_S)
                switch (bit_sub<31, 1>(sptbr)) {
                case 0:  return {0, 0, 0, 0}; // off
                case 1:  return {2, 10, 4, bit_sub<0, 22>(sptbr) << PGSHIFT}; // SV32
                default: abort();
                }
            abort();
            return {0, 0, 0, 0}; // dummy
        }
    };
    // specialization 64bit
    template <typename T>
    class hart_state<T, typename std::enable_if<std::is_same<T, uint64_t>::value>::type> {
    public:
        BEGIN_BF_DECL(mstatus_t, T);
        // SD bit is read-only and is set when either the FS or XS bits encode a Dirty state (i.e., SD=((FS==11) OR XS==11)))
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
        // status of additional user-mode extensions and associated state, All off/None dirty or clean, some on/None dirty, some clean/Some dirty
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

        T satp;

        static constexpr T get_misa() { return (2ULL << 62) | ISA_I | ISA_M | ISA_A | ISA_U | ISA_S | ISA_M; }

        static constexpr T get_mask(unsigned priv_lvl) {
            switch (priv_lvl) {
            case PRIV_U: return 0x8000000000000011ULL; // 0b1...0 1111 0000 0000 0111 1111 1111 1001 1011 1011
            case PRIV_S: return 0x80000003000de133ULL; // 0b1...0 0011 0000 0000 0000 1101 1110 0001 0011 0011
            default:     return 0x8000000f007ff9ddULL; // 0b1...0 1111 0000 0000 0111 1111 1111 1001 1011 1011
            }
        }

        static inline vm_info decode_vm_info(uint32_t state, T sptbr) {
            if (state == PRIV_M) return {0, 0, 0, 0};
            if (state <= PRIV_S)
                switch (bit_sub<60, 4>(sptbr)) {
                case 0: return {0, 0, 0, 0}; // off
                case 8: return {3, 9, 8, bit_sub<0, 44>(sptbr) << PGSHIFT};// SV39
                case 9: return {4, 9, 8, bit_sub<0, 44>(sptbr) << PGSHIFT};// SV48
                case 10: return {5, 9, 8, bit_sub<0, 44>(sptbr) << PGSHIFT};// SV57
                case 11: return {6, 9, 8, bit_sub<0, 44>(sptbr) << PGSHIFT};// SV64
                default: abort();
                }
            abort();
            return {0, 0, 0, 0}; // dummy
        }
    };

    const typename super::reg_t PGSIZE = 1 << PGSHIFT;
    const typename super::reg_t PGMASK = PGSIZE - 1;

    constexpr reg_t get_irq_mask(size_t mode) {
        const reg_t m[4] = {
            0b000100010001, // U mode
            0b001100110011, // S-mode
            0,
            0b101110111011 // M-mode
        };
        return m[mode];
    }

    riscv_hart_msu_vp();
    virtual ~riscv_hart_msu_vp() = default;

    void load_file(std::string name, int type = -1) override;

    virtual phys_addr_t v2p(const iss::addr_t &addr);

    iss::status read(const iss::addr_t &addr, unsigned length, uint8_t *const data) override;
    iss::status write(const iss::addr_t &addr, unsigned length, const uint8_t *const data) override;

    virtual uint64_t enter_trap(uint64_t flags) override { return riscv_hart_msu_vp::enter_trap(flags, fault_data); }
    virtual uint64_t enter_trap(uint64_t flags, uint64_t addr) override;
    virtual uint64_t leave_trap(uint64_t flags) override;
    void wait_until(uint64_t flags) override;

    void notify_phase(iss::arch_if::exec_phase phase);

    void disass_output(uint64_t pc, const std::string instr) override {
        std::stringstream s;
        s << "[p:" << lvl[this->reg.machine_state] << ";s:0x" << std::hex << std::setfill('0')
          << std::setw(sizeof(reg_t) * 2) << (reg_t)state.mstatus << std::dec << ";c:" << this->reg.icount << "]";
        CLOG(INFO, disass) << "0x"<<std::setw(16)<<std::setfill('0')<<std::hex<<pc<<"\t\t"<<instr<<"\t"<<s.str();
    };

protected:
    virtual iss::status read_mem(phys_addr_t addr, unsigned length, uint8_t *const data);
    virtual iss::status write_mem(phys_addr_t addr, unsigned length, const uint8_t *const data);

    virtual iss::status read_csr(unsigned addr, reg_t &val);
    virtual iss::status write_csr(unsigned addr, reg_t val);

    uint64_t tohost = tohost_dflt;
    uint64_t fromhost = fromhost_dflt;

    reg_t fault_data;
    using mem_type = util::sparse_array<uint8_t, 1ULL << 32>;
    using csr_type = util::sparse_array<typename traits<BASE>::reg_t, 1ULL << 12, 12>;
    using csr_page_type = typename csr_type::page_type;
    mem_type mem;
    csr_type csr;
    hart_state<reg_t> state;
    unsigned to_host_wr_cnt = 0;
    std::stringstream uart_buf;
    std::unordered_map<reg_t, uint64_t> ptw;
    std::unordered_map<uint64_t, uint8_t> atomic_reservation;
    std::unordered_map<unsigned, rd_csr_f> csr_rd_cb;
    std::unordered_map<unsigned, wr_csr_f> csr_wr_cb;

private:
    iss::status read_cycle(unsigned addr, reg_t &val);
    iss::status read_time(unsigned addr, reg_t &val);
    iss::status read_status(unsigned addr, reg_t &val);
    iss::status write_status(unsigned addr, reg_t val);
    iss::status read_ie(unsigned addr, reg_t &val);
    iss::status write_ie(unsigned addr, reg_t val);
    iss::status read_ip(unsigned addr, reg_t &val);
    iss::status write_ip(unsigned addr, reg_t val);
    iss::status read_satp(unsigned addr, reg_t &val);
    iss::status write_satp(unsigned addr, reg_t val);
    void check_interrupt();
};

template <typename BASE>
riscv_hart_msu_vp<BASE>::riscv_hart_msu_vp()
: state() {
    csr[misa] = hart_state<reg_t>::get_misa();
    uart_buf.str("");
    // read-only registers
    csr_wr_cb[misa] = nullptr;
    for (unsigned addr = mcycle; addr <= hpmcounter31; ++addr) csr_wr_cb[addr] = nullptr;
    for (unsigned addr = mcycleh; addr <= hpmcounter31h; ++addr) csr_wr_cb[addr] = nullptr;
    // special handling
    csr_rd_cb[time] = &riscv_hart_msu_vp<BASE>::read_time;
    csr_wr_cb[time] = nullptr;
    csr_rd_cb[timeh] = &riscv_hart_msu_vp<BASE>::read_time;
    csr_wr_cb[timeh] = nullptr;
    csr_rd_cb[mcycle] = &riscv_hart_msu_vp<BASE>::read_cycle;
    csr_rd_cb[mcycleh] = &riscv_hart_msu_vp<BASE>::read_cycle;
    csr_rd_cb[minstret] = &riscv_hart_msu_vp<BASE>::read_cycle;
    csr_rd_cb[minstreth] = &riscv_hart_msu_vp<BASE>::read_cycle;
    csr_rd_cb[mstatus] = &riscv_hart_msu_vp<BASE>::read_status;
    csr_wr_cb[mstatus] = &riscv_hart_msu_vp<BASE>::write_status;
    csr_rd_cb[sstatus] = &riscv_hart_msu_vp<BASE>::read_status;
    csr_wr_cb[sstatus] = &riscv_hart_msu_vp<BASE>::write_status;
    csr_rd_cb[ustatus] = &riscv_hart_msu_vp<BASE>::read_status;
    csr_wr_cb[ustatus] = &riscv_hart_msu_vp<BASE>::write_status;
    csr_rd_cb[mip] = &riscv_hart_msu_vp<BASE>::read_ip;
    csr_wr_cb[mip] = &riscv_hart_msu_vp<BASE>::write_ip;
    csr_rd_cb[sip] = &riscv_hart_msu_vp<BASE>::read_ip;
    csr_wr_cb[sip] = &riscv_hart_msu_vp<BASE>::write_ip;
    csr_rd_cb[uip] = &riscv_hart_msu_vp<BASE>::read_ip;
    csr_wr_cb[uip] = &riscv_hart_msu_vp<BASE>::write_ip;
    csr_rd_cb[mie] = &riscv_hart_msu_vp<BASE>::read_ie;
    csr_wr_cb[mie] = &riscv_hart_msu_vp<BASE>::write_ie;
    csr_rd_cb[sie] = &riscv_hart_msu_vp<BASE>::read_ie;
    csr_wr_cb[sie] = &riscv_hart_msu_vp<BASE>::write_ie;
    csr_rd_cb[uie] = &riscv_hart_msu_vp<BASE>::read_ie;
    csr_wr_cb[uie] = &riscv_hart_msu_vp<BASE>::write_ie;
    csr_rd_cb[satp] = &riscv_hart_msu_vp<BASE>::read_satp;
    csr_wr_cb[satp] = &riscv_hart_msu_vp<BASE>::write_satp;
}

template <typename BASE> void riscv_hart_msu_vp<BASE>::load_file(std::string name, int type) {
    FILE *fp = fopen(name.c_str(), "r");
    if (fp) {
        char buf[5];
        auto n = fread(buf, 1, 4, fp);
        if (n != 4) throw std::runtime_error("input file has insufficient size");
        buf[4] = 0;
        if (strcmp(buf + 1, "ELF") == 0) {
            fclose(fp);
            // Create elfio reader
            ELFIO::elfio reader;
            // Load ELF data
            if (!reader.load(name)) throw std::runtime_error("could not process elf file");
            // check elf properties
            if ( reader.get_class() != ELFCLASS32 )
                if(sizeof(reg_t) == 4) throw std::runtime_error("wrong elf class in file");
            if (reader.get_type() != ET_EXEC) throw std::runtime_error("wrong elf type in file");
            if ( reader.get_machine() != EM_RISCV ) throw std::runtime_error("wrong elf machine in file");
            for (const auto pseg : reader.segments) {
                const auto fsize = pseg->get_file_size(); // 0x42c/0x0
                const auto seg_data = pseg->get_data();
                if (fsize > 0) {
                    auto res = this->write(
                        typed_addr_t<PHYSICAL>(iss::DEBUG_WRITE, traits<BASE>::MEM, pseg->get_physical_address()),
                        fsize, reinterpret_cast<const uint8_t *const>(seg_data));
                    if (res != iss::Ok)
                        LOG(ERROR) << "problem writing " << fsize << "bytes to 0x" << std::hex
                                   << pseg->get_physical_address();
                }
            }
            for (const auto sec : reader.sections) {
                if (sec->get_name() == ".tohost") {
                    tohost = sec->get_address();
                    fromhost = tohost + 0x40;
                }
            }
            return;
        }
    }
}

template <typename BASE>
iss::status riscv_hart_msu_vp<BASE>::read(const iss::addr_t &addr, unsigned length, uint8_t *const data) {
#ifndef NDEBUG
    if (addr.type & iss::DEBUG) {
        LOG(DEBUG) << "debug read of " << length << " bytes @addr " << addr;
    } else {
        LOG(DEBUG) << "read of " << length << " bytes  @addr " << addr;
    }
#endif
    switch (addr.space) {
    case traits<BASE>::MEM: {
        if ((addr.type & (iss::ACCESS_TYPE - iss::DEBUG)) == iss::FETCH && (addr.val & 0x1) == 1) {
            fault_data = addr.val;
            if ((addr.type & iss::DEBUG)) throw trap_access(0, addr.val);
            this->reg.trap_state = (1 << 31); // issue trap 0
            return iss::Err;
        }
        try {
            if ((addr.val & ~PGMASK) != ((addr.val + length - 1) & ~PGMASK)) { // we may cross a page boundary
                vm_info vm = hart_state<reg_t>::decode_vm_info(this->reg.machine_state, state.satp);
                if (vm.levels != 0) { // VM is active
                    auto split_addr = (addr.val + length) & ~PGMASK;
                    auto len1 = split_addr - addr.val;
                    auto res = read(addr, len1, data);
                    if (res == iss::Ok)
                        res = read(iss::addr_t{addr.type, addr.space, split_addr}, length - len1, data + len1);
                    return res;
                }
            }
            phys_addr_t paddr = (addr.type & iss::ADDRESS_TYPE) == iss::PHYSICAL ? addr : v2p(addr);
            auto res = read_mem(paddr, length, data);
            if (res != iss::Ok) this->reg.trap_state = (1 << 31) | (5 << 16); // issue trap 5 (load access fault
            return res;
        } catch (trap_access &ta) {
            this->reg.trap_state = (1 << 31) | ta.id;
            return iss::Err;
        }
    } break;
    case traits<BASE>::CSR: {
        if (length != sizeof(reg_t)) return iss::Err;
        return read_csr(addr.val, *reinterpret_cast<reg_t *const>(data));
    } break;
    case traits<BASE>::FENCE: {
        if ((addr.val + length) > mem.size()) return iss::Err;
        switch (addr.val) {
        case 2:   // SFENCE:VMA lower
        case 3: { // SFENCE:VMA upper
            auto tvm = state.mstatus.TVM;
            if (this->reg.machine_state == PRIV_S & tvm != 0) {
                this->reg.trap_state = (1 << 31) | (2 << 16);
                this->fault_data = this->reg.PC;
                return iss::Err;
            }
            return iss::Ok;
        }
        }
    } break;
    case traits<BASE>::RES: {
        auto it = atomic_reservation.find(addr.val);
        if (it != atomic_reservation.end() && (*it).second != 0) {
            memset(data, 0xff, length);
            atomic_reservation.erase(addr.val);
        } else
            memset(data, 0, length);
    } break;
    default:
        return iss::Err; // assert("Not supported");
    }
    return iss::Ok;
}

template <typename BASE>
iss::status riscv_hart_msu_vp<BASE>::write(const iss::addr_t &addr, unsigned length, const uint8_t *const data) {
#ifndef NDEBUG
    const char *prefix = addr.type & iss::DEBUG ? "debug " : "";
    switch (length) {
    case 8:
        LOG(DEBUG) << prefix << "write of " << length << " bytes (0x" << std::hex << *(uint64_t *)&data[0] << std::dec
                   << ") @addr " << addr;
        break;
    case 4:
        LOG(DEBUG) << prefix << "write of " << length << " bytes (0x" << std::hex << *(uint32_t *)&data[0] << std::dec
                   << ") @addr " << addr;
        break;
    case 2:
        LOG(DEBUG) << prefix << "write of " << length << " bytes (0x" << std::hex << *(uint16_t *)&data[0] << std::dec
                   << ") @addr " << addr;
        break;
    case 1:
        LOG(DEBUG) << prefix << "write of " << length << " bytes (0x" << std::hex << (uint16_t)data[0] << std::dec
                   << ") @addr " << addr;
        break;
    default:
        LOG(DEBUG) << prefix << "write of " << length << " bytes @addr " << addr;
    }
#endif
    try {
        switch (addr.space) {
        case traits<BASE>::MEM: {
            if ((addr.type & (iss::ACCESS_TYPE - iss::DEBUG)) == iss::FETCH && (addr.val & 0x1) == 1) {
                fault_data = addr.val;
                if ((addr.type & iss::DEBUG)) throw trap_access(0, addr.val);
                this->reg.trap_state = (1 << 31); // issue trap 0
                return iss::Err;
            }
            try {
                if ((addr.val & ~PGMASK) != ((addr.val + length - 1) & ~PGMASK)) { // we may cross a page boundary
                    vm_info vm = hart_state<reg_t>::decode_vm_info(this->reg.machine_state, state.satp);
                    if (vm.levels != 0) { // VM is active
                        auto split_addr = (addr.val + length) & ~PGMASK;
                        auto len1 = split_addr - addr.val;
                        auto res = write(addr, len1, data);
                        if (res == iss::Ok)
                            res = write(iss::addr_t{addr.type, addr.space, split_addr}, length - len1, data + len1);
                        return res;
                    }
                }
                phys_addr_t paddr = (addr.type & iss::ADDRESS_TYPE) == iss::PHYSICAL ? addr : v2p(addr);
                auto res = write_mem(paddr, length, data);
                if (res != iss::Ok) this->reg.trap_state = (1 << 31) | (7 << 16); // issue trap 7 (load access fault
                return res;
            } catch (trap_access &ta) {
                this->reg.trap_state = (1 << 31) | ta.id;
                return iss::Err;
            }

            phys_addr_t paddr = (addr.type & iss::ADDRESS_TYPE) == iss::PHYSICAL ? addr : v2p(addr);
            if ((paddr.val + length) > mem.size()) return iss::Err;
            switch (paddr.val) {
            case 0x10013000: // UART0 base, TXFIFO reg
            case 0x10023000: // UART1 base, TXFIFO reg
                uart_buf << (char)data[0];
                if (((char)data[0]) == '\n' || data[0] == 0) {
                    // LOG(INFO)<<"UART"<<((paddr.val>>16)&0x3)<<" send
                    // '"<<uart_buf.str()<<"'";
                    std::cout << uart_buf.str();
                    uart_buf.str("");
                }
                return iss::Ok;
            case 0x10008000: { // HFROSC base, hfrosccfg reg
                auto &p = mem(paddr.val / mem.page_size);
                auto offs = paddr.val & mem.page_addr_mask;
                std::copy(data, data + length, p.data() + offs);
                auto &x = *(p.data() + offs + 3);
                if (x & 0x40) x |= 0x80; // hfroscrdy = 1 if hfroscen==1
                return iss::Ok;
            }
            case 0x10008008: { // HFROSC base, pllcfg reg
                auto &p = mem(paddr.val / mem.page_size);
                auto offs = paddr.val & mem.page_addr_mask;
                std::copy(data, data + length, p.data() + offs);
                auto &x = *(p.data() + offs + 3);
                x |= 0x80; // set pll lock upon writing
                return iss::Ok;
            } break;
            default: {}
            }
        } break;
        case traits<BASE>::CSR: {
            if (length != sizeof(reg_t)) return iss::Err;
            return write_csr(addr.val, *reinterpret_cast<const reg_t *>(data));
        } break;
        case traits<BASE>::FENCE: {
            if ((addr.val + length) > mem.size()) return iss::Err;
            switch (addr.val) {
            case 2:
            case 3: {
                ptw.clear();
                auto tvm = state.mstatus.TVM;
                if (this->reg.machine_state == PRIV_S & tvm != 0) {
                    this->reg.trap_state = (1 << 31) | (2 << 16);
                    this->fault_data = this->reg.PC;
                    return iss::Err;
                }
                return iss::Ok;
            }
            }
        } break;
        case traits<BASE>::RES: {
            atomic_reservation[addr.val] = data[0];
        } break;
        default:
            return iss::Err;
        }
        return iss::Ok;
    } catch (trap_access &ta) {
        this->reg.trap_state = (1 << 31) | ta.id;
        return iss::Err;
    }
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_csr(unsigned addr, reg_t &val) {
    if (addr >= csr.size()) return iss::Err;
    auto it = csr_rd_cb.find(addr);
    if (it == csr_rd_cb.end()) {
        val = csr[addr & csr.page_addr_mask];
        return iss::Ok;
    }
    rd_csr_f f = it->second;
    if (f == nullptr) throw illegal_instruction_fault(this->fault_data);
    return (this->*f)(addr, val);
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::write_csr(unsigned addr, reg_t val) {
    if (addr >= csr.size()) return iss::Err;
    auto it = csr_wr_cb.find(addr);
    if (it == csr_wr_cb.end()) {
        csr[addr & csr.page_addr_mask] = val;
        return iss::Ok;
    }
    wr_csr_f f = it->second;
    if (f == nullptr) throw illegal_instruction_fault(this->fault_data);
    return (this->*f)(addr, val);
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_cycle(unsigned addr, reg_t &val) {
    if (addr == mcycle) {
        val = static_cast<reg_t>(this->reg.icount);
    } else if (addr == mcycleh) {
        if (sizeof(typename traits<BASE>::reg_t) != 4) return iss::Err;
        val = static_cast<reg_t>((this->reg.icount) >> 32);
    }
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_time(unsigned addr, reg_t &val) {
	uint64_t time_val=this->reg.icount>>12;
    if (addr == time) {
        val = static_cast<reg_t>(time_val);
    } else if (addr == timeh) {
        if (sizeof(typename traits<BASE>::reg_t) != 4) return iss::Err;
        val = static_cast<reg_t>(time_val >> 32);
    }
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_status(unsigned addr, reg_t &val) {
    auto req_priv_lvl = addr >> 8;
    if (this->reg.machine_state < req_priv_lvl) throw illegal_instruction_fault(this->fault_data);
    val = state.mstatus & hart_state<reg_t>::get_mask(req_priv_lvl);
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::write_status(unsigned addr, reg_t val) {
    auto req_priv_lvl = addr >> 8;
    if (this->reg.machine_state < req_priv_lvl) throw illegal_instruction_fault(this->fault_data);
    auto mask = hart_state<reg_t>::get_mask(req_priv_lvl);
    auto old_val = state.mstatus;
    auto new_val = (old_val & ~mask) | (val & mask);
    state.mstatus = new_val;
    check_interrupt();
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_ie(unsigned addr, reg_t &val) {
    auto req_priv_lvl = addr >> 8;
    if (this->reg.machine_state < req_priv_lvl) throw illegal_instruction_fault(this->fault_data);
    val = csr[mie];
    if (addr < mie) val &= csr[mideleg];
    if (addr < sie) val &= csr[sideleg];
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::write_ie(unsigned addr, reg_t val) {
    auto req_priv_lvl = addr >> 8;
    if (this->reg.machine_state < req_priv_lvl) throw illegal_instruction_fault(this->fault_data);
    auto mask = get_irq_mask(req_priv_lvl);
    csr[mie] = (csr[mie] & ~mask) | (val & mask);
    check_interrupt();
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_ip(unsigned addr, reg_t &val) {
    auto req_priv_lvl = addr >> 8;
    if (this->reg.machine_state < req_priv_lvl) throw illegal_instruction_fault(this->fault_data);
    val = csr[mip];
    if (addr < mip) val &= csr[mideleg];
    if (addr < sip) val &= csr[sideleg];
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::write_ip(unsigned addr, reg_t val) {
    auto req_priv_lvl = addr >> 8;
    if (this->reg.machine_state < req_priv_lvl) throw illegal_instruction_fault(this->fault_data);
    auto mask = get_irq_mask(req_priv_lvl);
    csr[mip] = (csr[mip] & ~mask) | (val & mask);
    check_interrupt();
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::read_satp(unsigned addr, reg_t &val) {
    reg_t tvm = state.mstatus.TVM;
    if (this->reg.machine_state == PRIV_S & tvm != 0) {
        this->reg.trap_state = (1 << 31) | (2 << 16);
        this->fault_data = this->reg.PC;
        return iss::Err;
    }
    val = state.satp;
    return iss::Ok;
}

template <typename BASE> iss::status riscv_hart_msu_vp<BASE>::write_satp(unsigned addr, reg_t val) {
    reg_t tvm = state.mstatus.TVM;
    if (this->reg.machine_state == PRIV_S & tvm != 0) {
        this->reg.trap_state = (1 << 31) | (2 << 16);
        this->fault_data = this->reg.PC;
        return iss::Err;
    }
    state.satp = val;
    return iss::Ok;
}

template <typename BASE>
iss::status riscv_hart_msu_vp<BASE>::read_mem(phys_addr_t paddr, unsigned length, uint8_t *const data) {
    if ((paddr.val + length) > mem.size()) return iss::Err;
    switch (paddr.val) {
    case 0x0200BFF8: { // CLINT base, mtime reg
    	if(sizeof(reg_t)<length) return iss::Err;
    	reg_t time_val=this->csr[time];
        std::copy((uint8_t *)&time_val, ((uint8_t *)&time_val) + length, data);
    } break;
    case 0x10008000: {
        const mem_type::page_type &p = mem(paddr.val / mem.page_size);
        uint64_t offs = paddr.val & mem.page_addr_mask;
        std::copy(p.data() + offs, p.data() + offs + length, data);
        if (this->reg.icount > 30000) data[3] |= 0x80;
    } break;
    default: {
        const auto &p = mem(paddr.val / mem.page_size);
        auto offs = paddr.val & mem.page_addr_mask;
        std::copy(p.data() + offs, p.data() + offs + length, data);
    }
    }
    return iss::Ok;
}

template <typename BASE>
iss::status riscv_hart_msu_vp<BASE>::write_mem(phys_addr_t paddr, unsigned length, const uint8_t *const data) {
    if ((paddr.val + length) > mem.size()) return iss::Err;
    switch (paddr.val) {
    case 0x10013000: // UART0 base, TXFIFO reg
    case 0x10023000: // UART1 base, TXFIFO reg
        uart_buf << (char)data[0];
        if (((char)data[0]) == '\n' || data[0] == 0) {
            // LOG(INFO)<<"UART"<<((paddr.val>>16)&0x3)<<" send
            // '"<<uart_buf.str()<<"'";
            std::cout << uart_buf.str();
            uart_buf.str("");
        }
        break;
    case 0x10008000: { // HFROSC base, hfrosccfg reg
        mem_type::page_type &p = mem(paddr.val / mem.page_size);
        size_t offs = paddr.val & mem.page_addr_mask;
        std::copy(data, data + length, p.data() + offs);
        uint8_t &x = *(p.data() + offs + 3);
        if (x & 0x40) x |= 0x80; // hfroscrdy = 1 if hfroscen==1
    } break;
    case 0x10008008: { // HFROSC base, pllcfg reg
        mem_type::page_type &p = mem(paddr.val / mem.page_size);
        size_t offs = paddr.val & mem.page_addr_mask;
        std::copy(data, data + length, p.data() + offs);
        uint8_t &x = *(p.data() + offs + 3);
        x |= 0x80; // set pll lock upon writing
    } break;
    default: {
        mem_type::page_type &p = mem(paddr.val / mem.page_size);
        std::copy(data, data + length, p.data() + (paddr.val & mem.page_addr_mask));
        // tohost handling in case of riscv-test
        if ((paddr.type & iss::DEBUG) == 0) {
            auto tohost_upper = (traits<BASE>::XLEN == 32 && paddr.val == (tohost + 4)) ||
                                (traits<BASE>::XLEN == 64 && paddr.val == tohost);
            auto tohost_lower =
                (traits<BASE>::XLEN == 32 && paddr.val == tohost) || (traits<BASE>::XLEN == 64 && paddr.val == tohost);
            if (tohost_lower || tohost_upper) {
                uint64_t hostvar = *reinterpret_cast<uint64_t *>(p.data() + (tohost & mem.page_addr_mask));
                if (tohost_upper || (tohost_lower && to_host_wr_cnt > 0)) {
                    switch (hostvar >> 48) {
                    case 0:
                        if (hostvar != 0x1)
                            LOG(FATAL) << "tohost value is 0x" << std::hex << hostvar << std::dec << " (" << hostvar
                                       << "), stopping simulation";
                        else
                            LOG(INFO) << "tohost value is 0x" << std::hex << hostvar << std::dec << " (" << hostvar
                                      << "), stopping simulation";
                        throw(iss::simulation_stopped(hostvar));
                    case 0x0101: {
                        char c = static_cast<char>(hostvar & 0xff);
                        if (c == '\n' || c == 0) {
                            LOG(INFO) << "tohost send '" << uart_buf.str() << "'";
                            uart_buf.str("");
                        } else
                            uart_buf << c;
                        to_host_wr_cnt = 0;
                    } break;
                    default:
                        break;
                    }
                } else if (tohost_lower)
                    to_host_wr_cnt++;
            } else if ((traits<BASE>::XLEN == 32 && paddr.val == fromhost + 4) ||
                       (traits<BASE>::XLEN == 64 && paddr.val == fromhost)) {
                uint64_t fhostvar = *reinterpret_cast<uint64_t *>(p.data() + (fromhost & mem.page_addr_mask));
                *reinterpret_cast<uint64_t *>(p.data() + (tohost & mem.page_addr_mask)) = fhostvar;
            }
        }
    }
    }
    return iss::Ok;
}

template<typename BASE>
inline void riscv_hart_msu_vp<BASE>::notify_phase(iss::arch_if::exec_phase phase) {
	BASE::notify_phase(phase);
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

    auto mie = state.mstatus.MIE;
    auto m_enabled = this->reg.machine_state < PRIV_M || (this->reg.machine_state == PRIV_M && mie);
    auto enabled_interrupts = m_enabled ? ena_irq & ~ideleg : 0;

    if (enabled_interrupts == 0) {
        auto sie = state.mstatus.SIE;
        auto s_enabled = this->reg.machine_state < PRIV_S || (this->reg.machine_state == PRIV_S && sie);
        enabled_interrupts = s_enabled ? ena_irq & ideleg : 0;
    }
    if (enabled_interrupts != 0) {
        int res = 0;
        while ((enabled_interrupts & 1) == 0) enabled_interrupts >>= 1, res++;
        this->reg.pending_trap = res << 16 | 1; // 0x80 << 24 | (cause << 16) | trap_id
    }
}

template <typename BASE>
typename riscv_hart_msu_vp<BASE>::phys_addr_t riscv_hart_msu_vp<BASE>::v2p(const iss::addr_t &addr) {
    const uint64_t tmp = reg_t(1) << (traits<BASE>::XLEN - 1);
    const uint64_t msk = tmp | (tmp - 1);

    if (addr.space != traits<BASE>::MEM) { // non-memory access
        phys_addr_t ret(addr);
        ret.val &= msk;
        return ret;
    }

    const auto type = (access_type)(addr.getAccessType() & ~iss::DEBUG);
    uint32_t mode = type != iss::FETCH && state.mstatus.MPRV ? // MPRV
                        mode = state.mstatus.MPP:
                        this->reg.machine_state;

    const vm_info vm = hart_state<reg_t>::decode_vm_info(mode, state.satp);

    if (vm.levels == 0) {
        phys_addr_t ret(addr);
        ret.val &= msk;
        return ret;
    }

    const bool s_mode = mode == PRIV_S;
    const bool sum = state.mstatus.SUM;
    const bool mxr = state.mstatus.MXR;

    auto it = ptw.find(addr.val >> PGSHIFT);
    if (it != ptw.end()) {
        const reg_t pte = it->second;
        const reg_t ad = PTE_A | ((type == iss::WRITE) * PTE_D);
#ifdef RISCV_ENABLE_DIRTY
        // set accessed and possibly dirty bits.
        *(uint32_t *)ppte |= ad;
        return {addr.getAccessType(), addr.space, (pte & (~PGMASK)) | (addr.val & PGMASK)};
#else
        // take exception if access or possibly dirty bit is not set.
        if ((pte & ad) == ad)
            return {addr.getAccessType(), addr.space, (pte & (~PGMASK)) | (addr.val & PGMASK)};
        else
            ptw.erase(it);
#endif
    } else {
        // verify bits xlen-1:va_bits-1 are all equal
        const int va_bits = PGSHIFT + vm.levels * vm.idxbits;
        const reg_t mask = (reg_t(1) << (traits<BASE>::XLEN > -(va_bits - 1))) - 1;
        const reg_t masked_msbs = (addr.val >> (va_bits - 1)) & mask;
        const int levels = (masked_msbs != 0 && masked_msbs != mask) ? 0 : vm.levels;

        reg_t base = vm.ptbase;
        for (int i = levels - 1; i >= 0; i--) {
            const int ptshift = i * vm.idxbits;
            const reg_t idx = (addr.val >> (PGSHIFT + ptshift)) & ((1 << vm.idxbits) - 1);

            // check that physical address of PTE is legal
            reg_t pte = 0;
            const uint8_t res =
                this->read(phys_addr_t(addr.getAccessType(), traits<BASE>::MEM, base + idx * vm.ptesize), vm.ptesize,
                           (uint8_t *)&pte);
            if (res != 0) throw trap_load_access_fault(addr.val);
            const reg_t ppn = pte >> PTE_PPN_SHIFT;

            if (PTE_TABLE(pte)) { // next level of page table
                base = ppn << PGSHIFT;
            } else if ((pte & PTE_U) ? s_mode && (type == iss::FETCH || !sum) : !s_mode) {
                break;
            } else if (!(pte & PTE_V) || (!(pte & PTE_R) && (pte & PTE_W))) {
                break;
            } else if (type == iss::FETCH ? !(pte & PTE_X)
                                          : type == iss::READ ? !(pte & PTE_R) && !(mxr && (pte & PTE_X))
                                                              : !((pte & PTE_R) && (pte & PTE_W))) {
                break;
            } else if ((ppn & ((reg_t(1) << ptshift) - 1)) != 0) {
                break;
            } else {
                const reg_t ad = PTE_A | ((type == iss::WRITE) * PTE_D);
#ifdef RISCV_ENABLE_DIRTY
                // set accessed and possibly dirty bits.
                *(uint32_t *)ppte |= ad;
#else
                // take exception if access or possibly dirty bit is not set.
                if ((pte & ad) != ad) break;
#endif
                // for superpage mappings, make a fake leaf PTE for the TLB's benefit.
                const reg_t vpn = addr.val >> PGSHIFT;
                const reg_t value = (ppn | (vpn & ((reg_t(1) << ptshift) - 1))) << PGSHIFT;
                const reg_t offset = addr.val & PGMASK;
                ptw[vpn] = value | (pte & 0xff);
                return {addr.getAccessType(), addr.space, value | offset};
            }
        }
    }
    switch (type) {
    case FETCH:
        this->fault_data = addr.val;
        throw trap_instruction_page_fault(addr.val);
    case READ:
        this->fault_data = addr.val;
        throw trap_load_page_fault(addr.val);
    case WRITE:
        this->fault_data = addr.val;
        throw trap_store_page_fault(addr.val);
    default:
        abort();
    }
}

template <typename BASE> uint64_t riscv_hart_msu_vp<BASE>::enter_trap(uint64_t flags, uint64_t addr) {
    auto cur_priv = this->reg.machine_state;
    // flags are ACTIVE[31:31], CAUSE[30:16], TRAPID[15:0]
    // calculate and write mcause val
    auto trap_id = bit_sub<0, 16>(flags);
    auto cause = bit_sub<16, 15>(flags);
    if (trap_id == 0 && cause == 11) cause = 0x8 + cur_priv; // adjust environment call cause
    // calculate effective privilege level
    auto new_priv = PRIV_M;
    if (trap_id == 0) { // exception
        if (cur_priv != PRIV_M && ((csr[medeleg] >> cause) & 0x1) != 0)
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
        csr[utval | (new_priv << 8)] = fault_data;
        fault_data = 0;
    } else {
        if (cur_priv != PRIV_M && ((csr[mideleg] >> cause) & 0x1) != 0)
            new_priv = (csr[sideleg] >> cause) & 0x1 ? PRIV_U : PRIV_S;
        csr[uepc | (new_priv << 8)] = this->reg.NEXT_PC; // store next address if interrupt
        this->reg.pending_trap = 0;
    }
    csr[ucause | (new_priv << 8)] = cause;
    // update mstatus
    // xPP field of mstatus is written with the active privilege mode at the time
    // of the trap; the x PIE field of mstatus
    // is written with the value of the active interrupt-enable bit at the time of
    // the trap; and the x IE field of mstatus
    // is cleared
    // store the actual privilege level in yPP and store interrupt enable flags
    switch (new_priv) {
    case PRIV_M:
        state.mstatus.MPP=cur_priv;
        state.mstatus.MPIE=state.mstatus.MIE;
        break;
    case PRIV_S:
        state.mstatus.SPP = cur_priv;
        state.mstatus.SPIE=state.mstatus.SIE;
        break;
    case PRIV_U:
        state.mstatus.UPIE=state.mstatus.UIE;
        break;
    default:
        break;
    }

    // get trap vector
    auto ivec = csr[utvec | (new_priv << 8)];
    // calculate addr// set NEXT_PC to trap addressess to jump to based on MODE
    // bits in mtvec
    this->reg.NEXT_PC = ivec & ~0x1UL;
    if ((ivec & 0x1) == 1 && trap_id != 0) this->reg.NEXT_PC += 4 * cause;
    // reset trap state
    this->reg.machine_state = new_priv;
    this->reg.trap_state = 0;
    char buffer[32];
    sprintf(buffer, "0x%016lx", addr);
    CLOG(INFO, disass) << (trap_id ? "Interrupt " : "Trap ") << " with cause '" << (trap_id ? irq_str[cause] : trap_str[cause])
                       << "' at address " << buffer << " occurred, changing privilege level from " << lvl[cur_priv]
                       << " to " << lvl[new_priv];
    return this->reg.NEXT_PC;
}

template <typename BASE> uint64_t riscv_hart_msu_vp<BASE>::leave_trap(uint64_t flags) {
    auto cur_priv = this->reg.machine_state;
    auto inst_priv = flags & 0x3;
    auto status = state.mstatus;

    auto tsr = state.mstatus.TSR;
    if (cur_priv == PRIV_S && inst_priv == PRIV_S && tsr != 0) {
        this->reg.trap_state = (1 << 31) | (2 << 16);
        this->fault_data = this->reg.PC;
        return this->reg.PC;
    }

    // pop the relevant lower-privilege interrupt enable and privilege mode stack
    // clear respective yIE
    switch (inst_priv) {
    case PRIV_M:
        this->reg.machine_state = state.mstatus.MPP;
        state.mstatus.MPP=0; // clear mpp to U mode
        state.mstatus.MIE=state.mstatus.MPIE;
        break;
    case PRIV_S:
        this->reg.machine_state = state.mstatus.SPP;
        state.mstatus.SPP= 0; // clear spp to U mode
        state.mstatus.SIE=state.mstatus.SPIE;
        break;
    case PRIV_U:
        this->reg.machine_state = 0;
        state.mstatus.UIE=state.mstatus.UPIE;
        break;
    }
    // sets the pc to the value stored in the x epc register.
    this->reg.NEXT_PC = csr[uepc | inst_priv << 8];
    CLOG(INFO, disass) << "Executing xRET , changing privilege level from " << lvl[cur_priv] << " to " << lvl[this->reg.machine_state];
    return this->reg.NEXT_PC;
}

template <typename BASE> void riscv_hart_msu_vp<BASE>::wait_until(uint64_t flags) {
    auto status = state.mstatus;
    auto tw = status.TW;
    if (this->reg.machine_state == PRIV_S && tw != 0) {
        this->reg.trap_state = (1 << 31) | (2 << 16);
        this->fault_data = this->reg.PC;
    }
}
}
}

#endif /* _RISCV_CORE_H_ */
