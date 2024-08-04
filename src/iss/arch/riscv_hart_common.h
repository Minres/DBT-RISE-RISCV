/*******************************************************************************
 * Copyright (C) 2017, 2018, 2021 MINRES Technologies GmbH
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

#include <cstdint>
#include <elfio/elfio.hpp>
#include <fmt/format.h>
#include <iss/arch_if.h>
#include <iss/log_categories.h>
#include <string>
#include <unordered_map>
#include <util/logging.h>

#if defined(__GNUC__)
#define likely(x) ::__builtin_expect(!!(x), 1)
#define unlikely(x) ::__builtin_expect(!!(x), 0)
#else
#define likely(x) x
#define unlikely(x) x
#endif

namespace iss {
namespace arch {

enum { tohost_dflt = 0xF0001000, fromhost_dflt = 0xF0001040 };

enum features_e { FEAT_NONE, FEAT_PMP = 1, FEAT_EXT_N = 2, FEAT_CLIC = 4, FEAT_DEBUG = 8, FEAT_TCM = 16 };

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
    dscratch1 = 0x7B3
};

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

struct vm_info {
    int levels;
    int idxbits;
    int ptesize;
    uint64_t ptbase;
    bool is_active() { return levels; }
};

struct feature_config {
    uint64_t clic_base{0xc0000000};
    unsigned clic_int_ctl_bits{4};
    unsigned clic_num_irq{16};
    unsigned clic_num_trigger{0};
    uint64_t tcm_base{0x10000000};
    uint64_t tcm_size{0x8000};
    uint64_t io_address{0xf0000000};
    uint64_t io_addr_mask{0xf0000000};
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

inline void read_reg_uint32(uint64_t offs, uint32_t& reg, uint8_t* const data, unsigned length) {
    auto reg_ptr = reinterpret_cast<uint8_t*>(&reg);
    switch(offs & 0x3) {
    case 0:
        for(auto i = 0U; i < length; ++i)
            *(data + i) = *(reg_ptr + i);
        break;
    case 1:
        for(auto i = 0U; i < length; ++i)
            *(data + i) = *(reg_ptr + 1 + i);
        break;
    case 2:
        for(auto i = 0U; i < length; ++i)
            *(data + i) = *(reg_ptr + 2 + i);
        break;
    case 3:
        *data = *(reg_ptr + 3);
        break;
    }
}

inline void write_reg_uint32(uint64_t offs, uint32_t& reg, const uint8_t* const data, unsigned length) {
    auto reg_ptr = reinterpret_cast<uint8_t*>(&reg);
    switch(offs & 0x3) {
    case 0:
        for(auto i = 0U; i < length; ++i)
            *(reg_ptr + i) = *(data + i);
        break;
    case 1:
        for(auto i = 0U; i < length; ++i)
            *(reg_ptr + 1 + i) = *(data + i);
        break;
    case 2:
        for(auto i = 0U; i < length; ++i)
            *(reg_ptr + 2 + i) = *(data + i);
        break;
    case 3:
        *(reg_ptr + 3) = *data;
        break;
    }
}
struct riscv_hart_common {
    riscv_hart_common(){};
    ~riscv_hart_common(){};
    std::unordered_map<std::string, uint64_t> symbol_table;

    std::unordered_map<std::string, uint64_t> get_sym_table(std::string name) {
        if(!symbol_table.empty())
            return symbol_table;
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
                if(reader.get_type() != ELFIO::ET_EXEC)
                    throw std::runtime_error("wrong elf type in file");
                if(reader.get_machine() != ELFIO::EM_RISCV)
                    throw std::runtime_error("wrong elf machine in file");
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
                            CPPLOG(DEBUG) << "Found Symbol " << name;
#endif
                        }
                    }
                }
                return symbol_table;
            }
            throw std::runtime_error(fmt::format("memory load file {} is not a valid elf file", name));
        } else
            throw std::runtime_error(fmt::format("memory load file not found, check if {} is a valid file", name));
    };
};

} // namespace arch
} // namespace iss

#endif
