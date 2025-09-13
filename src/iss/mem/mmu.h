/*******************************************************************************
 * Copyright (C) 2025 MINRES Technologies GmbH
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

#include "iss/arch/riscv_hart_common.h"
#include "iss/arch_if.h"
#include "iss/vm_types.h"
#include "memory_if.h"
#include "util/ities.h"
#include <cstdint>
#include <optional>
#include <type_traits>
#include <util/logging.h>

namespace iss {
namespace mem {
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

struct vm_info {
    int levels;
    int idxbits;
    int ptesize;
    uint64_t ptbase;
};

template <typename WORD_TYPE> struct mmu : public memory_elem {
    using this_class = mmu<WORD_TYPE>;
    using reg_t = WORD_TYPE;

    constexpr static reg_t PGSIZE = 1 << PGSHIFT;
    constexpr static reg_t PGMASK = PGSIZE - 1;

    mmu(arch::priv_if<WORD_TYPE> hart_if)
    : hart_if(hart_if) {
        hart_if.csr_rd_cb[arch::riscv_csr::satp] = MK_CSR_RD_CB(read_satp);
        hart_if.csr_wr_cb[arch::riscv_csr::satp] = MK_CSR_WR_CB(write_satp);
    }

    virtual ~mmu() = default;

    memory_if get_mem_if() override {
        return memory_if{.rd_mem{util::delegate<rd_mem_func_sig>::from<this_class, &this_class::read_mem>(this)},
                         .wr_mem{util::delegate<wr_mem_func_sig>::from<this_class, &this_class::write_mem>(this)}};
    }

    void set_next(memory_if mem) override { down_stream_mem = mem; }
    void flush_tlb(std::optional<reg_t> vaddr, std::optional<reg_t> asid) {
        // we do not control the tlb to the granularity allowed by the spec (evicting only parts)
        // we flush the entire buffer
        tlb.clear();
    }

private:
    uint32_t effective_priv(iss::access_type type) {
        auto priv = hart_if.PRIV;
        if(priv == arch::PRIV_M && (type & iss::access_type::FUNC) != iss::access_type::FETCH && hart_if.state.mstatus.MPRV)
            priv = hart_if.state.mstatus.MPP;
        return priv;
    }

    bool needs_translation(iss::access_type type) {
        return (effective_priv(type) == arch::PRIV_U || effective_priv(type) == arch::PRIV_S) && vm_setting.levels;
    }

    iss::status read_mem(iss::access_type access, uint64_t addr, unsigned length, uint8_t* data) {
        if(unlikely((addr & ~PGMASK) != ((addr + length - 1) & ~PGMASK) && needs_translation(access))) { // we may cross a page boundary
            auto split_addr = (addr + length) & ~PGMASK;
            auto len1 = split_addr - addr;
            auto res = down_stream_mem.rd_mem(access, virt2phys(access, addr), len1, data);
            if(res == iss::Ok)
                res = down_stream_mem.rd_mem(access, virt2phys(access, split_addr), length - len1, data + len1);
            return res;
        }
        return down_stream_mem.rd_mem(access, needs_translation(access) ? virt2phys(access, addr) : addr, length, data);
    }

    iss::status write_mem(iss::access_type access, uint64_t addr, unsigned length, uint8_t const* data) {
        if(unlikely((addr & ~PGMASK) != ((addr + length - 1) & ~PGMASK) && needs_translation(access))) { // we may cross a page boundary
            auto split_addr = (addr + length) & ~PGMASK;
            auto len1 = split_addr - addr;
            auto res = down_stream_mem.wr_mem(access, virt2phys(access, addr), len1, data);
            if(res == iss::Ok)
                res = down_stream_mem.wr_mem(access, virt2phys(access, split_addr), length - len1, data + len1);
            return res;
        }
        return down_stream_mem.wr_mem(access, needs_translation(access) ? virt2phys(access, addr) : addr, length, data);
    }

    iss::status read_plain(unsigned addr, reg_t& val) {
        val = hart_if.csr[addr];
        return iss::Ok;
    }

    iss::status write_plain(unsigned addr, reg_t const& val) {
        hart_if.csr[addr] = val;
        return iss::Ok;
    }

    iss::status read_satp(unsigned addr, reg_t& val) {
        reg_t tvm = hart_if.state.mstatus.TVM;
        if(hart_if.PRIV == arch::PRIV_S && tvm != 0) {
            hart_if.raise_trap(0, 2, hart_if.PC);
            return iss::Err;
        }
        val = satp;
        return iss::Ok;
    }

    iss::status write_satp(unsigned addr, reg_t val) {
        reg_t tvm = hart_if.state.mstatus.TVM;
        if(hart_if.PRIV == arch::PRIV_S && tvm != 0) {
            hart_if.raise_trap(0, 2, hart_if.PC);
            return iss::Err;
        }
        CPPLOG(INFO) << "mmu: writing satp with 0x" << std::hex << val;
        satp = val;
        update_vm_info();
        return iss::Ok;
    }

    uint64_t virt2phys(iss::access_type access, uint64_t addr);

    template <typename T = reg_t, std::enable_if_t<std::is_same_v<T, uint32_t>, bool> = true> inline void update_vm_info() {
        switch(bit_sub<31, 1>(satp)) {
        case 0:
            vm_setting = {0, 0, 0, 0}; // off
            break;
        case 1:
            vm_setting = {2, 10, 4, bit_sub<0, 22>(satp) << PGSHIFT}; // SV32
            break;
        default:
            abort();
        }
    }
    template <typename T = reg_t, std::enable_if_t<std::is_same_v<T, uint64_t>, bool> = true> inline void update_vm_info() {
        switch(bit_sub<60, 4>(satp)) {
        case 0:
            vm_setting = {0, 0, 0, 0}; // off
            break;
        case 8:
            vm_setting = {3, 9, 8, bit_sub<0, 44>(satp) << PGSHIFT}; // SV39
            break;
        case 9:
            vm_setting = {4, 9, 8, bit_sub<0, 44>(satp) << PGSHIFT}; // SV48
            break;
        case 10:
            vm_setting = {5, 9, 8, bit_sub<0, 44>(satp) << PGSHIFT}; // SV57
            break;
        case 11:
            vm_setting = {6, 9, 8, bit_sub<0, 44>(satp) << PGSHIFT}; // SV64
            break;
        default:
            abort();
        }
    }
    void throw_page_fault(iss::access_type type, uint64_t bad_addr) {
        switch(type) {
        case access_type::FETCH:
            throw arch::trap_instruction_page_fault(bad_addr);
        case access_type::READ:
            throw arch::trap_load_page_fault(bad_addr);
        case access_type::WRITE:
            throw arch::trap_store_page_fault(bad_addr);
        default:
            abort();
        }
    }

protected:
    reg_t satp;
    std::unordered_map<reg_t, uint64_t> tlb;
    vm_info vm_setting{0, 0, 0, 0};
    arch::priv_if<WORD_TYPE> hart_if;
    memory_if down_stream_mem;
};

template <typename WORD_TYPE> uint64_t mmu<WORD_TYPE>::virt2phys(iss::access_type access, uint64_t addr) {
    const auto type = access & iss::access_type::FUNC;
    reg_t pte{0};
    if(auto it = tlb.find(addr >> PGSHIFT); it != tlb.end()) {
        pte = it->second;
    } else {
        update_vm_info();
        reg_t base = vm_setting.ptbase;
        const int va_bits = vm_setting.idxbits * vm_setting.levels + PGSHIFT;
        const reg_t mask = (reg_t(1) << (sizeof(reg_t) * 8 - (va_bits - 1))) - 1;
        const reg_t masked_msbs = (addr >> (va_bits - 1)) & mask;
        if(masked_msbs != 0 && masked_msbs != mask) {
            CPPLOG(DEBUG) << "Page fault for address 0x" << std::hex << addr << ": invalid unused address bits";
            throw_page_fault(type, addr);
        }
        for(int i = vm_setting.levels - 1; i >= 0; i--) {
            const int ptshift = i * vm_setting.idxbits;
            const reg_t idx = (addr >> (PGSHIFT + ptshift)) & ((1 << vm_setting.idxbits) - 1);
            const iss::status res =
                down_stream_mem.rd_mem(iss::access_type::READ, base + idx * vm_setting.ptesize, vm_setting.ptesize, (uint8_t*)&pte);
            if(res != iss::status::Ok) {
                CPPLOG(DEBUG) << "Access fault when trying to read next pte";
                switch(type) {
                case iss::access_type::READ:
                    throw arch::trap_load_access_fault(addr);
                case iss::access_type::WRITE:
                    throw arch::trap_store_access_fault(addr);
                case iss::access_type::FETCH:
                    throw arch::trap_instruction_access_fault(addr);
                default:
                    abort();
                };
            }
            if(bit_sub<63, 1>(static_cast<uint64_t>(pte))) {
                CPPLOG(DEBUG) << "Page fault for address 0x" << std::hex << addr << ": set 'N' bit without Svnapot extension present";
                throw_page_fault(type, addr);
            }
            if(bit_sub<61, 2>(static_cast<uint64_t>(pte))) {
                CPPLOG(DEBUG) << "Page fault for address 0x" << std::hex << addr << ": set 'PBMT' bit(s) without Svpbmt extension present";
                throw_page_fault(type, addr);
            }
            if(bit_sub<54, 7>(static_cast<uint64_t>(pte))) {
                CPPLOG(DEBUG) << "Page fault for address 0x" << std::hex << addr << ": set 'reserved' bits";
                throw_page_fault(type, addr);
            }

            if(!(pte & PTE_V) || (!(pte & PTE_R) && (pte & PTE_W))) {
                CPPLOG(INFO) << "Page fault for address 0x" << std::hex << addr << ": invalid page, PTE=0x" << std::hex << pte;
                throw_page_fault(type, addr);
            }
            const reg_t ppn = pte >> PTE_PPN_SHIFT;
            if(!(pte & PTE_R || pte & PTE_X)) {
                base = ppn << PGSHIFT;
                continue;
            }
            if((ppn & ((reg_t(1) << ptshift) - 1)) != 0) {
                CPPLOG(DEBUG) << "Page fault for address 0x" << std::hex << addr << ": page misalignment";
                throw_page_fault(type, addr);
            }
            if(!(pte & PTE_A) || (type == iss::access_type::WRITE && !(pte & PTE_D))) {
                // non-Svade
                // Perform the following steps atomically:
                //  ■ Compare pte to the value of the PTE at address a+va.vpn[i]×PTESIZE.
                //  ■ If the values match, set pte.a to 1 and, if the original memory access is a store, also set pte.d to 1.
                //  ■ If the comparison fails, return to step 2.
                // pte |= PTE_A;
                // if(type == iss::access_type::WRITE)
                //    pte |= PTE_D;

                // Svade behavior
                CPPLOG(DEBUG) << "Page fault for address 0x" << std::hex << addr << ": unset A or D bit";
                throw_page_fault(type, addr);
            }
            const reg_t vpn = addr >> PGSHIFT;
            const reg_t value = (ppn | (vpn & ((reg_t(1) << ptshift) - 1))) << PGSHIFT;
            const reg_t pte_entry = value | (pte & 0xff);
            tlb[vpn] = pte_entry;
            pte = pte_entry;
            break;
        }
    }
    const bool s_mode = effective_priv(type) == arch::PRIV_S;
    const bool sum = hart_if.state.mstatus.SUM;
    if((pte & PTE_U) ? s_mode && (type == iss::access_type::FETCH || !sum) : !s_mode) {
        CPPLOG(DEBUG) << "Page fault for address 0x" << std::hex << addr << ": SUM bit";
        throw_page_fault(type, addr);
    }
    const bool mxr = hart_if.state.mstatus.MXR;
    if(type == iss::access_type::FETCH   ? !(pte & PTE_X)
       : type == iss::access_type::WRITE ? !(pte & PTE_W)
                                         : !((pte & PTE_R) || (mxr && (pte & PTE_X)))) {
        CPPLOG(DEBUG) << "Page fault for address 0x" << std::hex << addr << ": Invalid request";
        throw_page_fault(type, addr);
    }

    return (pte & (~PGMASK)) | (addr & PGMASK);
}
} // namespace mem
} // namespace iss
