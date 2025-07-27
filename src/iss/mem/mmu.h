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
#include "iss/vm_types.h"
#include "memory_if.h"
#include <optional>
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
    bool is_active() { return levels; }
};

inline void read_reg_with_offset(uint32_t reg, uint8_t offs, uint8_t* const data, unsigned length) {
    auto reg_ptr = reinterpret_cast<uint8_t*>(&reg);
    switch(offs) {
    default:
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

inline void write_reg_with_offset(uint32_t& reg, uint8_t offs, const uint8_t* const data, unsigned length) {
    auto reg_ptr = reinterpret_cast<uint8_t*>(&reg);
    switch(offs) {
    default:
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
// TODO: update vminfo on trap enter and leave as well as mstatus write, reset
template <typename WORD_TYPE> struct mmu : public memory_elem {
    using this_class = mmu<WORD_TYPE>;
    using reg_t = WORD_TYPE;
    constexpr static unsigned WORD_LEN = sizeof(WORD_TYPE) * 8;

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
        ptw.clear();
    }

private:
    iss::status read_mem(iss::access_type access, uint64_t addr, unsigned length, uint8_t* data) {
        vm_info vm = decode_vm_info(hart_if.PRIV, satp);
        if(unlikely((addr & ~PGMASK) != ((addr + length - 1) & ~PGMASK)) && vm.levels) { // we may cross a page boundary
            auto split_addr = (addr + length) & ~PGMASK;
            auto len1 = split_addr - addr;
            auto res = down_stream_mem.rd_mem(access, virt2phys(access, addr), len1, data);
            if(res == iss::Ok)
                res = down_stream_mem.rd_mem(access, virt2phys(access, split_addr), length - len1, data + len1);
            return res;
        }
        return down_stream_mem.rd_mem(access, vm.levels ? virt2phys(access, addr) : addr, length, data);
    }

    iss::status write_mem(iss::access_type access, uint64_t addr, unsigned length, uint8_t const* data) {
        vm_info vm = decode_vm_info(hart_if.PRIV, satp);
        if(unlikely((addr & ~PGMASK) != ((addr + length - 1) & ~PGMASK)) && vm.levels) { // we may cross a page boundary
            auto split_addr = (addr + length) & ~PGMASK;
            auto len1 = split_addr - addr;
            auto res = down_stream_mem.wr_mem(access, virt2phys(access, addr), len1, data);
            if(res == iss::Ok)
                res = down_stream_mem.wr_mem(access, virt2phys(access, split_addr), length - len1, data + len1);
            return res;
        }
        return down_stream_mem.wr_mem(access, vm.levels ? virt2phys(access, addr) : addr, length, data);
    }
    void update_vm_info();

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
        satp = val;
        update_vm_info();
        return iss::Ok;
    }

    uint64_t virt2phys(iss::access_type access, uint64_t addr);

    static inline vm_info decode_vm_info(uint32_t state, uint32_t sptbr) {
        if(state == arch::PRIV_M)
            return {0, 0, 0, 0};
        if(state <= arch::PRIV_S)
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

    static inline vm_info decode_vm_info(uint32_t state, uint64_t sptbr) {
        if(state == arch::PRIV_M)
            return {0, 0, 0, 0};
        if(state <= arch::PRIV_S)
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

protected:
    reg_t satp;
    std::unordered_map<reg_t, uint64_t> ptw;
    std::array<vm_info, 2> vmt;
    std::array<address_type, 4> addr_mode;

    arch::priv_if<WORD_TYPE> hart_if;
    memory_if down_stream_mem;
};

template <typename WORD_TYPE> uint64_t mmu<WORD_TYPE>::virt2phys(iss::access_type access, uint64_t addr) {
    const auto type = access & iss::access_type::FUNC;
    auto it = ptw.find(addr >> PGSHIFT);
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
            return {(pte & (~PGMASK)) | (addr & PGMASK)};
        else
            ptw.erase(it); // throw an exception
#endif
    } else {
        uint32_t mode = type != iss::access_type::FETCH && hart_if.state.mstatus.MPRV ? // MPRV
                            hart_if.state.mstatus.MPP
                                                                                      : hart_if.PRIV;

        const vm_info& vm = vmt[static_cast<uint16_t>(type) / 2];

        const bool s_mode = mode == arch::PRIV_S;
        const bool sum = hart_if.state.mstatus.SUM;
        const bool mxr = hart_if.state.mstatus.MXR;

        // verify bits xlen-1:va_bits-1 are all equal
        const int va_bits = PGSHIFT + vm.levels * vm.idxbits;
        const reg_t mask = (reg_t(1) << (sizeof(reg_t) * 8 - (va_bits - 1))) - 1;
        const reg_t masked_msbs = (addr >> (va_bits - 1)) & mask;
        const int levels = (masked_msbs != 0 && masked_msbs != mask) ? 0 : vm.levels;

        reg_t base = vm.ptbase;
        for(int i = levels - 1; i >= 0; i--) {
            const int ptshift = i * vm.idxbits;
            const reg_t idx = (addr >> (PGSHIFT + ptshift)) & ((1 << vm.idxbits) - 1);

            // check that physical address of PTE is legal
            reg_t pte = 0;
            const iss::status res = down_stream_mem.rd_mem(iss::access_type::READ, base + idx * vm.ptesize, vm.ptesize, (uint8_t*)&pte);
            if(res != iss::status::Ok)
                throw arch::trap_load_access_fault(addr);
            const reg_t ppn = pte >> PTE_PPN_SHIFT;

            if(PTE_TABLE(pte)) { // next level of page table
                base = ppn << PGSHIFT;
            } else if((pte & PTE_U) ? s_mode && (type == iss::access_type::FETCH || !sum) : !s_mode) {
                break;
            } else if(!(pte & PTE_V) || (!(pte & PTE_R) && (pte & PTE_W))) {
                break;
            } else if(type == (type == iss::access_type::FETCH  ? !(pte & PTE_X)
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
                const reg_t vpn = addr >> PGSHIFT;
                const reg_t value = (ppn | (vpn & ((reg_t(1) << ptshift) - 1))) << PGSHIFT;
                const reg_t offset = addr & PGMASK;
                ptw[vpn] = value | (pte & 0xff);
                return value | offset;
            }
        }
    }
    switch(type) {
    case access_type::FETCH:
        hart_if.raise_trap(12, 0, addr);
        throw arch::trap_instruction_page_fault(addr);
    case access_type::READ:
        hart_if.raise_trap(13, 0, addr);
        throw arch::trap_load_page_fault(addr);
    case access_type::WRITE:
        hart_if.raise_trap(15, 0, addr);
        throw arch::trap_store_page_fault(addr);
    default:
        abort();
    }
}

template <typename WORD_TYPE> inline void mmu<WORD_TYPE>::update_vm_info() {
    vmt[1] = decode_vm_info(hart_if.PRIV, satp);
    addr_mode[3] = addr_mode[2] = vmt[1].is_active() ? iss::address_type::VIRTUAL : iss::address_type::PHYSICAL;
    if(hart_if.state.mstatus.MPRV)
        vmt[0] = decode_vm_info(hart_if.state.mstatus.MPP, satp);
    else
        vmt[0] = vmt[1];
    addr_mode[1] = addr_mode[0] = vmt[0].is_active() ? iss::address_type::VIRTUAL : iss::address_type::PHYSICAL;
    ptw.clear();
}

} // namespace mem
} // namespace iss
