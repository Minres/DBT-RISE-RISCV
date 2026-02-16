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
#include "iss/vm_if.h"
#include "iss/vm_types.h"
#include "riscv_hart_common.h"
#include "util/logging.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <elfio/elf_types.hpp>
#include <elfio/elfio.hpp>
#include <limits>
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>
#include <iss/mem/memory_with_htif.h>
#include <iss/mem/mmu.h>
#include <optional>
#include <type_traits>
#include <unordered_map>

namespace iss {
namespace arch {

template <typename BASE, features_e FEAT = FEAT_NONE> class riscv_hart_msu_vp : public riscv_hart_common<BASE> {
public:
    using core = BASE;
    using base = riscv_hart_common<BASE>;
    using this_class = riscv_hart_msu_vp<BASE>;
    using reg_t = typename core::reg_t;
    using phys_addr_t = typename core::phys_addr_t;

    void write_mstatus(reg_t val, unsigned priv_lvl) {
        reg_t old_val = this->state.mstatus;
        auto mask = base::get_msu_status_mask(priv_lvl);
        auto new_val = (old_val & ~mask) | (val & mask);
        if constexpr(sizeof(reg_t) == 8) {
            // Retain previous SXL and UXL value
            new_val |= old_val & this->state.mstatus.SXL.Mask;

            new_val |= old_val & this->state.mstatus.UXL.Mask;
        }

        if constexpr(base::extension_status_mask)
            if((new_val & base::extension_status_mask) == base::extension_status_mask)
                new_val |= reg_t(1) << (sizeof(reg_t) * 8 - 1);
        this->state.mstatus = new_val;
    }

    riscv_hart_msu_vp();

    virtual ~riscv_hart_msu_vp() = default;

    void reset(uint64_t address) override;

    void enable_disass(bool enable) {riscv_hart_common<BASE>::enable_disass_output(enable);}

    iss::status read(const addr_t& addr, const unsigned length, uint8_t* const data);
    iss::status write(const addr_t& addr, const unsigned length, const uint8_t* const data);

    uint64_t enter_trap(uint64_t flags) override { return riscv_hart_msu_vp::enter_trap(flags, this->fault_data, this->fault_data); }
    uint64_t enter_trap(uint64_t flags, uint64_t addr, uint64_t instr) override;
    uint64_t leave_trap(uint64_t flags) override;
    void wait_until(uint64_t flags) override;

    void set_csr(unsigned addr, reg_t val) { this->csr[addr] = val; }

protected:
    using mem_read_f = iss::status(iss::phys_addr_t addr, unsigned, uint8_t* const);
    using mem_write_f = iss::status(iss::phys_addr_t addr, unsigned, uint8_t const* const);

    std::unordered_map<uint64_t, uint8_t> atomic_reservation;

    iss::status read_status(unsigned addr, reg_t& val);
    iss::status write_status(unsigned addr, reg_t val);
    iss::status read_statush(unsigned addr, reg_t& val);
    iss::status write_statush(unsigned addr, reg_t val);
    iss::status read_ie(unsigned addr, reg_t& val);
    iss::status write_ie(unsigned addr, reg_t val);
    iss::status read_ip(unsigned addr, reg_t& val);
    iss::status write_ideleg(unsigned addr, reg_t val);
    iss::status write_edeleg(unsigned addr, uint32_t val);
    iss::status write_edeleg(unsigned addr, uint64_t val);
    iss::status write_edelegh(unsigned addr, uint32_t val);

    void check_interrupt();
    mem::mmu<BASE> mmu;
    mem::neumann_memory_with_htif<BASE> default_mem;
};

template <typename BASE, features_e FEAT>
riscv_hart_msu_vp<BASE, FEAT>::riscv_hart_msu_vp()
: mmu(base::get_priv_if())
, default_mem(base::get_priv_if()) {
    if constexpr(sizeof(reg_t) == 8)
        // set UXL and SXL to indicate 64 bits
        // FIXME: this value is not preserved in resets
        this->state.mstatus.backing.val |= 0x500000000;
    // common regs
    const std::array<unsigned, 16> rwaddrs{
        {mepc, mtvec, mscratch, mcause, mtval, sepc, stvec, sscratch, scause, stval, sscratch, uepc, utvec, uscratch, ucause, utval}};
    for(auto addr : rwaddrs) {
        this->csr_rd_cb[addr] = MK_CSR_RD_CB(read_plain);
        this->csr_wr_cb[addr] = MK_CSR_WR_CB(write_plain);
    }
    // special handling & overrides
    this->csr_rd_cb[mstatus] = MK_CSR_RD_CB(read_status);
    this->csr_wr_cb[mstatus] = MK_CSR_WR_CB(write_status);
    this->csr_rd_cb[sstatus] = MK_CSR_RD_CB(read_status);
    this->csr_wr_cb[sstatus] = MK_CSR_WR_CB(write_status);
    this->csr_rd_cb[scause] = MK_CSR_RD_CB(read_cause);
    this->csr_wr_cb[scause] = MK_CSR_WR_CB(write_cause);
    this->csr_rd_cb[stvec] = MK_CSR_RD_CB(read_tvec);
    this->csr_wr_cb[stvec] = MK_CSR_WR_CB(write_plain);
    this->csr_rd_cb[mip] = MK_CSR_RD_CB(read_ip);
    this->csr_wr_cb[mip] = MK_CSR_WR_CB(write_plain);
    this->csr_rd_cb[sip] = MK_CSR_RD_CB(read_ip);
    this->csr_wr_cb[sip] = MK_CSR_WR_CB(write_plain);
    this->csr_rd_cb[mie] = MK_CSR_RD_CB(read_ie);
    this->csr_wr_cb[mie] = MK_CSR_WR_CB(write_ie);
    this->csr_rd_cb[sie] = MK_CSR_RD_CB(read_ie);
    this->csr_wr_cb[sie] = MK_CSR_WR_CB(write_ie);
    if(FEAT & FEAT_EXT_N) {
        this->csr_rd_cb[ustatus] = MK_CSR_RD_CB(read_status);
        this->csr_wr_cb[ustatus] = MK_CSR_WR_CB(write_status);
        this->csr_wr_cb[ucause] = MK_CSR_WR_CB(write_cause);
        this->csr_rd_cb[utvec] = MK_CSR_RD_CB(read_tvec);
        this->csr_rd_cb[uip] = MK_CSR_RD_CB(read_ip);
        this->csr_wr_cb[uip] = MK_CSR_WR_CB(write_plain);
        this->csr_rd_cb[uie] = MK_CSR_RD_CB(read_ie);
        this->csr_wr_cb[uie] = MK_CSR_WR_CB(write_ie);
        this->csr[misa] |= extension_encoding::N;
    }
    this->csr_rd_cb[mcounteren] = MK_CSR_RD_CB(read_null);
    this->csr_wr_cb[mcounteren] = MK_CSR_WR_CB(write_null);
    this->csr_rd_cb[arch::riscv_csr::medeleg] = MK_CSR_RD_CB(read_plain);
    this->csr_wr_cb[arch::riscv_csr::medeleg] = MK_CSR_WR_CB(write_edeleg);
    this->csr_rd_cb[mideleg] = MK_CSR_RD_CB(read_plain);
    this->csr_wr_cb[mideleg] = MK_CSR_WR_CB(write_ideleg);
    this->csr_rd_cb[scounteren] = MK_CSR_RD_CB(read_null);
    this->csr_wr_cb[scounteren] = MK_CSR_WR_CB(write_null);
    if(traits<BASE>::XLEN == 32) {
        this->csr_rd_cb[mstatush] = MK_CSR_RD_CB(read_statush);
        this->csr_wr_cb[mstatush] = MK_CSR_WR_CB(write_statush);
        this->csr_rd_cb[medelegh] = MK_CSR_RD_CB(read_plain);
        this->csr_wr_cb[medelegh] = MK_CSR_WR_CB(write_edelegh);
    }
    if(FEAT & FEAT_DEBUG)
        this->add_debug_csrs();

    this->rd_func = util::delegate<arch_if::rd_func_sig>::from<this_class, &this_class::read>(this);
    this->wr_func = util::delegate<arch_if::wr_func_sig>::from<this_class, &this_class::write>(this);
    this->set_next(mmu.get_mem_if());
    this->memories.root(mmu);
    this->memories.append(default_mem);
    this->csr[misa] |= (extension_encoding::S | extension_encoding::U);
}

template <typename BASE, features_e FEAT>
iss::status riscv_hart_msu_vp<BASE, FEAT>::read(const addr_t& a, const unsigned length, uint8_t* const data) {
    auto& addr = a.val;
    auto& space = a.space;
    auto& access = a.access;
    auto& type = a.type;
#ifndef NDEBUG
    if(access && iss::access_type::DEBUG) {
        ILOG(isslogger, logging::TRACEALL, fmt::format("debug read of {} bytes @addr 0x{:x}", length, addr));
    } else if(is_fetch(access)) {
        ILOG(isslogger, logging::TRACEALL, fmt::format("fetch of {} bytes @addr 0x{:x}", length, addr));
    } else {
        ILOG(isslogger, logging::TRACEALL, fmt::format("read of {} bytes @addr 0x{:x}", length, addr));
    }
#endif
    try {
        switch(space) {
        case traits<BASE>::CSR: {
            if(length != sizeof(reg_t))
                return iss::Err;
            auto res = this->read_csr(addr, *reinterpret_cast<reg_t* const>(data));
            if(res != iss::Ok && !is_debug(access))
                this->reg.trap_state = (1U << 31) | traits<BASE>::RV_CAUSE_ILLEGAL_INSTRUCTION << 16;
            return res;
        } break;
        case traits<BASE>::FENCE: {
            switch(addr) {
            case traits<BASE>::fencevma: {
                auto tvm = this->state.mstatus.TVM;
                if(this->reg.PRIV == PRIV_S & tvm != 0) {
                    this->reg.trap_state = (1UL << 31) | (traits<BASE>::RV_CAUSE_ILLEGAL_INSTRUCTION << 16);
                    this->fault_data = this->reg.PC;
                    return iss::Err;
                }
                return iss::Ok;
            }
            default:
                return iss::Ok;
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
        default: {
            auto alignment = is_fetch(access) ? (this->has_compressed() ? 2 : 4) : std::min<unsigned>(length, sizeof(reg_t));
            if(unlikely(is_fetch(access) && (addr & (alignment - 1)))) {
                this->fault_data = addr;
                if(is_debug(access))
                    throw trap_access(0, addr);
                this->reg.trap_state = (1UL << 31) | traits<BASE>::RV_CAUSE_MISALIGNED_FETCH << 16;
                return iss::Err;
            }
            try {
                if(!is_debug(access) && (addr & (alignment - 1))) {
                    this->reg.trap_state = (1UL << 31) | traits<BASE>::RV_CAUSE_MISALIGNED_LOAD << 16;
                    this->fault_data = addr;
                    return iss::Err;
                }
                auto res = this->memory.rd_mem({address_type::VIRTUAL, a.access, a.space, a.val}, length, data);
                if(unlikely(res != iss::Ok && (access & access_type::DEBUG) == 0)) {
                    this->reg.trap_state = (1UL << 31) | traits<BASE>::RV_CAUSE_LOAD_ACCESS << 16;
                    this->fault_data = addr;
                }
                return res;
            } catch(trap_access& ta) {
                if((access & access_type::DEBUG) == 0) {
                    this->reg.trap_state = (1UL << 31) | ta.id;
                    this->fault_data = ta.addr;
                }
                return iss::Err;
            }
        } break;
        }
        return iss::Ok;
    } catch(trap_access& ta) {
        if((access & access_type::DEBUG) == 0) {
            this->reg.trap_state = (1UL << 31) | ta.id;
            this->fault_data = ta.addr;
        }
        return iss::Err;
    }
}

template <typename BASE, features_e FEAT>
iss::status riscv_hart_msu_vp<BASE, FEAT>::write(const addr_t& a, const unsigned length, const uint8_t* const data) {
    auto& addr = a.val;
    auto& space = a.space;
    auto& access = a.access;
    auto& type = a.type;
#ifndef NDEBUG
    const char* prefix = (access && iss::access_type::DEBUG) ? "debug " : "";
    switch(length) {
    case 8:
        ILOG(isslogger, logging::TRACEALL,
             fmt::format("{}write of {} bytes (0x{:x}) @addr 0x{:x}", prefix, length, *reinterpret_cast<const uint64_t*>(&data[0]), addr));
        break;
    case 4:
        ILOG(isslogger, logging::TRACEALL,
             fmt::format("{}write of {} bytes (0x{:x}) @addr 0x{:x}", prefix, length, *reinterpret_cast<const uint32_t*>(&data[0]), addr));
        break;
    case 2:
        ILOG(isslogger, logging::TRACEALL,
             fmt::format("{}write of {} bytes (0x{:x}) @addr 0x{:x}", prefix, length, *reinterpret_cast<const uint16_t*>(&data[0]), addr));
        break;
    case 1:
        ILOG(isslogger, logging::TRACEALL,
             fmt::format("{}write of {} bytes (0x{:x}) @addr 0x{:x}", prefix, length, (uint16_t)data[0], addr));
        break;
    default:
        ILOG(isslogger, logging::TRACEALL, fmt::format("{}write of {} bytes @addr 0x{:x}", prefix, length, addr));
    }
#endif
    try {
        switch(space) {
        case traits<BASE>::CSR: {
            if(length != sizeof(reg_t))
                return iss::Err;
            auto res = this->write_csr(addr, *reinterpret_cast<const reg_t*>(data));
            if(res != iss::Ok && !is_debug(access))
                this->reg.trap_state = (1U << 31) | traits<BASE>::RV_CAUSE_ILLEGAL_INSTRUCTION << 16;
            return res;
        } break;
        case traits<BASE>::FENCE: {
            switch(addr) {
            case 2:
            case 3: {
                auto tvm = this->state.mstatus.TVM;
                if(this->reg.PRIV == PRIV_S & tvm != 0) {
                    this->reg.trap_state = (1UL << 31) | (traits<BASE>::RV_CAUSE_ILLEGAL_INSTRUCTION << 16);
                    this->fault_data = this->reg.PC;
                    return iss::Err;
                }
                uint8_t asid_reg = *data;
                uint8_t vaddr_reg = *(data + 1);
                std::optional<reg_t> vaddr =
                    vaddr_reg ? std::make_optional(*(this->get_regs_base_ptr() + traits<BASE>::reg_byte_offsets.at(vaddr_reg)))
                              : std::nullopt;
                std::optional<reg_t> asid =
                    asid_reg ? std::make_optional(*(this->get_regs_base_ptr() + traits<BASE>::reg_byte_offsets.at(asid_reg)))
                             : std::nullopt;
                mmu.flush_tlb(vaddr, asid_reg);
                return iss::Ok;
            }
            }
        } break;
        case traits<BASE>::RES: {
            atomic_reservation[addr] = data[0];
        } break;
        default: {
            if(unlikely(is_fetch(access) && (addr & 0x1) == 1)) {
                this->fault_data = addr;
                if(access && iss::access_type::DEBUG)
                    throw trap_access(0, addr);
                this->reg.trap_state = (1UL << 31) | traits<BASE>::RV_CAUSE_MISALIGNED_FETCH << 16;
                return iss::Err;
            }
            try {
                auto alignment = std::min<unsigned>(length, sizeof(reg_t));
                if(length > 1 && (addr & (alignment - 1)) && !is_debug(access)) {
                    this->reg.trap_state = (1UL << 31) | traits<BASE>::RV_CAUSE_MISALIGNED_STORE << 16;
                    this->fault_data = addr;
                    return iss::Err;
                }
                auto res = this->memory.wr_mem({address_type::VIRTUAL, a.access, a.space, a.val}, length, data);
                if(unlikely(res != iss::Ok && !is_debug(access))) {
                    this->reg.trap_state = (1UL << 31) | traits<BASE>::RV_CAUSE_STORE_ACCESS << 16;
                    this->fault_data = addr;
                }
                return res;
            } catch(trap_access& ta) {
                this->reg.trap_state = (1UL << 31) | ta.id;
                this->fault_data = ta.addr;
                return iss::Err;
            }
        } break;
        }
        return iss::Ok;
    } catch(trap_access& ta) {
        if((access & access_type::DEBUG) == 0) {
            this->reg.trap_state = (1UL << 31) | ta.id;
            this->fault_data = ta.addr;
        }
        return iss::Err;
    }
}

template <typename BASE, features_e FEAT> iss::status riscv_hart_msu_vp<BASE, FEAT>::read_status(unsigned addr, reg_t& val) {
    auto req_priv_lvl = (addr >> 8) & 0x3;
    val = this->state.mstatus & base::get_msu_status_mask(req_priv_lvl);
    return iss::Ok;
}

template <typename BASE, features_e FEAT> iss::status riscv_hart_msu_vp<BASE, FEAT>::write_status(unsigned addr, reg_t val) {
    auto req_priv_lvl = (addr >> 8) & 0x3;
    write_mstatus(val, req_priv_lvl);
    check_interrupt();
    return iss::Ok;
}
template <typename BASE, features_e FEAT> iss::status riscv_hart_msu_vp<BASE, FEAT>::read_statush(unsigned addr, reg_t& val) {
    val = this->csr[mstatush] & 0b11'1111'1000;
    return iss::Ok;
}

template <typename BASE, features_e FEAT> iss::status riscv_hart_msu_vp<BASE, FEAT>::write_statush(unsigned addr, reg_t val) {
    this->csr[mstatush] = val & 0b11'1111'1000;
    return iss::Ok;
}

template <typename BASE, features_e FEAT> iss::status riscv_hart_msu_vp<BASE, FEAT>::read_ie(unsigned addr, reg_t& val) {
    val = this->csr[mie];
    if(addr < mie)
        val &= this->csr[mideleg];
    if(addr < sie)
        val &= this->csr[sideleg];
    return iss::Ok;
}

template <typename BASE, features_e FEAT> iss::status riscv_hart_msu_vp<BASE, FEAT>::write_ie(unsigned addr, reg_t val) {
    // generate mask from allowed writable bits, the number of custom interrupts and the available ie bits
    auto mask = riscv_hart_common<BASE>::get_irq_mask((addr >> 8) & 0x3) & this->clint_custom_irq_mask & (FEAT & FEAT_EXT_N?0xbbb:0xaaa);
    this->csr[mie] = (this->csr[mie] & ~mask) | (val & mask);
    check_interrupt();
    return iss::Ok;
}

template <typename BASE, features_e FEAT> iss::status riscv_hart_msu_vp<BASE, FEAT>::read_ip(unsigned addr, reg_t& val) {
    val = this->csr[mip];
    if(addr < mip)
        val &= this->csr[mideleg];
    if(addr < sip)
        val &= this->csr[sideleg];
    return iss::Ok;
}

template <typename BASE, features_e FEAT> iss::status riscv_hart_msu_vp<BASE, FEAT>::write_ideleg(unsigned addr, reg_t val) {
    // only U and S mode interrupts can be delegated
    auto mask = 0b0011'0011'0011;
    this->csr[mideleg] = (this->csr[mideleg] & ~mask) | (val & mask);
    return iss::Ok;
}

template <typename BASE, features_e FEAT> iss::status riscv_hart_msu_vp<BASE, FEAT>::write_edeleg(unsigned addr, uint32_t val) {
    // bit 3 (break), bit 10 (reserved), bit 11 (Ecall from M) bit 14 (reserved), bit 16-17 (reserved), bit 20-23 (reserved)
    uint32_t mask = 0b1111'1111'0000'1100'1011'0011'1111'0111;
    this->csr[arch::riscv_csr::medeleg] = (this->csr[arch::riscv_csr::medeleg] & ~mask) | (val & mask);
    return iss::Ok;
}

template <typename BASE, features_e FEAT> iss::status riscv_hart_msu_vp<BASE, FEAT>::write_edeleg(unsigned addr, uint64_t val) {
    // bit 3 (break), bit 10 (reserved), bit 11 (Ecall from M) bit 14 (reserved), bit 16-17 (reserved), bit 20-23 (reserved)
    uint32_t mask_lower = 0b1111'1111'0000'1100'1011'0011'1111'0111;
    // bit 32-47(reserved)
    uint64_t mask = ((uint64_t)0b1111'1111'1111'1111'0000'0000'0000'0000 << 32) | mask_lower;
    this->csr[arch::riscv_csr::medeleg] = (this->csr[arch::riscv_csr::medeleg] & ~mask) | (val & mask);
    return iss::Ok;
}

template <typename BASE, features_e FEAT> iss::status riscv_hart_msu_vp<BASE, FEAT>::write_edelegh(unsigned addr, uint32_t val) {
    // bit 32-47(reserved)
    auto mask = 0b1111'1111'1111'1111'0000'0000'0000'0000;
    this->csr[medelegh] = (this->csr[medelegh] & ~mask) | (val & mask);
    return iss::Ok;
}

template <typename BASE, features_e FEAT> inline void riscv_hart_msu_vp<BASE, FEAT>::reset(uint64_t address) {
    BASE::reset(address);
    this->state.mstatus = hart_state<reg_t>::mstatus_reset_val;
    if(sizeof(reg_t) == 8) // Set UXLEN and SXLEN to 64
        this->state.mstatus |= static_cast<reg_t>(0xa00000000ULL);
}

template <typename BASE, features_e FEAT> void riscv_hart_msu_vp<BASE, FEAT>::check_interrupt() {
    auto status = this->state.mstatus;
    auto ip = this->csr[mip];
    auto ie = this->csr[mie];
    auto ideleg = this->csr[mideleg];
    // Multiple simultaneous interrupts and traps at the same privilege level are
    // handled in the following decreasing priority order:
    // external interrupts, software interrupts, timer interrupts, then finally
    // any synchronous traps.
    auto ena_irq = ip & ie;

    bool mie = this->state.mstatus.MIE;
    auto m_enabled = this->reg.PRIV < PRIV_M || (this->reg.PRIV == PRIV_M && mie);
    auto enabled_interrupts = m_enabled ? ena_irq & ~ideleg : 0;

    if(enabled_interrupts == 0) {
        auto sie = this->state.mstatus.SIE;
        auto s_enabled = this->reg.PRIV < PRIV_S || (this->reg.PRIV == PRIV_S && sie);
        enabled_interrupts = s_enabled ? ena_irq & ideleg : 0;
    }
    if(enabled_interrupts != 0) {
        int res = 0;
        while((enabled_interrupts & 1) == 0)
            enabled_interrupts >>= 1, res++;
        this->reg.pending_trap = 0x80ULL << 24 | res << 16 | 1; // 0x80 << 24 | (cause << 16) | trap_id
    }
}

template <typename BASE, features_e FEAT>
uint64_t riscv_hart_msu_vp<BASE, FEAT>::enter_trap(uint64_t flags, uint64_t addr, uint64_t tval) {
    // flags are ACTIVE[31:31], CAUSE[30:16], TRAPID[15:0]
    // calculate and write mcause val
    if(flags == std::numeric_limits<uint32_t>::max())
        flags = this->reg.trap_state;
    auto trap_id = bit_sub<0, 16>(flags);
    auto cause = bit_sub<16, 15>(flags);
    if(trap_id == 0 && cause == 11)
        cause = 0x8 + this->reg.PRIV; // adjust environment call cause
    // calculate effective privilege level
    unsigned new_priv = PRIV_M;
    if(trap_id == 0) { // exception
        uint64_t medeleg = this->csr[arch::riscv_csr::medeleg];
        if(traits<BASE>::XLEN == 32)
            medeleg |= (static_cast<uint64_t>(this->csr[arch::riscv_csr::medeleg]) << 32);
        if(this->reg.PRIV != PRIV_M && ((medeleg >> cause) & 0x1) != 0)
            new_priv = (this->csr[sedeleg] >> cause) & 0x1 ? PRIV_U : PRIV_S;
        // store ret addr in xepc register
        this->csr[uepc | (new_priv << 8)] = static_cast<reg_t>(addr); // store actual address instruction of exception
        /*
         * write mtval if new_priv=M_MODE, spec says:
         * When a hardware breakpoint is triggered, or an instruction-fetch, load,
         * or store address-misaligned,
         * access, or page-fault exception occurs, mtval is written with the
         * faulting effective address.
         */
        switch(cause) {
        case traits<BASE>::RV_CAUSE_MISALIGNED_FETCH:
            this->csr[mtval] = static_cast<reg_t>(tval);
            break;
        case traits<BASE>::RV_CAUSE_ILLEGAL_INSTRUCTION:
#ifdef MTVAL_ILLEGAL_INFORMATIVE
            this->csr[utval | (new_priv << 8)] = (!this->has_compressed() || (tval & 0x3) == 3) ? tval : tval & 0xffff;
#endif
            break;
        case traits<BASE>::RV_CAUSE_BREAKPOINT:
            if((FEAT & FEAT_DEBUG) && (this->csr[dcsr] & 0x8000)) {
                this->reg.DPC = addr;
                this->csr[dcsr] = (this->csr[dcsr] & ~0x1c3) | (1 << 6) | PRIV_M; // FIXME: cause should not be 4 (stepi)
                new_priv = this->reg.PRIV | PRIV_D;
            } else {
                this->csr[utval | (new_priv << 8)] = addr;
            }
            if(this->semihosting_cb) {
                // Check for semihosting call
                std::array<uint8_t, 8> data;
                // check for SLLI_X0_X0_0X1F and SRAI_X0_X0_0X07
                this->memory.rd_mem({address_type::VIRTUAL, access_type::DEBUG_READ, traits<BASE>::IMEM, addr - 4}, 4, data.data());
                addr += 8;
                this->memory.rd_mem({address_type::VIRTUAL, access_type::DEBUG_READ, traits<BASE>::IMEM, addr - 4}, 4, data.data() + 4);

                const std::array<uint8_t, 8> ref_data = {0x13, 0x10, 0xf0, 0x01, 0x13, 0x50, 0x70, 0x40};
                if(data == ref_data) {
                    this->reg.NEXT_PC = addr + 8;

                    std::array<char, 32> buffer;
#if defined(_MSC_VER)
                    sprintf(buffer.data(), "0x%016llx", addr);
#else
                    sprintf(buffer.data(), "0x%016lx", addr);
#endif
                    this->disass_output(fmt::format("Semihosting call at address {} occurred ", buffer.data()));
                    this->semihosting_cb(this, &(this->reg.X10) /*a0*/, &(this->reg.X11) /*a1*/);
                    return this->reg.NEXT_PC;
                }
            }
            break;
        case traits<BASE>::RV_CAUSE_FETCH_ACCESS:
        case traits<BASE>::RV_CAUSE_MISALIGNED_LOAD:
        case traits<BASE>::RV_CAUSE_LOAD_ACCESS:
        case traits<BASE>::RV_CAUSE_MISALIGNED_STORE:
        case traits<BASE>::RV_CAUSE_STORE_ACCESS:
        case traits<BASE>::RV_CAUSE_FETCH_PAGE_FAULT:
        case traits<BASE>::RV_CAUSE_LOAD_PAGE_FAULT:
        case traits<BASE>::RV_CAUSE_STORE_PAGE_FAULT:
            this->csr[utval | (new_priv << 8)] = this->fault_data;
            break;
        default:
            this->csr[utval | (new_priv << 8)] = 0;
        }
        this->fault_data = 0;
    } else {
        if(this->reg.PRIV != PRIV_M && ((this->csr[mideleg] >> cause) & 0x1) != 0)
            new_priv = (this->csr[sideleg] >> cause) & 0x1 ? PRIV_U : PRIV_S;
        this->csr[uepc | (new_priv << 8)] = this->reg.NEXT_PC; // store next address if interrupt
        this->reg.pending_trap = 0;
    }
    size_t adr = ucause | (new_priv << 8);
    this->csr[adr] = (trap_id << (traits<BASE>::XLEN - 1)) + cause;
    // update mstatus
    // xPP field of mstatus is written with the active privilege mode at the time
    // of the trap; the x PIE field of mstatus
    // is written with the value of the active interrupt-enable bit at the time of
    // the trap; and the x IE field of mstatus
    // is cleared
    // store the actual privilege level in yPP and store interrupt enable flags
    switch(new_priv) {
    case PRIV_M:
        this->state.mstatus.MPP = this->reg.PRIV;
        this->state.mstatus.MPIE = this->state.mstatus.MIE;
        this->state.mstatus.MIE = false;
        break;
    case PRIV_S:
        this->state.mstatus.SPP = this->reg.PRIV;
        this->state.mstatus.SPIE = this->state.mstatus.SIE;
        this->state.mstatus.SIE = false;
        break;
    case PRIV_U:
        this->state.mstatus.UPIE = this->state.mstatus.UIE;
        this->state.mstatus.UIE = false;
        break;
    default:
        break;
    }

    // get trap vector
    auto xtvec = this->csr[utvec | (new_priv << 8)];
    // calculate addr// set NEXT_PC to trap addressess to jump to based on MODE
    // bits in mtvec
    this->reg.NEXT_PC = xtvec & ~0x3UL;
    if(trap_id != 0) {
        if((xtvec & 0x3UL) == 3UL) {
            reg_t data;
            auto ret = read({address_type::VIRTUAL, access_type::READ, traits<BASE>::MEM, this->csr[mtvt]}, sizeof(reg_t),
                            reinterpret_cast<uint8_t*>(&data));
            if(ret == iss::Err)
                return this->reg.PC;
            this->reg.NEXT_PC = data;
        } else if((xtvec & 0x3UL) == 1UL)
            this->reg.NEXT_PC += 4 * cause;
    }
    std::array<char, 32> buffer;
#if defined(_MSC_VER)
    sprintf(buffer.data(), "0x%016llx", addr);
#else
    sprintf(buffer.data(), "0x%016lx", addr);
#endif
    if((flags & 0xffffffff) != 0xffffffff) {
        if(trap_id) {
            auto irq_str = cause < this->irq_str.size() ? this->irq_str.at(cause) : "Unknown";
            this->disass_output(
                 fmt::format("Interrupt with cause '{}' ({}) occurred  at address {}", irq_str, cause, buffer.data()));
        } else {
            auto irq_str = cause < this->trap_str.size() ? this->trap_str.at(cause) : "Unknown";
            this->disass_output(
                 fmt::format("Trap with cause '{}' ({}) occurred  at address {}", irq_str, cause, buffer.data()));
        }
    }
    // reset trap this->state
    this->reg.PRIV = new_priv;
    this->reg.trap_state = 0;
    return this->reg.NEXT_PC;
}

template <typename BASE, features_e FEAT> uint64_t riscv_hart_msu_vp<BASE, FEAT>::leave_trap(uint64_t flags) {
    auto cur_priv = this->reg.PRIV;
    auto inst_priv = flags & 0x3;

    auto tsr = this->state.mstatus.TSR;
    if(cur_priv == PRIV_S && inst_priv == PRIV_S && tsr != 0) {
        this->reg.trap_state = 0x80ULL << 24 | (traits<BASE>::RV_CAUSE_ILLEGAL_INSTRUCTION << 16);
        this->reg.NEXT_PC = std::numeric_limits<uint32_t>::max();
    } else {
        auto status = this->state.mstatus;
        // pop the relevant lower-privilege interrupt enable and privilege mode stack
        // clear respective yIE
        switch(inst_priv) {
        case PRIV_M:
            this->reg.PRIV = this->state.mstatus.MPP;
            this->state.mstatus.MPP = 0; // clear mpp to U mode
            this->state.mstatus.MIE = this->state.mstatus.MPIE;
            this->state.mstatus.MPIE = 1;
            break;
        case PRIV_S:
            this->reg.PRIV = this->state.mstatus.SPP;
            this->state.mstatus.SPP = 0; // clear spp to U mode
            this->state.mstatus.SIE = this->state.mstatus.SPIE;
            this->state.mstatus.SPIE = 1;
            break;
        case PRIV_U:
            this->reg.PRIV = 0;
            this->state.mstatus.UIE = this->state.mstatus.UPIE;
            this->state.mstatus.UPIE = 1;
            break;
        }
        // sets the pc to the value stored in the x epc register.
        this->reg.NEXT_PC = this->csr[uepc | inst_priv << 8];
        this->disass_output(
             fmt::format("Executing xRET, changing privilege level from {} to {}", this->lvl[cur_priv], this->lvl[this->reg.PRIV]));
        check_interrupt();
    }
    this->reg.trap_state = this->reg.pending_trap;
    return this->reg.NEXT_PC;
}

template <typename BASE, features_e FEAT> void riscv_hart_msu_vp<BASE, FEAT>::wait_until(uint64_t flags) {
    auto status = this->state.mstatus;
    auto tw = status.TW;
    if(this->reg.PRIV < PRIV_M && tw != 0) {
        this->reg.trap_state = (1UL << 31) | (traits<BASE>::RV_CAUSE_ILLEGAL_INSTRUCTION << 16);
        this->fault_data = this->reg.PC;
    }
}
} // namespace arch
} // namespace iss

#endif /* _RISCV_HART_MSU_VP_H */
