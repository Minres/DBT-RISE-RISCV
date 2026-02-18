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
#include "iss/vm_if.h"
#include "iss/vm_types.h"
#include "util/logging.h"
#include <algorithm>
#include <cstdint>
#include <elfio/elf_types.hpp>
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <array>
#include <elfio/elfio.hpp>
#include <fmt/format.h>
#include <iss/mem/memory_with_htif.h>
#include <unordered_map>

namespace iss {
namespace arch {

template <typename BASE, features_e FEAT = FEAT_NONE> class riscv_hart_m_p : public riscv_hart_common<BASE> {
public:
    using core = BASE;
    using base = riscv_hart_common<BASE>;
    using this_class = riscv_hart_m_p<BASE, FEAT>;
    using reg_t = typename core::reg_t;
    using phys_addr_t = typename core::phys_addr_t;

    void write_mstatus(reg_t val) {
        constexpr auto mask = base::get_mstatus_mask();
        auto new_val = (this->state.mstatus() & ~mask) | (val & mask);
        if constexpr(base::extension_status_mask)
            if((new_val & base::extension_status_mask) == base::extension_status_mask)
                new_val |= reg_t(1) << (sizeof(reg_t) * 8 - 1);
        this->state.mstatus = new_val;
    }

    riscv_hart_m_p();

    virtual ~riscv_hart_m_p() = default;

    void reset(uint64_t address) override;

    void enable_disass(bool enable) { riscv_hart_common<BASE>::enable_disass_output(enable); }

    iss::status read(const addr_t& addr, const unsigned length, uint8_t* const data);
    iss::status write(const addr_t& addr, const unsigned length, const uint8_t* const data);

    uint64_t enter_trap(uint64_t flags) override { return riscv_hart_m_p::enter_trap(flags, this->fault_data, this->fault_data); }
    uint64_t enter_trap(uint64_t flags, uint64_t addr, uint64_t instr) override;
    uint64_t leave_trap(uint64_t flags) override;

    void set_csr(unsigned addr, reg_t val) { this->csr[addr] = val; }

protected:
    std::unordered_map<uint64_t, uint8_t> atomic_reservation;

    iss::status read_status(unsigned addr, reg_t& val);
    iss::status write_status(unsigned addr, reg_t val);
    iss::status read_ie(unsigned addr, reg_t& val);
    iss::status write_ie(unsigned addr, reg_t val);
    iss::status read_ip(unsigned addr, reg_t& val);

    void check_interrupt();
    mem::neumann_memory_with_htif<BASE> default_mem;
};

template <typename BASE, features_e FEAT>
riscv_hart_m_p<BASE, FEAT>::riscv_hart_m_p()
: default_mem(base::get_priv_if()) {
    this->csr_rd_cb[mstatus] = MK_CSR_RD_CB(read_status);
    this->csr_wr_cb[mstatus] = MK_CSR_WR_CB(write_status);
    this->csr_rd_cb[mip] = MK_CSR_RD_CB(read_ip);
    this->csr_wr_cb[mip] = MK_CSR_WR_CB(write_plain);
    this->csr_rd_cb[mie] = MK_CSR_RD_CB(read_ie);
    this->csr_wr_cb[mie] = MK_CSR_WR_CB(write_ie);

    if(FEAT & FEAT_DEBUG)
        this->add_debug_csrs();

    this->rd_func = util::delegate<arch_if::rd_func_sig>::from<this_class, &this_class::read>(this);
    this->wr_func = util::delegate<arch_if::wr_func_sig>::from<this_class, &this_class::write>(this);
    this->memories.root(*this);
    this->memories.append(default_mem);
}

template <typename BASE, features_e FEAT>
iss::status riscv_hart_m_p<BASE, FEAT>::read(const addr_t& a, const unsigned length, uint8_t* const data) {
    auto& addr = a.val;
    auto& space = a.space;
    auto& access = a.access;
    auto& type = a.type;

#ifndef NDEBUG
    if(is_debug(access)) {
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
            case traits<BASE>::fence:
            case traits<BASE>::fencei:
                break;
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
                auto res = this->memory.rd_mem({address_type::PHYSICAL, a.access, a.space, a.val}, length, data);
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
iss::status riscv_hart_m_p<BASE, FEAT>::write(const addr_t& a, const unsigned length, const uint8_t* const data) {
    auto& addr = a.val;
    auto& space = a.space;
    auto& access = a.access;
    auto& type = a.type;
#ifndef NDEBUG
    const char* prefix = (is_debug(access)) ? "debug " : "";
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
            case traits<BASE>::fence:
            case traits<BASE>::fencei:
                break;
            default:
                return iss::Ok;
            }
        }
        case traits<BASE>::RES: {
            atomic_reservation[addr] = data[0];
        } break;
        default: {
            if(unlikely(is_fetch(access) && (addr & 0x1) == 1)) {
                this->fault_data = addr;
                if(access && access_type::DEBUG)
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
                auto res = this->memory.wr_mem({address_type::PHYSICAL, a.access, a.space, a.val}, length, data);
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

template <typename BASE, features_e FEAT> iss::status riscv_hart_m_p<BASE, FEAT>::read_status(unsigned addr, reg_t& val) {
    val = this->state.mstatus & base::get_mstatus_mask();
    return iss::Ok;
}

template <typename BASE, features_e FEAT> iss::status riscv_hart_m_p<BASE, FEAT>::write_status(unsigned addr, reg_t val) {
    write_mstatus(val);
    check_interrupt();
    return iss::Ok;
}

template <typename BASE, features_e FEAT> iss::status riscv_hart_m_p<BASE, FEAT>::read_ie(unsigned addr, reg_t& val) {
    auto mask = riscv_hart_common<BASE>::get_irq_mask(3);
    val = this->csr[mie] & mask;
    return iss::Ok;
}

template <typename BASE, features_e FEAT> iss::status riscv_hart_m_p<BASE, FEAT>::write_ie(unsigned addr, reg_t val) {
    // generate mask from allowed writable bits, the number of custom interrupts and the available ie bits
    auto mask = riscv_hart_common<BASE>::get_irq_mask(3) & this->clint_custom_irq_mask & 0x0888;
    this->csr[mie] = (this->csr[mie] & ~mask) | (val & mask);
    check_interrupt();
    return iss::Ok;
}

template <typename BASE, features_e FEAT> iss::status riscv_hart_m_p<BASE, FEAT>::read_ip(unsigned addr, reg_t& val) {
    auto mask = riscv_hart_common<BASE>::get_irq_mask(3);
    val = this->csr[mip] & mask;
    return iss::Ok;
}

template <typename BASE, features_e FEAT> inline void riscv_hart_m_p<BASE, FEAT>::reset(uint64_t address) {
    BASE::reset(address);
    this->state.mstatus = hart_state<reg_t>::mstatus_reset_val;
}

template <typename BASE, features_e FEAT> void riscv_hart_m_p<BASE, FEAT>::check_interrupt() {
    // Multiple simultaneous interrupts and traps at the same privilege level are
    // handled in the following decreasing priority order:
    // external interrupts, software interrupts, timer interrupts, then finally
    // any synchronous traps.
    auto ena_irq = this->csr[mip] & this->csr[mie];

    bool mstatus_mie = this->state.mstatus.MIE;
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

template <typename BASE, features_e FEAT> uint64_t riscv_hart_m_p<BASE, FEAT>::enter_trap(uint64_t flags, uint64_t addr, uint64_t tval) {
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
        this->csr[mepc] = static_cast<reg_t>(addr) & this->get_pc_mask(); // store actual address instruction of exception
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
            this->csr[mtval] = (!this->has_compressed() || (tval & 0x3) == 3) ? tval : tval & 0xffff;
#endif
            break;
        case traits<BASE>::RV_CAUSE_BREAKPOINT:
            if((FEAT & FEAT_DEBUG) && (this->csr[dcsr] & 0x8000)) {
                this->reg.DPC = addr;
                this->csr[dcsr] = (this->csr[dcsr] & ~0x1c3) | (1 << 6) | PRIV_M; // FIXME: cause should not be 4 (stepi)
                new_priv = this->reg.PRIV | PRIV_D;
            } else {
                this->csr[mtval] = addr;
            }
            if(this->semihosting_cb) {
                // Check for semihosting call
                std::array<uint8_t, 8> data;
                // check for SLLI_X0_X0_0X1F and SRAI_X0_X0_0X07
                this->memory.rd_mem({address_type::PHYSICAL, access_type::DEBUG_READ, traits<BASE>::IMEM, addr - 4}, 4, data.data());
                addr += 8;
                this->memory.rd_mem({address_type::PHYSICAL, access_type::DEBUG_READ, traits<BASE>::IMEM, addr - 4}, 4, data.data() + 4);

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
            this->csr[mtval] = this->fault_data;
            break;
        default:
            this->csr[mtval] = 0;
        }
        this->fault_data = 0;
    } else {
        this->csr[mepc] = this->reg.NEXT_PC & this->get_pc_mask(); // store next address if interrupt
        this->reg.pending_trap = 0;
    }
    this->csr[mcause] = (trap_id << (traits<BASE>::XLEN - 1)) + cause;
    // update mstatus
    // xPP field of mstatus is written with the active privilege mode at the time
    // of the trap; the x PIE field of mstatus
    // is written with the value of the active interrupt-enable bit at the time of
    // the trap; and the x IE field of mstatus
    // is cleared
    // store the actual privilege level in yPP and store interrupt enable flags
    this->state.mstatus.MPP = PRIV_M;
    this->state.mstatus.MPIE = this->state.mstatus.MIE;
    this->state.mstatus.MIE = false;

    // get trap vector
    auto xtvec = this->csr[mtvec];
    // calculate adds// set NEXT_PC to trap addressess to jump to based on MODE
    if(trap_id != 0 && (xtvec & 0x3UL) == 3UL) {
        reg_t data;
        auto ret = this->memory.rd_mem({address_type::PHYSICAL, access_type::DEBUG_READ, traits<BASE>::MEM, this->csr[arch::mtvt]},
                                       sizeof(reg_t), reinterpret_cast<uint8_t*>(&data));
        if(ret == iss::Err)
            return this->reg.PC;
        this->reg.NEXT_PC = data;
    } else {
        // bits in mtvec
        this->reg.NEXT_PC = xtvec & ~0x3UL;
        if((xtvec & 0x1) == 1 && trap_id != 0)
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
            this->disass_output(fmt::format("Interrupt with cause '{}' ({}) occurred  at address {}", irq_str, cause, buffer.data()));
        } else {
            auto irq_str = cause < this->trap_str.size() ? this->trap_str.at(cause) : "Unknown";
            this->disass_output(fmt::format("Trap with cause '{}' ({}) occurred  at address {}", irq_str, cause, buffer.data()));
        }
    }
    // reset trap state
    this->reg.PRIV = new_priv;
    this->reg.trap_state = 0;
    return this->reg.NEXT_PC;
}

template <typename BASE, features_e FEAT> uint64_t riscv_hart_m_p<BASE, FEAT>::leave_trap(uint64_t flags) {
    this->state.mstatus.MIE = this->state.mstatus.MPIE;
    this->state.mstatus.MPIE = 1;
    // sets the pc to the value stored in the x epc register.
    this->reg.NEXT_PC = this->csr[mepc] & this->get_pc_mask();
    this->disass_output("Executing xRET");
    check_interrupt();
    this->reg.trap_state = this->reg.pending_trap;
    return this->reg.NEXT_PC;
}

} // namespace arch
} // namespace iss

#endif /* _RISCV_HART_M_P_H */
