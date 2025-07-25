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

#ifndef _RISCV_HART_MU_P_H
#define _RISCV_HART_MU_P_H

#include "iss/vm_if.h"
#include "iss/vm_types.h"
#include "riscv_hart_common.h"
#include "util/logging.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <elfio/elf_types.hpp>
#include <elfio/elfio.hpp>
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>
#include <iss/mem/memory_with_htif.h>
#include <unordered_map>

namespace iss {
namespace arch {

template <typename BASE, features_e FEAT = FEAT_NONE, typename LOGCAT = logging::disass>
class riscv_hart_mu_p : public riscv_hart_common<BASE> {
public:
    using core = BASE;
    using base = riscv_hart_common<BASE>;
    using this_class = riscv_hart_mu_p<BASE, FEAT, LOGCAT>;
    using reg_t = typename core::reg_t;

    static constexpr reg_t get_mstatus_mask(unsigned priv_lvl) {
        if(sizeof(reg_t) == 4) {
#if __cplusplus < 201402L
            return priv_lvl == PRIV_U ? 0x80000011UL : priv_lvl == PRIV_S ? 0x800de133UL : 0x807ff9ddUL;
#else
            switch(priv_lvl) {
            case PRIV_U:
                return FEAT & features_e::FEAT_EXT_N ? 0x00000011UL : 0UL; // 0b1...0 0001 0001
            default:
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
                //       |        |||||| | | |  ||  +-UPIE
                //       |        ||||||/|/|/|  ||  |+-MIE
                //       |        ||||||/|/|/|  ||  ||  +-UIE
                return 0b10000000001000000001100010011001;
            }
#endif
        } else if(sizeof(reg_t) == 8) {
#if __cplusplus < 201402L
            return priv_lvl == PRIV_U ? 0x011ULL : priv_lvl == PRIV_S ? 0x000de133ULL : 0x007ff9ddULL;
#else
            switch(priv_lvl) {
            case PRIV_U:
                return FEAT & features_e::FEAT_EXT_N ? 0x8000000000000011ULL : 0ULL; // 0b1...0 0001 0001
            default:
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
                //                |||||| | | |  ||  +-UPIE
                //                ||||||/|/|/|  ||  |+-MIE
                //                ||||||/|/|/|  ||  ||  +-UIE
                return 0b00000000001000000001100010011001 | 0x8000000000000000ULL;
            }
#endif
        } else
            assert(false && "Unsupported XLEN value");
    }

    void write_mstatus(reg_t val, unsigned priv_lvl) {
        auto mask = get_mstatus_mask(priv_lvl);
        auto new_val = (this->state.mstatus() & ~mask) | (val & mask);
        this->state.mstatus = new_val;
    }

    constexpr reg_t get_irq_mask(size_t mode) {
        std::array<const reg_t, 4> m = {{
            0b000100010001, // U mode
            0b001100110011, // S mode
            0,
            0b100110011001 // M mode
        }};
        return m[mode];
    }

    riscv_hart_mu_p();

    virtual ~riscv_hart_mu_p() = default;

    void reset(uint64_t address) override;

    iss::status read(const address_type type, const access_type access, const uint32_t space, const uint64_t addr, const unsigned length,
                     uint8_t* const data);
    iss::status write(const address_type type, const access_type access, const uint32_t space, const uint64_t addr, const unsigned length,
                      const uint8_t* const data);

    uint64_t enter_trap(uint64_t flags) override { return riscv_hart_mu_p::enter_trap(flags, this->fault_data, this->fault_data); }
    uint64_t enter_trap(uint64_t flags, uint64_t addr, uint64_t instr) override;
    uint64_t leave_trap(uint64_t flags) override;
    void wait_until(uint64_t flags) override;

    void set_csr(unsigned addr, reg_t val) { this->csr[addr & this->csr.page_addr_mask] = val; }

protected:
    using mem_read_f = iss::status(iss::phys_addr_t addr, unsigned, uint8_t* const);
    using mem_write_f = iss::status(iss::phys_addr_t addr, unsigned, uint8_t const* const);

    hart_state<reg_t> state;

    std::unordered_map<uint64_t, uint8_t> atomic_reservation;

    iss::status read_status(unsigned addr, reg_t& val);
    iss::status write_status(unsigned addr, reg_t val);
    iss::status read_cause(unsigned addr, reg_t& val);
    iss::status write_cause(unsigned addr, reg_t val);
    iss::status read_ie(unsigned addr, reg_t& val);
    iss::status write_ie(unsigned addr, reg_t val);
    iss::status read_ip(unsigned addr, reg_t& val);
    iss::status write_xtvt(unsigned addr, reg_t val);
    iss::status write_dcsr(unsigned addr, reg_t val);
    iss::status read_debug(unsigned addr, reg_t& val);
    iss::status write_dscratch(unsigned addr, reg_t val);
    iss::status read_dpc(unsigned addr, reg_t& val);
    iss::status write_dpc(unsigned addr, reg_t val);

    void check_interrupt();
    mem::memory_with_htif<reg_t> default_mem;
};

template <typename BASE, features_e FEAT, typename LOGCAT>
riscv_hart_mu_p<BASE, FEAT, LOGCAT>::riscv_hart_mu_p()
: state()
, default_mem(base::get_priv_if()) {
    const std::array<unsigned, 8> rwaddrs{{
        mepc,
        mtvec,
        mscratch,
        mtval,
        uepc,
        utvec,
        uscratch,
        utval,
    }};
    for(auto addr : rwaddrs) {
        this->csr_rd_cb[addr] = MK_CSR_RD_CB(read_plain);
        this->csr_wr_cb[addr] = MK_CSR_WR_CB(write_plain);
    }
    this->csr_rd_cb[mstatus] = MK_CSR_RD_CB(read_status);
    this->csr_wr_cb[mstatus] = MK_CSR_WR_CB(write_status);
    this->csr_rd_cb[mcause] = MK_CSR_RD_CB(read_cause);
    this->csr_wr_cb[mcause] = MK_CSR_WR_CB(write_cause);
    this->csr_rd_cb[mtvec] = MK_CSR_RD_CB(read_tvec);
    this->csr_wr_cb[mepc] = MK_CSR_WR_CB(write_epc);
    this->csr_rd_cb[mip] = MK_CSR_RD_CB(read_ip);
    this->csr_wr_cb[mip] = MK_CSR_WR_CB(write_null);
    this->csr_rd_cb[mie] = MK_CSR_RD_CB(read_ie);
    this->csr_wr_cb[mie] = MK_CSR_WR_CB(write_ie);
    this->csr_rd_cb[mcounteren] = MK_CSR_RD_CB(read_null);
    this->csr_wr_cb[mcounteren] = MK_CSR_WR_CB(write_null);
    this->csr_wr_cb[misa] = MK_CSR_WR_CB(write_null);
    this->csr_wr_cb[mvendorid] = MK_CSR_WR_CB(write_null);
    this->csr_wr_cb[marchid] = MK_CSR_WR_CB(write_null);
    this->csr_wr_cb[mimpid] = MK_CSR_WR_CB(write_null);

    if(FEAT & FEAT_EXT_N) {
        this->csr_rd_cb[uie] = MK_CSR_RD_CB(read_ie);
        this->csr_wr_cb[uie] = MK_CSR_WR_CB(write_ie);
        this->csr_rd_cb[uip] = MK_CSR_RD_CB(read_ip);
        this->csr_wr_cb[uip] = MK_CSR_WR_CB(write_null);
        this->csr_wr_cb[uepc] = MK_CSR_WR_CB(write_epc);
        this->csr_rd_cb[ustatus] = MK_CSR_RD_CB(read_status);
        this->csr_wr_cb[ustatus] = MK_CSR_WR_CB(write_status);
        this->csr_rd_cb[ucause] = MK_CSR_RD_CB(read_cause);
        this->csr_wr_cb[ucause] = MK_CSR_WR_CB(write_cause);
        this->csr_rd_cb[utvec] = MK_CSR_RD_CB(read_tvec);
    }
    if(FEAT & FEAT_DEBUG) {
        this->csr_wr_cb[dscratch0] = MK_CSR_WR_CB(write_dscratch);
        this->csr_rd_cb[dscratch0] = MK_CSR_RD_CB(read_debug);
        this->csr_wr_cb[dscratch1] = MK_CSR_WR_CB(write_dscratch);
        this->csr_rd_cb[dscratch1] = MK_CSR_RD_CB(read_debug);
        this->csr_wr_cb[dpc] = MK_CSR_WR_CB(write_dpc);
        this->csr_rd_cb[dpc] = MK_CSR_RD_CB(read_dpc);
        this->csr_wr_cb[dcsr] = MK_CSR_WR_CB(write_dcsr);
        this->csr_rd_cb[dcsr] = MK_CSR_RD_CB(read_debug);
    }
    this->rd_func = util::delegate<arch_if::rd_func_sig>::from<this_class, &this_class::read>(this);
    this->wr_func = util::delegate<arch_if::wr_func_sig>::from<this_class, &this_class::write>(this);
    this->memories.root(*this);
    this->memories.append(default_mem);
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_mu_p<BASE, FEAT, LOGCAT>::read(const address_type type, const access_type access, const uint32_t space,
                                                      const uint64_t addr, const unsigned length, uint8_t* const data) {
#ifndef NDEBUG
    if(access && iss::access_type::DEBUG) {
        CPPLOG(TRACEALL) << "debug read of " << length << " bytes @addr 0x" << std::hex << addr;
    } else if(is_fetch(access)) {
        CPPLOG(TRACEALL) << "fetch of " << length << " bytes  @addr 0x" << std::hex << addr;
    } else {
        CPPLOG(TRACE) << "read of " << length << " bytes  @addr 0x" << std::hex << addr;
    }
#endif
    try {
        switch(space) {
        case traits<BASE>::MEM: {
            auto alignment = is_fetch(access) ? (this->has_compressed() ? 2 : 4) : std::min<unsigned>(length, sizeof(reg_t));
            if(unlikely(is_fetch(access) && (addr & (alignment - 1)))) {
                this->fault_data = addr;
                if(is_debug(access))
                    throw trap_access(0, addr);
                this->reg.trap_state = (1UL << 31); // issue trap 0
                return iss::Err;
            }
            try {
                if(!is_debug(access) && (addr & (alignment - 1))) {
                    this->reg.trap_state = (1UL << 31) | 4 << 16;
                    this->fault_data = addr;
                    return iss::Err;
                }
                auto res = this->memory.rd_mem(access, addr, length, data);
                if(unlikely(res != iss::Ok && (access & access_type::DEBUG) == 0)) {
                    this->reg.trap_state = (1UL << 31) | (5 << 16); // issue trap 5 (load access fault
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
        case traits<BASE>::CSR: {
            if(length != sizeof(reg_t))
                return iss::Err;
            return this->read_csr(addr, *reinterpret_cast<reg_t* const>(data));
        } break;
        case traits<BASE>::FENCE: {
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
            this->fault_data = ta.addr;
        }
        return iss::Err;
    }
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_mu_p<BASE, FEAT, LOGCAT>::write(const address_type type, const access_type access, const uint32_t space,
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
        CPPLOG(TRACE) << prefix << "write of " << length << " bytes @addr 0x" << std::hex << addr;
    }
#endif
    try {
        switch(space) {
        case traits<BASE>::MEM: {
            if(unlikely(is_fetch(access) && (addr & 0x1) == 1)) {
                this->fault_data = addr;
                if(access && iss::access_type::DEBUG)
                    throw trap_access(0, addr);
                this->reg.trap_state = (1UL << 31); // issue trap 0
                return iss::Err;
            }
            try {
                auto alignment = std::min<unsigned>(length, sizeof(reg_t));
                if(length > 1 && (addr & (alignment - 1)) && !is_debug(access)) {
                    this->reg.trap_state = (1UL << 31) | 6 << 16;
                    this->fault_data = addr;
                    return iss::Err;
                }
                auto res = this->memory.wr_mem(access, addr, length, data);
                if(unlikely(res != iss::Ok && !is_debug(access))) {
                    this->reg.trap_state = (1UL << 31) | (7UL << 16); // issue trap 7 (Store/AMO access fault)
                    this->fault_data = addr;
                }
                return res;
            } catch(trap_access& ta) {
                this->reg.trap_state = (1UL << 31) | ta.id;
                this->fault_data = ta.addr;
                return iss::Err;
            }
        } break;
        case traits<BASE>::CSR: {
            if(length != sizeof(reg_t))
                return iss::Err;
            return this->write_csr(addr, *reinterpret_cast<const reg_t*>(data));
        } break;
        case traits<BASE>::FENCE:
            break;
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
            this->fault_data = ta.addr;
        }
        return iss::Err;
    }
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_mu_p<BASE, FEAT, LOGCAT>::read_status(unsigned addr, reg_t& val) {
    val = state.mstatus & get_mstatus_mask((addr >> 8) & 0x3);
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_mu_p<BASE, FEAT, LOGCAT>::write_status(unsigned addr, reg_t val) {
    write_mstatus(val, (addr >> 8) & 0x3);
    check_interrupt();
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_mu_p<BASE, FEAT, LOGCAT>::read_cause(unsigned addr, reg_t& val) {
    val = this->csr[addr] & ((1UL << (traits<BASE>::XLEN - 1)) | (this->mcause_max_irq - 1));
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_mu_p<BASE, FEAT, LOGCAT>::write_cause(unsigned addr, reg_t val) {
    auto mask = ((1UL << (traits<BASE>::XLEN - 1)) | (this->mcause_max_irq - 1));
    this->csr[addr] = (val & mask) | (this->csr[addr] & ~mask);
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_mu_p<BASE, FEAT, LOGCAT>::read_ie(unsigned addr, reg_t& val) {
    auto mask = get_irq_mask((addr >> 8) & 0x3);
    val = this->csr[mie] & mask;
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_mu_p<BASE, FEAT, LOGCAT>::write_ie(unsigned addr, reg_t val) {
    auto mask = get_irq_mask((addr >> 8) & 0x3);
    this->csr[mie] = (this->csr[mie] & ~mask) | (val & mask);
    check_interrupt();
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT>
iss::status riscv_hart_mu_p<BASE, FEAT, LOGCAT>::read_ip(unsigned addr, reg_t& val) {
    auto mask = get_irq_mask((addr >> 8) & 0x3);
    val = this->csr[mip] & mask;
    return iss::Ok;
}

template <typename BASE, features_e FEAT, typename LOGCAT> inline void riscv_hart_mu_p<BASE, FEAT, LOGCAT>::reset(uint64_t address) {
    BASE::reset(address);
    state.mstatus = hart_state<reg_t>::mstatus_reset_val;
}

template <typename BASE, features_e FEAT, typename LOGCAT> void riscv_hart_mu_p<BASE, FEAT, LOGCAT>::check_interrupt() {
    // TODO: Implement CLIC functionality
    // Multiple simultaneous interrupts and traps at the same privilege level are
    // handled in the following decreasing priority order:
    // external interrupts, software interrupts, timer interrupts, then finally
    // any synchronous traps.
    auto ena_irq = this->csr[mip] & this->csr[mie];

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
uint64_t riscv_hart_mu_p<BASE, FEAT, LOGCAT>::enter_trap(uint64_t flags, uint64_t addr, uint64_t tval) {
    // flags are ACTIVE[31:31], CAUSE[30:16], TRAPID[15:0]
    // calculate and write mcause val
    if(flags == std::numeric_limits<uint64_t>::max())
        flags = this->reg.trap_state;
    auto const trap_id = bit_sub<0, 16>(flags);
    auto cause = bit_sub<16, 15>(flags);
    if(trap_id == 0 && cause == 11)
        cause = 0x8 + this->reg.PRIV; // adjust environment call cause
    // calculate effective privilege level
    unsigned new_priv = PRIV_M;
    if(trap_id == 0) { // exception
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
        case 0:
            this->csr[utval | (new_priv << 8)] = static_cast<reg_t>(addr);
            break;
        case 2:
            this->csr[utval | (new_priv << 8)] = (!this->has_compressed() || (tval & 0x3) == 3) ? tval : tval & 0xffff;
            break;
        case 3:
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
                this->memory.rd_mem(iss::access_type::DEBUG_READ, addr - 4, 4, data.data());
                addr += 8;
                this->memory.rd_mem(iss::access_type::DEBUG_READ, addr - 4, 4, data.data() + 4);

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

                    this->semihosting_cb(this, &(this->reg.X10) /*a0*/, &(this->reg.X11) /*a1*/);
                    return this->reg.NEXT_PC;
                }
            }
            break;
        case 4:
        case 6:
        case 7:
            this->csr[utval | (new_priv << 8)] = this->fault_data;
            break;
        default:
            this->csr[utval | (new_priv << 8)] = 0;
        }
        this->fault_data = 0;
    } else {
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
        state.mstatus.MPP = this->reg.PRIV;
        state.mstatus.MPIE = state.mstatus.MIE;
        state.mstatus.MIE = false;
        break;
    case PRIV_U:
        state.mstatus.UPIE = state.mstatus.UIE;
        state.mstatus.UIE = false;
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
            auto ret = read(address_type::LOGICAL, access_type::READ, 0, this->csr[mtvt], sizeof(reg_t), reinterpret_cast<uint8_t*>(&data));
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
    if((flags & 0xffffffff) != 0xffffffff)
        NSCLOG(INFO, LOGCAT) << (trap_id ? "Interrupt" : "Trap") << " with cause '"
                             << (trap_id ? this->irq_str[cause] : this->trap_str[cause]) << "' (" << cause << ")"
                             << " at address " << buffer.data() << " occurred, changing privilege level from " << this->lvl[this->reg.PRIV]
                             << " to " << this->lvl[new_priv];
    // reset trap state
    this->reg.PRIV = new_priv;
    this->reg.trap_state = 0;
    return this->reg.NEXT_PC;
}

template <typename BASE, features_e FEAT, typename LOGCAT> uint64_t riscv_hart_mu_p<BASE, FEAT, LOGCAT>::leave_trap(uint64_t flags) {
    auto cur_priv = this->reg.PRIV;
    auto inst_priv = (flags & 0x3) ? 3 : 0;
    if(inst_priv > cur_priv) {
        this->reg.trap_state = 0x80ULL << 24 | (2 << 16); // illegal instruction
        this->reg.NEXT_PC = std::numeric_limits<uint32_t>::max();
    } else {
        auto status = state.mstatus;
        // pop the relevant lower-privilege interrupt enable and privilege mode stack
        // clear respective yIE
        switch(inst_priv) {
        case PRIV_M:
            this->reg.PRIV = state.mstatus.MPP;
            state.mstatus.MPP = 0; // clear mpp to U mode
            state.mstatus.MIE = state.mstatus.MPIE;
            state.mstatus.MPIE = 1;
            break;
        case PRIV_U:
            this->reg.PRIV = 0;
            state.mstatus.UIE = state.mstatus.UPIE;
            state.mstatus.UPIE = 1;
            break;
        }
        // sets the pc to the value stored in the x epc register.
        this->reg.NEXT_PC = this->csr[uepc | inst_priv << 8];
        NSCLOG(INFO, LOGCAT) << "Executing xRET , changing privilege level from " << this->lvl[cur_priv] << " to "
                             << this->lvl[this->reg.PRIV];
        check_interrupt();
    }
    this->reg.trap_state = this->reg.pending_trap;
    return this->reg.NEXT_PC;
}

template <typename BASE, features_e FEAT, typename LOGCAT> void riscv_hart_mu_p<BASE, FEAT, LOGCAT>::wait_until(uint64_t flags) {
    auto status = state.mstatus;
    auto tw = status.TW;
    if(this->reg.PRIV == PRIV_S && tw != 0) {
        this->reg.trap_state = (1 << 31) | (2 << 16);
        this->fault_data = this->reg.PC;
    }
}
} // namespace arch
} // namespace iss

#endif /* _RISCV_HART_MU_P_H */
