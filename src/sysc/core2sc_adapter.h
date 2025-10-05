/*******************************************************************************
 * Copyright (C) 2023 - 2025 MINRES Technologies GmbH
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

#ifndef _SYSC_CORE2SC_ADAPTER_H_
#define _SYSC_CORE2SC_ADAPTER_H_

#include "core_complex.h"
#include "sc2core_if.h"
#include <iostream>
#include <iss/arch/riscv_hart_common.h>
#include <iss/iss.h>
#include <iss/mem/memory_if.h>
#include <iss/vm_types.h>
#include <scc/report.h>
#include <util/ities.h>
namespace sysc {
template <typename PLAT> class core2sc_adapter : public PLAT, public sc2core_if {
public:
    using this_class = core2sc_adapter<PLAT>;
    using core = typename PLAT::core;
    using reg_t = typename PLAT::reg_t;
    using phys_addr_t = typename PLAT::phys_addr_t;
    core2sc_adapter(sysc::riscv::core_complex_if* owner)
    : owner(owner) {
        this->csr_rd_cb[iss::arch::time] = MK_CSR_RD_CB(read_time);
        if(sizeof(reg_t) == 4)
            this->csr_rd_cb[iss::arch::timeh] = MK_CSR_RD_CB(read_time);
        this->memories.replace_last(*this);
        this->set_hartid = util::delegate<void(unsigned)>::from<this_class, &this_class::_set_mhartid>(this);
        this->set_irq_count = util::delegate<void(unsigned)>::from<this_class, &this_class::_set_irq_num>(this);
        this->get_mode = util::delegate<uint32_t()>::from<this_class, &this_class::_get_mode>(this);
        this->get_state = util::delegate<uint64_t()>::from<this_class, &this_class::_get_state>(this);
        this->get_interrupt_execution = util::delegate<bool()>::from<this_class, &this_class::_get_interrupt_execution>(this);
        this->set_interrupt_execution = util::delegate<void(bool)>::from<this_class, &this_class::_set_interrupt_execution>(this);
        this->local_irq = util::delegate<void(short, bool)>::from<this_class, &this_class::_local_irq>(this);
        this->register_csr_rd = util::delegate<void(unsigned, rd_csr_f)>::from<this_class, &this_class::_register_csr_rd>(this);
        this->register_csr_wr = util::delegate<void(unsigned, wr_csr_f)>::from<this_class, &this_class::_register_csr_wr>(this);
    }

    virtual ~core2sc_adapter() {}

    void notify_phase(iss::arch_if::exec_phase p) {
        if(p == iss::arch_if::ISTART && !first) {
            auto cycle_incr = owner->get_last_bus_cycles();
            if(cycle_incr > 1)
                this->instr_if.update_last_instr_cycles(cycle_incr);
            owner->sync(this->instr_if.get_total_cycles());
        }
        first = false;
    }

    iss::sync_type needed_sync() const { return iss::PRE_SYNC; }

    void disass_output(uint64_t pc, const std::string instr) {
        static constexpr std::array<const char, 4> lvl = {{'U', 'S', 'H', 'M'}};
        if(!owner->disass_output(pc, instr)) {
            std::stringstream s;
            s << "[p:" << lvl[this->reg.PRIV] << ";s:0x" << std::hex << std::setfill('0') << std::setw(sizeof(reg_t) * 2)
              << (reg_t)this->state.mstatus << std::dec << ";c:" << this->reg.icount + this->cycle_offset << "]";
            SCCDEBUG(owner->hier_name()) << "disass: " << "0x" << std::setw(16) << std::right << std::setfill('0') << std::hex << pc
                                         << "\t\t" << std::setw(40) << std::setfill(' ') << std::left << instr << s.str();
        }
    };

    iss::mem::memory_if get_mem_if() {
        return iss::mem::memory_if{.rd_mem{util::delegate<iss::mem::rd_mem_func_sig>::from<this_class, &this_class::read_mem>(this)},
                                   .wr_mem{util::delegate<iss::mem::wr_mem_func_sig>::from<this_class, &this_class::write_mem>(this)}};
    }

    iss::status read_mem(iss::access_type access, uint64_t addr, unsigned length, uint8_t* data) {
        if(access && iss::access_type::DEBUG)
            return owner->read_mem_dbg(addr, length, data) ? iss::Ok : iss::Err;
        else {
            return owner->read_mem(addr, length, data, is_fetch(access)) ? iss::Ok : iss::Err;
        }
    }

    iss::status write_mem(iss::access_type access, uint64_t addr, unsigned length, uint8_t const* data) {
        if(access && iss::access_type::DEBUG)
            return owner->write_mem_dbg(addr, length, data) ? iss::Ok : iss::Err;
        if(addr == this->tohost) {
            reg_t cur_data = *reinterpret_cast<const reg_t*>(data);
            // Extract Device (bits 63:56)
            uint8_t device = sizeof(reg_t) == 4 ? 0 : (cur_data >> 56) & 0xFF;
            // Extract Command (bits 55:48)
            uint8_t command = sizeof(reg_t) == 4 ? 0 : (cur_data >> 48) & 0xFF;
            // Extract payload (bits 47:0)
            uint64_t payload_addr = cur_data & 0xFFFFFFFFFFFFULL; // 24bits
            if(payload_addr & 1) {
                if(payload_addr != 0x1) {
                    SCCERR(owner->hier_name()) << "tohost value is 0x" << std::hex << payload_addr << std::dec << " (" << payload_addr
                                               << "), stopping simulation";
                } else {
                    SCCINFO(owner->hier_name())
                        << "tohost value is 0x" << std::hex << payload_addr << std::dec << " (" << payload_addr << "), stopping simulation";
                }
                this->reg.trap_state = std::numeric_limits<uint32_t>::max();
                this->interrupt_sim = payload_addr;
                return iss::Ok;
            }
            if(device == 0 && command == 0) {
                std::array<uint64_t, 8> loaded_payload;
                auto res = owner->read_mem(payload_addr, 8 * sizeof(uint64_t), reinterpret_cast<uint8_t*>(loaded_payload.data()), false)
                               ? iss::Ok
                               : iss::Err;
                if(res == iss::Err) {
                    SCCERR(owner->hier_name()) << "Syscall read went wrong";
                    return iss::Ok;
                }
                uint64_t syscall_num = loaded_payload.at(0);
                if(syscall_num == 64) // SYS_WRITE
                    return this->execute_sys_write(this, loaded_payload, PLAT::MEM);
                SCCERR(owner->hier_name()) << "tohost syscall with number 0x" << std::hex << syscall_num << std::dec << " (" << syscall_num
                                           << ") not implemented";
                this->reg.trap_state = std::numeric_limits<uint32_t>::max();
                this->interrupt_sim = payload_addr;
                return iss::Ok;
            }
            SCCERR(owner->hier_name()) << "tohost functionality not implemented for device " << device << " and command " << command;
            this->reg.trap_state = std::numeric_limits<uint32_t>::max();
            this->interrupt_sim = payload_addr;
            return iss::Ok;
        }
        auto res = owner->write_mem(addr, length, data) ? iss::Ok : iss::Err;
        return res;
    }

    iss::status read_time(unsigned addr, reg_t& val) {
        uint64_t time_val = owner->mtime_i.get_interface() ? owner->mtime_i.read() : 0;
        if(addr == iss::arch::time) {
            val = static_cast<reg_t>(time_val);
        } else if(addr == iss::arch::timeh) {
            if(sizeof(reg_t) != 4)
                return iss::Err;
            val = static_cast<reg_t>(time_val >> 32);
        }
        return iss::Ok;
    }

    void wait_until(uint64_t flags) {
        SCCDEBUG(owner->hier_name()) << "Sleeping until interrupt";
        PLAT::wait_until(flags);
        while(this->reg.pending_trap == 0 && (this->csr[iss::arch::mip] & this->csr[iss::arch::mie]) == 0) {
            sc_core::wait(wfi_evt);
        }
    }

private:
    void _set_mhartid(unsigned id) { PLAT::set_mhartid(id); }

    void _set_irq_num(unsigned num) { PLAT::set_irq_num(num); }

    uint32_t _get_mode() { return this->reg.PRIV; }

    void _set_interrupt_execution(bool v) { this->interrupt_sim = v ? 1 : 0; }

    bool _get_interrupt_execution() { return this->interrupt_sim; }

    uint64_t _get_state() { return this->state.mstatus.backing.val; }

    void _local_irq(short id, bool value) {
        reg_t mask = 0;
        switch(id) {
        case 3: // SW
            mask = 1 << 3;
            break;
        case 7: // timer
            mask = 1 << 7;
            break;
        case 11: // external
            mask = 1 << 11;
            break;
        default:
            if(id > 15)
                mask = 1 << id;
            break;
        }
        if(value) {
            this->csr[iss::arch::mip] |= mask;
            wfi_evt.notify();
        } else
            this->csr[iss::arch::mip] &= ~mask;
        this->check_interrupt();
        if(value)
            SCCTRACE(owner->hier_name()) << "Triggering interrupt " << id << " Pending trap: " << this->reg.pending_trap;
    }

    void _register_csr_rd(unsigned addr, rd_csr_f cb) {
        std::function<iss::status(unsigned addr, reg_t&)> lambda = [cb](unsigned addr, reg_t& r) -> iss::status {
            uint64_t temp = r;
            auto ret = cb(addr, temp);
            r = temp;
            return ret;
        };
        this->register_csr(addr, lambda);
    }
    void _register_csr_wr(unsigned addr, wr_csr_f cb) {
        std::function<iss::status(unsigned addr, reg_t)> lambda = [cb](unsigned addr, reg_t r) -> iss::status { return cb(addr, r); };
        this->register_csr(addr, lambda);
    }

    sysc::riscv::core_complex_if* const owner{nullptr};
    sc_core::sc_event wfi_evt;
    unsigned to_host_wr_cnt = 0;
    bool first{true};
};
} // namespace sysc
#endif /* _SYSC_CORE2SC_ADAPTER_H_ */
