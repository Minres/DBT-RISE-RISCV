/*
 * sc_core_adapter.h
 *
 *  Created on: Jul 5, 2023
 *      Author: eyck
 */

#ifndef _SYSC_SC_CORE_ADAPTER_H_
#define _SYSC_SC_CORE_ADAPTER_H_

#include "sc_core_adapter_if.h"
#include <iostream>
#include <iss/iss.h>
#include <iss/vm_types.h>
#include <scc/report.h>
#include <util/ities.h>

namespace sysc {
template <typename PLAT> class sc_core_adapter : public PLAT, public sc_core_adapter_if {
public:
    using reg_t = typename iss::arch::traits<typename PLAT::core>::reg_t;
    using phys_addr_t = typename iss::arch::traits<typename PLAT::core>::phys_addr_t;
    using heart_state_t = typename PLAT::hart_state_type;
    sc_core_adapter(sysc::riscv_vp::core_complex_if* owner)
    : owner(owner) {}

    iss::arch_if* get_arch_if() override { return this; }

    void set_mhartid(unsigned id) override { PLAT::set_mhartid(id); }

    uint32_t get_mode() override { return this->reg.PRIV; }

    void set_interrupt_execution(bool v) override { this->interrupt_sim = v ? 1 : 0; }

    bool get_interrupt_execution() override { return this->interrupt_sim; }

    uint64_t get_state() override { return this->state.mstatus.backing.val; }

    void notify_phase(iss::arch_if::exec_phase p) override {
        if(p == iss::arch_if::ISTART && !first) {
            auto cycle_incr = owner->get_last_bus_cycles();
            if(cycle_incr > 1)
                this->instr_if.update_last_instr_cycles(cycle_incr);
            owner->sync(this->instr_if.get_total_cycles());
        }
        first = false;
    }

    iss::sync_type needed_sync() const override { return iss::PRE_SYNC; }

    void disass_output(uint64_t pc, const std::string instr) override {
        static constexpr std::array<const char, 4> lvl = {{'U', 'S', 'H', 'M'}};
        if(!owner->disass_output(pc, instr)) {
            std::stringstream s;
            s << "[p:" << lvl[this->reg.PRIV] << ";s:0x" << std::hex << std::setfill('0') << std::setw(sizeof(reg_t) * 2)
              << (reg_t)this->state.mstatus << std::dec << ";c:" << this->reg.icount + this->cycle_offset << "]";
            SCCDEBUG(owner->hier_name()) << "disass: "
                                         << "0x" << std::setw(16) << std::right << std::setfill('0') << std::hex << pc << "\t\t"
                                         << std::setw(40) << std::setfill(' ') << std::left << instr << s.str();
        }
    };

    iss::status read_mem(phys_addr_t addr, unsigned length, uint8_t* const data) override {
        if(addr.access && iss::access_type::DEBUG)
            return owner->read_mem_dbg(addr.val, length, data) ? iss::Ok : iss::Err;
        else {
            return owner->read_mem(addr.val, length, data, is_fetch(addr.access)) ? iss::Ok : iss::Err;
        }
    }

    iss::status write_mem(phys_addr_t addr, unsigned length, const uint8_t* const data) override {
        if(addr.access && iss::access_type::DEBUG)
            return owner->write_mem_dbg(addr.val, length, data) ? iss::Ok : iss::Err;
        else {
            auto tohost_upper = (sizeof(reg_t) == 4 && addr.val == (this->tohost + 4)) || (sizeof(reg_t) == 8 && addr.val == this->tohost);
            auto tohost_lower = (sizeof(reg_t) == 4 && addr.val == this->tohost) || (sizeof(reg_t) == 64 && addr.val == this->tohost);
            if(tohost_lower || tohost_upper) {
                if(tohost_upper || (tohost_lower && to_host_wr_cnt > 0)) {
                    switch(hostvar >> 48) {
                    case 0:
                        if(hostvar != 0x1) {
                            SCCINFO(owner->hier_name())
                                << "tohost value is 0x" << std::hex << hostvar << std::dec << " (" << hostvar << "), stopping simulation";
                        } else {
                            SCCINFO(owner->hier_name())
                                << "tohost value is 0x" << std::hex << hostvar << std::dec << " (" << hostvar << "), stopping simulation";
                        }
                        this->reg.trap_state = std::numeric_limits<uint32_t>::max();
                        this->interrupt_sim = hostvar;
#ifndef WITH_TCC
                        throw(iss::simulation_stopped(hostvar));
#endif
                        break;
                    default:
                        break;
                    }
                } else if(tohost_lower)
                    to_host_wr_cnt++;
                return iss::Ok;
            } else {
                auto res = owner->write_mem(addr.val, length, data) ? iss::Ok : iss::Err;
                // clear MTIP on mtimecmp write
                if(addr.val == 0x2004000) {
                    reg_t val;
                    this->read_csr(iss::arch::mip, val);
                    if(val & (1ULL << 7))
                        this->write_csr(iss::arch::mip, val & ~(1ULL << 7));
                }
                return res;
            }
        }
    }

    iss::status read_csr(unsigned addr, reg_t& val) override {
        if((addr == iss::arch::time || addr == iss::arch::timeh)) {
            uint64_t time_val = owner->mtime_i.get_interface() ? owner->mtime_i.read() : 0;
            if(addr == iss::arch::time) {
                val = static_cast<reg_t>(time_val);
            } else if(addr == iss::arch::timeh) {
                if(sizeof(reg_t) != 4)
                    return iss::Err;
                val = static_cast<reg_t>(time_val >> 32);
            }
            return iss::Ok;
        } else {
            return PLAT::read_csr(addr, val);
        }
    }

    void wait_until(uint64_t flags) override {
        SCCDEBUG(owner->hier_name()) << "Sleeping until interrupt";
        while(this->reg.pending_trap == 0 && (this->csr[iss::arch::mip] & this->csr[iss::arch::mie]) == 0) {
            sc_core::wait(wfi_evt);
        }
        PLAT::wait_until(flags);
    }

    void local_irq(short id, bool value) override {
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

private:
    sysc::riscv_vp::core_complex_if* const owner{nullptr};
    sc_core::sc_event wfi_evt;
    uint64_t hostvar{std::numeric_limits<uint64_t>::max()};
    unsigned to_host_wr_cnt = 0;
    bool first{true};
};
} // namespace sysc
#endif /* _SYSC_SC_CORE_ADAPTER_H_ */
