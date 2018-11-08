/*******************************************************************************
 * Copyright (C) 2017, 2018 MINRES Technologies GmbH
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
 *******************************************************************************/

#include <sysc/SiFive/plic.h>

#include <scc/report.h>
#include <scc/utilities.h>
#include <sysc/SiFive/gen/plic_regs.h>

namespace sysc {

plic::plic(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, tlm_target<>(clk)
, NAMED(clk_i)
, NAMED(rst_i)
, NAMED(global_interrupts_i, 256)
, NAMED(core_interrupt_o)
, NAMEDD(regs, plic_regs)

{
    regs->registerResources(*this);
    // register callbacks
    regs->claim_complete.set_write_cb([this](scc::sc_register<uint32_t> reg, uint32_t v, sc_core::sc_time d) -> bool {
        reg.put(v);
        reset_pending_int(v);
        // std::cout << "Value of register: 0x" << std::hex << reg << std::endl;
        // todo: reset related interrupt and find next high-prio interrupt
        return true;
    });

    // port callbacks
    SC_METHOD(global_int_port_cb);
    for (uint8_t i = 0; i < 255; i++) {
        sensitive << global_interrupts_i[i].pos();
    }
    dont_initialize();

    // register event callbacks
    SC_METHOD(clock_cb);
    sensitive << clk_i;
    SC_METHOD(reset_cb);
    sensitive << rst_i;
    dont_initialize();
}

plic::~plic() {}// NOLINT

void plic::clock_cb() { this->clk = clk_i.read(); }

void plic::reset_cb() {
    if (rst_i.read())
        regs->reset_start();
    else
        regs->reset_stop();
}

// Functional handling of interrupts:
// - global_int_port_cb()
//   - set pending register bits
//   - called by: incoming global_int
// - handle_pending_int()
//   - update claim register content
//   - generate core-interrupt pulse
//   - called by:
//     - incoming global_int
//     - complete-register write access
// - reset_pending_int(int-id)
//   - reset pending bit
//   - call next handle_pending_int()
//   - called by:
//     - complete-reg write register content

void plic::global_int_port_cb() {
    auto handle_pending = false;
    // set related pending bit if enable is set for incoming global_interrupt
    for (uint32_t i = 1; i < 256; i++) {
        auto reg_idx = i >> 5;
        auto bit_ofs = i & 0x1F;
        bool enable = regs->r_enabled[reg_idx] & (0x1 << bit_ofs); // read enable bit

        if (enable && global_interrupts_i[i].read() == 1) {
            regs->r_pending[reg_idx] = regs->r_pending[reg_idx] | (0x1 << bit_ofs);
            handle_pending = true;
            SCDEBUG(this->name()) << "pending interrupt identified: " << i;
        }
    }

    if (handle_pending) handle_pending_int();
}

void plic::handle_pending_int() {
    // identify high-prio pending interrupt and raise a core-interrupt
    auto claim_int = 0U;  // claim interrupt
    auto claim_prio = 0U; // related priority (highest prio interrupt wins the race)
    auto raise_int = false;
    auto thold = regs->r_threshold.threshold; // threshold value

    for (size_t i = 1; i < 255; i++) {
        auto reg_idx = i >> 5;
        auto bit_ofs = i & 0x1F;
        bool pending = (regs->r_pending[reg_idx] & (0x1 << bit_ofs)) ? true : false;
        auto prio = regs->r_priority[i].priority; // read priority value

        if (pending && thold < prio) {
            // below condition ensures implicitly that lowest id is selected in case of multiple identical
            // priority-interrupts
            if (prio > claim_prio) {
                claim_prio = prio;
                claim_int = i;
                raise_int = true;
                SCDEBUG(this->name()) << "pending interrupt activated: " << i;
            }
        }
    }

    if (raise_int) {
        regs->r_claim_complete = claim_int;
        core_interrupt_o.write(true);
        // todo: evluate clock period
    } else {
        regs->r_claim_complete = 0;
        SCDEBUG(this->name()) << "no further pending interrupt.";
    }
}

void plic::reset_pending_int(uint32_t irq) {
    // todo: evaluate enable register (see spec)
    // todo: make sure that pending is set, otherwise don't reset irq ... read spec.
    SCTRACE(this->name()) << "reset pending interrupt: " << irq;
    // reset related pending bit
    auto reg_idx = irq >> 5;
    auto bit_ofs = irq & 0x1F;
    regs->r_pending[reg_idx] &= ~(0x1 << bit_ofs);
    core_interrupt_o.write(false);

    // evaluate next pending interrupt
    handle_pending_int();
}

} /* namespace sysc */
