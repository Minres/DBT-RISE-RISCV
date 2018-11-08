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

#ifndef _PLIC_H_
#define _PLIC_H_

#include <scc/register.h>
#include <scc/tlm_target.h>

namespace sysc {

class plic_regs;

class plic : public sc_core::sc_module, public scc::tlm_target<> {
public:
    SC_HAS_PROCESS(plic);// NOLINT
    sc_core::sc_in<sc_core::sc_time> clk_i;
    sc_core::sc_in<bool> rst_i;
    sc_core::sc_vector<sc_core::sc_in<bool>> global_interrupts_i;
    sc_core::sc_out<bool> core_interrupt_o;
    sc_core::sc_event raise_int_ev;
    sc_core::sc_event clear_int_ev;
    plic(sc_core::sc_module_name nm);
    ~plic() override;

protected:
    void clock_cb();
    void reset_cb();

    void global_int_port_cb();
    void handle_pending_int();
    void reset_pending_int(uint32_t irq);

    void raise_core_interrupt();
    void clear_core_interrupt();
    sc_core::sc_time clk;
    std::unique_ptr<plic_regs> regs;
    std::function<bool(scc::sc_register<uint32_t>, uint32_t)> m_claim_complete_write_cb;
};

} /* namespace sysc */

#endif /* _PLIC_H_ */
