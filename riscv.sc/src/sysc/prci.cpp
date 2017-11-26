////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, MINRES Technologies GmbH
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Contributors:
//       eyck@minres.com - initial implementation
//
//
////////////////////////////////////////////////////////////////////////////////

#include "sysc/SiFive/prci.h"

#include "scc/utilities.h"
#include "sysc/SiFive/gen/prci_regs.h"

namespace sysc {

prci::prci(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, tlm_target<>(clk)
, NAMED(clk_i)
, NAMED(rst_i)
, NAMEDD(prci_regs, regs) {
    regs->registerResources(*this);
    SC_METHOD(clock_cb);
    sensitive << clk_i;
    SC_METHOD(reset_cb);
    sensitive << rst_i;
    dont_initialize();
    SC_METHOD(hfrosc_en_cb);
    sensitive << hfrosc_en_evt;
    dont_initialize();

    regs->hfrosccfg.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data) -> bool {
        reg.put(data);
        if (this->regs->r_hfrosccfg & (1 << 30)) { // check rosc_en
            this->hfrosc_en_evt.notify(1, sc_core::SC_US);
        }
        return true;
    });
    regs->pllcfg.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data) -> bool {
        reg.put(data);
        auto &pllcfg = this->regs->r_pllcfg;
        if (pllcfg.pllbypass == 0 && pllcfg.pllq != 0) { // set pll_lock if pll is selected
            pllcfg.plllock = 1;
        }
        return true;
    });
}

void prci::clock_cb() {
	this->clk = clk_i.read();

}

prci::~prci() {}

void prci::reset_cb() {
    if (rst_i.read())
        regs->reset_start();
    else
        regs->reset_stop();
}

void prci::hfrosc_en_cb() {
    regs->r_hfrosccfg |= (1 << 31); // set rosc_rdy
}

} /* namespace sysc */
