////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 eyck@minres.com
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
// License for the specific language governing permissions and limitations under
// the License.
////////////////////////////////////////////////////////////////////////////////

#include "sysc/SiFive/prci.h"
#include "sysc/SiFive/gen/prci_regs.h"
#include "sysc/utilities.h"

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

    regs->hfrosccfg.set_write_cb([this](sysc::sc_register<uint32_t> &reg, uint32_t data) -> bool {
        reg.put(data);
        if (this->regs->r_hfrosccfg & (1 << 30)) { // check rosc_en
            this->hfrosc_en_evt.notify(1, sc_core::SC_US);
        }
        return true;
    });
    regs->pllcfg.set_write_cb([this](sysc::sc_register<uint32_t> &reg, uint32_t data) -> bool {
        reg.put(data);
        auto &pllcfg = this->regs->r_pllcfg;
        if (pllcfg.pllbypass == 0 && pllcfg.pllq != 0) { // set pll_lock if pll is selected
            pllcfg.plllock = 1;
        }
        return true;
    });
}

void prci::clock_cb() {}

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
