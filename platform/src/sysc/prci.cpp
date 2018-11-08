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

#include "sysc/SiFive/prci.h"

#include "scc/utilities.h"
#include "sysc/SiFive/gen/prci_regs.h"

namespace sysc {
using namespace sc_core;

prci::prci(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, tlm_target<>(hfclk)
, NAMED(rst_i)
, NAMED(hfclk_o)
, NAMEDD(regs, prci_regs) {
    regs->registerResources(*this);
    SC_METHOD(reset_cb);
    sensitive << rst_i;
    dont_initialize();
    SC_METHOD(hfxosc_cb);
    sensitive << hfxosc_i;
    SC_METHOD(hfrosc_en_cb);
    sensitive << hfrosc_en_evt;
    dont_initialize();

    regs->hfxosccfg.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data, sc_core::sc_time d) -> bool {
        reg.put(data);
        if (this->regs->r_hfxosccfg.hfxoscen == 1) { // check rosc_en
            this->hfxosc_en_evt.notify(1, sc_core::SC_US);
        } else {
            this->hfxosc_en_evt.notify(SC_ZERO_TIME);
        }
        return true;
    });
    regs->hfrosccfg.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data, sc_core::sc_time d) -> bool {
        reg.put(data);
        if (this->regs->r_hfrosccfg.hfroscen == 1) { // check rosc_en
            this->hfrosc_en_evt.notify(1, sc_core::SC_US);
        } else {
            this->hfrosc_en_evt.notify(SC_ZERO_TIME);
        }
        return true;
    });
    regs->pllcfg.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data, sc_core::sc_time d) -> bool {
        reg.put(data);
        auto &pllcfg = this->regs->r_pllcfg;
        if (pllcfg.pllbypass == 0 && pllcfg.pllq != 0) { // set pll_lock if pll is selected
            pllcfg.plllock = 1;
        }
        update_hfclk();
        return true;
    });
    regs->plloutdiv.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data, sc_core::sc_time d) -> bool {
        reg.put(data);
        update_hfclk();
        return true;
    });
    hfxosc_clk = 62.5_ns;
}

prci::~prci() = default;

void prci::reset_cb() {
    if (rst_i.read())
        regs->reset_start();
    else {
        regs->reset_stop();
        this->hfxosc_en_evt.notify(1, sc_core::SC_US);
    }
}

void prci::hfxosc_cb() {
    this->regs->r_hfxosccfg.hfxoscrdy = 0;
    this->hfxosc_en_evt.notify(1, sc_core::SC_US);
}

void prci::hfxosc_en_cb() {
    update_hfclk();
    if (regs->r_hfxosccfg.hfxoscen == 1) // set rosc_rdy
        regs->r_hfxosccfg.hfxoscrdy = 1;
    else
        regs->r_hfxosccfg.hfxoscrdy = 0;
}

void prci::hfrosc_en_cb() {
    update_hfclk();
    auto &hfrosccfg = regs->r_hfrosccfg;
    if (regs->r_hfrosccfg.hfroscen == 1) { // set rosc_rdy
        regs->r_hfrosccfg.hfroscrdy = 1;
    } else {
        regs->r_hfrosccfg.hfroscrdy = 0;
    }
}

void prci::update_hfclk() {
    auto &hfrosccfg = regs->r_hfrosccfg;
    auto &pllcfg = regs->r_pllcfg;
    auto &plldiv = regs->r_plloutdiv;
    uint32_t trim = hfrosccfg.hfrosctrim;
    uint32_t div = hfrosccfg.hfroscdiv;
    hfrosc_clk = sc_core::sc_time(((div + 1) * 1.0) / (70000000 + 12000.0 * trim), sc_core::SC_SEC);
    auto pll_ref = pllcfg.pllrefsel == 1 ? hfxosc_clk : hfrosc_clk;
    auto r = pllcfg.pllr + 1;
    auto f = 2 * (pllcfg.pllf + 1);
    auto q = 1 << pllcfg.pllq;
    auto pll_out = pllcfg.pllbypass == 1 || pllcfg.plllock == 0 ? pll_ref : ((pll_ref * r) / f) * q;
    auto pll_res = plldiv & 0x100 ? pll_out : 2 * pll_out * ((plldiv & 0x3f) + 1);
    hfclk = pllcfg.pllsel ? pll_res : hfrosc_clk;
    hfclk_o.write(hfclk);
}

} /* namespace sysc */
