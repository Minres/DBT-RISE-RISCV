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

#include "sysc/SiFive/clint.h"

#include "scc/utilities.h"
#include "scc/report.h"
#include "sysc/SiFive/gen/clint_regs.h"

namespace sysc {

const int lfclk_mutiplier = 1 << 12;

clint::clint(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, tlm_target<>(clk)
, NAMED(clk_i)
, NAMED(rst_i)
, NAMED(mtime_int_o)
, NAMED(msip_int_o)
, NAMEDD(clint_regs, regs)
, cnt_fraction(0) {
    regs->registerResources(*this);
    SC_METHOD(clock_cb);
    sensitive << clk_i;
    SC_METHOD(reset_cb);
    sensitive << rst_i;
    dont_initialize();
    regs->mtimecmp.set_write_cb([this](scc::sc_register<uint64_t> &reg, uint64_t data) -> bool {
        if (!regs->in_reset()) {
            reg.put(data);
            this->update_mtime();
        }
        return true;
    });
    regs->mtime.set_read_cb([this](const scc::sc_register<uint64_t> &reg, uint64_t &data) -> bool {
        this->update_mtime();
        data = reg.get();
        return true;
    });
    regs->mtime.set_write_cb([this](scc::sc_register<uint64_t> &reg, uint64_t data) -> bool { return false; });
    regs->msip.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data) -> bool {
        reg.put(data);
        msip_int_o.write(regs->r_msip.msip);
        return true;
    });
    SC_METHOD(update_mtime);
    sensitive << mtime_evt;
    dont_initialize();
}

void clint::clock_cb() {
    update_mtime();
    this->clk = clk_i.read();
    update_mtime();
}

clint::~clint() {}

void clint::reset_cb() {
    if (rst_i.read()) {
        regs->reset_start();
        msip_int_o.write(false);
        mtime_int_o.write(false);
        cnt_fraction = 0;
    } else
        regs->reset_stop();
}

void clint::update_mtime() {
    auto diff = (sc_core::sc_time_stamp() - last_updt) / clk;
    auto diffi = (int)diff;
    regs->r_mtime += (diffi + cnt_fraction) / lfclk_mutiplier;
    cnt_fraction = (cnt_fraction + diffi) % lfclk_mutiplier;
    mtime_evt.cancel();
    if (regs->r_mtimecmp > 0)
    	if(regs->r_mtimecmp > regs->r_mtime && clk > sc_core::SC_ZERO_TIME) {
    		sc_core::sc_time next_trigger = (clk * lfclk_mutiplier) * (regs->r_mtimecmp - regs->mtime) - cnt_fraction * clk;
    		LOG(DEBUG)<<"Timer fires at "<< sc_time_stamp()+next_trigger;
    		mtime_evt.notify(next_trigger);
    		mtime_int_o.write(false);
    	} else
    		mtime_int_o.write(true);
    last_updt = sc_core::sc_time_stamp();
}

} /* namespace sysc */
