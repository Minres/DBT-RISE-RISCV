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

#include "sysc/SiFive/clint.h"
#include "sysc/SiFive/gen/clint_regs.h"
#include "sysc/utilities.h"

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
    regs->mtimecmp.set_write_cb([this](sc_register<uint64_t> &reg, uint64_t data) -> bool {
        if (!regs->in_reset()) {
            reg.put(data);
            this->update_mtime();
        }
        return true;
    });
    regs->mtime.set_read_cb([this](const sc_register<uint64_t> &reg, uint64_t &data) -> bool {
        this->update_mtime();
        data = reg.get();
        return true;
    });
    regs->mtime.set_write_cb([this](sc_register<uint64_t> &reg, uint64_t data) -> bool { return false; });
    regs->msip.set_write_cb([this](sc_register<uint32_t> &reg, uint32_t data) -> bool {
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
    clk = clk_i.read();
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
    auto diff = (sc_time_stamp() - last_updt) / clk;
    auto diffi = (int)diff;
    regs->r_mtime += (diffi + cnt_fraction) / lfclk_mutiplier;
    cnt_fraction = (cnt_fraction + diffi) % lfclk_mutiplier;
    mtime_evt.cancel();
    if (regs->r_mtimecmp > regs->r_mtime && clk > SC_ZERO_TIME) {
        sc_core::sc_time next_trigger = (clk * lfclk_mutiplier) * (regs->r_mtimecmp - regs->mtime) - cnt_fraction * clk;
        mtime_evt.notify(next_trigger);
    } else
        mtime_int_o.write(true);
    last_updt = sc_time_stamp();
}

} /* namespace sysc */
