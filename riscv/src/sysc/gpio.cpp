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

#include "sysc/SiFive/gpio.h"
#include "sysc/SiFive/gen/gpio_regs.h"
#include "sysc/utilities.h"

namespace sysc {

gpio::gpio(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, tlm_target<>(clk)
, NAMED(clk_i)
, NAMED(rst_i)
, NAMEDD(gpio_regs, regs)
{
    regs->registerResources(*this);
    SC_METHOD(clock_cb);
    sensitive<<clk_i;
    SC_METHOD(reset_cb);
    sensitive<<rst_i;
}

gpio::~gpio() {
}

void gpio::clock_cb() {
}

void gpio::reset_cb() {
    if(rst_i.read())
        regs->reset_start();
    else
        regs->reset_stop();
}

} /* namespace sysc */
