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
/*
 * simplesystem.cpp
 *
 *  Created on: 17.09.2017
 *      Author: eyck@minres.com
 */

#include <sysc/SiFive/platform.h>

namespace sysc {

platform::platform(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, NAMED(i_master)
, NAMED(i_router, 4, 1)
, NAMED(i_uart)
, NAMED(i_spi)
, NAMED(i_gpio)
, NAMED(i_plic)
, NAMED(s_clk)
, NAMED(s_rst) {
    i_master.initiator(i_router.target[0]);
    size_t i = 0;
    for (const auto &e : e300_plat_map) {
        i_router.initiator.at(i)(e.target->socket);
        i_router.add_target_range(i, e.start, e.size);
        i++;
    }
    i_uart.clk_i(s_clk);
    i_spi.clk_i(s_clk);
    i_gpio.clk_i(s_clk);
    i_plic.clk_i(s_clk);
    s_clk.write(10_ns);

    i_uart.rst_i(s_rst);
    i_spi.rst_i(s_rst);
    i_gpio.rst_i(s_rst);
    i_plic.rst_i(s_rst);
    i_master.rst_i(s_rst);

    SC_THREAD(gen_reset);
}

void platform::gen_reset() {
    s_rst = true;
    wait(10_ns);
    s_rst = false;
}

} /* namespace sysc */
