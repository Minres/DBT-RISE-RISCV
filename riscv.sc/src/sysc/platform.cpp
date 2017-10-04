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
, NAMED(i_core_complex)
, NAMED(i_router, 10, 1)
, NAMED(i_uart0)
, NAMED(i_uart1)
, NAMED(i_spi)
, NAMED(i_gpio)
, NAMED(i_plic)
, NAMED(i_aon)
, NAMED(i_prci)
, NAMED(i_clint)
, NAMED(i_mem_qspi)
, NAMED(i_mem_ram)
, NAMED(s_clk)
, NAMED(s_rst)
, NAMED(s_global_int, 256)
, NAMED(s_core_int) {
    i_core_complex.initiator(i_router.target[0]);
    size_t i = 0;
    for (const auto &e : e300_plat_map) {
        i_router.initiator.at(i)(e.target->socket);
        i_router.add_target_range(i, e.start, e.size);
        i++;
    }
    i_router.initiator.at(i)(i_mem_qspi.target);
    i_router.add_target_range(i, 0x20000000, 512_MB);
    i_router.initiator.at(++i)(i_mem_ram.target);
    i_router.add_target_range(i, 0x80000000, 128_kB);

    i_uart0.clk_i(s_clk);
    i_uart1.clk_i(s_clk);
    i_spi.clk_i(s_clk);
    i_gpio.clk_i(s_clk);
    i_plic.clk_i(s_clk);
    i_aon.clk_i(s_clk);
    i_prci.clk_i(s_clk);
    i_clint.clk_i(s_clk);
    i_core_complex.clk_i(s_clk);

    i_uart0.rst_i(s_rst);
    i_uart1.rst_i(s_rst);
    i_spi.rst_i(s_rst);
    i_gpio.rst_i(s_rst);
    i_plic.rst_i(s_rst);
    i_aon.rst_i(s_rst);
    i_prci.rst_i(s_rst);
    i_clint.rst_i(s_rst);
    i_core_complex.rst_i(s_rst);

    i_clint.mtime_int_o(s_mtime_int);
    i_clint.msip_int_o(s_msie_int);

    i_plic.global_interrupts_i(s_global_int);
    i_plic.core_interrupt_o(s_core_int);
    SC_THREAD(gen_reset);
}

void platform::gen_reset() {
    s_clk = 10_ns;
    s_rst = true;
    wait(10_ns);
    s_rst = false;
}

} /* namespace sysc */
