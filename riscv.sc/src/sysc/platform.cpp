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

#include <sysc/SiFive/platform.h>

namespace sysc {

platform::platform(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, NAMED(i_core_complex)
, NAMED(i_router, 12, 1)
, NAMED(i_uart0)
, NAMED(i_uart1)
, NAMED(i_qspi0)
, NAMED(i_qspi1)
, NAMED(i_qspi2)
, NAMED(i_gpio0)
, NAMED(i_plic)
, NAMED(i_aon)
, NAMED(i_prci)
, NAMED(i_clint)
, NAMED(i_mem_qspi)
, NAMED(i_mem_ram)
, NAMED(s_clk)
, NAMED(s_rst)
, NAMED(s_global_int, 256)
, NAMED(s_local_int, 16)
, NAMED(s_core_int)
, NAMED(s_gpio_pins) {
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
    i_qspi0.clk_i(s_clk);
    i_qspi1.clk_i(s_clk);
    i_qspi2.clk_i(s_clk);
    i_gpio0.clk_i(s_clk);
    i_plic.clk_i(s_clk);
    i_aon.clk_i(s_clk);
    i_prci.clk_i(s_clk);
    i_clint.clk_i(s_clk);
    i_core_complex.clk_i(s_clk);

    i_uart0.rst_i(s_rst);
    i_uart1.rst_i(s_rst);
    i_qspi0.rst_i(s_rst);
    i_qspi1.rst_i(s_rst);
    i_qspi2.rst_i(s_rst);
    i_gpio0.rst_i(s_rst);
    i_plic.rst_i(s_rst);
    i_aon.rst_i(s_rst);
    i_prci.rst_i(s_rst);
    i_clint.rst_i(s_rst);
    i_core_complex.rst_i(s_rst);

    i_clint.mtime_int_o(s_mtime_int);
    i_clint.msip_int_o(s_msie_int);

    i_plic.global_interrupts_i(s_global_int);
    i_plic.core_interrupt_o(s_core_int);

    i_core_complex.sw_irq_i(s_msie_int);
    i_core_complex.timer_irq_i(s_mtime_int);
    i_core_complex.global_irq_i(s_core_int);
    i_core_complex.local_irq_i(s_local_int);

    i_gpio0.pins_io(s_gpio_pins);

    SC_THREAD(gen_reset);
}

void platform::gen_reset() {
    s_clk = 10_ns;
    s_rst = true;
    wait(10_ns);
    s_rst = false;
}

} /* namespace sysc */
