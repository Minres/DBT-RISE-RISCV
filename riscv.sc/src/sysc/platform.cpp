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
, NAMED(pins_o, 32)
, NAMED(pins_i, 32)
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
, NAMED(s_tlclk)
, NAMED(s_rst)
, NAMED(s_global_int, 256)
, NAMED(s_local_int, 16)
, NAMED(s_core_int)
, NAMED(s_dummy, 16)
, NAMED(s_dummy_sck_i, 16)
, NAMED(s_dummy_sck_o, 16)
{
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

    i_uart0.clk_i(s_tlclk);
    i_uart1.clk_i(s_tlclk);
    i_qspi0.clk_i(s_tlclk);
    i_qspi1.clk_i(s_tlclk);
    i_qspi2.clk_i(s_tlclk);
    i_gpio0.clk_i(s_tlclk);
    i_plic.clk_i(s_tlclk);
    i_aon.clk_i(s_tlclk);
    i_prci.clk_i(s_tlclk);
    i_clint.tlclk_i(s_tlclk);
    i_clint.lfclk_i(s_lfclk);
    i_core_complex.clk_i(s_tlclk);

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

    pins_i(i_gpio0.pins_i);
    i_gpio0.pins_o(pins_o);

    i_gpio0.iof0_i[17](i_uart0.tx_o);
    i_uart0.rx_i(i_gpio0.iof0_o[16]);
    i_uart0.irq_o(s_global_int[3]);

    s_dummy_sck_i[0](i_uart1.tx_o);
    i_uart1.rx_i(s_dummy_sck_o[0]);
    i_uart1.irq_o(s_dummy[0]);

    SC_THREAD(gen_reset);

    for(auto& sock:s_dummy_sck_i) sock.error_if_no_callback=false;
}

void platform::gen_reset() {
    s_tlclk = 10_ns;
    s_lfclk = 30517_ns;
    s_rst = true;
    wait(10_ns);
    s_rst = false;
}

} /* namespace sysc */
