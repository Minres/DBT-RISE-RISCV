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

#include "sysc/SiFive/fe310.h"

namespace sysc {
using namespace sc_core;
using namespace SiFive;

#ifdef HAS_VERILATOR
inline std::unique_ptr<spi> create_spi(sc_module_name nm, bool use_rtl) {
    return use_rtl ? spi::create<spi_impl::rtl>("i_qspi1") : spi::create<spi_impl::beh>("i_qspi1");
}
#else
inline std::unique_ptr<spi> create_spi(sc_module_name nm, bool use_rtl) {
    return spi::create<spi_impl::beh>("i_qspi1");
}
#endif

fe310::fe310(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, NAMED(pins_o, 32)
, NAMED(pins_i, 32)
, NAMED(erst_n)
, NAMED(use_rtl, false)
, NAMEDD(i_core_complex, core_complex)
, NAMEDD(i_router, scc::router<>, e300_plat_t_map.size() + 2, 1)
, NAMEDD(i_uart0, uart)
, NAMEDD(i_uart1, uart)
, NAMEDC(i_qspi0, spi, spi_impl::beh)
, i_qspi1(create_spi("i_qspi1", use_rtl))
, NAMEDC(i_qspi2, spi, spi_impl::beh)
, NAMEDD(i_pwm0, pwm)
, NAMEDD(i_pwm1, pwm)
, NAMEDD(i_pwm2, pwm)
, NAMEDD(i_gpio0, gpio)
, NAMEDD(i_plic, plic)
, NAMEDD(i_aon, aon)
, NAMEDD(i_prci, prci)
, NAMEDD(i_clint, clint)
, NAMEDD(i_mem_qspi, mem_qspi_t)
, NAMEDD(i_mem_ram, mem_ram_t)
, NAMED(s_tlclk)
, NAMED(s_lfclk)
, NAMED(s_rst)
, NAMED(s_mtime_int)
, NAMED(s_msie_int)
, NAMED(s_global_int, 256)
, NAMED(s_local_int, 16)
, NAMED(s_core_int)
, NAMED(s_dummy_sck_i, 16)
, NAMED(s_dummy_sck_o, 16) {
    i_core_complex->initiator(i_router->target[0]);
    size_t i = 0;
    for (const auto &e : e300_plat_t_map) {
        i_router->initiator.at(i)(e.target);
        i_router->set_target_range(i, e.start, e.size);
        i++;
    }
    i_router->initiator.at(i)(i_mem_qspi->target);
    i_router->set_target_range(i, 0x20000000, 512_MB);
    i_router->initiator.at(++i)(i_mem_ram->target);
    i_router->set_target_range(i, 0x80000000, 128_kB);

    i_uart0->clk_i(s_tlclk);
    i_uart1->clk_i(s_tlclk);
    i_qspi0->clk_i(s_tlclk);
    i_qspi1->clk_i(s_tlclk);
    i_qspi2->clk_i(s_tlclk);
    i_pwm0->clk_i(s_tlclk);
    i_pwm1->clk_i(s_tlclk);
    i_pwm2->clk_i(s_tlclk);
    i_gpio0->clk_i(s_tlclk);
    i_plic->clk_i(s_tlclk);
    i_aon->clk_i(s_tlclk);
    i_aon->lfclkc_o(s_lfclk);
    i_prci->hfclk_o(s_tlclk); // clock driver
    i_clint->tlclk_i(s_tlclk);
    i_clint->lfclk_i(s_lfclk);
    i_core_complex->clk_i(s_tlclk);

    i_uart0->rst_i(s_rst);
    i_uart1->rst_i(s_rst);
    i_qspi0->rst_i(s_rst);
    i_qspi1->rst_i(s_rst);
    i_qspi2->rst_i(s_rst);
    i_pwm0->rst_i(s_rst);
    i_pwm1->rst_i(s_rst);
    i_pwm2->rst_i(s_rst);
    i_gpio0->rst_i(s_rst);
    i_plic->rst_i(s_rst);
    i_aon->rst_o(s_rst);
    i_prci->rst_i(s_rst);
    i_clint->rst_i(s_rst);
    i_core_complex->rst_i(s_rst);

    i_aon->erst_n_i(erst_n);

    i_clint->mtime_int_o(s_mtime_int);
    i_clint->msip_int_o(s_msie_int);

    i_plic->global_interrupts_i(s_global_int);
    i_plic->core_interrupt_o(s_core_int);

    i_core_complex->sw_irq_i(s_msie_int);
    i_core_complex->timer_irq_i(s_mtime_int);
    i_core_complex->global_irq_i(s_core_int);
    i_core_complex->local_irq_i(s_local_int);

    pins_i(i_gpio0->pins_i);
    i_gpio0->pins_o(pins_o);

    i_gpio0->iof0_i[17](i_uart0->tx_o);
    i_uart0->rx_i(i_gpio0->iof0_o[16]);
    i_uart0->irq_o(s_global_int[3]);

    i_gpio0->iof0_i[5](i_qspi1->sck_o);
    i_gpio0->iof0_i[3](i_qspi1->mosi_o);
    i_qspi1->miso_i(i_gpio0->iof0_o[4]);
    i_gpio0->iof0_i[2](i_qspi1->scs_o[0]);
    i_gpio0->iof0_i[9](i_qspi1->scs_o[2]);
    i_gpio0->iof0_i[10](i_qspi1->scs_o[3]);

    i_qspi0->irq_o(s_global_int[5]);
    i_qspi1->irq_o(s_global_int[6]);
    i_qspi2->irq_o(s_global_int[7]);

    s_dummy_sck_i[0](i_uart1->tx_o);
    i_uart1->rx_i(s_dummy_sck_o[0]);
    i_uart1->irq_o(s_global_int[4]);

    i_gpio0->iof1_i[0](i_pwm0->cmpgpio_o[0]);
    i_gpio0->iof1_i[1](i_pwm0->cmpgpio_o[1]);
    i_gpio0->iof1_i[2](i_pwm0->cmpgpio_o[2]);
    i_gpio0->iof1_i[3](i_pwm0->cmpgpio_o[3]);

    i_gpio0->iof1_i[10](i_pwm2->cmpgpio_o[0]);
    i_gpio0->iof1_i[11](i_pwm2->cmpgpio_o[1]);
    i_gpio0->iof1_i[12](i_pwm2->cmpgpio_o[2]);
    i_gpio0->iof1_i[13](i_pwm2->cmpgpio_o[3]);

    i_gpio0->iof1_i[19](i_pwm1->cmpgpio_o[0]);
    i_gpio0->iof1_i[20](i_pwm1->cmpgpio_o[1]);
    i_gpio0->iof1_i[21](i_pwm1->cmpgpio_o[2]);
    i_gpio0->iof1_i[22](i_pwm1->cmpgpio_o[3]);

    i_pwm0->cmpip_o[0](s_global_int[40]);
    i_pwm0->cmpip_o[1](s_global_int[41]);
    i_pwm0->cmpip_o[2](s_global_int[42]);
    i_pwm0->cmpip_o[3](s_global_int[43]);

    i_pwm1->cmpip_o[0](s_global_int[44]);
    i_pwm1->cmpip_o[1](s_global_int[45]);
    i_pwm1->cmpip_o[2](s_global_int[46]);
    i_pwm1->cmpip_o[3](s_global_int[47]);

    i_pwm2->cmpip_o[0](s_global_int[48]);
    i_pwm2->cmpip_o[1](s_global_int[49]);
    i_pwm2->cmpip_o[2](s_global_int[50]);
    i_pwm2->cmpip_o[3](s_global_int[51]);

    for (auto &sock : s_dummy_sck_i) sock.error_if_no_callback = false;
}

} /* namespace sysc */
