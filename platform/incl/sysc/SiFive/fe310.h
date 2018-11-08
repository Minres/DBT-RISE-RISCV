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

#ifndef _PLATFORM_H_
#define _PLATFORM_H_

#include "aon.h"
#include "clint.h"
#include "gpio.h"
#include "plic.h"
#include "prci.h"
#include "pwm.h"
#include "spi.h"
#include "sysc/core_complex.h"
#include "uart.h"

#include "cci_configuration"
#include "scc/memory.h"
#include "scc/router.h"
#include "scc/utilities.h"
#include "tlm/tlm_signal_sockets.h"
#include <array>
#include <memory>
#include <sysc/kernel/sc_module.h>

namespace sysc {

class fe310 : public sc_core::sc_module {
public:
    SC_HAS_PROCESS(fe310);// NOLINT

    sc_core::sc_vector<tlm::tlm_signal_initiator_socket<sc_dt::sc_logic>> pins_o;
    sc_core::sc_vector<tlm::tlm_signal_target_socket<sc_dt::sc_logic>> pins_i;

    sc_core::sc_in<bool> erst_n;

    fe310(sc_core::sc_module_name nm);

    cci::cci_param<bool> use_rtl;

private:
    std::unique_ptr<SiFive::core_complex> i_core_complex;
    std::unique_ptr<scc::router<>> i_router;
    std::unique_ptr<uart> i_uart0, i_uart1;
    std::unique_ptr<spi> i_qspi0, i_qspi1, i_qspi2;
    std::unique_ptr<pwm> i_pwm0, i_pwm1, i_pwm2;
    std::unique_ptr<gpio> i_gpio0;
    std::unique_ptr<plic> i_plic;
    std::unique_ptr<aon> i_aon;
    std::unique_ptr<prci> i_prci;
    std::unique_ptr<clint> i_clint;

    using mem_qspi_t = scc::memory<512_MB, 32>;
    std::unique_ptr<mem_qspi_t> i_mem_qspi;
    using mem_ram_t = scc::memory<128_kB, 32>;
    std::unique_ptr<mem_ram_t> i_mem_ram;

    sc_core::sc_signal<sc_core::sc_time, sc_core::SC_MANY_WRITERS> s_tlclk;
    sc_core::sc_signal<sc_core::sc_time, sc_core::SC_MANY_WRITERS> s_lfclk;
    
    sc_core::sc_signal<bool, sc_core::SC_MANY_WRITERS> s_rst, s_mtime_int, s_msie_int;
    
    sc_core::sc_vector<sc_core::sc_signal<bool, sc_core::SC_MANY_WRITERS>> s_global_int, s_local_int;
    sc_core::sc_signal<bool, sc_core::SC_MANY_WRITERS> s_core_int;
    
    sc_core::sc_vector<scc::tlm_signal_bool_opt_in> s_dummy_sck_i;
    sc_core::sc_vector<scc::tlm_signal_bool_opt_out> s_dummy_sck_o;

protected:
    void gen_reset();

#include "gen/e300_plat_t.h"
};

} /* namespace sysc */

#endif /* _PLATFORM_H_ */
