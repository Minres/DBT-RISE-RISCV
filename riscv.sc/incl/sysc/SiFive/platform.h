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

#ifndef _PLATFORM_H_
#define _PLATFORM_H_

#include "aon.h"
#include "clint.h"
#include "gpio.h"
#include "plic.h"
#include "prci.h"
#include "spi.h"
#include "uart.h"
#include "core_complex.h"

#include "scc/memory.h"
#include "scc/router.h"
#include "scc/utilities.h"
#include "tlm/tlm_signal_sockets.h"
#include <sysc/kernel/sc_module.h>
#include <array>


namespace sysc {

class platform : public sc_core::sc_module {
public:
    SC_HAS_PROCESS(platform);

    sc_core::sc_vector<tlm::tlm_signal_initiator_socket<sc_dt::sc_logic>> pins_o;
    sc_core::sc_vector<tlm::tlm_signal_target_socket<sc_dt::sc_logic>>    pins_i;

    sc_core::sc_in<bool> erst_n;

    platform(sc_core::sc_module_name nm);

private:
    SiFive::core_complex i_core_complex;
    scc::router<> i_router;
    uart i_uart0, i_uart1;
    spi i_qspi0, i_qspi1, i_qspi2;
    gpio i_gpio0;
    plic i_plic;
    aon i_aon;
    prci i_prci;
    clint i_clint;

    scc::memory<512_MB, 32> i_mem_qspi;
    scc::memory<128_kB, 32> i_mem_ram;
    sc_core::sc_signal<sc_core::sc_time> s_tlclk;
    sc_core::sc_signal<sc_core::sc_time> s_lfclk;
    sc_core::sc_signal<bool> s_rst, s_mtime_int, s_msie_int;
    sc_core::sc_vector<sc_core::sc_signal<bool, SC_MANY_WRITERS>> s_global_int, s_local_int;
    sc_core::sc_signal<bool> s_core_int;
    sc_core::sc_vector<sc_core::sc_signal<bool>> s_dummy;
    sc_core::sc_vector<scc::tlm_signal_bool_opt_in>  s_dummy_sck_i;
    sc_core::sc_vector<scc::tlm_signal_bool_opt_out> s_dummy_sck_o;


protected:
    void gen_reset();

#include "gen/e300_plat_t.h"
};

} /* namespace sysc */

#endif /* _PLATFORM_H_ */
