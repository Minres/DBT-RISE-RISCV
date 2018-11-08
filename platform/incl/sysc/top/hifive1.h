/*******************************************************************************
 * Copyright (C) 2018 MINRES Technologies GmbH
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

#ifndef _SIFIVE_HIFIVE1_H_
#define _SIFIVE_HIFIVE1_H_

#include <sysc/top/terminal.h>
#include <sysc/top/mcp_adc.h>
#include "tlm/tlm_signal_sockets.h"
#include <boost/preprocessor.hpp>
#include <systemc>
#include <sysc/SiFive/fe310.h>

namespace sysc {

struct hifive1 : public sc_core::sc_module {

    SC_HAS_PROCESS(hifive1);

    sc_core::sc_in<bool> erst_n;
    sc_core::sc_in<double> vref_i;
#define PORT_DECL(z, n, _)    sc_core::sc_in<double> adc_ch##n##_i;
    BOOST_PP_REPEAT(8, PORT_DECL, _);
#undef PORT_DECL
    sc_core::sc_out<sc_dt::sc_logic> ha_o, la_o, hb_o, lb_o,hc_o, lc_o;

    hifive1(sc_core::sc_module_name nm);

protected:
    sc_core::sc_vector<tlm::tlm_signal<sc_dt::sc_logic>> s_gpio;
    sc_core::sc_vector<scc::tlm_signal_logic_in> h_bridge;
    fe310 i_fe310;
    terminal i_terminal;
    mcp_3208 i_adc;
};

}

#endif /* _SYSC_SIFIVE_HIFIVE1_H_ */
