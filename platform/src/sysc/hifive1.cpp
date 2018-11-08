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

#include <sysc/top/hifive1.h>

using namespace sc_core;
using namespace sc_dt;
using namespace sysc;

hifive1::hifive1(sc_module_name nm)
: sc_module(nm)
, NAMED(erst_n)
, NAMED(vref_i)
#define PORT_NAMING(z, n, _) , NAMED(adc_ch##n##_i)
BOOST_PP_REPEAT(8, PORT_NAMING, _)
#undef PORT_NAMING
, NAMED(ha_o)
, NAMED(la_o)
, NAMED(hb_o)
, NAMED(lb_o)
, NAMED(hc_o)
, NAMED(lc_o)
, NAMED(s_gpio, 32)
, NAMED(h_bridge, 6)
, NAMED(i_fe310)
, NAMED(i_terminal)
, NAMED(i_adc)
{
    i_fe310.erst_n(erst_n);
    for (auto i = 0U; i < s_gpio.size(); ++i) {
        s_gpio[i].in(i_fe310.pins_o[i]);
        i_fe310.pins_i[i](s_gpio[i].out);
    }
    // connect other units
    // terminal
    i_terminal.tx_o(s_gpio[16].in);
    s_gpio[17].out(i_terminal.rx_i);
    // adc digital io
    s_gpio[2].out(i_adc.cs_i);
    s_gpio[3].out(i_adc.mosi_i);
    i_adc.miso_o(s_gpio[4].in);
    s_gpio[5].out(i_adc.sck_i);
    // adc analog inputs
    i_adc.vref_i(vref_i);
    i_adc.ch_i[0](adc_ch0_i);
    i_adc.ch_i[1](adc_ch1_i);
    i_adc.ch_i[2](adc_ch2_i);
    i_adc.ch_i[3](adc_ch3_i);
    i_adc.ch_i[4](adc_ch4_i);
    i_adc.ch_i[5](adc_ch5_i);
    i_adc.ch_i[6](adc_ch6_i);
    i_adc.ch_i[7](adc_ch7_i);
    // H-Bridge signal proxies
    s_gpio[0].out(h_bridge[0]);
    s_gpio[1].out(h_bridge[1]);
    s_gpio[10].out(h_bridge[2]);
    s_gpio[11].out(h_bridge[3]);
    s_gpio[20].out(h_bridge[4]);
    s_gpio[19].out(h_bridge[5]);
    // proxy callbacks
    h_bridge[0].register_nb_transport([this](tlm::tlm_signal_gp<sc_logic> &gp, tlm::tlm_phase &phase, sc_time &delay) -> tlm::tlm_sync_enum {
        ha_o.write(gp.get_value());
    });
    h_bridge[1].register_nb_transport([this](tlm::tlm_signal_gp<sc_logic> &gp, tlm::tlm_phase &phase, sc_time &delay) -> tlm::tlm_sync_enum {
        la_o.write(gp.get_value());
    });
    h_bridge[2].register_nb_transport([this](tlm::tlm_signal_gp<sc_logic> &gp, tlm::tlm_phase &phase, sc_time &delay) -> tlm::tlm_sync_enum {
        hb_o.write(gp.get_value());
    });
    h_bridge[3].register_nb_transport([this](tlm::tlm_signal_gp<sc_logic> &gp, tlm::tlm_phase &phase, sc_time &delay) -> tlm::tlm_sync_enum {
        lb_o.write(gp.get_value());
    });
    h_bridge[4].register_nb_transport([this](tlm::tlm_signal_gp<sc_logic> &gp, tlm::tlm_phase &phase, sc_time &delay) -> tlm::tlm_sync_enum {
        hc_o.write(gp.get_value());
    });
    h_bridge[5].register_nb_transport([this](tlm::tlm_signal_gp<sc_logic> &gp, tlm::tlm_phase &phase, sc_time &delay) -> tlm::tlm_sync_enum {
        lc_o.write(gp.get_value());
    });


}

