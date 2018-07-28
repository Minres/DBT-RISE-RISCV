/*
 * system.cpp
 *
 *  Created on: 11.07.2018
 *      Author: eyck
 */

#include "sysc/top/system.h"

using namespace sysc;

system::system(sc_core::sc_module_name nm)
: sc_module(nm)
, NAMED(s_gpio, 32)
, NAMED(s_rst_n)
, NAMED(s_vref)
, NAMED(s_va)
, NAMED(s_vb)
, NAMED(s_vc)
, NAMED(s_ana, 5)
, NAMED(i_platform)
, NAMED(i_terminal)
, NAMED(i_adc)
, NAMED(i_h_bridge)
, NAMED(i_motor)
{
    // connect platform
    i_platform.erst_n(s_rst_n);

    for(auto i=0U; i<s_gpio.size(); ++i){
      s_gpio[i].in(i_platform.pins_o[i]);
      i_platform.pins_i[i](s_gpio[i].out);
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
    i_adc.vref_i(s_vref);
    i_adc.ch_i[0](s_vasens);
    i_adc.ch_i[1](s_vbsens);
    i_adc.ch_i[2](s_vcsens);
    i_adc.ch_i[3](s_ana[0]);
    i_adc.ch_i[4](s_ana[1]);
    i_adc.ch_i[5](s_ana[2]);
    i_adc.ch_i[6](s_ana[3]);
    i_adc.ch_i[7](s_ana[4]);

    i_h_bridge.ha_i(s_gpio[0]);
    i_h_bridge.la_i(s_gpio[1]);
    i_h_bridge.hb_i(s_gpio[10]);
    i_h_bridge.lb_i(s_gpio[11]);
    i_h_bridge.hc_i(s_gpio[19]);
    i_h_bridge.lc_i(s_gpio[20]);
    i_h_bridge.va_o(s_va);
    i_h_bridge.vb_o(s_vb);
    i_h_bridge.vc_o(s_vc);

    i_motor.va_i(s_va);
    i_motor.vb_i(s_vb);
    i_motor.vc_i(s_vc);
    i_motor.va_o(s_vasens);
    i_motor.vb_o(s_vbsens);
    i_motor.vc_o(s_vcsens);

    SC_THREAD(gen_por);
}

system::~system() {
}

void sysc::system::gen_por() {
    // single shot
    s_rst_n = false;
    wait(10_ns);
    s_rst_n = true;
    s_vref=1.024;
    double val=0.1;
    for(auto& sig:s_ana){
        sig=val;
        val+=0.12;
    }
}
