/*
 * system.cpp
 *
 *  Created on: 11.07.2018
 *      Author: eyck
 */

#include "sysc/General/system.h"

using namespace sysc;

system::system(sc_core::sc_module_name nm)
: sc_module(nm)
, NAMED(s_gpio, 32)
, NAMED(s_rst_n)
, NAMED(s_vref)
, NAMED(s_ana, 8)
, NAMED(i_platform)
, NAMED(i_terminal)
, NAMED(i_adc)
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
    i_adc.ch_i(s_ana);
    i_adc.vref_i(s_vref);

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
