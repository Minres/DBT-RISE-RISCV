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
, NAMED(i_platform)
, NAMED(i_terminal)
{
    // connect platform
    i_platform.erst_n(s_rst_n);

    for(auto i=0U; i<s_gpio.size(); ++i){
      s_gpio[i].in(i_platform.pins_o[i]);
      i_platform.pins_i[i](s_gpio[i].out);
    }
    // connect other units
    s_gpio[17].out(i_terminal.rx_i);
    i_terminal.tx_o(s_gpio[16].in);
    SC_THREAD(gen_por);
}

system::~system() {
}

void sysc::system::gen_por() {
    // single shot
    s_rst_n = false;
    wait(10_ns);
    s_rst_n = true;
}
