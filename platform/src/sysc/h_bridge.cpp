/*
 * h_bridge.cpp
 *
 *  Created on: 25.07.2018
 *      Author: eyck
 */

#include "sysc/top/h_bridge.h"
#include "scc/utilities.h"

namespace sysc {
using namespace sc_core;

h_bridge::h_bridge(const sc_module_name& nm)
:sc_module(nm)
, NAMED(ha_i)
, NAMED(la_i)
, NAMED(hb_i)
, NAMED(lb_i)
, NAMED(hc_i)
, NAMED(lc_i)
, NAMED(va_o)
, NAMED(vb_o)
, NAMED(vc_o)
, NAMED(vcc, 48.0)
{
    SC_METHOD(ain_cb);
    sensitive<<ha_i<<la_i;
    SC_METHOD(bin_cb);
    sensitive<<hb_i<<lb_i;
    SC_METHOD(cin_cb);
    sensitive<<hc_i<<lc_i;
}

h_bridge::~h_bridge() {
}

void h_bridge::ain_cb() {
}

void h_bridge::bin_cb() {
}

void h_bridge::cin_cb() {
}

} /* namespace sysc */
