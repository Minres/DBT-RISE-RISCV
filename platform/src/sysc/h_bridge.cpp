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

#include "sysc/top/h_bridge.h"
#include "scc/utilities.h"
#include <cmath>

namespace sysc {
using namespace sc_core;
using namespace sc_dt;

h_bridge::h_bridge(const sc_module_name &nm)
: sc_module(nm)
, NAMED(ha_i)
, NAMED(la_i)
, NAMED(hb_i)
, NAMED(lb_i)
, NAMED(hc_i)
, NAMED(lc_i)
, NAMED(va_o)
, NAMED(vb_o)
, NAMED(vc_o)
, NAMED(vcc, 48.0) {
    SC_METHOD(ain_cb);
    sensitive << ha_i << la_i;
    SC_METHOD(bin_cb);
    sensitive << hb_i << lb_i;
    SC_METHOD(cin_cb);
    sensitive << hc_i << lc_i;
}

h_bridge::~h_bridge() = default;

void h_bridge::ain_cb() { write_output(ha_i.read(), la_i.read(), va_o); }

void h_bridge::bin_cb() { write_output(hb_i.read(), lb_i.read(), vb_o); }

void h_bridge::cin_cb() { write_output(hc_i.read(), lc_i.read(), vc_o); }

void h_bridge::write_output(sc_logic h_i, sc_logic l_i, sc_out<double> &v_o) {
    if (h_i == sc_dt::Log_1 && l_i == sc_dt::Log_0)
        v_o.write(vcc);
    else if (h_i == sc_dt::Log_0 && l_i == sc_dt::Log_1)
        v_o.write(0.0);
    else
        v_o.write(nan(""));
    /*
        auto v = v_o.read();
        if(h_i==Log_1 && l_i==Log_0){
            if(isnan(v)){
                v_o.write(0.75*vcc);
                next_trigger(2, SC_US);
            } else
                v_o.write(vcc);
        } else if(h_i==Log_0 && l_i==Log_1){
            if(isnan(v)){
                v_o.write(0.25*vcc);
                next_trigger(2, SC_US);
            } else
                v_o.write(0.0);
        } else {
            if(v_o.read()>0.8*vcc) {
                v_o.write(0.75*vcc);
                next_trigger(2, SC_US);
            } else if(v_o.read()>0.8*vcc) {
                v_o.write(0.25*vcc);
                next_trigger(2, SC_US);
            } else
                v_o.write(nan(""));
        }
    */
}

} /* namespace sysc */
