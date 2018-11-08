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

#include "sysc/top/system.h"

using namespace sysc;
using namespace sc_core;

system::system(sc_module_name nm)
: sc_module(nm)
, NAMED(s_ha)
, NAMED(s_la)
, NAMED(s_hb)
, NAMED(s_lb)
, NAMED(s_hc)
, NAMED(s_lc)
, NAMED(s_rst_n)
, NAMED(s_vref)
, NAMED(s_va)
, NAMED(s_vb)
, NAMED(s_vc)
, NAMED(s_vasens)
, NAMED(s_vbsens)
, NAMED(s_vcsens)
, NAMED(s_vcentersens)
, NAMED(s_ana, 4)
, NAMED(i_hifive1)
, NAMED(i_h_bridge)
, NAMED(i_motor) {
    // connect platform
    i_hifive1.erst_n(s_rst_n);
    // HiFive1 digital out
    i_hifive1.ha_o(s_ha);
    i_hifive1.la_o(s_la);
    i_hifive1.hb_o(s_hb);
    i_hifive1.lb_o(s_lb);
    i_hifive1.hc_o(s_hc);
    i_hifive1.lc_o(s_lc);
    // HiFive1 analog in
    i_hifive1.vref_i(s_vref);
    i_hifive1.adc_ch0_i(s_vasens);
    i_hifive1.adc_ch1_i(s_vbsens);
    i_hifive1.adc_ch2_i(s_vcsens);
    i_hifive1.adc_ch3_i(s_vcentersens);
    i_hifive1.adc_ch4_i(s_ana[0]);
    i_hifive1.adc_ch5_i(s_ana[1]);
    i_hifive1.adc_ch6_i(s_ana[2]);
    i_hifive1.adc_ch7_i(s_ana[3]);
    // H-bridge digital in
    i_h_bridge.ha_i(s_ha);
    i_h_bridge.la_i(s_la);
    i_h_bridge.hb_i(s_hb);
    i_h_bridge.lb_i(s_lb);
    i_h_bridge.hc_i(s_hc);
    i_h_bridge.lc_i(s_lc);
    // H-bridge analog out
    i_h_bridge.va_o(s_va);
    i_h_bridge.vb_o(s_vb);
    i_h_bridge.vc_o(s_vc);
    // motor analog in
    i_motor.va_i(s_va);
    i_motor.vb_i(s_vb);
    i_motor.vc_i(s_vc);
    // motor analog out
    i_motor.va_o(s_vasens);
    i_motor.vb_o(s_vbsens);
    i_motor.vc_o(s_vcsens);
    i_motor.vcenter_o(s_vcentersens);

    SC_THREAD(gen_por);
}

system::~system() = default;

void sysc::system::gen_por() {
    // single shot
    s_rst_n = false;
    wait(1_us);
    s_rst_n = true;
    s_vref = 4.8;
    double val = 0.1;
    for (auto &sig : s_ana) {
        sig = val;
        val += 0.12;
    }
}
