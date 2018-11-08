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

#include "sysc/top/dcmotor.h"
#include "scc/utilities.h"
#include <future>

namespace sysc {

using namespace sc_core;

auto get_config = []() -> BLDC::Config {
    BLDC::Config config{};
    config.Ke = 1. / 4000., // 0.01; // V/rad/s, = 1/Kv
        config.R = 0.5;     // Ohm
    config.Ke = 0.01;
    config.inertia = 0.0005;
    config.NbPoles = 2;
    config.damping = 0.00001;
    return config;
};

dc_motor::dc_motor(const sc_module_name &nm)
: sc_module(nm)
, NAMED(va_i)
, NAMED(vb_i)
, NAMED(vc_i)
, NAMED(va_o)
, NAMED(vb_o)
, NAMED(vc_o)
, NAMED(vcenter_o)
, NAMED(max_integ_step, sc_time(10, SC_US))
, NAMED(load, 0.1)
, bldc_model(get_config())
, bldc_state(bldc_model.getState()) {
    bldc_model.setLoad(0.0001);
    SC_THREAD(thread);
}

dc_motor::~dc_motor() = default;

void dc_motor::trace(sc_trace_file *trf) const {
    auto &ia = bldc_state.ia;
    sc_core::sc_trace(trf, bldc_state.ia, std::string(this->name()) +"." "ia");
    sc_core::sc_trace(trf, bldc_state.ib, std::string(this->name()) +"." "ib");
    sc_core::sc_trace(trf, bldc_state.ic, std::string(this->name()) +"." "ic");
    sc_core::sc_trace(trf, bldc_state.theta, std::string(this->name()) +"." "theta");
    sc_core::sc_trace(trf, bldc_state.omega, std::string(this->name()) +"." "omega");
    sc_core::sc_trace(trf, vout[0], std::string(this->name()) + "." "va");
    sc_core::sc_trace(trf, vout[1], std::string(this->name()) + "." "vb");
    sc_core::sc_trace(trf, vout[2], std::string(this->name()) + "." "vc");
    sc_core::sc_trace(trf, vout[3], std::string(this->name()) + "." "vcenter");
    sc_core::sc_trace(trf, vout[4], std::string(this->name()) + "." "ea");
    sc_core::sc_trace(trf, vout[5], std::string(this->name()) + "." "eb");
    sc_core::sc_trace(trf, vout[6], std::string(this->name()) + "." "ec");
}

void dc_motor::thread() {
    const auto divider = 10.0;
    wait(SC_ZERO_TIME);
    std::array<double, 3> vin{0., 0., 0.};
    auto eval_model = [this](std::array<double, 3> vin, const sc_time step) -> std::array<double, 7> {
        bldc_model.set_input(vin);
        bldc_model.run(step.to_seconds());
        return bldc_model.get_voltages();
    };
    while (true) {
        vin[0] = va_i.read();
        vin[1] = vb_i.read();
        vin[2] = vc_i.read();
        //        auto sim_res=std::async(std::launch::async, eval_model, vin, step);
        auto start = sc_time_stamp();
        wait(max_integ_step, va_i.value_changed_event() | vb_i.value_changed_event() | vc_i.value_changed_event());
        auto diff = sc_time_stamp() - start;
        if (diff.to_seconds() >= bldc_model.dt) {
            vout = eval_model(vin, diff); // sim_res.get();
            va_o = vout[0] / divider;
            vb_o = vout[1] / divider;
            vc_o = vout[2] / divider;
            vcenter_o = vout[3] / divider;
        }
    }
}

} /* namespace sysc */
