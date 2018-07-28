/*
 * dcmotor.cpp
 *
 *  Created on: 25.07.2018
 *      Author: eyck
 */

#include "sysc/top/dcmotor.h"
#include "scc/utilities.h"
#include <future>

namespace sysc {

using namespace sc_core;

auto get_config = []() -> BLDC::Config {
    BLDC::Config config{};
    config.Ke=1./4000. ,//0.01; // V/rad/s, = 1/Kv
    config.R=0.5; // Ohm
    config.inertia = 0.0005;
    config.NbPoles = 2;
    config.damping = 0.00001;
    return config;
};

dc_motor::dc_motor(const sc_module_name& nm )
: sc_module(nm)
, bldc_model(get_config())
, bldc_state(bldc_model.getState())
{
    bldc_model.setLoad(0.0001);
    SC_THREAD(thread);
}

dc_motor::~dc_motor() {
}

void dc_motor::trace(sc_trace_file* trf) {
    auto ia=bldc_state.ia;       TRACE_VAR(trf, ia);
    auto ib=bldc_state.ib;       TRACE_VAR(trf, ib);
    auto ic=bldc_state.ic;       TRACE_VAR(trf, ic);
    auto theta=bldc_state.theta; TRACE_VAR(trf, theta);
    auto omega=bldc_state.omega; TRACE_VAR(trf, omega);
}

void dc_motor::thread(void) {
    wait(SC_ZERO_TIME);
    std::array<double, 3> vin{0., 0., 0.};
    const sc_time step(1, SC_US);
    auto eval_model = [this](std::array<double, 3> vin, const sc_time step)->std::tuple<double, double, double> {
        bldc_model.set_input(vin);
        bldc_model.run(step.to_seconds());
        return bldc_model.get_voltages();
    };
    while(true){
        vin[0]=va_i.read();
        vin[1]=vb_i.read();
        vin[2]=vc_i.read();
//        auto sim_res=std::async(std::launch::async, eval_model, vin, step);
        wait(step);
//        auto vout=sim_res.get();
        auto vout = eval_model(vin, step);
        va_o=std::get<0>(vout);
        vb_o=std::get<1>(vout);
        vc_o=std::get<2>(vout);
    }
}

} /* namespace sysc */
