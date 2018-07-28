/*
 * h_bridge.h
 *
 *  Created on: 25.07.2018
 *      Author: eyck
 */

#ifndef RISCV_SC_INCL_SYSC_TOP_H_BRIDGE_H_
#define RISCV_SC_INCL_SYSC_TOP_H_BRIDGE_H_

#include "cci_configuration"
#include <sysc/kernel/sc_module.h>

namespace sysc {

class h_bridge: public sc_core::sc_module {
public:
    SC_HAS_PROCESS(h_bridge);

    sc_core::sc_in<sc_dt::sc_logic> ha_i, la_i;
    sc_core::sc_in<sc_dt::sc_logic> hb_i, lb_i;
    sc_core::sc_in<sc_dt::sc_logic> hc_i, lc_i;

    sc_core::sc_out<double> va_o, vb_o, vc_o;

    cci::cci_param<double> vcc;

    h_bridge(const sc_core::sc_module_name& nm);

    virtual ~h_bridge();
private:
    void ain_cb();
    void bin_cb();
    void cin_cb();
};

} /* namespace sysc */

#endif /* RISCV_SC_INCL_SYSC_TOP_H_BRIDGE_H_ */
