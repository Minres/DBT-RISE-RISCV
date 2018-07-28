/*
 * dcmotor.h
 *
 *  Created on: 25.07.2018
 *      Author: eyck
 */

#ifndef _SYSC_TOP_DCMOTOR_H_
#define _SYSC_TOP_DCMOTOR_H_

#include "BLDC.h"
#include "scc/traceable.h"
#include <systemc>

namespace sysc {

class dc_motor: public sc_core::sc_module, public scc::traceable {
public:
    SC_HAS_PROCESS(dc_motor);

    sc_core::sc_in<double> va_i, vb_i, vc_i;
    sc_core::sc_out<double> va_o, vb_o, vc_o;

    dc_motor(const sc_core::sc_module_name& nm );

    virtual ~dc_motor();

    void trace(sc_core::sc_trace_file *trf) override;

private:
    void thread(void);
    BLDC bldc_model;
    const BLDC::State& bldc_state;
};

} /* namespace sysc */

#endif /* RISCV_SC_INCL_SYSC_TOP_DCMOTOR_H_ */
