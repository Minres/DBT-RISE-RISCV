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

#ifndef _SYSC_TOP_DCMOTOR_H_
#define _SYSC_TOP_DCMOTOR_H_

#include "BLDC.h"
#include "cci_configuration"
#include "scc/traceable.h"
#include <systemc>

namespace sysc {

class dc_motor : public sc_core::sc_module, public scc::traceable {
public:
    SC_HAS_PROCESS(dc_motor);// NOLINT

    sc_core::sc_in<double> va_i, vb_i, vc_i;
    sc_core::sc_out<double> va_o, vb_o, vc_o, vcenter_o;

    dc_motor(const sc_core::sc_module_name &nm);

    virtual ~dc_motor();

    void trace(sc_core::sc_trace_file *trf) const override;

    cci::cci_param<sc_core::sc_time> max_integ_step;
    cci::cci_param<double> load;

private:
    void thread(void);
    BLDC bldc_model;
    const BLDC::State &bldc_state;
    std::array<double, 7> vout;
};

} /* namespace sysc */

#endif /* RISCV_SC_INCL_SYSC_TOP_DCMOTOR_H_ */
