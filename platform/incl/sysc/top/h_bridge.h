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

#ifndef RISCV_SC_INCL_SYSC_TOP_H_BRIDGE_H_
#define RISCV_SC_INCL_SYSC_TOP_H_BRIDGE_H_

#include "cci_configuration"
#include <sysc/kernel/sc_module.h>

namespace sysc {

class h_bridge : public sc_core::sc_module {
public:
    SC_HAS_PROCESS(h_bridge);// NOLINT

    sc_core::sc_in<sc_dt::sc_logic> ha_i, la_i;
    sc_core::sc_in<sc_dt::sc_logic> hb_i, lb_i;
    sc_core::sc_in<sc_dt::sc_logic> hc_i, lc_i;

    sc_core::sc_out<double> va_o, vb_o, vc_o;

    cci::cci_param<double> vcc;

    h_bridge(const sc_core::sc_module_name &nm);

    virtual ~h_bridge();

private:
    void ain_cb();
    void bin_cb();
    void cin_cb();
    void write_output(sc_dt::sc_logic h_i, sc_dt::sc_logic l_i, sc_core::sc_out<double> &v_o);
};

} /* namespace sysc */

#endif /* RISCV_SC_INCL_SYSC_TOP_H_BRIDGE_H_ */
