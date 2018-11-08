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

#ifndef __SYSC_GENERAL_SYSTEM_H_
#define __SYSC_GENERAL_SYSTEM_H_

#include "dcmotor.h"
#include "h_bridge.h"
#include <memory>
#include <systemc>
#include "hifive1.h"

namespace sysc {

class system : sc_core::sc_module {
public:
    SC_HAS_PROCESS(system);// NOLINT

    system(sc_core::sc_module_name nm);
    virtual ~system();

private:
    sc_core::sc_signal<sc_dt::sc_logic> s_ha, s_la, s_hb, s_lb, s_hc, s_lc;
    sc_core::sc_signal<bool> s_rst_n;
    sc_core::sc_signal<double> s_vref, s_va, s_vb, s_vc, s_vasens, s_vbsens, s_vcsens, s_vcentersens;
    sc_core::sc_vector<sc_core::sc_signal<double>> s_ana;
    sysc::hifive1 i_hifive1;
    sysc::h_bridge i_h_bridge;
    sysc::dc_motor i_motor;
    void gen_por();
};
}
#endif /* __SYSC_GENERAL_SYSTEM_H_ */
