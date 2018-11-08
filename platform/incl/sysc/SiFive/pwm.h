/*******************************************************************************
 * Copyright (C) 2017,2018 MINRES Technologies GmbH
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

#ifndef _PWM_H_
#define _PWM_H_

#include "cci_configuration"
#include "scc/signal_initiator_mixin.h"
#include "scc/signal_target_mixin.h"
#include "scc/tlm_target.h"
#include <tlm/tlm_signal.h>

namespace sysc {

class pwm_regs;

class pwm : public sc_core::sc_module, public scc::tlm_target<> {
public:
    SC_HAS_PROCESS(pwm);// NOLINT
    sc_core::sc_in<sc_core::sc_time> clk_i;
    sc_core::sc_in<bool> rst_i;

    sc_core::sc_vector<scc::tlm_signal_bool_opt_out> cmpgpio_o;
    sc_core::sc_vector<sc_core::sc_out<bool>> cmpip_o;

    pwm(sc_core::sc_module_name nm);
    virtual ~pwm() override; // need to keep it in source file because of fwd declaration of gpio_regs

protected:
    sc_core::sc_time clk, last_clk;
    void clock_cb();
    void reset_cb();
    inline double get_pulses(sc_core::sc_time d) {
        auto t = sc_core::sc_time_stamp() + d;
        return last_clk > sc_core::SC_ZERO_TIME ? (t - last_cnt_update) / last_clk : 0.;
    }
    void update_counter();
    void write_cmpgpio(size_t, bool);
    std::unique_ptr<pwm_regs> regs;
    uint64_t current_cnt;
    sc_core::sc_time last_cnt_update;
    double clk_remainder = 0.0;
    bool last_enable = false, reset_cnt = false;
    sc_core::sc_event update_counter_evt;
    std::array<bool, 4> pwmcmp_ip;
};

} /* namespace sysc */

#endif /* _GPIO_H_ */
