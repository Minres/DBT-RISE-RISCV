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

#ifndef _SYSC_TOP_TERMINAL_H_
#define _SYSC_TOP_TERMINAL_H_

#include "cci_configuration"
#include "scc/signal_initiator_mixin.h"
#include "scc/signal_target_mixin.h"
#include "tlm/tlm_signal.h"
#include <memory>
#include <sysc/kernel/sc_module.h>

namespace sysc {
class WsHandler;

class terminal : public sc_core::sc_module {
public:
    scc::tlm_signal_logic_out tx_o;
    scc::tlm_signal_logic_in rx_i;

    terminal();

    terminal(const sc_core::sc_module_name &nm);

    virtual ~terminal();

    cci::cci_param<bool> write_to_ws;

protected:
    void before_end_of_elaboration();
    void receive(tlm::tlm_signal_gp<sc_dt::sc_logic> &gp, sc_core::sc_time &delay);

    std::vector<uint8_t> queue;
    std::shared_ptr<sysc::WsHandler> handler;
    sc_core::sc_time last_tx_start = sc_core::SC_ZERO_TIME;
};
}

#endif /* _SYSC_TOP_TERMINAL_H_ */
