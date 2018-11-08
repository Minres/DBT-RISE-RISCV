/*******************************************************************************
 * Copyright (C) 2017, 2018 MINRES Technologies GmbH
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

#ifndef _UART_H_
#define _UART_H_

#include "cci_configuration"
#include "scc/signal_initiator_mixin.h"
#include "scc/signal_target_mixin.h"
#include "scc/tlm_target.h"
#include <tlm/tlm_signal.h>

namespace sysc {
class tlm_signal_uart_extension;
class uart_regs;
class WsHandler;

class uart : public sc_core::sc_module, public scc::tlm_target<> {
public:
    SC_HAS_PROCESS(uart);// NOLINT
    sc_core::sc_in<sc_core::sc_time> clk_i;
    sc_core::sc_in<bool> rst_i;
    scc::tlm_signal_bool_out tx_o;
    scc::tlm_signal_bool_in rx_i;

    sc_core::sc_out<bool> irq_o;

    cci::cci_param<bool> bit_true_transfer;

    uart(sc_core::sc_module_name nm);
    virtual ~uart() override;

protected:
    void clock_cb();
    void reset_cb();
    void transmit_data();
    void receive_data(tlm::tlm_signal_gp<> &gp, sc_core::sc_time &delay);
    void update_irq();
    sc_core::sc_time clk{sc_core::SC_ZERO_TIME}, rx_last_start{sc_core::SC_ZERO_TIME};
    std::unique_ptr<uart_regs> regs;
    sc_core::sc_fifo<uint8_t> rx_fifo, tx_fifo;
};

} /* namespace sysc */

#endif /* _UART_H_ */
