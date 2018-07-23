////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, MINRES Technologies GmbH
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Contributors:
//       eyck@minres.com - initial implementation
//
//
////////////////////////////////////////////////////////////////////////////////

#ifndef _SPI_H_
#define _SPI_H_

#include "scc/tlm_target.h"
#include "scc/signal_target_mixin.h"
#include "scc/signal_initiator_mixin.h"
#include <tlm/tlm_signal.h>
#include "cci_configuration"
#include <sysc/utils/sc_vector.h>

namespace sysc {

class spi_regs;

class spi : public sc_core::sc_module, public scc::tlm_target<> {
public:
    SC_HAS_PROCESS(spi);
    sc_core::sc_in<sc_core::sc_time> clk_i;
    sc_core::sc_in<bool>             rst_i;
    scc::tlm_signal_bool_opt_out         sck_o;
    scc::tlm_signal_bool_opt_out         mosi_o;
    scc::tlm_signal_bool_opt_in          miso_i;
    sc_core::sc_vector<scc::tlm_signal_bool_opt_out> scs_o;

    sc_core::sc_out<bool> irq_o;

    cci::cci_param<bool> bit_true_transfer;

    spi(sc_core::sc_module_name nm);
    virtual ~spi() override;

protected:
    void clock_cb();
    void reset_cb();
    void transmit_data();
    void receive_data(tlm::tlm_signal_gp<>& gp, sc_core::sc_time& delay);
    void update_irq();
    sc_core::sc_time clk;
    std::unique_ptr<spi_regs> regs;
    sc_core::sc_fifo<uint8_t> rx_fifo, tx_fifo;
};

} /* namespace sysc */

#endif /* _SPI_H_ */
