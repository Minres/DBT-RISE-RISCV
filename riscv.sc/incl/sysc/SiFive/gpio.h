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

#ifndef _GPIO_H_
#define _GPIO_H_

#include "scc/tlm_target.h"
#include "scc/ext_attribute.h"
#include <sysc/communication/sc_signal_rv_ports.h>

namespace sysc {

class gpio_regs;
class WsHandler;

class gpio : public sc_core::sc_module, public scc::tlm_target<> {
public:
    SC_HAS_PROCESS(gpio);
    sc_core::sc_in<sc_core::sc_time> clk_i;
    sc_core::sc_in<bool> rst_i;
    sc_core::sc_inout_rv<32> pins_io;

    gpio(sc_core::sc_module_name nm);
    virtual ~gpio() override; // need to keep it in source file because of fwd declaration of gpio_regs

    scc::ext_attribute<bool> write_to_ws;
protected:
    void clock_cb();
    void reset_cb();
    void update_pins();
    void pins_cb();
    void before_end_of_elaboration();
    sc_core::sc_time clk;
    std::unique_ptr<gpio_regs> regs;
    std::shared_ptr<sysc::WsHandler> handler;

private:
	void update_value_reg();
};

} /* namespace sysc */

#endif /* _GPIO_H_ */
