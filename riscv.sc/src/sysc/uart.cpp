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

#include "sysc/SiFive/uart.h"

#include "sysc/sc_comm_singleton.h"
#include "scc/report.h"
#include "scc/utilities.h"
#include "sysc/SiFive/gen/uart_regs.h"

using namespace std;


namespace sysc {
namespace {

using namespace seasocks;


}
uart::uart(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, tlm_target<>(clk)
, NAMED(clk_i)
, NAMED(rst_i)
, NAMEDD(uart_regs, regs)
, NAMED(write_to_ws, false, this) {
    regs->registerResources(*this);
    SC_METHOD(clock_cb);
    sensitive << clk_i;
    SC_METHOD(reset_cb);
    sensitive << rst_i;
    dont_initialize();
    regs->txdata.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data) -> bool {
        if (!this->regs->in_reset()) {
            reg.put(data);
            this->transmit_data();
        }
        return true;
    });
}

uart::~uart() {}

void uart::before_end_of_elaboration() {
	if(write_to_ws.value) {
	    LOG(TRACE)<<"Adding WS handler for "<<(std::string{"/ws/"}+name());
	    handler=std::make_shared<WsHandler>();
	    sc_comm_singleton::inst().registerWebSocketHandler((std::string{"/ws/"}+name()).c_str(), handler);
	}
}

void uart::clock_cb() {
	this->clk = clk_i.read();
}

void uart::reset_cb() {
    if (rst_i.read())
        regs->reset_start();
    else
        regs->reset_stop();
}

void uart::transmit_data() {
    if(regs->r_txdata.data != '\r') queue.push_back(regs->r_txdata.data);
    if (queue.size() >> 0 && (regs->r_txdata.data == '\n' || regs->r_txdata.data == 0)) {
    	std::string msg(queue.begin(), queue.end()-1);
        LOG(INFO) << this->name() << " transmit: '" << msg << "'";
        sc_core::sc_time now = sc_core::sc_time_stamp();
		if(handler)
			sc_comm_singleton::inst().execute([this, msg, now](){
				std::stringstream os;
				os << "{\"time\":\"" << now << "\",\"message\":\""<<msg<<"\"}";
				this->handler->send(os.str());
			});
        queue.clear();
    }
}

} /* namespace sysc */
