////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 eyck@minres.com
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
// License for the specific language governing permissions and limitations under
// the License.
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
, NAMEDD(uart_regs, regs) {
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
    LOG(TRACE)<<"Adding WS handler for "<<(std::string{"/ws/"}+name());
    handler=std::make_shared<WsHandler>();
    sc_comm_singleton::inst().registerWebSocketHandler((std::string{"/ws/"}+name()).c_str(), handler);
}

uart::~uart() {}

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
		sc_comm_singleton::inst().execute([this, msg, now](){
			std::stringstream os;
			os << "{\"time\":\"" << now << "\",\"message\":\""<<msg<<"\"}";
			this->handler->send(os.str());
		});
        queue.clear();
    }
}

} /* namespace sysc */
