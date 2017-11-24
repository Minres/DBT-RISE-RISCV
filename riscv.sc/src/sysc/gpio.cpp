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

#include "sysc/SiFive/gpio.h"

#include "sysc/sc_comm_singleton.h"
#include "scc/report.h"
#include "scc/utilities.h"
#include "sysc/SiFive/gen/gpio_regs.h"

namespace sysc {

gpio::gpio(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, tlm_target<>(clk)
, NAMED(clk_i)
, NAMED(rst_i)
, NAMED(pins_io)
, NAMEDD(gpio_regs, regs)
, NAMED(write_to_ws, true, this){
    regs->registerResources(*this);
    SC_METHOD(clock_cb);
    sensitive << clk_i;
    SC_METHOD(reset_cb);
    sensitive << rst_i;
    dont_initialize();
    SC_METHOD(pins_cb);
    sensitive << pins_io;

    regs->port.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data) -> bool {
        if (!this->regs->in_reset()) {
            reg.put(data);
            // read r_ports and update pins_io
            update_pins();
        }
        return true;
    });

    regs->value.set_read_cb([this](const scc::sc_register<uint32_t> &reg, uint32_t& data) -> bool {
        if (!this->regs->in_reset()) {
        	// read pins_io and update r_value
        	update_value_reg();
            data=reg.get();
        }
        return true;
    });
}

gpio::~gpio() {}

void gpio::before_end_of_elaboration() {
	if(write_to_ws.value) {
		LOG(TRACE)<<"Adding WS handler for "<<(std::string{"/ws/"}+name());
		handler=std::make_shared<WsHandler>();
		sc_comm_singleton::inst().registerWebSocketHandler((std::string{"/ws/"}+name()).c_str(), handler);
	}
}

void gpio::reset_cb() {
    if (rst_i.read())
        regs->reset_start();
    else
        regs->reset_stop();
}

void gpio::clock_cb() {
	this->clk = clk_i.read();
}

void gpio::pins_cb(){
	auto inval=pins_io.read();
	std::string msg(inval.to_string());
    sc_core::sc_time now = sc_core::sc_time_stamp();
    if(handler) sc_comm_singleton::inst().execute([this, msg, now](){
		std::stringstream os;
		os << "{\"time\":\"" << now << "\",\"data\":\""<<msg<<"\"}";
		this->handler->send(os.str());
	});
}

void gpio::update_value_reg() {
	// read pins_io and update r_value reg
	auto inval = pins_io.read();
	uint32_t res = 0;
	for (size_t i = 0, msk = 1; i < 32; ++i, msk = msk << 1) {
		bool bit_set = false;
		if ((regs->r_input_en & msk) != 0) {
			if (inval.get_bit(1) == sc_dt::Log_1)
				bit_set = true;
			else if (inval.get_bit(1) == sc_dt::Log_Z
					&& (regs->r_pue & msk) != 0)
				bit_set = true;
		}
		if (bit_set) res |= msk;
	}
	regs->r_value = res;
}

void gpio::update_pins() {
	sc_core::sc_inout_rv<32>::data_type out_val;
	for(size_t i=0, msk = 1; i<32; ++i, msk=msk<<1)
		out_val.set_bit(i, regs->r_output_en&msk?regs->r_port&msk?sc_dt::Log_1:sc_dt::Log_0:sc_dt::Log_Z);
	pins_io.write(out_val);
}

} /* namespace sysc */

