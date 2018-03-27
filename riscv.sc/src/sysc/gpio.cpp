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
, NAMED(write_to_ws, false){
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
	if(write_to_ws.get_value()) {
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

