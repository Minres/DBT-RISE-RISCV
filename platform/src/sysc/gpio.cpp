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

#include "sysc/SiFive/gpio.h"
#include "scc/report.h"
#include "scc/utilities.h"
#include "sysc/SiFive/gen/gpio_regs.h"
#include "sysc/sc_comm_singleton.h"
#include <limits>

namespace sysc {
using namespace sc_core;
using namespace sc_dt;

gpio::gpio(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, tlm_target<>(clk)
, NAMED(clk_i)
, NAMED(rst_i)
, NAMED(pins_o, 32)
, NAMED(pins_i, 32)
, NAMED(iof0_o, 32)
, NAMED(iof1_o, 32)
, NAMED(iof0_i, 32)
, NAMED(iof1_i, 32)
, NAMEDD(regs, gpio_regs)
, NAMED(write_to_ws, false) {
    regs->registerResources(*this);
    SC_METHOD(clock_cb);
    sensitive << clk_i;
    SC_METHOD(reset_cb);
    sensitive << rst_i;
    dont_initialize();
    auto pins_i_cb = [this](unsigned int tag, tlm::tlm_signal_gp<sc_logic> &gp, tlm::tlm_phase &phase,
                            sc_core::sc_time &delay) -> tlm::tlm_sync_enum {
        this->pin_input(tag, gp, delay);
        return tlm::TLM_COMPLETED;
    };
    auto i = 0U;
    for (auto &s : pins_i) {
        s.register_nb_transport(pins_i_cb, i);
        ++i;
    }
    auto iof0_i_cb = [this](unsigned int tag, tlm::tlm_signal_gp<bool> &gp, tlm::tlm_phase &phase,
                            sc_core::sc_time &delay) -> tlm::tlm_sync_enum {
        last_iof0[tag] = gp.get_value();
        this->iof_input(tag, 0, gp, delay);
        return tlm::TLM_COMPLETED;
    };
    i = 0;
    for (auto &s : iof0_i) {
        s.register_nb_transport(iof0_i_cb, i);
        ++i;
    }
    auto iof1_i_cb = [this](unsigned int tag, tlm::tlm_signal_gp<bool> &gp, tlm::tlm_phase &phase,
                            sc_core::sc_time &delay) -> tlm::tlm_sync_enum {
        last_iof1[tag] = gp.get_value();
        this->iof_input(tag, 1, gp, delay);
        return tlm::TLM_COMPLETED;
    };
    i = 0;
    for (auto &s : iof1_i) {
        s.register_nb_transport(iof1_i_cb, i);
        ++i;
    }
    auto update_pins_cb = [this](scc::sc_register<uint32_t> &reg, uint32_t data, sc_core::sc_time d) -> bool {
        if (!this->regs->in_reset()) {
            auto changed_bits = (reg.get() ^ data);
            reg.put(data);
            update_pins(changed_bits);
        }
        return true;
    };
    regs->port.set_write_cb(update_pins_cb);
    regs->output_en.set_write_cb(update_pins_cb);
    regs->out_xor.set_write_cb(update_pins_cb);
    regs->iof_en.set_write_cb(update_pins_cb);
    regs->iof_sel.set_write_cb(update_pins_cb);
}

gpio::~gpio() = default;

void gpio::before_end_of_elaboration() {
    if (write_to_ws.get_value()) {
        SCTRACE() << "Adding WS handler for " << (std::string{"/ws/"} + name());
        handler = std::make_shared<WsHandler>();
        sc_comm_singleton::inst().registerWebSocketHandler((std::string{"/ws/"} + name()).c_str(), handler);
    }
}

void gpio::reset_cb() {
    if (rst_i.read()) {
        regs->reset_start();
    } else {
        regs->reset_stop();
    }
    update_pins(std::numeric_limits<uint32_t>::max());
}

void gpio::clock_cb() { this->clk = clk_i.read(); }

tlm::tlm_phase gpio::write_output(tlm::tlm_signal_gp<sc_dt::sc_logic> &gp, size_t i, sc_dt::sc_logic val) {
    sc_core::sc_time delay{SC_ZERO_TIME};
    tlm::tlm_phase phase{tlm::BEGIN_REQ};
    gp.set_command(tlm::TLM_WRITE_COMMAND);
    gp.set_response_status(tlm::TLM_OK_RESPONSE);
    gp.set_value(val);
    pins_o.at(i)->nb_transport_fw(gp, phase, delay);
    return phase;
}

void gpio::update_pins(uint32_t changed_bits) {
    sc_core::sc_inout_rv<32>::data_type out_val;
    tlm::tlm_signal_gp<sc_dt::sc_logic> gp;
    sc_logic val;
    for (size_t i = 0, mask = 1; i < 32; ++i, mask <<= 1) {
        if (changed_bits & mask) {
            if ((regs->r_iof_en & mask != 0) && (iof0_i[i].size() == 0 || iof1_i[i].size() == 0)) {
                if ((regs->r_iof_sel & mask) == 0 && iof0_i[i].size() > 0) {
                    val = last_iof0[i] ? sc_dt::Log_1 : sc_dt::Log_0;
                } else if ((regs->r_iof_sel & mask) == 1 && iof1_i[i].size() > 0)
                    val = last_iof1[i] ? sc_dt::Log_1 : sc_dt::Log_0;
            } else {
                if (regs->r_output_en & mask)
                    val = regs->r_port & mask ? sc_dt::Log_1 : sc_dt::Log_0;
                else
                    val = sc_dt::Log_Z;
                if (regs->r_out_xor & mask) val = ~val;
            }
            tlm::tlm_phase phase = write_output(gp, i, val);
        }
    }
}

void gpio::pin_input(unsigned int tag, tlm::tlm_signal_gp<sc_logic> &gp, sc_core::sc_time &delay) {
    if (delay > SC_ZERO_TIME) {
        wait(delay);
        delay = SC_ZERO_TIME;
    }
    auto mask = 1u << tag;
    switch (gp.get_value().value()) {
    case sc_dt::Log_1:
        if (regs->r_output_en & mask == 0) regs->r_value |= mask;
        forward_pin_input(tag, gp);
        break;
    case sc_dt::Log_0:
        if (regs->r_output_en & mask == 0) regs->r_value &= ~mask;
        forward_pin_input(tag, gp);
        break;
    }
}

void gpio::forward_pin_input(unsigned int tag, tlm::tlm_signal_gp<sc_logic> &gp) {
    const auto mask = 1U << tag;
    if (regs->iof_en & mask) {
        auto &socket = regs->iof_sel & mask ? iof1_o[tag] : iof0_o[tag];
        tlm::tlm_signal_gp<> new_gp;
        for (size_t i = 0; i < socket.size(); ++i) {
            sc_core::sc_time delay{SC_ZERO_TIME};
            tlm::tlm_phase phase{tlm::BEGIN_REQ};
            new_gp.set_command(tlm::TLM_WRITE_COMMAND);
            new_gp.set_response_status(tlm::TLM_OK_RESPONSE);
            new_gp.set_value(gp.get_value().value() == sc_dt::Log_1);
            new_gp.update_extensions_from(gp);
            socket->nb_transport_fw(new_gp, phase, delay); // we don't care about phase and sync enum
        }
    }
}

void gpio::iof_input(unsigned int tag, unsigned iof_idx, tlm::tlm_signal_gp<> &gp, sc_core::sc_time &delay) {
    if (delay > SC_ZERO_TIME) {
        wait(delay);
        delay = SC_ZERO_TIME;
    }
    const auto mask = 1U << tag;
    if (regs->r_iof_en & mask) {
        const auto idx = regs->r_iof_sel & mask ? 1 : 0;
        if (iof_idx == idx) {
            auto &socket = pins_o[tag];
            for (size_t i = 0; i < socket.size(); ++i) {
                sc_core::sc_time delay{SC_ZERO_TIME};
                tlm::tlm_phase phase{tlm::BEGIN_REQ};
                tlm::tlm_signal_gp<sc_logic> new_gp;
                new_gp.set_command(tlm::TLM_WRITE_COMMAND);
                auto val = gp.get_value();
                new_gp.set_value(val ? sc_dt::Log_1 : sc_dt::Log_0);
                new_gp.copy_extensions_from(gp);
                socket->nb_transport_fw(new_gp, phase, delay); // we don't care about phase and sync enum
                gp.update_extensions_from(new_gp);
            }
        }
    }
}

} /* namespace sysc */
