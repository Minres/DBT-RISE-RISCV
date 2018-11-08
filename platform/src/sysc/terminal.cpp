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

#include "sysc/top/terminal.h"

#include "scc/report.h"
#include "sysc/sc_comm_singleton.h"
#include "sysc/tlm_extensions.h"

using namespace sysc;

terminal::terminal()
: terminal(sc_core::sc_gen_unique_name("terminal")) {}

terminal::terminal(const sc_core::sc_module_name &nm)
: sc_core::sc_module(nm)
, NAMED(tx_o)
, NAMED(rx_i)
, NAMED(write_to_ws, false) {
    rx_i.register_nb_transport([this](tlm::tlm_signal_gp<sc_dt::sc_logic> &gp, tlm::tlm_phase &phase,
                                      sc_core::sc_time &delay) -> tlm::tlm_sync_enum {
        this->receive(gp, delay);
        return tlm::TLM_COMPLETED;
    });
}

terminal::~terminal() = default;

void terminal::before_end_of_elaboration() {
    if (write_to_ws.get_value()) {
        SCTRACE() << "Adding WS handler for " << (std::string{"/ws/"} + name());
        handler = std::make_shared<WsHandler>();
        sc_comm_singleton::inst().registerWebSocketHandler((std::string{"/ws/"} + name()).c_str(), handler);
    }
}

void terminal::receive(tlm::tlm_signal_gp<sc_dt::sc_logic> &gp, sc_core::sc_time &delay) {
    sysc::tlm_signal_uart_extension *ext;
    gp.get_extension(ext);
    if (ext && ext->start_time != last_tx_start) {
        auto txdata = static_cast<uint8_t>(ext->tx.data);
        last_tx_start = ext->start_time;
        if (txdata != '\r') queue.push_back(txdata);
        if (queue.size() >> 0 && (txdata == '\n' || txdata == 0)) {
            std::string msg(queue.begin(), queue.end() - 1);
            sc_core::sc_time now = sc_core::sc_time_stamp();
            if (handler)
                sysc::sc_comm_singleton::inst().execute([this, msg, now]() {
                    std::stringstream os;
                    os << R"({"time":")" << now << R"(","message":")" << msg << R"("})";
                    this->handler->send(os.str());
                });
            else
                SCINFO(this->name()) << " receive: '" << msg << "'";
            queue.clear();
        }
    }
}
