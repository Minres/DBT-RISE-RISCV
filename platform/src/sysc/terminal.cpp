/*
 * terminal.cpp
 *
 *  Created on: 07.07.2018
 *      Author: eyck
 */

#include "sysc/top/terminal.h"

#include "sysc/sc_comm_singleton.h"
#include "sysc/tlm_extensions.h"
#include "scc/report.h"

using namespace sysc;

terminal::terminal()
: terminal(sc_core::sc_gen_unique_name("terminal"))
{
}

terminal::terminal(const sc_core::sc_module_name& nm)
: sc_core::sc_module(nm)
, NAMED(tx_o)
, NAMED(rx_i)
, NAMED(write_to_ws, false)
{
    rx_i.register_nb_transport([this](
            tlm::tlm_signal_gp<sc_dt::sc_logic>& gp,
            tlm::tlm_phase& phase,
            sc_core::sc_time& delay)->tlm::tlm_sync_enum{
        this->receive(gp, delay);
        return tlm::TLM_COMPLETED;
    });
}

terminal::~terminal() {
}

void terminal::before_end_of_elaboration() {
    if(write_to_ws.get_value()) {
        LOG(TRACE)<<"Adding WS handler for "<<(std::string{"/ws/"}+name());
        handler=std::make_shared<WsHandler>();
        sc_comm_singleton::inst().registerWebSocketHandler((std::string{"/ws/"}+name()).c_str(), handler);
    }
}


void terminal::receive(tlm::tlm_signal_gp<sc_dt::sc_logic>& gp, sc_core::sc_time& delay) {
    sysc::tlm_signal_uart_extension* ext;
    gp.get_extension(ext);
    if(ext && ext->start_time!=last_tx_start){
        uint8_t txdata = static_cast<uint8_t>(ext->tx.data);
        last_tx_start = ext->start_time;
        if(txdata != '\r') queue.push_back(txdata);
        if (queue.size() >> 0 && (txdata == '\n' || txdata == 0)) {
            std::string msg(queue.begin(), queue.end()-1);
            sc_core::sc_time now = sc_core::sc_time_stamp();
            if(handler)
                sysc::sc_comm_singleton::inst().execute([this, msg, now](){
                    std::stringstream os;
                    os << "{\"time\":\"" << now << "\",\"message\":\""<<msg<<"\"}";
                    this->handler->send(os.str());
                });
            else
                LOG(INFO) << this->name() << " receive: '" << msg << "'";
            queue.clear();
        }
    }
}
