/*
 * terminal.h
 *
 *  Created on: 07.07.2018
 *      Author: eyck
 */

#ifndef _SYSC_TOP_TERMINAL_H_
#define _SYSC_TOP_TERMINAL_H_

#include "scc/signal_target_mixin.h"
#include "scc/signal_initiator_mixin.h"
#include "tlm/tlm_signal.h"
#include "cci_configuration"
#include <sysc/kernel/sc_module.h>
#include <memory>

namespace sysc {
class WsHandler;

class terminal: public sc_core::sc_module {
public:
    scc::tlm_signal_logic_out tx_o;
    scc::tlm_signal_logic_in  rx_i;

    terminal();

    terminal(const sc_core::sc_module_name& nm);

    virtual ~terminal();

    cci::cci_param<bool> write_to_ws;

protected:
    void before_end_of_elaboration();
    void receive(tlm::tlm_signal_gp<sc_dt::sc_logic>& gp, sc_core::sc_time& delay);

    std::vector<uint8_t> queue;
    std::shared_ptr<sysc::WsHandler> handler;
    sc_core::sc_time last_tx_start=sc_core::SC_ZERO_TIME;
};
}

#endif /* _SYSC_TOP_TERMINAL_H_ */
