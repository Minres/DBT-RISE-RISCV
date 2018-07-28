/*
 * mcp3008.h
 *
 *  Created on: 17.07.2018
 *      Author: eyck
 */

#ifndef _SYSC_TOP_MCP3008_H_
#define _SYSC_TOP_MCP3008_H_

#include "scc/signal_target_mixin.h"
#include "scc/signal_initiator_mixin.h"
#include "sysc/tlm_extensions.h"
#include <tlm/tlm_signal.h>
#include "cci_configuration"
#include <sysc/utils/sc_vector.h>
#include <sysc/kernel/sc_module.h>

namespace sysc {

class mcp3008: public sc_core::sc_module {
public:
    SC_HAS_PROCESS(mcp3008);
    scc::tlm_signal_logic_in  sck_i;
    scc::tlm_signal_logic_out miso_o;
    scc::tlm_signal_logic_in  mosi_i;
    scc::tlm_signal_logic_in  cs_i;

    sc_core::sc_in<double> vref_i;
    sc_core::sc_vector<sc_core::sc_in<double>> ch_i;

    mcp3008(sc_core::sc_module_name nm);
    virtual ~mcp3008();

private:
    tlm::tlm_sync_enum receive(tlm::tlm_signal_gp<sc_dt::sc_logic> &, tlm::tlm_phase &, sc_core::sc_time &);
    void do_conversion();
    unsigned idx, rx_bits;
    std::array<uint8_t, 3> rx_bytes, tx_bytes;
    sc_dt::sc_logic mosi_v, miso_v, cs_v;
    sysc::tlm_signal_spi_extension* ext, tx_ext;
    sc_core::sc_time last_tx_start;
};

} /* namespace sysc */

#endif /* _SYSC_TOP_MCP3008_H_ */
