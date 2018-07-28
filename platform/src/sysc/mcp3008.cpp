/*
 * mcp3008.cpp
 *
 *  Created on: 17.07.2018
 *      Author: eyck
 */

#include "sysc/top/mcp3008.h"

#include <scc/report.h>
#include <util/ities.h>

namespace sysc {

mcp3008::mcp3008(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, NAMED(sck_i)
, NAMED(miso_o)
, NAMED(mosi_i)
, NAMED(cs_i)
, NAMED(vref_i)
, NAMED(ch_i, 8)
, last_tx_start(sc_core::SC_ZERO_TIME)
{
    sck_i.register_nb_transport([this](tlm::tlm_signal_gp<sc_dt::sc_logic>& gp, tlm::tlm_phase& phase, sc_core::sc_time& delay)
            -> tlm::tlm_sync_enum{
        return tlm::TLM_COMPLETED;
    });

    mosi_i.register_nb_transport([this](tlm::tlm_signal_gp<sc_dt::sc_logic>& gp, tlm::tlm_phase& phase, sc_core::sc_time& delay)
            -> tlm::tlm_sync_enum{
        if(cs_v==sc_dt::Log_0)
            return receive(gp, phase, delay);
    });

    cs_i.register_nb_transport([this](tlm::tlm_signal_gp<sc_dt::sc_logic>& gp, tlm::tlm_phase& phase, sc_core::sc_time& delay)
            -> tlm::tlm_sync_enum{
        if(cs_v!=sc_dt::Log_0 && gp.get_value()==sc_dt::Log_0){
            idx=0; // falling edge
            rx_bits=0;
        }
        cs_v=gp.get_value();
        return tlm::TLM_COMPLETED;
    });
}

mcp3008::~mcp3008() {
}

tlm::tlm_sync_enum mcp3008::receive(tlm::tlm_signal_gp<sc_dt::sc_logic>& gp, tlm::tlm_phase& phase, sc_core::sc_time& delay) {
    gp.get_extension(ext);
    if(ext){
        if( ext->start_time!=last_tx_start){
            assert(ext->tx.data_bits==8);
            rx_bytes[idx]=bit_sub<0,8>(ext->tx.m2s_data);
            if(idx==1)
                do_conversion();
            ext->tx.s2m_data=tx_bytes[idx];
            ext->tx.s2m_data_valid=true;
            idx++;
            last_tx_start=ext->start_time;
        }
    }
    return tlm::TLM_COMPLETED;
}

void mcp3008::do_conversion() {
    if(rx_bytes[0]==0x1){
        auto mode = bit_sub<7,1>(rx_bytes[1]);
        auto channel = bit_sub<4,3>(rx_bytes[1]);
        auto vref=vref_i.read();
        if(mode){ // single ended
            auto inp = ch_i[channel].read();
            auto norm = inp/vref*1024.0;
            auto res = static_cast<int>(norm);
            CLOG(DEBUG, SystemC)<<"Converting "<<inp<<" to "<<norm<<" as int "<<res;
            tx_bytes[1]=bit_sub<8,2>(res);
            tx_bytes[2]=bit_sub<0,8>(res);
        } else {
            tx_bytes[1]=0;
            tx_bytes[2]=0;
        }
    }
}

} /* namespace sysc */
