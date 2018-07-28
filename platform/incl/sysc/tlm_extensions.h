/*
 * tlm_extensions.h
 *
 *  Created on: 12.07.2018
 *      Author: eyck
 */

#ifndef _SYSC_TLM_EXTENSIONS_H_
#define _SYSC_TLM_EXTENSIONS_H_

#include "tlm/tlm_extensions.h"

namespace sysc {
struct tlm_signal_uart_extension : public tlm::tlm_unmanaged_extension<tlm_signal_uart_extension> {

    struct uart_tx {
        unsigned data_bits:4;
        unsigned stop_bits:2;
        bool parity:1;
        unsigned baud_rate:24;
        unsigned data;
    } tx;
    sc_core::sc_time start_time;

};

struct tlm_signal_spi_extension : public tlm::tlm_unmanaged_extension<tlm_signal_spi_extension> {

    struct spi_tx {
        unsigned data_bits:5;
        bool msb_first:1;
        bool s2m_data_valid:1;
        unsigned m2s_data, s2m_data;
    } tx;
    sc_core::sc_time start_time;

    void copy_from(tlm_extension_base const & other) override {
        auto& o = static_cast<const type&>(other);
        this->tx=o.tx;
        this->start_time=o.start_time;
    }
};

}



#endif /* _SYSC_TLM_EXTENSIONS_H_ */
