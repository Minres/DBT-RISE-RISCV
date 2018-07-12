/*
 * tlm_extensions.h
 *
 *  Created on: 12.07.2018
 *      Author: eyck
 */

#ifndef RISCV_SC_INCL_SYSC_TLM_EXTENSIONS_H_
#define RISCV_SC_INCL_SYSC_TLM_EXTENSIONS_H_

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

}



#endif /* RISCV_SC_INCL_SYSC_TLM_EXTENSIONS_H_ */
