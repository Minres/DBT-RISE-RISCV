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

#ifndef _SYSC_TLM_EXTENSIONS_H_
#define _SYSC_TLM_EXTENSIONS_H_

#include "tlm/tlm_extensions.h"

namespace sysc {
struct tlm_signal_uart_extension : public tlm::tlm_unmanaged_extension<tlm_signal_uart_extension> {

    struct uart_tx {
        unsigned data_bits : 4;
        unsigned stop_bits : 2;
        bool parity : 1;
        unsigned baud_rate : 24;
        unsigned data;
    } tx;
    sc_core::sc_time start_time;
};

struct tlm_signal_spi_extension : public tlm::tlm_unmanaged_extension<tlm_signal_spi_extension> {

    struct spi_tx {
        unsigned data_bits : 5;
        bool msb_first : 1;
        bool m2s_data_valid : 1;
        bool s2m_data_valid : 1;
        unsigned m2s_data, s2m_data;
    } tx;
    sc_core::sc_time start_time;

    void copy_from(tlm_extension_base const &other) override {
        auto &o = static_cast<const type &>(other);
        this->tx = o.tx;
        this->start_time = o.start_time;
    }
};
}

#endif /* _SYSC_TLM_EXTENSIONS_H_ */
