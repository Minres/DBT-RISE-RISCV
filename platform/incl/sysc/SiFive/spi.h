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

#ifndef _SPI_H_
#define _SPI_H_

#include <sysc/utils/sc_vector.h>
#include <tlm/tlm_signal.h>

namespace sysc {

namespace spi_impl {
class beh;
class rtl;
}

class spi : public sc_core::sc_module {
public:
    template <typename TYPE>
    static std::unique_ptr<spi> create(sc_core::sc_module_name nm);

    template <typename T> using tlm_in = tlm::tlm_signal_opt_target_socket<T>;
    template <typename T> using tlm_out = tlm::tlm_signal_opt_initiator_socket<T>;

    tlm::tlm_target_socket<> socket;
    sc_core::sc_in<sc_core::sc_time> clk_i;
    sc_core::sc_in<bool> rst_i;
    tlm_out<bool> sck_o;
    tlm_out<bool> mosi_o;
    tlm_in<bool> miso_i;
    sc_core::sc_vector<tlm_out<bool>> scs_o;

    sc_core::sc_out<bool> irq_o;

    spi(spi &other) = delete;

    spi(spi &&other) = delete;

    spi &operator=(spi &other) = delete;

    spi &operator=(spi &&other) = delete;

    ~spi() override = default;

protected:
    spi(sc_core::sc_module_name nm)
    : sc_core::sc_module(nm)
    , NAMED(clk_i)
    , NAMED(rst_i)
    , NAMED(sck_o)
    , NAMED(mosi_o)
    , NAMED(miso_i)
    , NAMED(scs_o, 4)
    , NAMED(irq_o){};
};

} /* namespace sysc */

#endif /* _SPI_H_ */
