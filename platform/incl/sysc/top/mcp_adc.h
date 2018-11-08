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

#ifndef _SYSC_TOP_MCP3008_H_
#define _SYSC_TOP_MCP3008_H_

#include "cci_configuration"
#include "scc/signal_initiator_mixin.h"
#include "scc/signal_target_mixin.h"
#include "sysc/tlm_extensions.h"
#include <sysc/kernel/sc_module.h>
#include <sysc/utils/sc_vector.h>
#include <tlm/tlm_signal.h>

namespace sysc {

class mcp_adc : public sc_core::sc_module {
public:

    template <typename TYPE>
    static std::unique_ptr<mcp_adc> create(sc_core::sc_module_name nm);

    scc::tlm_signal_logic_in sck_i;
    scc::tlm_signal_logic_out miso_o;
    scc::tlm_signal_logic_in mosi_i;
    scc::tlm_signal_logic_in cs_i;

    sc_core::sc_in<double> vref_i;
    sc_core::sc_vector<sc_core::sc_in<double>> ch_i;

    mcp_adc(mcp_adc &other) = delete;

    mcp_adc(mcp_adc &&other) = delete;

    mcp_adc &operator=(mcp_adc &other) = delete;

    mcp_adc &operator=(mcp_adc &&other) = delete;

    ~mcp_adc() override = default;

protected:
    mcp_adc(sc_core::sc_module_name nm, size_t channel_no)
    : sc_core::sc_module(nm)
    , NAMED(sck_i)
    , NAMED(miso_o)
    , NAMED(mosi_i)
    , NAMED(cs_i)
    , NAMED(vref_i)
    , NAMED(ch_i, channel_no) {}
};

class mcp_3008 : public mcp_adc {
public:
    SC_HAS_PROCESS(mcp_3008);// NOLINT

    mcp_3008(sc_core::sc_module_name nm);
    ~mcp_3008() override = default;

private:
    tlm::tlm_sync_enum receive(tlm::tlm_signal_gp<sc_dt::sc_logic> &, tlm::tlm_phase &, sc_core::sc_time &);
    void do_conversion();
    unsigned idx, rx_bits;
    std::array<uint8_t, 3> rx_bytes, tx_bytes;
    sc_dt::sc_logic mosi_v, miso_v, cs_v;
    sysc::tlm_signal_spi_extension *ext, tx_ext;
    sc_core::sc_time last_tx_start;
};

class mcp_3208 : public mcp_adc {
public:
    SC_HAS_PROCESS(mcp_3208);// NOLINT

    mcp_3208(sc_core::sc_module_name nm);
    ~mcp_3208() override = default;

private:
    tlm::tlm_sync_enum receive(tlm::tlm_signal_gp<sc_dt::sc_logic> &, tlm::tlm_phase &, sc_core::sc_time &);
    void sample_inputs();
    void do_conversion();
    unsigned idx, rx_bits, byte_offs, bit_offs;
    std::array<uint8_t, 3> rx_bytes, tx_bytes;
    sc_dt::sc_logic mosi_v, sck_v, cs_v;
    sysc::tlm_signal_spi_extension *ext, tx_ext;
    sc_core::sc_time last_tx_start;
    sc_core::sc_event clk_sample_evt;
};


} /* namespace sysc */

#endif /* _SYSC_TOP_MCP3008_H_ */
