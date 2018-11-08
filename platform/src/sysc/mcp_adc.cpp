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

#include <scc/report.h>
#include <sysc/top/mcp_adc.h>
#include <util/ities.h>

namespace sysc {


mcp_3008::mcp_3008(sc_core::sc_module_name nm)
: sysc::mcp_adc(nm, 8)
, last_tx_start(sc_core::SC_ZERO_TIME) {
    sck_i.register_nb_transport([this](tlm::tlm_signal_gp<sc_dt::sc_logic> &gp, tlm::tlm_phase &phase,
                                       sc_core::sc_time &delay) -> tlm::tlm_sync_enum { return tlm::TLM_COMPLETED; });

    mosi_i.register_nb_transport([this](tlm::tlm_signal_gp<sc_dt::sc_logic> &gp, tlm::tlm_phase &phase,
                                        sc_core::sc_time &delay) -> tlm::tlm_sync_enum {
        if (cs_v == sc_dt::Log_0) return receive(gp, phase, delay);
        return tlm::TLM_COMPLETED;
    });

    cs_i.register_nb_transport([this](tlm::tlm_signal_gp<sc_dt::sc_logic> &gp, tlm::tlm_phase &phase,
                                      sc_core::sc_time &delay) -> tlm::tlm_sync_enum {
        if (cs_v != sc_dt::Log_0 && gp.get_value() == sc_dt::Log_0) {
            idx = 0; // falling edge
            rx_bits = 0;
        }
        cs_v = gp.get_value();
        return tlm::TLM_COMPLETED;
    });
}

tlm::tlm_sync_enum mcp_3008::receive(tlm::tlm_signal_gp<sc_dt::sc_logic> &gp, tlm::tlm_phase &phase,
                                     sc_core::sc_time &delay) {
    gp.get_extension(ext);
    if (ext) {
        if (ext->start_time != last_tx_start) {
            assert(ext->tx.data_bits == 8);
            rx_bytes[idx] = bit_sub<0, 8>(ext->tx.m2s_data);
            if (idx == 1) do_conversion();
            ext->tx.s2m_data = tx_bytes[idx];
            ext->tx.s2m_data_valid = true;
            idx++;
            last_tx_start = ext->start_time;
        }
    }
    return tlm::TLM_COMPLETED;
}

void mcp_3008::do_conversion() {
    if (rx_bytes[0] == 0x1) {
        auto mode = bit_sub<7, 1>(rx_bytes[1]);
        auto channel = bit_sub<4, 3>(rx_bytes[1]);
        auto vref = vref_i.read();
        if (mode) { // single ended
            auto inp = ch_i[channel].read();
            auto norm = 1024.0 * inp / vref;
            auto res = static_cast<int>(norm);
            SCDEBUG(this->name()) << "Converting " << inp << " to " << norm << " as int " << res;
            tx_bytes[1] = bit_sub<8, 2>(res);
            tx_bytes[2] = bit_sub<0, 8>(res);
        } else {
            tx_bytes[1] = 0;
            tx_bytes[2] = 0;
        }
    }
}

mcp_3208::mcp_3208(sc_core::sc_module_name nm)
: sysc::mcp_adc(nm, 8)
, ext(nullptr)
, last_tx_start(sc_core::SC_ZERO_TIME) {
    sck_i.register_nb_transport([this](tlm::tlm_signal_gp<sc_dt::sc_logic> &gp, tlm::tlm_phase &phase,
                                       sc_core::sc_time &delay) -> tlm::tlm_sync_enum {
        auto ret = tlm::TLM_COMPLETED;
        if (cs_v == sc_dt::Log_0) ret = receive(gp, phase, delay);
        sck_v = gp.get_value();
        return ret;
    });

    mosi_i.register_nb_transport([this](tlm::tlm_signal_gp<sc_dt::sc_logic> &gp, tlm::tlm_phase &phase,
                                        sc_core::sc_time &delay) -> tlm::tlm_sync_enum {
        mosi_v = gp.get_value();
        return tlm::TLM_COMPLETED;
    });

    cs_i.register_nb_transport([this](tlm::tlm_signal_gp<sc_dt::sc_logic> &gp, tlm::tlm_phase &phase,
                                      sc_core::sc_time &delay) -> tlm::tlm_sync_enum {
        if (cs_v != sc_dt::Log_0 && gp.get_value() == sc_dt::Log_0) { // falling edge of CS
            idx = 0;
            rx_bits = byte_offs = 0;
            bit_offs = 7;
        }
        cs_v = gp.get_value();
        return tlm::TLM_COMPLETED;
    });
    SC_METHOD(sample_inputs);
    sensitive << clk_sample_evt;
}

tlm::tlm_sync_enum mcp_3208::receive(tlm::tlm_signal_gp<sc_dt::sc_logic> &gp, tlm::tlm_phase &phase,
                                     sc_core::sc_time &delay) {
    gp.get_extension(ext);
    if (ext) {
        if (ext->start_time != last_tx_start) {
            assert(ext->tx.data_bits == 8);
            if (ext->tx.m2s_data_valid) {
                rx_bytes[idx] = bit_sub<0, 8>(ext->tx.m2s_data);
                if (idx == 1) do_conversion();
                ext->tx.s2m_data = tx_bytes[idx];
                ext->tx.s2m_data_valid = true;
                last_tx_start = ext->start_time;
                idx++;
            }
        }
    } else if (gp.get_value() == sc_dt::SC_LOGIC_1 && sck_v == sc_dt::SC_LOGIC_0) // sample an rising edge
        clk_sample_evt.notify(sc_core::SC_ZERO_TIME);
    return tlm::TLM_COMPLETED;
}

void mcp_3208::sample_inputs() {
    if (byte_offs >= 3) return;
    if (bit_offs == 7) {
        rx_bytes[byte_offs] = 0;
        if (byte_offs == 0) tx_bytes[0] = tx_bytes[1] = tx_bytes[2] = 0;
    }
    auto mask = 1 << bit_offs;
    if (mosi_v == sc_dt::SC_LOGIC_1) rx_bytes[byte_offs] |= mask;
    miso_o.write_now(tx_bytes[byte_offs] & mask ? sc_dt::SC_LOGIC_1 : sc_dt::SC_LOGIC_0);
    // increment counters
    if (bit_offs == 0) {
        bit_offs = 7;
        byte_offs++;
    } else
        bit_offs--;
    // sample if in the middle of second byte
    if (byte_offs == 1 && bit_offs == 4) do_conversion();
}

void mcp_3208::do_conversion() {
    if (rx_bytes[0] & 0x4) {
        auto mode = bit_sub<1, 1>(rx_bytes[0]);
        auto channel = bit_sub<0, 1>(rx_bytes[0]) * 4 + bit_sub<6, 2>(rx_bytes[1]);
        auto vref = vref_i.read();
        if (mode) { // single ended
            auto inp = ch_i[channel].read();
            auto norm = 4096.0 * inp / vref;
            auto res = static_cast<int>(norm);
            SCDEBUG(this->name()) << "Converting channel " << channel << " " << inp << "V to " << norm << " as int "
                                  << res;
            tx_bytes[1] = bit_sub<8, 4>(res);
            tx_bytes[2] = bit_sub<0, 8>(res);
        } else {
            tx_bytes[1] = 0;
            tx_bytes[2] = 0;
        }
    }
}

template <>
std::unique_ptr<mcp_adc> mcp_adc::create<mcp_3008>(sc_core::sc_module_name nm) {
    auto *res = new mcp_3008(nm);
    return std::unique_ptr<mcp_adc>(res);
}

template <>
std::unique_ptr<mcp_adc> mcp_adc::create<mcp_3208>(sc_core::sc_module_name nm) {
    auto *res = new mcp_3208(nm);
    return std::unique_ptr<mcp_adc>(res);
}

} /* namespace sysc */
