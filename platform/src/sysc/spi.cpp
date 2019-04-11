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

#include "sysc/SiFive/spi.h"
#include "cci_configuration"
#include "scc/signal_initiator_mixin.h"
#include "scc/signal_target_mixin.h"
#include "scc/tlm_target.h"

#include "scc/utilities.h"
#include "sysc/SiFive/gen/spi_regs.h"
#include "sysc/tlm_extensions.h"
#include <util/ities.h>

namespace sysc {
namespace spi_impl {
using namespace sc_core;

class beh : public sysc::spi, public scc::tlm_target<> {
public:
    SC_HAS_PROCESS(beh); // NOLINT

    cci::cci_param<bool> bit_true_transfer;

    beh(sc_core::sc_module_name nm);
    ~beh() override;

protected:
    scc::tlm_signal_bool_opt_out _sck_o;
    scc::tlm_signal_bool_opt_out _mosi_o;
    scc::tlm_signal_bool_opt_in _miso_i;
    sc_core::sc_vector<scc::tlm_signal_bool_opt_out> _scs_o;

    void clock_cb();
    void reset_cb();
    void transmit_data();
    void receive_data(tlm::tlm_signal_gp<> &gp, sc_core::sc_time &delay);
    void update_irq();
    sc_core::sc_event update_irq_evt;
    sc_core::sc_time clk;
    std::unique_ptr<spi_regs> regs;
    sc_core::sc_fifo<uint8_t> rx_fifo, tx_fifo;
};

beh::beh(sc_core::sc_module_name nm)
: sysc::spi(nm)
, tlm_target<>(clk)
, NAMED(_sck_o)
, NAMED(_mosi_o)
, NAMED(_miso_i)
, NAMED(_scs_o, 4)
, NAMED(bit_true_transfer, false)
, NAMEDD(regs, spi_regs)
, rx_fifo(8)
, tx_fifo(8) {
    spi::socket(scc::tlm_target<>::socket);
    _sck_o(sck_o);
    _mosi_o(mosi_o);
    miso_i(_miso_i);
    _scs_o(scs_o);

    regs->registerResources(*this);
    SC_METHOD(clock_cb);
    sensitive << clk_i;
    SC_METHOD(reset_cb);
    sensitive << rst_i;
    dont_initialize();
    SC_THREAD(transmit_data);
    _miso_i.register_nb_transport(
        [this](tlm::tlm_signal_gp<bool> &gp, tlm::tlm_phase &phase, sc_core::sc_time &delay) -> tlm::tlm_sync_enum {
            this->receive_data(gp, delay);
            return tlm::TLM_COMPLETED;
        });
    regs->txdata.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data, sc_core::sc_time d) -> bool {
        if (!this->regs->in_reset()) {
            reg.put(data);
            tx_fifo.nb_write(static_cast<uint8_t>(regs->r_txdata.data));
        }
        return true;
    });
    regs->rxdata.set_read_cb([this](const scc::sc_register<uint32_t> &reg, uint32_t &data, sc_core::sc_time d) -> bool {
        if (!this->regs->in_reset()) {
            uint8_t val;
            if (rx_fifo.nb_read(val)) {
                regs->r_rxdata.empty = 0;
                regs->r_rxdata.data = val;
                if (regs->r_rxmark.rxmark <= rx_fifo.num_available()) {
                    regs->r_ip.rxwm = 1;
                    update_irq();
                }
            } else
                regs->r_rxdata.empty = 1;
            data = reg.get() & reg.rdmask;
        }
        return true;
    });
    regs->csmode.set_write_cb(
        [this](const scc::sc_register<uint32_t> &reg, const uint32_t &data, sc_core::sc_time d) -> bool {
            if (regs->r_csmode.mode == 2 && regs->r_csmode.mode != bit_sub<0, 2>(data) && regs->r_csid < 4) {
                tlm::tlm_phase phase(tlm::BEGIN_REQ);
                sc_core::sc_time delay(SC_ZERO_TIME);
                tlm::tlm_signal_gp<> gp;
                gp.set_command(tlm::TLM_WRITE_COMMAND);
                gp.set_value(true);
                _scs_o[regs->r_csid]->nb_transport_fw(gp, phase, delay);
            }
            reg.put(data);
            return true;
        });
    regs->csid.set_write_cb([this](const scc::sc_register<uint32_t> &reg, const uint32_t &data, sc_core::sc_time d) -> bool {
        if (regs->r_csmode.mode == 2 && regs->csid != data && regs->r_csid < 4) {
            tlm::tlm_phase phase(tlm::BEGIN_REQ);
            sc_core::sc_time delay(SC_ZERO_TIME);
            tlm::tlm_signal_gp<> gp;
            gp.set_command(tlm::TLM_WRITE_COMMAND);
            gp.set_value(true);
            _scs_o[regs->r_csid]->nb_transport_fw(gp, phase, delay);
        }
        reg.put(data);
        return true;
    });
    regs->csdef.set_write_cb([this](const scc::sc_register<uint32_t> &reg, const uint32_t &data, sc_core::sc_time d) -> bool {
        auto diff = regs->csdef ^ data;
        if (regs->r_csmode.mode == 2 && diff != 0 && (regs->r_csid < 4) && (diff & (1 << regs->r_csid)) != 0) {
            tlm::tlm_phase phase(tlm::BEGIN_REQ);
            sc_core::sc_time delay(SC_ZERO_TIME);
            tlm::tlm_signal_gp<> gp;
            gp.set_command(tlm::TLM_WRITE_COMMAND);
            gp.set_value(true);
            _scs_o[regs->r_csid]->nb_transport_fw(gp, phase, delay);
        }
        reg.put(data);
        return true;
    });
    regs->ie.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data, sc_core::sc_time d) -> bool {
        reg.put(data);
        update_irq_evt.notify();
        return true;
    });
    regs->ip.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data, sc_core::sc_time d) -> bool {
        reg.put(data);
        update_irq_evt.notify();
        return true;
    });

    SC_METHOD(update_irq);
    sensitive << update_irq_evt << rx_fifo.data_written_event() << rx_fifo.data_read_event()
              << tx_fifo.data_written_event() << tx_fifo.data_read_event();
}

beh::~beh() = default;

void beh::clock_cb() { this->clk = clk_i.read(); }

void beh::reset_cb() {
    if (rst_i.read())
        regs->reset_start();
    else
        regs->reset_stop();
}

void beh::transmit_data() {
    uint8_t txdata;
    tlm::tlm_phase phase(tlm::BEGIN_REQ);
    sc_core::sc_time delay(SC_ZERO_TIME);
    sc_core::sc_time bit_duration(SC_ZERO_TIME);
    sc_core::sc_time start_time;

    auto set_bit = [&](bool val, scc::tlm_signal_bool_opt_out &socket,
                       bool data_valid = false) -> std::pair<bool, uint32_t> {
        if (socket.get_interface() == nullptr) return std::pair<bool, uint32_t>{false, 0};
        auto *gp = tlm::tlm_signal_gp<>::create();
        auto *ext = new sysc::tlm_signal_spi_extension();
        ext->tx.data_bits = 8;
        ext->start_time = start_time;
        ext->tx.m2s_data = txdata;
        ext->tx.m2s_data_valid = data_valid;
        ext->tx.s2m_data_valid = false;
        gp->set_extension(ext);
        gp->set_command(tlm::TLM_WRITE_COMMAND);
        gp->set_value(val);
        tlm::tlm_phase phase(tlm::BEGIN_REQ);
        gp->acquire();
        phase = tlm::BEGIN_REQ;
        delay = SC_ZERO_TIME;
        socket->nb_transport_fw(*gp, phase, delay);
        std::pair<bool, uint32_t> ret{ext->tx.s2m_data_valid != 0, ext->tx.s2m_data};
        gp->release();
        return ret;
    };

    wait(delay); // intentionally 0ns;
    while (true) {
        wait(tx_fifo.data_written_event());
        if (regs->r_csmode.mode != 3 && regs->r_csid < 4) // not in OFF mode
            set_bit(false, _scs_o[regs->r_csid]);
        set_bit(regs->r_sckmode.pol, _sck_o);
        while (tx_fifo.nb_read(txdata)) {
            regs->r_txdata.full = tx_fifo.num_free() == 0;
            regs->r_ip.txwm = regs->r_txmark.txmark <= (7 - tx_fifo.num_free()) ? 1 : 0;
            update_irq_evt.notify();
            bit_duration = 2 * (regs->r_sckdiv.div + 1) * clk;
            start_time = sc_core::sc_time_stamp();
            set_bit(txdata & 0x80, _mosi_o); // 8 data bits, MSB first
            auto s2m = set_bit(1 - regs->r_sckmode.pol, _sck_o, true);
            wait(bit_duration / 2);
            set_bit(regs->r_sckmode.pol, _sck_o, true);
            wait(bit_duration / 2);
            if (bit_true_transfer.get_value()) {
                for (size_t i = 0, mask = 0x40; i < 7; ++i, mask >= 1) {
                    set_bit(txdata & mask, _mosi_o); // 8 data bits, MSB first
                    set_bit(1 - regs->r_sckmode.pol, _sck_o);
                    wait(bit_duration / 2);
                    set_bit(regs->r_sckmode.pol, _sck_o);
                    wait(bit_duration / 2);
                }
            } else
                wait(7 * bit_duration);
            if (s2m.first) rx_fifo.nb_write(s2m.second & 0xff);
            update_irq_evt.notify();
        }
        if (regs->r_csmode.mode == 0 && regs->r_csid < 4) // in AUTO mode
            set_bit(false, _scs_o[regs->r_csid]);
    }
}

void beh::receive_data(tlm::tlm_signal_gp<> &gp, sc_core::sc_time &delay) {}

void beh::update_irq() {
    regs->r_ip.rxwm = regs->r_rxmark.rxmark < rx_fifo.num_available();
    regs->r_ip.txwm = regs->r_txmark.txmark <= tx_fifo.num_available();
    regs->r_txdata.full = tx_fifo.num_free() == 0;
    irq_o.write((regs->r_ie.rxwm > 0 && regs->r_ip.rxwm > 0) || (regs->r_ie.txwm > 0 && regs->r_ip.txwm > 0));
}
} /* namespace spi:impl */

template <> std::unique_ptr<spi> spi::create<sysc::spi_impl::beh>(sc_core::sc_module_name nm) {
    auto *res = new sysc::spi_impl::beh(nm);
    return std::unique_ptr<spi>(res);
}

} /* namespace sysc */
