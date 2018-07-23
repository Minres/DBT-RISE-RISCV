////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, MINRES Technologies GmbH
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Contributors:
//       eyck@minres.com - initial implementation
//
//
////////////////////////////////////////////////////////////////////////////////

#include "sysc/SiFive/spi.h"

#include "sysc/tlm_extensions.h"
#include "scc/utilities.h"
#include "sysc/SiFive/gen/spi_regs.h"
#include <util/ities.h>

namespace sysc {

spi::spi(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, tlm_target<>(clk)
, NAMED(clk_i)
, NAMED(rst_i)
, NAMED(sck_o)
, NAMED(mosi_o)
, NAMED(miso_i)
, NAMED(scs_o, 4)
, NAMED(irq_o)
, NAMED(bit_true_transfer, false)
, NAMEDD(spi_regs, regs) {
    regs->registerResources(*this);
    SC_METHOD(clock_cb);
    sensitive << clk_i;
    SC_METHOD(reset_cb);
    sensitive << rst_i;
    dont_initialize();
    SC_THREAD(transmit_data);
    miso_i.register_nb_transport([this](tlm::tlm_signal_gp<bool>& gp,
            tlm::tlm_phase& phase, sc_core::sc_time& delay)->tlm::tlm_sync_enum{
       this->receive_data(gp, delay);
       return tlm::TLM_COMPLETED;
    });
    regs->txdata.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data) -> bool {
        if (!this->regs->in_reset()) {
            reg.put(data);
            tx_fifo.nb_write(static_cast<uint8_t>(regs->r_txdata.data));
            regs->r_txdata.full=tx_fifo.num_free()==0;
            update_irq();
        }
        return true;
    });
    regs->rxdata.set_read_cb([this](const scc::sc_register<uint32_t> &reg, uint32_t& data) -> bool {
        if (!this->regs->in_reset()) {
            uint8_t val;
            if(rx_fifo.nb_read(val)){
                regs->r_rxdata.empty=0;
                regs->r_rxdata.data=val;
                if(regs->r_rxmark.rxmark<=rx_fifo.num_available()){
                    regs->r_ip.rxwm=1;
                    update_irq();
                }
            } else
                regs->r_rxdata.empty=1;
            data = reg.get()&reg.rdmask;
        }
        return true;
    });
    regs->csmode.set_write_cb([this](const scc::sc_register<uint32_t> &reg, uint32_t& data) -> bool {
        if(regs->r_csmode.mode==2 && regs->r_csmode.mode != bit_sub<0, 2>(data) && regs->r_csid<4){
            tlm::tlm_phase phase(tlm::BEGIN_REQ);
            sc_core::sc_time delay(SC_ZERO_TIME);
            tlm::tlm_signal_gp<> gp;
            gp.set_command(tlm::TLM_WRITE_COMMAND);
            gp.set_value(true);
            scs_o[regs->r_csid]->nb_transport_fw(gp, phase, delay);
        }
        reg.put(data);
        return true;
    });
    regs->csid.set_write_cb([this](const scc::sc_register<uint32_t> &reg, uint32_t& data) -> bool {
        if(regs->r_csmode.mode==2 && regs->csid != data && regs->r_csid<4){
            tlm::tlm_phase phase(tlm::BEGIN_REQ);
            sc_core::sc_time delay(SC_ZERO_TIME);
            tlm::tlm_signal_gp<> gp;
            gp.set_command(tlm::TLM_WRITE_COMMAND);
            gp.set_value(true);
            scs_o[regs->r_csid]->nb_transport_fw(gp, phase, delay);
        }
        reg.put(data);
        return true;
    });
    regs->csdef.set_write_cb([this](const scc::sc_register<uint32_t> &reg, uint32_t& data) -> bool {
        auto diff = regs->csdef ^ data;
        if(regs->r_csmode.mode==2 && diff!=0 && (regs->r_csid<4) && (diff & (1<<regs->r_csid))!=0){
            tlm::tlm_phase phase(tlm::BEGIN_REQ);
            sc_core::sc_time delay(SC_ZERO_TIME);
            tlm::tlm_signal_gp<> gp;
            gp.set_command(tlm::TLM_WRITE_COMMAND);
            gp.set_value(true);
            scs_o[regs->r_csid]->nb_transport_fw(gp, phase, delay);
        }
        reg.put(data);
        return true;
    });
    regs->ie.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data) -> bool {
        update_irq();
    });
    regs->ip.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data) -> bool {
        update_irq();
    });
}

spi::~spi() {}

void spi::clock_cb() {
	this->clk = clk_i.read();
}

void spi::reset_cb() {
    if (rst_i.read())
        regs->reset_start();
    else
        regs->reset_stop();
}

void spi::transmit_data() {
    uint8_t txdata;
    sysc::tlm_signal_spi_extension ext;
    tlm::tlm_phase phase(tlm::BEGIN_REQ);
    tlm::tlm_signal_gp<> gp;
    sc_core::sc_time delay(SC_ZERO_TIME);
    sc_core::sc_time bit_duration(SC_ZERO_TIME);

    gp.set_extension(&ext);
    ext.tx.data_bits=8;

    auto set_bit = [&](bool val, scc::tlm_signal_bool_opt_out& socket){
        if(socket.get_interface()==nullptr) return;
        gp.set_command(tlm::TLM_WRITE_COMMAND);
        gp.set_value(val);
        tlm::tlm_phase phase(tlm::BEGIN_REQ);
        socket->nb_transport_fw(gp, phase, delay);
    };

    wait(delay); //intentionally 0ns;
    while(true){
        wait(tx_fifo.data_written_event());
        if(regs->r_csmode.mode != 3 && regs->r_csid<4) // not in OFF mode
            set_bit(false, scs_o[regs->r_csid]);
        set_bit(regs->r_sckmode.pol, sck_o);
        while(tx_fifo.nb_read(txdata)){
            regs->r_txdata.full=tx_fifo.num_free()==0;
            regs->r_ip.txwm=regs->r_txmark.txmark<=(7-tx_fifo.num_free())?1:0;
            bit_duration = 2*(regs->r_sckdiv.div+1)*clk;
            ext.start_time = sc_core::sc_time_stamp();
            ext.tx.m2s_data=txdata;
            ext.tx.s2m_data_valid=false;
            set_bit(txdata&0x80, mosi_o); // 8 data bits, MSB first
            set_bit(1-regs->r_sckmode.pol, sck_o);
            wait(bit_duration/2);
            set_bit(regs->r_sckmode.pol, sck_o);
            wait(bit_duration/2);
            if(bit_true_transfer.get_value()){
                for(size_t i = 0, mask=0x40; i<7; ++i, mask>=1){
                    set_bit(txdata&mask, mosi_o); // 8 data bits, MSB first
                    set_bit(1-regs->r_sckmode.pol, sck_o);
                    wait(bit_duration/2);
                    set_bit(regs->r_sckmode.pol, sck_o);
                    wait(bit_duration/2);

                }
            } else
                wait(7*bit_duration);
            rx_fifo.nb_write(ext.tx.s2m_data&0xff);
            if(regs->r_rxmark.rxmark<=rx_fifo.num_available()){
                regs->r_ip.rxwm=1;
                update_irq();
            }
        }
        if(regs->r_csmode.mode == 0 && regs->r_csid<4) // in AUTO mode
            set_bit(false, scs_o[regs->r_csid]);
    }
}

void spi::receive_data(tlm::tlm_signal_gp<>& gp, sc_core::sc_time& delay) {
}

void spi::update_irq() {
}

} /* namespace sysc */

