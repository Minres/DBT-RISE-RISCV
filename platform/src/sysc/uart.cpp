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

#include "sysc/SiFive/uart.h"

#include "sysc/tlm_extensions.h"
#include "scc/report.h"
#include "scc/utilities.h"
#include "sysc/SiFive/gen/uart_regs.h"

using namespace std;


namespace sysc {

uart::uart(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, tlm_target<>(clk)
, NAMED(clk_i)
, NAMED(rst_i)
, NAMED(tx_o)
, NAMED(rx_i)
, NAMED(irq_o)
, NAMED(bit_true_transfer, false)
, NAMEDD(uart_regs, regs)
, NAMED(rx_fifo, 8)
, NAMED(tx_fifo, 8)
{
    regs->registerResources(*this);
    SC_METHOD(clock_cb);
    sensitive << clk_i;
    SC_METHOD(reset_cb);
    sensitive << rst_i;
    dont_initialize();
    SC_THREAD(transmit_data);
    rx_i.register_nb_transport([this](tlm::tlm_signal_gp<bool>& gp,
            tlm::tlm_phase& phase, sc_core::sc_time& delay)->tlm::tlm_sync_enum{
       this->receive_data(gp, delay);
       return tlm::TLM_COMPLETED;
    });
    regs->txdata.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data) -> bool {
        if (!this->regs->in_reset()) {
            reg.put(data);
            tx_fifo.nb_write(static_cast<uint8_t>(regs->r_txdata.data));
            regs->r_txdata.full=tx_fifo.num_free()==0;
            regs->r_ip.txwm=regs->r_txctrl.txcnt<=(7-tx_fifo.num_free())?1:0;
            update_irq();
        }
        return true;
    });
    regs->rxdata.set_read_cb([this](const scc::sc_register<uint32_t> &reg, uint32_t& data) -> bool {
        if (!this->regs->in_reset()) {
            uint8_t val;
            if(rx_fifo.nb_read(val)){
                regs->r_rxdata.data=val;
                if(regs->r_rxctrl.rxcnt<=rx_fifo.num_available()){
                    regs->r_ip.rxwm=1;
                    update_irq();
                }
            }
            data = reg.get()&reg.rdmask;
        }
        return true;
    });
    regs->ie.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data) -> bool {
        update_irq();
    });
    regs->ip.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data) -> bool {
        update_irq();
    });
}

uart::~uart() {}

void uart::update_irq() {
    irq_o=(regs->r_ip.rxwm==1 && regs->r_ie.rxwm==1) || (regs->r_ip.txwm==1 && regs->r_ie.txwm==1);
}

void uart::clock_cb() {
	this->clk = clk_i.read();
}

void uart::reset_cb() {
    if (rst_i.read())
        regs->reset_start();
    else
        regs->reset_stop();
}

void uart::transmit_data() {
    uint8_t txdata;
    sysc::tlm_signal_uart_extension ext;
    tlm::tlm_phase phase(tlm::BEGIN_REQ);
    tlm::tlm_signal_gp<> gp;
    sc_core::sc_time delay(SC_ZERO_TIME);
    sc_core::sc_time bit_duration(SC_ZERO_TIME);

    gp.set_extension(&ext);
    ext.tx.data_bits=8;
    ext.tx.parity=false;

    auto set_bit = [&](bool val){
        gp.set_command(tlm::TLM_WRITE_COMMAND);
        gp.set_value(val);
        tlm::tlm_phase phase(tlm::BEGIN_REQ);
        tx_o->nb_transport_fw(gp, phase, delay);
        if(delay<bit_duration) wait(bit_duration-delay);
    };
    wait(delay);
    while(true){
        set_bit(true);
        wait(tx_fifo.data_written_event());
        while(tx_fifo.nb_read(txdata)){
            regs->r_txdata.full=tx_fifo.num_free()==0;
            regs->r_ip.txwm=regs->r_txctrl.txcnt<=(7-tx_fifo.num_free())?1:0;
            bit_duration = (regs->r_div.div+1)*clk;
            ext.start_time = sc_core::sc_time_stamp();
            ext.tx.stop_bits=1+regs->r_txctrl.nstop;
            ext.tx.baud_rate=static_cast<unsigned>(1/bit_duration.to_seconds());
            ext.tx.data=txdata;
            set_bit(false); // start bit
            if(bit_true_transfer.get_value()){
                for(int i = 8; i>0; --i)
                    set_bit(txdata&(1<<(i-1))); // 8 data bits, MSB first
                if(regs->r_txctrl.nstop) set_bit(true); // stop bit 1
            } else
                wait(8*bit_duration);
            set_bit(true); // stop bit 1/2
        }
    }
}

void uart::receive_data(tlm::tlm_signal_gp<>& gp, sc_core::sc_time& delay) {
    sysc::tlm_signal_uart_extension* ext{nullptr};
    gp.get_extension(ext);
    if(ext && ext->start_time != rx_last_start){
        auto data = static_cast<uint8_t>(ext->tx.data);
        if(ext->tx.parity || ext->tx.data_bits!=8) data = rand(); // random value if wrong config
        rx_fifo.write(data);
        if(regs->r_rxctrl.rxcnt<=rx_fifo.num_available()){
            regs->r_ip.rxwm=1;
            update_irq();
        }
        rx_last_start=ext->start_time; // omit repeated handling of signal changes
    }
    gp.set_response_status(tlm::TLM_OK_RESPONSE);
}

} /* namespace sysc */

