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

#include "sysc/SiFive/pwm.h"

#include "scc/utilities.h"
#include "sysc/SiFive/gen/pwm_regs.h"

using namespace sysc;
using namespace sc_core;

pwm::pwm(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, tlm_target<>(clk)
, NAMED(clk_i)
, NAMED(rst_i)
, NAMED(cmpgpio_o, 4)
, NAMED(cmpip_o, 4)
, NAMEDD(regs, pwm_regs)
, current_cnt(0)
, last_cnt_update() {
    regs->registerResources(*this);

    regs->pwmcfg.set_write_cb(
        [this](const scc::sc_register<uint32_t> &reg, const uint32_t &data, sc_core::sc_time d) -> bool {
            if (d.value()) wait(d);
            reg.put(data);
            update_counter();
            return true;
        });
    regs->pwmcount.set_write_cb(
        [this](const scc::sc_register<uint32_t> &reg, const uint32_t &data, sc_core::sc_time d) -> bool {
            if (d.value()) wait(d);
            reg.put(data);
            update_counter();
            current_cnt = data;
            clk_remainder = 0.;
            return true;
        });
    regs->pwmcount.set_read_cb([this](const scc::sc_register<uint32_t> &reg, uint32_t &data,
                                      sc_core::sc_time d) -> bool {
        auto offset = regs->r_pwmcfg.pwmenalways || regs->r_pwmcfg.pwmenoneshot ? static_cast<int>(get_pulses(d)) : 0;
        data = current_cnt + offset;
        regs->r_pwmcount.pwmcount = data;
        return true;
    });
    regs->pwms.set_write_cb(
        [this](scc::sc_register<uint32_t> &reg, uint32_t data, sc_core::sc_time d) -> bool { return false; });
    regs->pwms.set_read_cb([this](const scc::sc_register<uint32_t> &reg, uint32_t &data, sc_core::sc_time d) -> bool {
        auto offset = regs->r_pwmcfg.pwmenalways || regs->r_pwmcfg.pwmenoneshot ? static_cast<int>(get_pulses(d)) : 0;
        auto cnt = current_cnt + offset;
        data = (cnt >> regs->r_pwmcfg.pwmscale) & 0xffff;
        regs->r_pwms.pwms = static_cast<uint16_t>(data);
        return true;
    });
    regs->pwmcmp0.set_write_cb(
        [this](const scc::sc_register<uint32_t> &reg, const uint32_t &data, sc_core::sc_time d) -> bool {
            reg.put(data);
            update_counter();
            return true;
        });
    regs->pwmcmp1.set_write_cb(
        [this](const scc::sc_register<uint32_t> &reg, const uint32_t &data, sc_core::sc_time d) -> bool {
            reg.put(data);
            update_counter();
            return true;
        });
    regs->pwmcmp2.set_write_cb(
        [this](const scc::sc_register<uint32_t> &reg, const uint32_t &data, sc_core::sc_time d) -> bool {
            reg.put(data);
            update_counter();
            return true;
        });
    regs->pwmcmp3.set_write_cb(
        [this](const scc::sc_register<uint32_t> &reg, const uint32_t &data, sc_core::sc_time d) -> bool {
            reg.put(data);
            update_counter();
            return true;
        });

    SC_METHOD(clock_cb);
    sensitive << clk_i;
    SC_METHOD(reset_cb);
    sensitive << rst_i;
    SC_METHOD(update_counter);
    sensitive << update_counter_evt;
    dont_initialize();
}

void pwm::clock_cb() {
    update_counter();
    clk = clk_i.read();
}

pwm::~pwm() = default;

void pwm::reset_cb() {
    if (rst_i.read()) {
        regs->reset_start();
    } else {
        regs->reset_stop();
    }
}

void pwm::update_counter() {
    auto now = sc_time_stamp();
    if (now == SC_ZERO_TIME) return;
    update_counter_evt.cancel();
    if (regs->r_pwmcfg.pwmenalways || regs->r_pwmcfg.pwmenoneshot) {
        std::array<bool, 4> pwmcmp_new_ip{false, false, false, false};
        auto dpulses = get_pulses(SC_ZERO_TIME);
        auto pulses = static_cast<int>(dpulses);
        clk_remainder += dpulses - pulses;
        if (clk_remainder > 1) {
            pulses++;
            clk_remainder -= 1.0;
        }
        if (reset_cnt) {
            current_cnt = 0;
            reset_cnt = false;
        } else if (last_enable)
            current_cnt += pulses;
        auto pwms = (current_cnt >> regs->r_pwmcfg.pwmscale) & 0xffff;
        auto next_trigger_time =
            (0xffff - pwms) * (1 << regs->r_pwmcfg.pwmscale) * clk; // next trigger based on wrap around
        if (pwms == 0xffff) {                                       // wrap around calculation
            reset_cnt = true;
            next_trigger_time = clk;
            regs->r_pwmcfg.pwmenoneshot = 0;
        }
        auto pwms0 = (regs->r_pwmcfg.pwmcmp0center && (pwms & 0x8000) == 1) ? pwms ^ 0xffff : pwms;
        if (pwms0 >= regs->r_pwmcmp0.pwmcmp0) {
            pwmcmp_new_ip[0] = true;
            regs->r_pwmcfg.pwmenoneshot = 0;
            if (regs->r_pwmcfg.pwmzerocmp) {
                reset_cnt = true;
                next_trigger_time = clk;
            }
        } else {
            pwmcmp_new_ip[0] = false;
            // TODO: add correct calculation for regs->r_pwmcfg.pwmcmpXcenter==1
            auto nt = (regs->r_pwmcmp0.pwmcmp0 - pwms0) * (1 << regs->r_pwmcfg.pwmscale) * clk;
            next_trigger_time = nt < next_trigger_time ? nt : next_trigger_time;
        }
        auto pwms1 = (regs->r_pwmcfg.pwmcmp0center && (pwms & 0x8000) == 1) ? pwms ^ 0xffff : pwms;
        if (pwms1 >= regs->r_pwmcmp1.pwmcmp0) {
            pwmcmp_new_ip[1] = true;
        } else {
            pwmcmp_new_ip[1] = false;
            // TODO: add correct calculation for regs->r_pwmcfg.pwmcmpXcenter==1
            auto nt = (regs->r_pwmcmp0.pwmcmp0 - pwms0) * (1 << regs->r_pwmcfg.pwmscale) * clk;
            next_trigger_time = nt < next_trigger_time ? nt : next_trigger_time;
        }
        auto pwms2 = (regs->r_pwmcfg.pwmcmp0center && (pwms & 0x8000) == 1) ? pwms ^ 0xffff : pwms;
        if (pwms2 >= regs->r_pwmcmp2.pwmcmp0) {
            pwmcmp_new_ip[2] = true;
        } else {
            pwmcmp_new_ip[2] = false;
            // TODO: add correct calculation for regs->r_pwmcfg.pwmcmpXcenter==1
            auto nt = (regs->r_pwmcmp0.pwmcmp0 - pwms0) * regs->r_pwmcfg.pwmscale * clk;
            next_trigger_time = nt < next_trigger_time ? nt : next_trigger_time;
        }
        auto pwms3 = (regs->r_pwmcfg.pwmcmp0center && (pwms & 0x8000) == 1) ? pwms ^ 0xffff : pwms;
        if (pwms3 >= regs->r_pwmcmp3.pwmcmp0) {
            pwmcmp_new_ip[3] = true;
        } else {
            pwmcmp_new_ip[3] = false;
            // TODO: add correct calculation for regs->r_pwmcfg.pwmcmpXcenter==1
            auto nt = (regs->r_pwmcmp0.pwmcmp0 - pwms0) * (1 << regs->r_pwmcfg.pwmscale) * clk;
            next_trigger_time = nt < next_trigger_time ? nt : next_trigger_time;
        }
        for (size_t i = 0; i < 4; ++i) {
            // write gpio bits depending of gang bit
            if (regs->r_pwmcfg & (1 < (24 + i)))
                write_cmpgpio(i, pwmcmp_new_ip[i] && !pwmcmp_new_ip[(i + 1) % 4]);
            else
                write_cmpgpio(i, pwmcmp_new_ip[i]);
            // detect rising edge and set ip bit if found
            if (!pwmcmp_ip[i] && pwmcmp_new_ip[i]) regs->r_pwmcfg |= 1 << (28 + i);
            pwmcmp_ip[i] = pwmcmp_new_ip[i];
        }
        last_enable = true;
        update_counter_evt.notify(next_trigger_time);
    } else
        last_enable = false;
    cmpip_o[0].write(regs->r_pwmcfg.pwmcmp0ip != 0);
    cmpip_o[1].write(regs->r_pwmcfg.pwmcmp1ip != 0);
    cmpip_o[2].write(regs->r_pwmcfg.pwmcmp2ip != 0);
    cmpip_o[3].write(regs->r_pwmcfg.pwmcmp3ip != 0);
    last_cnt_update = now;
    last_clk = clk;
}

void pwm::write_cmpgpio(size_t index, bool val) {
    if (cmpgpio_o[index].get_interface()) {
        tlm::tlm_phase phase(tlm::BEGIN_REQ);
        tlm::tlm_signal_gp<> gp;
        sc_core::sc_time delay(SC_ZERO_TIME);
        gp.set_value(val);
        cmpgpio_o[index]->nb_transport_fw(gp, phase, delay);
    }
}
