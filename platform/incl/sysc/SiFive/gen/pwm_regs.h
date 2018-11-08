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

#ifndef _PWM_REGS_H_
#define _PWM_REGS_H_

#include <scc/register.h>
#include <scc/tlm_target.h>
#include <scc/utilities.h>
#include <util/bit_field.h>

namespace sysc {

class pwm_regs : public sc_core::sc_module, public scc::resetable {
public:
    // storage declarations
    BEGIN_BF_DECL(pwmcfg_t, uint32_t);
    BF_FIELD(pwmscale, 0, 4);
    BF_FIELD(pwmsticky, 8, 1);
    BF_FIELD(pwmzerocmp, 9, 1);
    BF_FIELD(pwmdeglitch, 10, 1);
    BF_FIELD(pwmenalways, 12, 1);
    BF_FIELD(pwmenoneshot, 13, 1);
    BF_FIELD(pwmcmp0center, 16, 1);
    BF_FIELD(pwmcmp1center, 17, 1);
    BF_FIELD(pwmcmp2center, 18, 1);
    BF_FIELD(pwmcmp3center, 19, 1);
    BF_FIELD(pwmcmp0gang, 24, 1);
    BF_FIELD(pwmcmp1gang, 25, 1);
    BF_FIELD(pwmcmp2gang, 26, 1);
    BF_FIELD(pwmcmp3gang, 27, 1);
    BF_FIELD(pwmcmp0ip, 28, 1);
    BF_FIELD(pwmcmp1ip, 29, 1);
    BF_FIELD(pwmcmp2ip, 30, 1);
    BF_FIELD(pwmcmp3ip, 31, 1);
    END_BF_DECL() r_pwmcfg;

    BEGIN_BF_DECL(pwmcount_t, uint32_t);
    BF_FIELD(pwmcount, 0, 31);
    END_BF_DECL() r_pwmcount;

    BEGIN_BF_DECL(pwms_t, uint32_t);
    BF_FIELD(pwms, 0, 16);
    END_BF_DECL() r_pwms;

    BEGIN_BF_DECL(pwmcmp0_t, uint32_t);
    BF_FIELD(pwmcmp0, 0, 16);
    END_BF_DECL() r_pwmcmp0;

    BEGIN_BF_DECL(pwmcmp1_t, uint32_t);
    BF_FIELD(pwmcmp0, 0, 16);
    END_BF_DECL() r_pwmcmp1;

    BEGIN_BF_DECL(pwmcmp2_t, uint32_t);
    BF_FIELD(pwmcmp0, 0, 16);
    END_BF_DECL() r_pwmcmp2;

    BEGIN_BF_DECL(pwmcmp3_t, uint32_t);
    BF_FIELD(pwmcmp0, 0, 16);
    END_BF_DECL() r_pwmcmp3;

    // register declarations
    scc::sc_register<pwmcfg_t> pwmcfg;
    scc::sc_register<pwmcount_t> pwmcount;
    scc::sc_register<pwms_t> pwms;
    scc::sc_register<pwmcmp0_t> pwmcmp0;
    scc::sc_register<pwmcmp1_t> pwmcmp1;
    scc::sc_register<pwmcmp2_t> pwmcmp2;
    scc::sc_register<pwmcmp3_t> pwmcmp3;

    pwm_regs(sc_core::sc_module_name nm);

    template <unsigned BUSWIDTH = 32> void registerResources(scc::tlm_target<BUSWIDTH> &target);
};
}
//////////////////////////////////////////////////////////////////////////////
// member functions
//////////////////////////////////////////////////////////////////////////////

inline sysc::pwm_regs::pwm_regs(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, NAMED(pwmcfg, r_pwmcfg, 0, *this)
, NAMED(pwmcount, r_pwmcount, 0, *this)
, NAMED(pwms, r_pwms, 0, *this)
, NAMED(pwmcmp0, r_pwmcmp0, 0, *this)
, NAMED(pwmcmp1, r_pwmcmp1, 0, *this)
, NAMED(pwmcmp2, r_pwmcmp2, 0, *this)
, NAMED(pwmcmp3, r_pwmcmp3, 0, *this) {}

template <unsigned BUSWIDTH> inline void sysc::pwm_regs::registerResources(scc::tlm_target<BUSWIDTH> &target) {
    target.addResource(pwmcfg, 0x0UL);
    target.addResource(pwmcount, 0x8UL);
    target.addResource(pwms, 0x10UL);
    target.addResource(pwmcmp0, 0x20UL);
    target.addResource(pwmcmp1, 0x24UL);
    target.addResource(pwmcmp2, 0x28UL);
    target.addResource(pwmcmp3, 0x2cUL);
}

#endif // _PWM_REGS_H_
