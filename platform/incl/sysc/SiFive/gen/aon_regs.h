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

#ifndef _AON_REGS_H_
#define _AON_REGS_H_

#include <scc/register.h>
#include <scc/tlm_target.h>
#include <scc/utilities.h>
#include <util/bit_field.h>

namespace sysc {

class aon_regs : public sc_core::sc_module, public scc::resetable {
public:
    // storage declarations
    uint32_t r_wdogcfg;

    uint32_t r_wdogcount;

    uint32_t r_wdogs;

    uint32_t r_wdogfeed;

    uint32_t r_wdogkey;

    uint32_t r_wdogcmp;

    uint32_t r_rtccfg;

    uint32_t r_rtclo;

    uint32_t r_rtchi;

    uint32_t r_rtcs;

    uint32_t r_rtccmp;

    uint32_t r_lfrosccfg;

    std::array<uint32_t, 32> r_backup;

    BEGIN_BF_DECL(pmuwakeupi_t, uint32_t);
    BF_FIELD(delay, 0, 4);
    BF_FIELD(vddpaden, 5, 1);
    BF_FIELD(corerst, 7, 1);
    BF_FIELD(hfclkrst, 8, 1);
    END_BF_DECL();
    std::array<pmuwakeupi_t, 8> r_pmuwakeupi;

    BEGIN_BF_DECL(pmusleepi_t, uint32_t);
    BF_FIELD(delay, 0, 4);
    BF_FIELD(vddpaden, 5, 1);
    BF_FIELD(corerst, 7, 1);
    BF_FIELD(hfclkrst, 8, 1);
    END_BF_DECL();
    std::array<pmusleepi_t, 8> r_pmusleepi;

    uint32_t r_pmuie;

    uint32_t r_pmucause;

    uint32_t r_pmusleep;

    uint32_t r_pmukey;

    // register declarations
    scc::sc_register<uint32_t> wdogcfg;
    scc::sc_register<uint32_t> wdogcount;
    scc::sc_register<uint32_t> wdogs;
    scc::sc_register<uint32_t> wdogfeed;
    scc::sc_register<uint32_t> wdogkey;
    scc::sc_register<uint32_t> wdogcmp;
    scc::sc_register<uint32_t> rtccfg;
    scc::sc_register<uint32_t> rtclo;
    scc::sc_register<uint32_t> rtchi;
    scc::sc_register<uint32_t> rtcs;
    scc::sc_register<uint32_t> rtccmp;
    scc::sc_register<uint32_t> lfrosccfg;
    scc::sc_register_indexed<uint32_t, 32> backup;
    scc::sc_register_indexed<pmuwakeupi_t, 8> pmuwakeupi;
    scc::sc_register_indexed<pmusleepi_t, 8> pmusleepi;
    scc::sc_register<uint32_t> pmuie;
    scc::sc_register<uint32_t> pmucause;
    scc::sc_register<uint32_t> pmusleep;
    scc::sc_register<uint32_t> pmukey;

    aon_regs(sc_core::sc_module_name nm);

    template <unsigned BUSWIDTH = 32> void registerResources(scc::tlm_target<BUSWIDTH> &target);
};
}
//////////////////////////////////////////////////////////////////////////////
// member functions
//////////////////////////////////////////////////////////////////////////////

inline sysc::aon_regs::aon_regs(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, NAMED(wdogcfg, r_wdogcfg, 0, *this)
, NAMED(wdogcount, r_wdogcount, 0, *this)
, NAMED(wdogs, r_wdogs, 0, *this)
, NAMED(wdogfeed, r_wdogfeed, 0, *this)
, NAMED(wdogkey, r_wdogkey, 0, *this)
, NAMED(wdogcmp, r_wdogcmp, 0, *this)
, NAMED(rtccfg, r_rtccfg, 0, *this)
, NAMED(rtclo, r_rtclo, 0, *this)
, NAMED(rtchi, r_rtchi, 0, *this)
, NAMED(rtcs, r_rtcs, 0, *this)
, NAMED(rtccmp, r_rtccmp, 0, *this)
, NAMED(lfrosccfg, r_lfrosccfg, 0, *this)
, NAMED(backup, r_backup, 0, *this)
, NAMED(pmuwakeupi, r_pmuwakeupi, 0, *this)
, NAMED(pmusleepi, r_pmusleepi, 0, *this)
, NAMED(pmuie, r_pmuie, 0, *this)
, NAMED(pmucause, r_pmucause, 0, *this)
, NAMED(pmusleep, r_pmusleep, 0, *this)
, NAMED(pmukey, r_pmukey, 0, *this) {}

template <unsigned BUSWIDTH> inline void sysc::aon_regs::registerResources(scc::tlm_target<BUSWIDTH> &target) {
    target.addResource(wdogcfg, 0x0UL);
    target.addResource(wdogcount, 0x8UL);
    target.addResource(wdogs, 0x10UL);
    target.addResource(wdogfeed, 0x18UL);
    target.addResource(wdogkey, 0x1cUL);
    target.addResource(wdogcmp, 0x20UL);
    target.addResource(rtccfg, 0x40UL);
    target.addResource(rtclo, 0x48UL);
    target.addResource(rtchi, 0x4cUL);
    target.addResource(rtcs, 0x50UL);
    target.addResource(rtccmp, 0x60UL);
    target.addResource(lfrosccfg, 0x70UL);
    target.addResource(backup, 0x80UL);
    target.addResource(pmuwakeupi, 0x100UL);
    target.addResource(pmusleepi, 0x120UL);
    target.addResource(pmuie, 0x140UL);
    target.addResource(pmucause, 0x144UL);
    target.addResource(pmusleep, 0x148UL);
    target.addResource(pmukey, 0x14cUL);
}

#endif // _AON_REGS_H_
