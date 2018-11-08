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

#ifndef _PRCI_REGS_H_
#define _PRCI_REGS_H_

#include <scc/register.h>
#include <scc/tlm_target.h>
#include <scc/utilities.h>
#include <util/bit_field.h>

namespace sysc {

class prci_regs : public sc_core::sc_module, public scc::resetable {
public:
    // storage declarations
    BEGIN_BF_DECL(hfrosccfg_t, uint32_t);
    BF_FIELD(hfroscdiv, 0, 6);
    BF_FIELD(hfrosctrim, 16, 5);
    BF_FIELD(hfroscen, 30, 1);
    BF_FIELD(hfroscrdy, 31, 1);
    END_BF_DECL() r_hfrosccfg;

    BEGIN_BF_DECL(hfxosccfg_t, uint32_t);
    BF_FIELD(hfxoscrdy, 31, 1);
    BF_FIELD(hfxoscen, 30, 1);
    END_BF_DECL() r_hfxosccfg;

    BEGIN_BF_DECL(pllcfg_t, uint32_t);
    BF_FIELD(pllr, 0, 3);
    BF_FIELD(pllf, 4, 6);
    BF_FIELD(pllq, 10, 2);
    BF_FIELD(pllsel, 16, 1);
    BF_FIELD(pllrefsel, 17, 1);
    BF_FIELD(pllbypass, 18, 1);
    BF_FIELD(plllock, 31, 1);
    END_BF_DECL() r_pllcfg;

    uint32_t r_plloutdiv;

    uint32_t r_coreclkcfg;

    // register declarations
    scc::sc_register<hfrosccfg_t> hfrosccfg;
    scc::sc_register<hfxosccfg_t> hfxosccfg;
    scc::sc_register<pllcfg_t> pllcfg;
    scc::sc_register<uint32_t> plloutdiv;
    scc::sc_register<uint32_t> coreclkcfg;

    prci_regs(sc_core::sc_module_name nm);

    template <unsigned BUSWIDTH = 32> void registerResources(scc::tlm_target<BUSWIDTH> &target);
};
}
//////////////////////////////////////////////////////////////////////////////
// member functions
//////////////////////////////////////////////////////////////////////////////

inline sysc::prci_regs::prci_regs(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, NAMED(hfrosccfg, r_hfrosccfg, 0, *this)
, NAMED(hfxosccfg, r_hfxosccfg, 0x40000000, *this)
, NAMED(pllcfg, r_pllcfg, 0, *this)
, NAMED(plloutdiv, r_plloutdiv, 0, *this)
, NAMED(coreclkcfg, r_coreclkcfg, 0, *this) {}

template <unsigned BUSWIDTH> inline void sysc::prci_regs::registerResources(scc::tlm_target<BUSWIDTH> &target) {
    target.addResource(hfrosccfg, 0x0UL);
    target.addResource(hfxosccfg, 0x4UL);
    target.addResource(pllcfg, 0x8UL);
    target.addResource(plloutdiv, 0xcUL);
    target.addResource(coreclkcfg, 0x10UL);
}

#endif // _PRCI_REGS_H_
