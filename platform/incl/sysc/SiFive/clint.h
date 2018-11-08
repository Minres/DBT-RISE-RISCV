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

#ifndef _CLINT_H_
#define _CLINT_H_

#include "scc/tlm_target.h"

namespace iss {
namespace arch {
template <typename BASE> class riscv_hart_msu_vp;
}
}

namespace sysc {

class clint_regs;
namespace SiFive {
class core_complex;
}

class clint : public sc_core::sc_module, public scc::tlm_target<> {
public:
    SC_HAS_PROCESS(clint);// NOLINT
    sc_core::sc_in<sc_core::sc_time> tlclk_i;
    sc_core::sc_in<sc_core::sc_time> lfclk_i;
    sc_core::sc_in<bool> rst_i;
    sc_core::sc_out<bool> mtime_int_o;
    sc_core::sc_out<bool> msip_int_o;
    clint(sc_core::sc_module_name nm);
    virtual ~clint() override; // NOLINT // need to keep it in source file because of fwd declaration of clint_regs

protected:
    void clock_cb();
    void reset_cb();
    void update_mtime();
    sc_core::sc_time clk, last_updt;
    unsigned cnt_fraction;
    std::unique_ptr<clint_regs> regs;
    sc_core::sc_event mtime_evt;
};

} /* namespace sysc */

#endif /* _CLINT_H_ */
