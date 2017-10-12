/*******************************************************************************
 * Copyright 2017 eyck@minres.com
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
 * License for the specific language governing permissions and limitations under
 * the License.
 ******************************************************************************/

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
    SC_HAS_PROCESS(clint);
    sc_core::sc_in<sc_core::sc_time> clk_i;
    sc_core::sc_in<bool> rst_i;
    sc_core::sc_out<bool> mtime_int_o;
    sc_core::sc_out<bool> msip_int_o;
    clint(sc_core::sc_module_name nm);
    virtual ~clint() override; // need to keep it in source file because of fwd declaration of clint_regs

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
