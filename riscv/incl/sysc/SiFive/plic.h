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

#ifndef _PLIC_H_
#define _PLIC_H_

#include <sysc/tlm_target.h>

namespace sysc {

class plic_regs;

class plic: public sc_core::sc_module, public tlm_target<> {
public:
    SC_HAS_PROCESS(plic);
    sc_core::sc_in<sc_core::sc_time> clk_i;
    sc_core::sc_in<bool>             rst_i;
    plic(sc_core::sc_module_name nm);
    virtual ~plic();
protected:
    void clock_cb();
    void reset_cb();
    sc_core::sc_time clk;
    std::unique_ptr<plic_regs> regs;
};

} /* namespace sysc */

#endif /* _PLIC_H_ */
