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

#ifndef _PRCI_H_
#define _PRCI_H_

#include "scc/tlm_target.h"

namespace sysc {

class prci_regs;

class prci : public sc_core::sc_module, public scc::tlm_target<> {
public:
    SC_HAS_PROCESS(prci);
    sc_core::sc_in<sc_core::sc_time> clk_i;
    sc_core::sc_in<bool> rst_i;
    prci(sc_core::sc_module_name nm);
    virtual ~prci() override; // need to keep it in source file because of fwd declaration of prci_regs

protected:
    void clock_cb();
    void reset_cb();
    void hfrosc_en_cb();
    sc_core::sc_time clk;
    std::unique_ptr<prci_regs> regs;
    sc_core::sc_event hfrosc_en_evt;
};

} /* namespace sysc */

#endif /* _GPIO_H_ */
