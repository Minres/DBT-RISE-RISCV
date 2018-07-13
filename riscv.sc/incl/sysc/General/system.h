/*
 * system.h
 *
 *  Created on: 11.07.2018
 *      Author: eyck
 */

#ifndef __SYSC_GENERAL_SYSTEM_H_
#define __SYSC_GENERAL_SYSTEM_H_

#include "sysc/SiFive/platform.h"
#include "sysc/General/terminal.h"
#include <systemc>

namespace sysc {

class system: sc_core::sc_module {
public:
    SC_HAS_PROCESS(system);

    system(sc_core::sc_module_name nm);
    virtual ~system();

private:
    sc_core::sc_vector<tlm::tlm_signal<sc_dt::sc_logic>> s_gpio;
    sc_core::sc_signal<bool> s_rst_n;
    sysc::platform i_platform;
    sysc::terminal i_terminal;
    void gen_por();
};

}
#endif /* __SYSC_GENERAL_SYSTEM_H_ */
