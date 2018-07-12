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
    system(sc_core::sc_module_name nm);
    virtual ~system();

    sc_core::sc_vector<tlm::tlm_signal<sc_dt::sc_logic>> s_gpio;

private:
    sysc::platform i_platform;
    sysc::terminal i_terminal;
};

}
#endif /* __SYSC_GENERAL_SYSTEM_H_ */
