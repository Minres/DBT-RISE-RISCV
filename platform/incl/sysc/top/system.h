/*
 * system.h
 *
 *  Created on: 11.07.2018
 *      Author: eyck
 */

#ifndef __SYSC_GENERAL_SYSTEM_H_
#define __SYSC_GENERAL_SYSTEM_H_

#include <systemc>
#include "sysc/SiFive/hifive1.h"
#include "mcp3008.h"
#include "terminal.h"
#include "h_bridge.h"
#include "dcmotor.h"

namespace sysc {

class system: sc_core::sc_module {
public:
    SC_HAS_PROCESS(system);

    system(sc_core::sc_module_name nm);
    virtual ~system();


private:
    sc_core::sc_vector<tlm::tlm_signal<sc_dt::sc_logic>> s_gpio;
    sc_core::sc_signal<bool> s_rst_n;
    sc_core::sc_signal<double> s_vref, s_va, s_vb, s_vc, s_vasens, s_vbsens, s_vcsens;
    sc_core::sc_vector<sc_core::sc_signal<double>> s_ana;
    sysc::hifive1 i_platform;
    sysc::terminal i_terminal;
    sysc::mcp3008 i_adc;
    sysc::h_bridge i_h_bridge;
    sysc::dc_motor i_motor;
    void gen_por();
};

}
#endif /* __SYSC_GENERAL_SYSTEM_H_ */
