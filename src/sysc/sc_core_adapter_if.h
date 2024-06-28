/*
 * sc_core_adapter.h
 *
 *  Created on: Jul 5, 2023
 *      Author: eyck
 */

#ifndef _SYSC_SC_CORE_ADAPTER_IF_H_
#define _SYSC_SC_CORE_ADAPTER_IF_H_

#include "core_complex.h"
#include <iostream>
#include <iss/iss.h>
#include <iss/vm_types.h>
#include <scc/report.h>
#include <util/ities.h>

namespace sysc {
struct sc_core_adapter_if {
    virtual iss::arch_if* get_arch_if() = 0;
    virtual void set_mhartid(unsigned) = 0;
    virtual uint32_t get_mode() = 0;
    virtual uint64_t get_state() = 0;
    virtual bool get_interrupt_execution() = 0;
    virtual void set_interrupt_execution(bool v) = 0;
    virtual void local_irq(short id, bool value) = 0;
    virtual ~sc_core_adapter_if() = default;
};
} // namespace sysc
#endif /* _SYSC_SC_CORE_ADAPTER_IF_H_ */
