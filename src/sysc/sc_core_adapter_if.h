/*******************************************************************************
 * Copyright (C) 2023 MINRES Technologies GmbH
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
 * Contributors:
 *       eyck@minres.com - initial implementation
 ******************************************************************************/

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
    virtual void set_irq_num(unsigned) = 0;
    virtual uint32_t get_mode() = 0;
    virtual uint64_t get_state() = 0;
    virtual bool get_interrupt_execution() = 0;
    virtual void set_interrupt_execution(bool v) = 0;
    virtual void local_irq(short id, bool value) = 0;
    virtual ~sc_core_adapter_if() = default;
};
} // namespace sysc
#endif /* _SYSC_SC_CORE_ADAPTER_IF_H_ */
