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

#ifndef _SYSC_SC2CORE_IF_H_
#define _SYSC_SC2CORE_IF_H_

#include "iss/arch_if.h"
#include <iss/iss.h>
#include <iss/vm_types.h>
#include <scc/report.h>
#include <util/delegate.h>
#include <util/ities.h>

namespace sysc {
//! an adpater to call specific RISC-V core function without knowing the exact type.
struct sc2core_if {
    // this is needed since we want to call the destructor with a pointer-to-base
    virtual ~sc2core_if() = default;

    virtual void setup_mt() = 0;

    virtual void enable_disass(bool enable) = 0;
    util::delegate<void(unsigned)> set_hartid;
    util::delegate<void(unsigned)> set_irq_count;
    util::delegate<uint32_t()> get_mode;
    util::delegate<uint64_t()> get_state;
    util::delegate<bool()> get_interrupt_execution;
    util::delegate<void(bool)> set_interrupt_execution;
    util::delegate<void(short, bool)> local_irq; // id, value
    //! sets a callback for CSR read requests. The callback gets always 64bit data passed no matter what size the actual CSR has
    using rd_csr_f = std::function<iss::status(unsigned addr, uint64_t&)>;
    util::delegate<void(unsigned, rd_csr_f)> register_csr_rd;
    //! sets a callback for CSR write requests. The callback gets always 64bit data passed no matter what size the actual CSR has
    using wr_csr_f = std::function<iss::status(unsigned addr, uint64_t)>;
    util::delegate<void(unsigned, wr_csr_f)> register_csr_wr;
    virtual void register_unknown_instr_handler(util::delegate<iss::arch_if::unknown_instr_cb_t>) = 0;
};
} // namespace sysc
#endif /* _SYSC_SC2CORE_IF_H_ */
