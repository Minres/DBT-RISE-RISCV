/*******************************************************************************
 * Copyright (C) 2022 MINRES Technologies GmbH
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

#ifndef _RISCV_HART_M_P_HWL_H
#define _RISCV_HART_M_P_HWL_H

#include "riscv_hart_common.h"
#include <iss/vm_types.h>

namespace iss {
namespace arch {

template <typename BASE> class hwl : public BASE {
public:
    using base_class = BASE;
    using this_class = hwl<BASE>;
    using reg_t = typename BASE::reg_t;

    hwl(feature_config cfg = feature_config{});
    virtual ~hwl() = default;

protected:
    iss::status read_custom_csr(unsigned addr, reg_t& val) override;
    iss::status write_custom_csr(unsigned addr, reg_t val) override;
};

template <typename BASE>
inline hwl<BASE>::hwl(feature_config cfg)
: BASE(cfg) {
    for(unsigned addr = 0x800; addr < 0x803; ++addr) {
        this->register_custom_csr_rd(addr);
        this->register_custom_csr_wr(addr);
    }
    for(unsigned addr = 0x804; addr < 0x807; ++addr) {
        this->register_custom_csr_rd(addr);
        this->register_custom_csr_wr(addr);
    }
}

template <typename BASE> inline iss::status iss::arch::hwl<BASE>::read_custom_csr(unsigned addr, reg_t& val) {
    switch(addr) {
    case 0x800:
        val = this->reg.lpstart0;
        break;
    case 0x801:
        val = this->reg.lpend0;
        break;
    case 0x802:
        val = this->reg.lpcount0;
        break;
    case 0x804:
        val = this->reg.lpstart1;
        break;
    case 0x805:
        val = this->reg.lpend1;
        break;
    case 0x806:
        val = this->reg.lpcount1;
        break;
    }
    return iss::Ok;
}

template <typename BASE> inline iss::status iss::arch::hwl<BASE>::write_custom_csr(unsigned addr, reg_t val) {
    switch(addr) {
    case 0x800:
        this->reg.lpstart0 = val;
        break;
    case 0x801:
        this->reg.lpend0 = val;
        break;
    case 0x802:
        this->reg.lpcount0 = val;
        break;
    case 0x804:
        this->reg.lpstart1 = val;
        break;
    case 0x805:
        this->reg.lpend1 = val;
        break;
    case 0x806:
        this->reg.lpcount1 = val;
        break;
    }
    return iss::Ok;
}

} // namespace arch
} // namespace iss

#endif /* _RISCV_HART_M_P_H */
