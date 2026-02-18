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
 *******************************************************************************/

// clang-format off
#include <iss/arch/rv32i.h>
#include <iss/arch/rv32imac.h>
#include <iss/arch/rv32gc.h>
#include <iss/arch/rv64i.h>
#include <iss/arch/rv64gc.h>
#include <iss/arch/tgc5c.h>
#include <iss/arch/riscv_hart_m_p.h>
#include <iss/arch/riscv_hart_mu_p.h>
#include <iss/arch/riscv_hart_msu_vp.h>
#include <iss/mem/pmp.h>
#include "iss_factory.h"
#include "core2sc_adapter.h"
#include <array>
// clang-format on

namespace iss {
namespace interp {
using namespace sysc;
__attribute__((used)) volatile std::array<bool, 15> riscv_init = {
    iss_factory::instance().register_creator("rv32i_m:interp",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_m_p<arch::rv32i>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32i*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32i_mu:interp",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_m_p<arch::rv32i>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32i*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32imac_m:interp",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_m_p<arch::rv32imac>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32imac*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32imac_mp:interp", // rv32imac_m with PMP
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_m_p<arch::rv32imac>>(cc);
                                                 cpu->memories.insert_before_last(
                                                     std::make_unique<iss::mem::pmp<iss::arch::rv32imac>>(cpu->get_priv_if()));
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32imac*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32imac_mu:interp",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_mu_p<arch::rv32imac>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32imac*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32gc_m:interp",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_m_p<arch::rv32gc>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32gc_mu:interp",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_mu_p<arch::rv32gc>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32gc_msu:interp",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_msu_vp<arch::rv32gc>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv64i_m:interp",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_m_p<arch::rv64i>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv64i*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv64i_mu:interp",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_m_p<arch::rv64i>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv64i*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv64gc_m:interp",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_m_p<arch::rv64gc>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv64gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv64gc_mu:interp",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_mu_p<arch::rv64gc>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv64gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv64gc_msu:interp",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_msu_vp<arch::rv64gc>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv64gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("tgc5c_m:interp",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_m_p<arch::tgc5c>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::tgc5c*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("tgc5c_mu:interp",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_mu_p<arch::tgc5c>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::tgc5c*>(cpu), gdb_port)}};
                                             })}; // namespace interp
} // namespace interp
#if defined(WITH_LLVM)
namespace llvm {
using namespace sysc;
volatile std::array<bool, 4> riscv_init = {
    iss_factory::instance().register_creator("rv32imac_m:llvm",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_m_p<arch::rv32imc>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32imac*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32imac_mu:llvm",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_mu_p<arch::rv32imc>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32imac*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("tgc5c_m:llvm",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_m_p<arch::tgc5c>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::tgc5c*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("tgc5c_mu:llvm",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_mu_p<arch::tgc5c>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::tgc5c*>(cpu), gdb_port)}};
                                             })};
} // namespace llvm
#endif
#if defined(WITH_ASMJIT)
namespace asmjit {
using namespace sysc;
volatile std::array<bool, 14> riscv_init = {
    iss_factory::instance().register_creator("rv32i_m:asmjit",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_m_p<arch::rv32i>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32i*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32i_mu:asmjit",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_mu_p<arch::rv32i>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32i*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32imac_m:asmjit",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_m_p<arch::rv32imac>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32imac*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32imac_mu:asmjit",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_mu_p<arch::rv32imac>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32imac*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32gc_m:asmjit",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_m_p<arch::rv32gc>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32gc_mu:asmjit",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_mu_p<arch::rv32gc>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32gc_msu:asmjit",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_msu_vp<arch::rv32gc>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv64i_m:asmjit",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_m_p<arch::rv64i>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv64i*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv64i_mu:asmjit",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_mu_p<arch::rv64i>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv64i*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv64gc_m:asmjit",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_m_p<arch::rv64gc>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv64gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv64gc_mu:asmjit",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_mu_p<arch::rv64gc>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv64gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv64gc_msu:asmjit",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_msu_vp<arch::rv64gc>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv64gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("tgc5c_m:asmjit",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_m_p<arch::tgc5c>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::tgc5c*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("tgc5c_mu:asmjit",
                                             [](unsigned gdb_port, sysc::riscv::core_complex_if* cc) -> iss_factory::base_t {
                                                 auto* cpu = new core2sc_adapter<arch::riscv_hart_mu_p<arch::tgc5c>>(cc);
                                                 return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::tgc5c*>(cpu), gdb_port)}};
                                             })};
} // namespace asmjit
#endif
} // namespace iss
