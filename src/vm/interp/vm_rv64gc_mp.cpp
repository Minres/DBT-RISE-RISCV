// rv64gc_mp:interp — rv64gc with PMP enforcement.
// Kept separate from the generated vm_rv64gc.cpp to survive template re-generation.
#include <array>
#include <cstdint>
#include <iss/arch/riscv_hart_m_p.h>
#include <iss/arch/rv64gc.h>
#include <iss/factory.h>
#include <iss/mem/pmp.h>
#include <iss/semihosting/semihosting.h>

namespace iss {

// rv64gc_mp_hart inserts pmp between the hart and default_mem.
// The base constructor calls set_next(default_mem) via memories.append(), but
// virtual dispatch during base construction routes to the base implementation,
// so the rewiring must be done explicitly in the derived constructor body once
// all members (including pmp_obj) are fully initialized.
struct rv64gc_mp_hart : public arch::riscv_hart_m_p<arch::rv64gc> {
    mem::pmp<arch::rv64gc> pmp_obj{this->get_priv_if()};

    rv64gc_mp_hart() {
        pmp_obj.set_next(this->default_mem.get_mem_if());
        memory = pmp_obj.get_mem_if();
    }
};

namespace {

volatile std::array<bool, 1> rv64gc_mp_dummy = {
    core_factory::instance().register_creator("rv64gc_mp:interp",
        [](unsigned port, void* init_data) -> std::tuple<cpu_ptr, vm_ptr> {
            auto* cpu = new rv64gc_mp_hart();
            if (init_data) {
                auto* cb = reinterpret_cast<semihosting_cb_t<arch::traits<arch::rv64gc>::reg_t>*>(init_data);
                cpu->set_semihosting_callback(*cb);
            }
            return {cpu_ptr{cpu}, vm_ptr{iss::interp::create<arch::rv64gc>(cpu, port, false)}};
        })
};

} // namespace
} // namespace iss
