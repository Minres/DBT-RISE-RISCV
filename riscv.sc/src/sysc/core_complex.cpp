////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, MINRES Technologies GmbH
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Contributors:
//       eyck@minres.com - initial implementation
//
//
////////////////////////////////////////////////////////////////////////////////

#include "scc/report.h"
#include "iss/arch/riscv_hart_msu_vp.h"
#include "iss/arch/rv32imac.h"
#include "iss/iss.h"
#include "iss/vm_types.h"
#include "iss/debugger/server.h"
#include "iss/debugger/gdb_session.h"
#include "iss/debugger/target_adapter_if.h"
#include "iss/debugger/encoderdecoder.h"
#include "sysc/SiFive/core_complex.h"

#ifdef WITH_SCV
#include <scv.h>
#include <array>
#endif


namespace sysc {
namespace SiFive {
using namespace std;
using namespace iss;

namespace {
iss::debugger::encoder_decoder encdec;

}


namespace {

std::array<const char, 4> lvl = { { 'U', 'S', 'H', 'M' } };

std::array<const char*, 16> trap_str = { {
		"Instruction address misaligned",
		"Instruction access fault",
		"Illegal instruction",
		"Breakpoint",
		"Load address misaligned",
		"Load access fault",
		"Store/AMO address misaligned",
		"Store/AMO access fault",
		"Environment call from U-mode",
		"Environment call from S-mode",
		"Reserved",
		"Environment call from M-mode",
		"Instruction page fault",
		"Load page fault",
		"Reserved",
		"Store/AMO page fault"
} };
std::array<const char*, 12> irq_str = { {
		"User software interrupt", "Supervisor software interrupt", "Reserved", "Machine software interrupt",
		"User timer interrupt",    "Supervisor timer interrupt",    "Reserved", "Machine timer interrupt",
		"User external interrupt", "Supervisor external interrupt", "Reserved", "Machine external interrupt" } };
}

class core_wrapper : public iss::arch::riscv_hart_msu_vp<iss::arch::rv32imac> {
public:
    using core_type = arch::rv32imac;
    using base_type = arch::riscv_hart_msu_vp<arch::rv32imac>;
    using phys_addr_t = typename arch::traits<arch::rv32imac>::phys_addr_t;
    core_wrapper(core_complex *owner)
    : owner(owner)
    {}

    uint32_t get_mode(){ return this->reg.machine_state; }

    base_type::hart_state<base_type::reg_t>& get_state() { return this->state; }

    void notify_phase(exec_phase p) override {
        if(p == ISTART) owner->sync(this->reg.icount+cycle_offset);
    }

    sync_type needed_sync() const override { return PRE_SYNC; }

    void disass_output(uint64_t pc, const std::string instr) override {
    	if (logging::INFO <= logging::Log<logging::Output2FILE<logging::disass>>::reporting_level() && logging::Output2FILE<logging::disass>::stream()){
			std::stringstream s;
			s << "[p:" << lvl[this->reg.machine_state] << ";s:0x" << std::hex << std::setfill('0')
			  << std::setw(sizeof(reg_t) * 2) << (reg_t)state.mstatus << std::dec << ";c:" << this->reg.icount << "]";
			scc::Log<logging::Output2FILE<logging::disass>>().get(logging::INFO, "disass")
					<< "0x"<<std::setw(16)<<std::setfill('0')<<std::hex<<pc<<"\t\t"<<std::setw(40)<<std::setfill(' ')<<std::left<<instr<<s.str();
    	}
        owner->disass_output(pc,instr);
    };

    status read_mem(phys_addr_t addr, unsigned length, uint8_t *const data) {
    	if (addr.access && access_type::DEBUG)
    		return owner->read_mem_dbg(addr.val, length, data) ? Ok : Err;
    	else {
    		return owner->read_mem(addr.val, length, data,addr.access && access_type::FETCH) ? Ok : Err;
    	}
    }

    status write_mem(phys_addr_t addr, unsigned length, const uint8_t *const data) {
    	if (addr.access && access_type::DEBUG)
    		return owner->write_mem_dbg(addr.val, length, data) ? Ok : Err;
    	else{
    		auto res = owner->write_mem(addr.val, length, data) ? Ok : Err;
    		// clear MTIP on mtimecmp write
    		if(addr.val==0x2004000){
    		    reg_t val;
    		    this->read_csr(arch::mip, val);
    			this->write_csr(arch::mip, val & ~(1ULL<<7));
    		}
    		return res;
    	}
    }

    void wait_until(uint64_t flags) {
    	do{
    			wait(wfi_evt);
    			this->check_interrupt();
    	} while(this->reg.pending_trap==0);
    	base_type::wait_until(flags);
    }

    void local_irq(short id){
    	switch(id){
    	case 16: // SW
    		this->csr[arch::mip] |= 1<<3;
    		break;
    	case 17: // timer
    		this->csr[arch::mip] |= 1<<7;
    		break;
    	case 18: //external
    		this->csr[arch::mip] |= 1<<11;
    		break;
    	default:
    		/* do nothing*/
    		break;
    	}
    	wfi_evt.notify();
    }
private:
    core_complex *const owner;
    sc_event wfi_evt;
};

int cmd_sysc(int argc, char* argv[], debugger::out_func of, debugger::data_func df, debugger::target_adapter_if* tgt_adapter){
    if(argc>1) {
        if(strcasecmp(argv[1], "print_time")==0){
            std::string t = sc_core::sc_time_stamp().to_string();
            of(t.c_str());
			std::array<char, 64> buf;
			encdec.enc_string(t.c_str(), buf.data(), 63);
			df(buf.data());
            return Ok;
        } else if(strcasecmp(argv[1], "break")==0){
            sc_core::sc_time t;
            if(argc==4){
                 t= scc::parse_from_string(argv[2], argv[3]);
            } else if(argc==3){
                t= scc::parse_from_string(argv[2]);
            } else
                return Err;
            // no check needed as it is only called if debug server is active
            tgt_adapter->add_break_condition([t]()->unsigned{
                LOG(TRACE)<<"Checking condition at "<<sc_core::sc_time_stamp();
                return sc_core::sc_time_stamp()>=t?std::numeric_limits<unsigned>::max():0;
            });
            return Ok;
        }
        return Err;
    }
    return Err;

}

core_complex::core_complex(sc_core::sc_module_name name)
: sc_core::sc_module(name)
, NAMED(initiator)
, NAMED(clk_i)
, NAMED(rst_i)
, NAMED(global_irq_i)
, NAMED(timer_irq_i)
, NAMED(local_irq_i, 16)
, NAMED(elf_file, "")
, NAMED(enable_disass, true)
, NAMED(reset_address, 0ULL)
, NAMED(gdb_server_port, 0)
, NAMED(dump_ir, false)
, read_lut(tlm_dmi_ext())
, write_lut(tlm_dmi_ext())
, tgt_adapter(nullptr)
#ifdef WITH_SCV
, m_db(scv_tr_db::get_default_db())
, stream_handle(nullptr)
, instr_tr_handle(nullptr)
, fetch_tr_handle(nullptr)
#endif
{

    initiator.register_invalidate_direct_mem_ptr([=](uint64_t start, uint64_t end) -> void {
        auto lut_entry = read_lut.getEntry(start);
        if (lut_entry.get_granted_access() != tlm::tlm_dmi::DMI_ACCESS_NONE && end <= lut_entry.get_end_address() + 1) {
            read_lut.removeEntry(lut_entry);
        }
        lut_entry = write_lut.getEntry(start);
        if (lut_entry.get_granted_access() != tlm::tlm_dmi::DMI_ACCESS_NONE && end <= lut_entry.get_end_address() + 1) {
            write_lut.removeEntry(lut_entry);
        }
    });

    SC_THREAD(run);
    SC_METHOD(clk_cb);
    sensitive << clk_i;
    SC_METHOD(sw_irq_cb);
    sensitive<<sw_irq_i;
    SC_METHOD(timer_irq_cb);
    sensitive<<timer_irq_i;
    SC_METHOD(global_irq_cb);
    sensitive<<global_irq_i;
}

core_complex::~core_complex() = default;

void core_complex::trace(sc_core::sc_trace_file *trf) {}

void core_complex::before_end_of_elaboration() {
    cpu = make_unique<core_wrapper>(this);
    vm = create<arch::rv32imac>(cpu.get(), gdb_server_port.get_value(), dump_ir.get_value());
    vm->setDisassEnabled(enable_disass.get_value());
    auto* srv = debugger::server<debugger::gdb_session>::get();
    if(srv) tgt_adapter = srv->get_target();
    if(tgt_adapter)
        tgt_adapter->add_custom_command({
        "sysc",
        [this](int argc, char* argv[], debugger::out_func of, debugger::data_func df)-> int {
            return cmd_sysc(argc, argv, of, df, tgt_adapter);
        },
        "SystemC sub-commands: break <time>, print_time"});
}

void core_complex::start_of_simulation() {
    quantum_keeper.reset();
    if (elf_file.get_value().size() > 0){
    	std::pair<uint64_t,bool> start_addr=cpu->load_file(elf_file.get_value());
    	if(reset_address.is_default_value() && start_addr.second==true) reset_address.set_value(start_addr.first);
    }
#ifdef WITH_SCV
        if (m_db!=nullptr && stream_handle == nullptr) {
            string basename(this->name());
            stream_handle = new scv_tr_stream((basename + ".instr").c_str(), "TRANSACTOR", m_db);
            instr_tr_handle = new scv_tr_generator<>("execute", *stream_handle);
            fetch_tr_handle = new scv_tr_generator<uint64_t>("fetch", *stream_handle);
        }
#endif
}

void core_complex::disass_output(uint64_t pc, const std::string instr_str) {
#ifdef WITH_SCV
	if (m_db==nullptr) return;
    if(tr_handle.is_active()) tr_handle.end_transaction();
    tr_handle = instr_tr_handle->begin_transaction();
    tr_handle.record_attribute("PC", pc);
    tr_handle.record_attribute("INSTR", instr_str);
    tr_handle.record_attribute("MODE", lvl[cpu->get_mode()]);
    tr_handle.record_attribute("MSTATUS", cpu->get_state().mstatus.st.value);
    tr_handle.record_attribute("LTIME_START", quantum_keeper.get_current_time().value()/1000);
#endif
}

void core_complex::clk_cb() {
	curr_clk = clk_i.read();
}

void core_complex::sw_irq_cb(){
	if(sw_irq_i.read()) cpu->local_irq(16);
}

void core_complex::timer_irq_cb(){
	if(timer_irq_i.read()) cpu->local_irq(17);
}

void core_complex::global_irq_cb(){
	if(timer_irq_i.read()) cpu->local_irq(18);
}

void core_complex::run() {
    wait(sc_core::SC_ZERO_TIME);
    cpu->reset(reset_address.get_value());
    try {
        vm->start(-1);
    } catch (simulation_stopped &e) {
    }
    sc_core::sc_stop();
}

bool core_complex::read_mem(uint64_t addr, unsigned length, uint8_t *const data, bool is_fetch) {
    auto lut_entry = read_lut.getEntry(addr);
    if (lut_entry.get_granted_access() != tlm::tlm_dmi::DMI_ACCESS_NONE &&
        addr + length <= lut_entry.get_end_address() + 1) {
        auto offset = addr - lut_entry.get_start_address();
        std::copy(lut_entry.get_dmi_ptr() + offset, lut_entry.get_dmi_ptr() + offset + length, data);
        quantum_keeper.inc(lut_entry.get_read_latency());
        return true;
    } else {
        tlm::tlm_generic_payload gp;
        gp.set_command(tlm::TLM_READ_COMMAND);
        gp.set_address(addr);
        gp.set_data_ptr(data);
        gp.set_data_length(length);
        gp.set_streaming_width(length);
        auto delay{quantum_keeper.get_local_time()};
#ifdef WITH_SCV
        if(m_db!=nullptr && tr_handle.is_valid()){
            if(is_fetch && tr_handle.is_active()){
                tr_handle.end_transaction();
            }
            auto preExt = new scv4tlm::tlm_recording_extension(tr_handle, this);
            gp.set_extension(preExt);
        }
#endif
        initiator->b_transport(gp, delay);
        LOG(TRACE) << "read_mem(0x" << std::hex << addr << ") : " << data;
        if (gp.get_response_status() != tlm::TLM_OK_RESPONSE) {
            return false;
        }
        if (gp.is_dmi_allowed()) {
            gp.set_command(tlm::TLM_READ_COMMAND);
            gp.set_address(addr);
            tlm_dmi_ext dmi_data;
            if (initiator->get_direct_mem_ptr(gp, dmi_data)) {
                if (dmi_data.is_read_allowed())
                    read_lut.addEntry(dmi_data, dmi_data.get_start_address(),
                                      dmi_data.get_end_address() - dmi_data.get_start_address() + 1);
                if (dmi_data.is_write_allowed())
                    write_lut.addEntry(dmi_data, dmi_data.get_start_address(),
                                       dmi_data.get_end_address() - dmi_data.get_start_address() + 1);
            }
        }
        return true;
    }
}

bool core_complex::write_mem(uint64_t addr, unsigned length, const uint8_t *const data) {
    auto lut_entry = write_lut.getEntry(addr);
    if (lut_entry.get_granted_access() != tlm::tlm_dmi::DMI_ACCESS_NONE &&
        addr + length <= lut_entry.get_end_address() + 1) {
        auto offset = addr - lut_entry.get_start_address();
        std::copy(data, data + length, lut_entry.get_dmi_ptr() + offset);
        quantum_keeper.inc(lut_entry.get_read_latency());
        return true;
    } else {
        write_buf.resize(length);
        std::copy(data, data + length, write_buf.begin()); // need to copy as TLM does not guarantee data integrity
        tlm::tlm_generic_payload gp;
        gp.set_command(tlm::TLM_WRITE_COMMAND);
        gp.set_address(addr);
        gp.set_data_ptr(write_buf.data());
        gp.set_data_length(length);
        gp.set_streaming_width(length);
        auto delay{quantum_keeper.get_local_time()};
#ifdef WITH_SCV
        if(m_db!=nullptr && tr_handle.is_valid()){
            auto preExt = new scv4tlm::tlm_recording_extension(tr_handle, this);
            gp.set_extension(preExt);
        }
#endif
        initiator->b_transport(gp, delay);
        quantum_keeper.set(delay);
        LOG(TRACE) << "write_mem(0x" << std::hex << addr << ") : " << data;
        if (gp.get_response_status() != tlm::TLM_OK_RESPONSE) {
            return false;
        }
        if (gp.is_dmi_allowed()) {
            gp.set_command(tlm::TLM_READ_COMMAND);
            gp.set_address(addr);
            tlm_dmi_ext dmi_data;
            if (initiator->get_direct_mem_ptr(gp, dmi_data)) {
                if (dmi_data.is_read_allowed())
                    read_lut.addEntry(dmi_data, dmi_data.get_start_address(),
                                      dmi_data.get_end_address() - dmi_data.get_start_address() + 1);
                if (dmi_data.is_write_allowed())
                    write_lut.addEntry(dmi_data, dmi_data.get_start_address(),
                                       dmi_data.get_end_address() - dmi_data.get_start_address() + 1);
            }
        }
        return true;
    }
}

bool core_complex::read_mem_dbg(uint64_t addr, unsigned length, uint8_t *const data) {
    auto lut_entry = read_lut.getEntry(addr);
    if (lut_entry.get_granted_access() != tlm::tlm_dmi::DMI_ACCESS_NONE &&
        addr + length <= lut_entry.get_end_address() + 1) {
        auto offset = addr - lut_entry.get_start_address();
        std::copy(lut_entry.get_dmi_ptr() + offset, lut_entry.get_dmi_ptr() + offset + length, data);
        quantum_keeper.inc(lut_entry.get_read_latency());
        return true;
    } else {
        tlm::tlm_generic_payload gp;
        gp.set_command(tlm::TLM_READ_COMMAND);
        gp.set_address(addr);
        gp.set_data_ptr(data);
        gp.set_data_length(length);
        gp.set_streaming_width(length);
        return initiator->transport_dbg(gp) == length;
    }
}

bool core_complex::write_mem_dbg(uint64_t addr, unsigned length, const uint8_t *const data) {
    auto lut_entry = write_lut.getEntry(addr);
    if (lut_entry.get_granted_access() != tlm::tlm_dmi::DMI_ACCESS_NONE &&
        addr + length <= lut_entry.get_end_address() + 1) {
        auto offset = addr - lut_entry.get_start_address();
        std::copy(data, data + length, lut_entry.get_dmi_ptr() + offset);
        quantum_keeper.inc(lut_entry.get_read_latency());
        return true;
    } else {
        write_buf.resize(length);
        std::copy(data, data + length, write_buf.begin()); // need to copy as TLM does not guarantee data integrity
        tlm::tlm_generic_payload gp;
        gp.set_command(tlm::TLM_WRITE_COMMAND);
        gp.set_address(addr);
        gp.set_data_ptr(write_buf.data());
        gp.set_data_length(length);
        gp.set_streaming_width(length);
        return initiator->transport_dbg(gp) == length;
    }
}

} /* namespace SiFive */
} /* namespace sysc */
