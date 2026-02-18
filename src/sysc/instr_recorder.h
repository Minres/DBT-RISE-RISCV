#pragma once

#include "scv-tr/scv_tr.h"
#include <rigtorp/SPSCQueue.h>
#include <scc/utilities.h>
#include <tlm/scc/quantum_keeper.h>
#include <tlm/scc/scv/tlm_recording_extension.h>
#ifdef HAS_SCV
#include <scv.h>
#else
#include <scv-tr.h>
#ifndef SCVNS
#define SCVNS ::scv_tr::
#endif
#endif

namespace sysc {
namespace riscv {

struct instr_recorder_b {
    bool init(std::string const& basename) {
        if(tx_db != nullptr) {
            stream_hnd = new SCVNS scv_tr_stream((basename + ".instr").c_str(), "TRANSACTOR", tx_db);
            gen_hndl = new SCVNS scv_tr_generator<>("execute", *stream_hnd);
            gen_timed_hndl = new SCVNS scv_tr_generator<>("execute(timed)", *stream_hnd);
            return true;
        }
        return false;
    }
    instr_recorder_b(SCVNS scv_tr_db* tx_db = SCVNS scv_tr_db::get_default_db())
    : tx_db(tx_db) {}
    void record_instr(uint64_t pc, std::string const& instr_str, char mode, uint64_t status, sc_core::sc_time ltime) {
        tx_hndl = gen_hndl->begin_transaction();
        tx_hndl.record_attribute("PC", pc);
        tx_hndl.record_attribute("INSTR", instr_str);
        tx_hndl.record_attribute("MODE", mode);
        tx_hndl.record_attribute("MSTATUS", status);
        tx_hndl.record_attribute("LTIME_START(ps)", static_cast<uint64_t>(ltime / 1_ps));
        tx_hndl.end_transaction();
        if(tx_timed_hndl.is_active())
            gen_timed_hndl->end_transaction(tx_timed_hndl, ltime);
        tx_timed_hndl = gen_timed_hndl->begin_transaction(ltime);
        tx_timed_hndl.record_attribute("PC", pc);
        tx_timed_hndl.record_attribute("INSTR", instr_str);
        tx_timed_hndl.record_attribute("MODE", mode);
        tx_timed_hndl.record_attribute("MSTATUS", status);
        tx_timed_hndl.add_relation("PARENT/CHILD", tx_hndl);
    }

    tlm::scc::scv::tlm_recording_extension* get_recording_extension(bool finish_instr) {
        if(tx_db != nullptr) {
            if(tx_timed_hndl.is_valid()) {
                return new tlm::scc::scv::tlm_recording_extension(tx_timed_hndl, this);
            } else if(tx_hndl.is_valid()) {
                return new tlm::scc::scv::tlm_recording_extension(tx_hndl, this);
            }
        }
        return nullptr;
    }

protected:
    //! transaction recording database
    SCVNS scv_tr_db* tx_db{nullptr};
    //! blocking transaction recording stream handle
    SCVNS scv_tr_stream* stream_hnd{nullptr};
    //! transaction generator handle for blocking transactions
    SCVNS scv_tr_generator<SCVNS _scv_tr_generator_default_data, SCVNS _scv_tr_generator_default_data>* gen_hndl{nullptr};
    SCVNS scv_tr_generator<SCVNS _scv_tr_generator_default_data, SCVNS _scv_tr_generator_default_data>* gen_timed_hndl{nullptr};
    SCVNS scv_tr_handle tx_hndl;
    SCVNS scv_tr_handle tx_timed_hndl;
};

template <typename QK> struct instr_recorder : instr_recorder_b {};

template <> struct instr_recorder<tlm::scc::quantumkeeper> : instr_recorder_b {
    void disass_output(uint64_t pc, std::string const& instr_str, char mode, uint64_t status) {
        if(stream_hnd == nullptr)
            return;
        record_instr(pc, instr_str, mode, status, quantum_keeper.get_local_absolute_time());
    }
    instr_recorder(tlm::scc::quantumkeeper& quantum_keeper)
    : quantum_keeper(quantum_keeper) {}

protected:
    tlm::scc::quantumkeeper& quantum_keeper;
};
#if SC_VERSION_MAJOR > 2
template <> struct instr_recorder<tlm::scc::quantumkeeper_mt> : instr_recorder_b {
    void disass_output(uint64_t pc, std::string const& instr_str, char mode, uint64_t status) {
        if(stream_hnd == nullptr)
            return;
        que.push(instr_record{pc, instr_str, mode, status, quantum_keeper.get_local_absolute_time()});
    }

    void record() {
        while(auto r = que.front()) {
            record_instr(r->pc, r->instr_str, r->mode, r->status, r->start_time);
            que.pop();
        }
    }

    struct callback : sc_core::sc_stage_callback_if {
        void stage_callback(const sc_core::sc_stage& stage) override { owner->record(); }
        callback(instr_recorder<tlm::scc::quantumkeeper_mt>* owner)
        : owner(owner) {}
        instr_recorder<tlm::scc::quantumkeeper_mt>* owner;
    } stage_cb{this};

    instr_recorder(tlm::scc::quantumkeeper_mt& quantum_keeper)
    : quantum_keeper(quantum_keeper)
    , que(8 * 1024) {
        sc_core::sc_register_stage_callback(stage_cb, sc_core::SC_PRE_TIMESTEP);
    }

protected:
    struct instr_record {
        uint64_t pc;
        std::string const instr_str;
        char mode;
        uint64_t status;
        sc_core::sc_time start_time;
    };
    tlm::scc::quantumkeeper_mt& quantum_keeper;
    rigtorp::SPSCQueue<instr_record> que;
};
#endif
} // namespace riscv
} // namespace sysc
