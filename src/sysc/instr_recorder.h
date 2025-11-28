#pragma once

#include "scv-tr/scv_tr.h"
#include <rigtorp/SPSCQueue.h>
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
        if(m_db != nullptr) {
            stream_handle = new SCVNS scv_tr_stream((basename + ".instr").c_str(), "TRANSACTOR", m_db);
            instr_tr_handle = new SCVNS scv_tr_generator<>("execute", *stream_handle);
            return true;
        }
        return false;
    }
    instr_recorder_b(SCVNS scv_tr_db* m_db = SCVNS scv_tr_db::get_default_db())
    : m_db(m_db) {}
    void record_instr(uint64_t pc, std::string const& instr_str, char mode, uint64_t status, sc_core::sc_time ltime) {
        if(stream_handle == nullptr)
            return;
        if(tr_handle.is_active())
            tr_handle.end_transaction();
        tr_handle = instr_tr_handle->begin_transaction();
        tr_handle.record_attribute("PC", pc);
        tr_handle.record_attribute("INSTR", instr_str);
        tr_handle.record_attribute("MODE", mode);
        tr_handle.record_attribute("MSTATUS", status);
        tr_handle.record_attribute("LTIME_START", ltime / 1_ns);
    }

    tlm::scc::scv::tlm_recording_extension* get_recording_extension(bool finish_instr) {
        if(m_db != nullptr && tr_handle.is_valid()) {
            if(finish_instr && tr_handle.is_active())
                tr_handle.end_transaction();
            return new tlm::scc::scv::tlm_recording_extension(tr_handle, this);
        }
        return nullptr;
    }

protected:
    //! transaction recording database
    SCVNS scv_tr_db* m_db{nullptr};
    //! blocking transaction recording stream handle
    SCVNS scv_tr_stream* stream_handle{nullptr};
    //! transaction generator handle for blocking transactions
    SCVNS scv_tr_generator<SCVNS _scv_tr_generator_default_data, SCVNS _scv_tr_generator_default_data>* instr_tr_handle{nullptr};
    SCVNS scv_tr_handle tr_handle;
};

template <typename QK> struct instr_recorder : instr_recorder_b {};

template <> struct instr_recorder<tlm::scc::quantumkeeper> : instr_recorder_b {
    void disass_output(uint64_t pc, std::string const& instr_str, char mode, uint64_t status) {
        if(stream_handle == nullptr)
            return;
        record_instr(pc, instr_str, mode, status, quantum_keeper.get_local_absolute_time());
    }
    instr_recorder(tlm::scc::quantumkeeper& quantum_keeper)
    : quantum_keeper(quantum_keeper) {}

protected:
    tlm::scc::quantumkeeper& quantum_keeper;
};

template <> struct instr_recorder<tlm::scc::quantumkeeper_mt> : instr_recorder_b {
    void disass_output(uint64_t pc, std::string const& instr_str, char mode, uint64_t status) {
        if(stream_handle == nullptr)
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
} // namespace riscv
} // namespace sysc