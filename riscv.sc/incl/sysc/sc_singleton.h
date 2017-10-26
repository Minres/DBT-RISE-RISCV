/*
 * sc_singleton.h
 *
 *  Created on: 09.10.2017
 *      Author: eyck
 */

#ifndef RISCV_SC_INCL_SYSC_SC_SINGLETON_H_
#define RISCV_SC_INCL_SYSC_SC_SINGLETON_H_

#include <sysc/kernel/sc_module.h>
#include <memory>
#include <thread>

namespace seasocks {
class Server;
}

namespace sysc {

class sc_singleton: public sc_core::sc_module {
public:
	sc_singleton() = delete;

	sc_singleton(const sc_singleton&) = delete;

	sc_singleton& operator=(sc_singleton& o) = delete;

	virtual ~sc_singleton();

	static sc_singleton& inst(){
		static sc_singleton i("__sc_singleton");
		return i;
	}

	seasocks::Server& get_server();
protected:
	void start_of_simulation();

private:
	sc_singleton(sc_core::sc_module_name nm);
	std::unique_ptr<seasocks::Server> m_serv;
	std::thread t;
	void thread_func();
};

} /* namespace sysc */

#endif /* RISCV_SC_INCL_SYSC_SC_SINGLETON_H_ */
