/*
 * sc_singleton.h
 *
 *  Created on: 09.10.2017
 *      Author: eyck
 */

#ifndef RISCV_SC_INCL_SYSC_SC_COMM_SINGLETON_H_
#define RISCV_SC_INCL_SYSC_SC_COMM_SINGLETON_H_

#include <sysc/kernel/sc_module.h>
#include <seasocks/PageHandler.h>
#include "seasocks/WebSocket.h"

#include <memory>
#include <thread>
#include <cstring>

namespace sysc {

class WsHandler: public seasocks::WebSocket::Handler {
public:
    explicit WsHandler() { }

    void onConnect(seasocks::WebSocket* connection) override;

    void onData(seasocks::WebSocket* connection, const char* data) override;

    void onDisconnect(seasocks::WebSocket* connection) override;

    void send(std::string msg) { for (auto *con : _connections) con->send(msg); }

    void set_receive_callback(std::function<void(const char* data)> cb){callback=cb;}

private:
    std::set<seasocks::WebSocket*> _connections;
    std::function<void(const char* data)> callback;
};

class sc_comm_singleton: public sc_core::sc_module {
	struct DefaultPageHandler: public seasocks::PageHandler {
		DefaultPageHandler(sc_comm_singleton& o):owner(o){}
	    virtual std::shared_ptr<seasocks::Response> handle(const seasocks::Request& request);
	    sc_comm_singleton& owner;
	};
public:
	sc_comm_singleton() = delete;

	sc_comm_singleton(const sc_comm_singleton&) = delete;

	sc_comm_singleton& operator=(sc_comm_singleton& o) = delete;

	virtual ~sc_comm_singleton();

	static sc_comm_singleton& inst(){
		static sc_comm_singleton i("__sc_singleton");
		return i;
	}

	seasocks::Server& get_server();

    void registerWebSocketHandler(const char* endpoint, std::shared_ptr<seasocks::WebSocket::Handler> handler, bool allowCrossOriginRequests = false);

    void execute(std::function<void()> f);

    void start_client();

protected:
	void start_of_simulation() override;
	void end_of_simulation() override;

private:
	sc_comm_singleton(sc_core::sc_module_name nm);
	std::unique_ptr<seasocks::Server> m_serv;
	std::thread t;
	void thread_func();
	bool client_started;
};

} /* namespace sysc */

#endif /* RISCV_SC_INCL_SYSC_SC_COMM_SINGLETON_H_ */
