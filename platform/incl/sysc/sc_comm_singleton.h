/*******************************************************************************
 * Copyright (C) 2017, 2018 MINRES Technologies GmbH
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

#ifndef _SYSC_SC_COMM_SINGLETON_H_
#define _SYSC_SC_COMM_SINGLETON_H_

#include "seasocks/WebSocket.h"
#include <seasocks/PageHandler.h>
#include <sysc/kernel/sc_module.h>

#include <cstring>
#include <functional>
#include <memory>
#include <thread>

namespace sysc {

class WsHandler : public seasocks::WebSocket::Handler {
public:
    explicit WsHandler() {}

    void onConnect(seasocks::WebSocket *connection) override;

    void onData(seasocks::WebSocket *connection, const char *data) override;

    void onDisconnect(seasocks::WebSocket *connection) override;

    void send(std::string msg) {
        for (auto *con : _connections) con->send(msg);
    }

    void set_receive_callback(std::function<void(const char *data)> cb) { callback = cb; }

private:
    std::set<seasocks::WebSocket *> _connections;
    std::function<void(const char *data)> callback;
};

class sc_comm_singleton : public sc_core::sc_module {
    struct DefaultPageHandler : public seasocks::PageHandler {
        DefaultPageHandler(sc_comm_singleton &o)
        : owner(o) {}
        virtual std::shared_ptr<seasocks::Response> handle(const seasocks::Request &request);
        sc_comm_singleton &owner;
    };

public:
    sc_comm_singleton() = delete;

    sc_comm_singleton(const sc_comm_singleton &) = delete;

    sc_comm_singleton &operator=(sc_comm_singleton &o) = delete;

    virtual ~sc_comm_singleton();

    static sc_comm_singleton &inst() {
        static sc_comm_singleton i("__sc_singleton");
        return i;
    }

    seasocks::Server &get_server();

    void registerWebSocketHandler(const char *endpoint, std::shared_ptr<seasocks::WebSocket::Handler> handler,
                                  bool allowCrossOriginRequests = false);

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
    bool needs_client, client_started;
    std::vector<std::string> endpoints;
};

} /* namespace sysc */

#endif /* _SYSC_SC_COMM_SINGLETON_H_ */
