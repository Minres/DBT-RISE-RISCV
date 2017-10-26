////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 eyck@minres.com
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
// License for the specific language governing permissions and limitations under
// the License.
////////////////////////////////////////////////////////////////////////////////

#include "sysc/SiFive/uart.h"

#include "scc/report.h"
#include "scc/utilities.h"
#include "sysc/SiFive/gen/uart_regs.h"
#include "sysc/sc_singleton.h"

#include "seasocks/PrintfLogger.h"
#include "seasocks/Server.h"
#include "seasocks/StringUtil.h"
#include "seasocks/WebSocket.h"
#include "seasocks/util/Json.h"

namespace sysc {
namespace {

using namespace seasocks;

class MyHandler: public WebSocket::Handler {
public:
    explicit MyHandler(Server* server) : _server(server), _currentValue(0) {
        setValue(1);
    }

    virtual void onConnect(WebSocket* connection) {
        _connections.insert(connection);
        connection->send(_currentSetValue.c_str());
        cout << "Connected: " << connection->getRequestUri()
                << " : " << formatAddress(connection->getRemoteAddress())
                << endl;
        cout << "Credentials: " << *(connection->credentials()) << endl;
    }

    virtual void onData(WebSocket* connection, const char* data) {
        if (0 == strcmp("die", data)) {
            _server->terminate();
            return;
        }
        if (0 == strcmp("close", data)) {
            cout << "Closing.." << endl;
            connection->close();
            cout << "Closed." << endl;
            return;
        }

        int value = atoi(data) + 1;
        if (value > _currentValue) {
            setValue(value);
            for (auto c : _connections) {
                c->send(_currentSetValue.c_str());
            }
        }
    }

    virtual void onDisconnect(WebSocket* connection) {
        _connections.erase(connection);
        cout << "Disconnected: " << connection->getRequestUri()
                << " : " << formatAddress(connection->getRemoteAddress())
                << endl;
    }

private:
    set<WebSocket*> _connections;
    Server* _server;
    int _currentValue;
    string _currentSetValue;

    void setValue(int value) {
        _currentValue = value;
        _currentSetValue = makeExecString("set", _currentValue);
    }
};

}
uart::uart(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, tlm_target<>(clk)
, NAMED(clk_i)
, NAMED(rst_i)
, NAMEDD(uart_regs, regs) {
    regs->registerResources(*this);
    SC_METHOD(clock_cb);
    sensitive << clk_i;
    SC_METHOD(reset_cb);
    sensitive << rst_i;
    dont_initialize();
    regs->txdata.set_write_cb([this](scc::sc_register<uint32_t> &reg, uint32_t data) -> bool {
        if (!this->regs->in_reset()) {
            reg.put(data);
            this->transmit_data();
        }
        return true;
    });
    auto& server = sc_singleton::inst().get_server();
    auto handler = std::make_shared<MyHandler>(&server);
    server.addWebSocketHandler((std::string{"/ws/"}+name()).c_str(), handler);

}

uart::~uart() {}

void uart::clock_cb() { this->clk = clk_i.read(); }

void uart::reset_cb() {
    if (rst_i.read())
        regs->reset_start();
    else
        regs->reset_stop();
}

void uart::transmit_data() {
    if(regs->r_txdata.data != '\r') queue.push_back(regs->r_txdata.data);
    if (queue.size() >> 0 && (regs->r_txdata.data == '\n' || regs->r_txdata.data == 0)) {
        LOG(INFO) << this->name() << " transmit: '" << std::string(queue.begin(), queue.end()) << "'";
        queue.clear();
    }
}

} /* namespace sysc */
