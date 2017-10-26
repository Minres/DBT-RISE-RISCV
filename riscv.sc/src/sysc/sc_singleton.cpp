/*
 * sc_singleton.cpp
 *
 *  Created on: 09.10.2017
 *      Author: eyck
 */

#include "sysc/sc_singleton.h"

#include "seasocks/PrintfLogger.h"
#include "seasocks/Server.h"
#include "seasocks/StringUtil.h"
#include "seasocks/util/Json.h"
#include "seasocks/ResponseWriter.h"
#include "seasocks/util/RootPageHandler.h"
#include "seasocks/util/CrackedUriPageHandler.h"
#include "seasocks/util/StaticResponseHandler.h"

namespace sysc {

using namespace seasocks;
using namespace std;

namespace {
const std::string MSG_TXT { "Hello World"};
}

void sc_singleton::start_of_simulation() {
	//Launch a thread
	t=std::thread(&sc_singleton::thread_func, this);

}

sc_singleton::sc_singleton(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, m_serv(new seasocks::Server(std::make_shared<PrintfLogger>(Logger::Level::DEBUG))){
	auto rootHandler = make_shared<seasocks::RootPageHandler>();
	rootHandler->add(std::shared_ptr<CrackedUriPageHandler>(new StaticResponseHandler("/", Response::textResponse(MSG_TXT))));
	m_serv->addPageHandler(rootHandler);
}

sc_singleton::~sc_singleton() {
	//Join the thread with the main thread
	t.join();
}

void sc_singleton::thread_func() {
	get_server().serve(".", 9090);
}

seasocks::Server& sc_singleton::get_server() {
	return *m_serv.get();
}


} /* namespace sysc */

