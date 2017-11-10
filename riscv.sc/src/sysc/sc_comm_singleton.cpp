/*
 * sc_singleton.cpp
 *
 *  Created on: 09.10.2017
 *      Author: eyck
 */

#include "sysc/sc_comm_singleton.h"

#include "seasocks/PrintfLogger.h"
#include "seasocks/Server.h"
#include "seasocks/StringUtil.h"
#include "seasocks/util/Json.h"
#include "seasocks/ResponseWriter.h"
#include "seasocks/util/RootPageHandler.h"
#include "seasocks/util/CrackedUriPageHandler.h"
#include "seasocks/util/StaticResponseHandler.h"

#include <cstdio>
#include <csignal>
#include <sys/stat.h>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

namespace sysc {

using namespace seasocks;
using namespace std;

namespace {
inline void die(){perror(nullptr);exit(errno);}
}

sc_comm_singleton::sc_comm_singleton(sc_core::sc_module_name nm)
: sc_core::sc_module(nm)
, m_serv(new Server(std::make_shared<PrintfLogger>(Logger::Level::WARNING)))
, client_started(false){
	m_serv->addPageHandler(std::make_shared<DefaultPageHandler>(*this));
}

sc_comm_singleton::~sc_comm_singleton() {
	//Join the thread with the main thread
	t.join();
}

void sc_comm_singleton::start_of_simulation() {
	//Launch a thread
	t=std::thread(&sc_comm_singleton::thread_func, this);
	start_client();
}

void sc_comm_singleton::end_of_simulation() {
	get_server().terminate();
}

void sc_comm_singleton::start_client() {
	if(client_started) return;
	std::stringstream ss;
#ifndef WIN32
	if(fork()==0){
		// daemonizing, see http://www.microhowto.info/howto/cause_a_process_to_become_a_daemon_in_c.html#id2407077
	    // Fork, allowing the parent process to terminate.
	    pid_t pid = fork();
	    if (pid == -1) {
	        die();
	    } else if (pid != 0) {
	        _exit(0);
	    }
	    // Start a new session for the daemon.
	    if (setsid()==-1) die();
	    // Fork again, allowing the parent process to terminate.
	    signal(SIGHUP,SIG_IGN);
	    pid=fork();
	    if (pid == -1) {
	        die();
	    } else if (pid != 0) {
	        _exit(0);
	    }
	    // Set the current working directory to the root directory.
	    if (chdir("/") == -1) die();
	    // Set the user file creation mask to zero.
	    umask(0);

	    // Close then reopen standard file descriptors.
	    close(STDIN_FILENO);
	    close(STDOUT_FILENO);
	    close(STDERR_FILENO);
	    if (open("/dev/null",O_RDONLY) == -1) die();
	    if (open("/dev/null",O_WRONLY) == -1) die();
	    if (open("/dev/null",O_RDWR) == -1)  die();
		// now do what is needed
		ss<<"x-www-browser http://localhost:9090/ws.html"; //Linux
		auto res = system (ss.str().c_str());
		if(res==0) exit(0);
		ss.str("");
		ss<<"xdg-open  http://localhost:9090/ws.html"; // Linux
		res=system (ss.str().c_str());
		if(res==0) exit(0);
		ss.str("");
		ss<<"open  http://localhost:9090/ws.html"; // MacOS
		res=system (ss.str().c_str());
		exit(0);
	}
	// #else
	// on windows should be open, see https://www.experts-exchange.com/articles/1595/Execute-a-Program-with-C.html
#endif
	client_started=true;
}

void sc_comm_singleton::registerWebSocketHandler(const char* endpoint,
		std::shared_ptr<WebSocket::Handler> handler,
		bool allowCrossOriginRequests) {
	get_server().addWebSocketHandler(endpoint, handler, allowCrossOriginRequests);
}

void sc_comm_singleton::execute(std::function<void()> f) {
	get_server().execute(f);
}

void sc_comm_singleton::thread_func() {
	get_server().serve("./html", 9090);
}

Server& sc_comm_singleton::get_server() {
	return *m_serv.get();
}


std::shared_ptr<Response> sc_comm_singleton::DefaultPageHandler::handle(const Request& request) {
	if(request.verb() == Request::Verb::Get && request.getRequestUri()=="conf.json"){
		return Response::htmlResponse("{}");
	}
	return Response::unhandled();
}

void WsHandler::onConnect(WebSocket* connection) {
	_connections.insert(connection);
	//connection->send("Connection established");;
}

void WsHandler::onData(WebSocket* connection, const char* data) {
	if (0 == strcmp("close", data)) {
		connection->close();
	} else if(callback)
		callback(data);
}

void WsHandler::onDisconnect(WebSocket* connection) {
	_connections.erase(connection);
}

} /* namespace sysc */
