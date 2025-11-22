/*******************************************************************************
 * Copyright (C) 2025 MINRES Technologies GmbH
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
 * Contributors:
 *       eyck@minres.com - initial implementation
 ******************************************************************************/

#include "semihosting.h"
#include <chrono>
#include <cstdint>
#include <iss/vm_types.h>
#include <map>
#include <stdexcept>
// explanation of syscalls can be found at https://github.com/SpinalHDL/openocd_riscv/blob/riscv_spinal/src/target/semihosting_common.h

const char* SYS_OPEN_MODES_STRS[] = {"r", "rb", "r+", "r+b", "w", "wb", "w+", "w+b", "a", "ab", "a+", "a+b"};

template <typename T> T sh_read_field(iss::arch_if* arch_if_ptr, T addr, int len = 4) {
    uint8_t bytes[4];
    auto res = arch_if_ptr->read({iss::address_type::LOGICAL, iss::access_type::DEBUG_READ, 0, addr}, 4, &bytes[0]);
    // auto res = arch_if_ptr->read(iss::address_type::PHYSICAL, iss::access_type::DEBUG_READ, 0, *parameter, 1, &character);

    if(res != iss::Ok) {
        return 0; // TODO THROW ERROR
    } else
        return static_cast<T>(bytes[0]) | (static_cast<T>(bytes[1]) << 8) | (static_cast<T>(bytes[2]) << 16) |
               (static_cast<T>(bytes[3]) << 24);
}

template <typename T> std::string sh_read_string(iss::arch_if* arch_if_ptr, T addr, T str_len) {
    std::vector<uint8_t> buffer(str_len);
    for(int i = 0; i < str_len; i++) {
        buffer[i] = sh_read_field(arch_if_ptr, addr + i, 1);
    }
    std::string str(buffer.begin(), buffer.end());
    return str;
}

template <typename T> void semihosting_callback<T>::operator()(iss::arch_if* arch_if_ptr, T* call_number, T* parameter) {
    static std::map<T, FILE*> openFiles;
    static T file_count = 3;
    static T semihostingErrno;

    switch(static_cast<semihosting_syscalls>(*call_number)) {
    case semihosting_syscalls::SYS_CLOCK: {
        auto end = std::chrono::high_resolution_clock::now(); // end measurement
        auto elapsed = end - timeVar;
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
        *call_number = millis; // TODO get time now
        break;
    }
    case semihosting_syscalls::SYS_CLOSE: {
        T file_handle = *parameter;
        if(openFiles.size() <= file_handle && file_handle < 0) {
            semihostingErrno = EBADF;
            return;
        }
        auto file = openFiles[file_handle];
        openFiles.erase(file_handle);
        if(!(file == stdin || file == stdout || file == stderr)) {
            int i = fclose(file);
            *call_number = i;
        } else {
            *call_number = -1;
            semihostingErrno = EINTR;
        }
        break;
    }
    case semihosting_syscalls::SYS_ELAPSED: {
        throw std::runtime_error("Semihosting Call not Implemented");
        break;
    }
    case semihosting_syscalls::SYS_ERRNO: {
        *call_number = semihostingErrno;
        break;
    }
    case semihosting_syscalls::SYS_EXIT: {

        throw std::runtime_error("ISS terminated by Semihost: SYS_EXIT");
        break;
    }
    case semihosting_syscalls::SYS_EXIT_EXTENDED: {
        throw std::runtime_error("ISS terminated by Semihost: SYS_EXIT_EXTENDED");
        break;
    }
    case semihosting_syscalls::SYS_FLEN: {
        T file_handle = *parameter;
        auto file = openFiles[file_handle];

        size_t currentPos = ftell(file);
        if(currentPos < 0)
            throw std::runtime_error("SYS_FLEN negative value");
        fseek(file, 0, SEEK_END);
        size_t length = ftell(file);
        fseek(file, currentPos, SEEK_SET);
        *call_number = (T)length;
        break;
    }
    case semihosting_syscalls::SYS_GET_CMDLINE: {
        throw std::runtime_error("Semihosting Call not Implemented");
        break;
    }
    case semihosting_syscalls::SYS_HEAPINFO: {
        throw std::runtime_error("Semihosting Call not Implemented");
        break;
    }
    case semihosting_syscalls::SYS_ISERROR: {
        T value = *parameter;
        *call_number = (value != 0);
        break;
    }
    case semihosting_syscalls::SYS_ISTTY: {
        T file_handle = *parameter;
        *call_number = (file_handle == 0 || file_handle == 1 || file_handle == 2);
        break;
    }
    case semihosting_syscalls::SYS_OPEN: {
        T path_str_addr = sh_read_field<T>(arch_if_ptr, *parameter);
        T mode = sh_read_field<T>(arch_if_ptr, 4 + (*parameter));
        T path_len = sh_read_field<T>(arch_if_ptr, 8 + (*parameter));

        std::string path_str = sh_read_string<T>(arch_if_ptr, path_str_addr, path_len);

        // TODO LOG INFO

        if(mode >= 12) {
            // TODO throw ERROR
            return;
        }

        FILE* file = nullptr;
        if(path_str == ":tt") {
            if(mode < 4)
                file = stdin;
            else if(mode < 8)
                file = stdout;
            else
                file = stderr;
        } else {
            file = fopen(path_str.c_str(), SYS_OPEN_MODES_STRS[mode]);
            if(file == nullptr) {
                // TODO throw error
                return;
            }
        }
        T file_handle = file_count++;
        openFiles[file_handle] = file;
        *call_number = file_handle;
        break;
    }
    case semihosting_syscalls::SYS_READ: {
        T file_handle = sh_read_field<T>(arch_if_ptr, (*parameter) + 4);
        T addr = sh_read_field<T>(arch_if_ptr, *parameter);
        T count = sh_read_field<T>(arch_if_ptr, (*parameter) + 8);

        auto file = openFiles[file_handle];

        std::vector<uint8_t> buffer(count);
        size_t num_read = 0;
        if(file == stdin) {
            // when reading from stdin: mimic behaviour from read syscall
            // and return on newline.
            while(num_read < count) {
                char c = fgetc(file);
                buffer[num_read] = c;
                num_read++;
                if(c == '\n')
                    break;
            }
        } else {
            num_read = fread(buffer.data(), 1, count, file);
        }
        buffer.resize(num_read);
        for(int i = 0; i < num_read; i++) {
            auto res = arch_if_ptr->write({iss::address_type::LOGICAL, iss::access_type::DEBUG_READ, 0, addr + i}, 1, &buffer[i]);
            if(res != iss::Ok)
                return;
        }
        *call_number = count - num_read;
        break;
    }
    case semihosting_syscalls::SYS_READC: {
        uint8_t character = getchar();
        // character = getchar();
        /*if(character != iss::Ok)
            std::cout << "Not OK";
            return;*/
        *call_number = character;
        break;
    }
    case semihosting_syscalls::SYS_REMOVE: {
        T path_str_addr = sh_read_field<T>(arch_if_ptr, *parameter);
        T path_len = sh_read_field<T>(arch_if_ptr, (*parameter) + 4);
        std::string path_str = sh_read_string<T>(arch_if_ptr, path_str_addr, path_len);

        if(remove(path_str.c_str()) < 0)
            *call_number = -1;
        break;
    }
    case semihosting_syscalls::SYS_RENAME: {
        T path_str_addr_old = sh_read_field<T>(arch_if_ptr, *parameter);
        T path_len_old = sh_read_field<T>(arch_if_ptr, (*parameter) + 4);
        T path_str_addr_new = sh_read_field<T>(arch_if_ptr, (*parameter) + 8);
        T path_len_new = sh_read_field<T>(arch_if_ptr, (*parameter) + 12);

        std::string path_str_old = sh_read_string<T>(arch_if_ptr, path_str_addr_old, path_len_old);
        std::string path_str_new = sh_read_string<T>(arch_if_ptr, path_str_addr_new, path_len_new);
        rename(path_str_old.c_str(), path_str_new.c_str());
        break;
    }
    case semihosting_syscalls::SYS_SEEK: {
        T file_handle = sh_read_field<T>(arch_if_ptr, *parameter);
        T pos = sh_read_field<T>(arch_if_ptr, (*parameter) + 1);
        auto file = openFiles[file_handle];

        int retval = fseek(file, pos, SEEK_SET);
        if(retval < 0)
            throw std::runtime_error("SYS_SEEK negative return value");

        break;
    }
    case semihosting_syscalls::SYS_SYSTEM: {
        T cmd_addr = sh_read_field<T>(arch_if_ptr, *parameter);
        T cmd_len = sh_read_field<T>(arch_if_ptr, (*parameter) + 1);
        std::string cmd = sh_read_string<T>(arch_if_ptr, cmd_addr, cmd_len);
        auto _ = system(cmd.c_str());
        break;
    }
    case semihosting_syscalls::SYS_TICKFREQ: {
        throw std::runtime_error("Semihosting Call not Implemented");
        break;
    }
    case semihosting_syscalls::SYS_TIME: {
        // returns time in seconds scince 01.01.1970 00:00
        *call_number = time(NULL);
        break;
    }
    case semihosting_syscalls::SYS_TMPNAM: {
        T buffer_addr = sh_read_field<T>(arch_if_ptr, *parameter);
        T identifier = sh_read_field<T>(arch_if_ptr, (*parameter) + 1);
        T buffer_len = sh_read_field<T>(arch_if_ptr, (*parameter) + 2);

        if(identifier > 255) {
            *call_number = -1;
            return;
        }
        std::stringstream ss;
        ss << "tmp/file-" << std::setfill('0') << std::setw(3) << identifier;
        std::string filename = ss.str();

        for(int i = 0; i < buffer_len; i++) {
            uint8_t character = filename[i];
            auto res = arch_if_ptr->write({iss::address_type::LOGICAL, iss::access_type::DEBUG_READ, 0, (*parameter) + i}, 1, &character);
            if(res != iss::Ok)
                return;
        }
        break;
    }
    case semihosting_syscalls::SYS_WRITE: {
        T file_handle = sh_read_field<T>(arch_if_ptr, (*parameter) + 4);
        T addr = sh_read_field<T>(arch_if_ptr, *parameter);
        T count = sh_read_field<T>(arch_if_ptr, (*parameter) + 8);

        auto file = openFiles[file_handle];
        std::string str = sh_read_string<T>(arch_if_ptr, addr, count);
        fwrite(&str[0], 1, count, file);
        break;
    }
    case semihosting_syscalls::SYS_WRITEC: {
        uint8_t character;
        auto res = arch_if_ptr->read({iss::address_type::LOGICAL, iss::access_type::DEBUG_READ, 0, *parameter}, 1, &character);
        if(res != iss::Ok)
            return;
        putchar(character);
        break;
    }
    case semihosting_syscalls::SYS_WRITE0: {
        uint8_t character;
        while(1) {
            auto res = arch_if_ptr->read({iss::address_type::LOGICAL, iss::access_type::DEBUG_READ, 0, *parameter}, 1, &character);
            if(res != iss::Ok)
                return;
            if(character == 0)
                break;
            putchar(character);
            (*parameter)++;
        }
        break;
    }
    case semihosting_syscalls::USER_CMD_0x100: {
        throw std::runtime_error("Semihosting Call not Implemented");
        break;
    }
    case semihosting_syscalls::USER_CMD_0x1FF: {
        throw std::runtime_error("Semihosting Call not Implemented");
        break;
    }
    default:
        throw std::runtime_error("Semihosting Call not Implemented");
        break;
    }
}
template class semihosting_callback<uint32_t>;
template class semihosting_callback<uint64_t>;
