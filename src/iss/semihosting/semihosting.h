#ifndef _SEMIHOSTING_H_
#define _SEMIHOSTING_H_
#include <chrono>
#include <functional>
#include <iss/arch_if.h>
/*
 * According to:
 * "Semihosting for AArch32 and AArch64, Release 2.0"
 * https://static.docs.arm.com/100863/0200/semihosting.pdf
 * from ARM Ltd.
 *
 * The available semihosting operation numbers passed in A0 are allocated
 * as follows:
 * - 0x00-0x31 Used by ARM.
 * - 0x32-0xFF Reserved for future use by ARM.
 * - 0x100-0x1FF Reserved for user applications. These are not used by ARM.
 *   However, if you are writing your own SVC operations, you are advised
 *   to use a different SVC number rather than using the semihosted
 *   SVC number and these operation type numbers.
 * - 0x200-0xFFFFFFFF Undefined and currently unused. It is recommended
 *   that you do not use these.
 */
enum class semihosting_syscalls {

    SYS_OPEN = 0x01,
    SYS_CLOSE = 0x02,
    SYS_WRITEC = 0x03,
    SYS_WRITE0 = 0x04,
    SYS_WRITE = 0x05,
    SYS_READ = 0x06,
    SYS_READC = 0x07,
    SYS_ISERROR = 0x08,
    SYS_ISTTY = 0x09,
    SYS_SEEK = 0x0A,
    SYS_FLEN = 0x0C,
    SYS_TMPNAM = 0x0D,
    SYS_REMOVE = 0x0E,
    SYS_RENAME = 0x0F,
    SYS_CLOCK = 0x10,
    SYS_TIME = 0x11,
    SYS_SYSTEM = 0x12,
    SYS_ERRNO = 0x13,
    SYS_GET_CMDLINE = 0x15,
    SYS_HEAPINFO = 0x16,
    SYS_EXIT = 0x18,
    SYS_EXIT_EXTENDED = 0x20,
    SYS_ELAPSED = 0x30,
    SYS_TICKFREQ = 0x31,
    USER_CMD_0x100 = 0x100,
    USER_CMD_0x1FF = 0x1FF,
};

template <typename T> struct semihosting_callback {
    std::chrono::high_resolution_clock::time_point timeVar;
    semihosting_callback()
    : timeVar(std::chrono::high_resolution_clock::now()) {}
    void operator()(iss::arch_if* arch_if_ptr, T* call_number, T* parameter);
};

template <typename T> using semihosting_cb_t = std::function<void(iss::arch_if*, T*, T*)>;
#endif