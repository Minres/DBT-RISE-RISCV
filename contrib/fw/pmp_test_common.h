// Shared helpers for the PMP firmware tests in contrib/fw/pmp-*.
//
// Result convention used by every PMP test:
//   Pass: reach a `j .` self-loop, which trips the ISS JUMP_TO_SELF finish
//         condition and exits 0.
//   Fail: semihosting SYS_EXIT, which makes riscv-sim exit 2.

#ifndef PMP_TEST_COMMON_H
#define PMP_TEST_COMMON_H

// Terminate the run as a failure via ARM-style semihosting SYS_EXIT.
// The ebreak must be the 4-byte encoding rather than c.ebreak, because the ISS
// identifies the semihosting call by the magic instructions at -4 and +4 around
// it, so norvc is required here even when the caller allows compressed code.
.macro semihosting_fail
    li   a0, 0x18       // SYS_EXIT
    .option push
    .option norvc
    slli zero, zero, 0x1f
    ebreak
    srai zero, zero, 7
    .option pop
    j    .              // fallback if semihosting is not configured
.endm

// Start a trap handler. mtvec's low two bits hold the vectoring mode, so a
// handler that is only 2-byte aligned has its address truncated and the hart
// vectors two bytes early. That lands inside a preceding semihosting_fail, whose
// trailing `j .` reads as the pass condition, so a misaligned handler makes a
// test pass without its body ever running. Always declare handlers with this.
.macro trap_entry name
    .align 2
\name\():
.endm

#endif // PMP_TEST_COMMON_H
