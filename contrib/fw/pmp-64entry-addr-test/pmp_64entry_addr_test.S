// Verify pmpaddr32 (CSR 0x3D0) is accessible - tests 64-entry pmpaddr support.
// Pass: readback matches written sentinel -> j . (exit 0)
// Fail: any trap (unregistered CSR) or readback mismatch -> semihosting SYS_EXIT (exit 2)

.section .text
.globl _start
_start:
    la   t0, fail
    csrw mtvec, t0

    li   t0, 0xDEAD
    csrw 0x3D0, t0          // pmpaddr32
    csrr t1, 0x3D0
    bne  t0, t1, fail

    j    .                  // pass

fail:
    li   a0, 0x18
    .option push
    .option norvc
    slli zero, zero, 0x1f
    ebreak
    srai zero, zero, 7
    .option pop
    j    .
