// Verify that RV64 pmpcfg0 upper 32 bits (cfg bytes 4-7) are preserved on write.
// On RV64 a pmpcfg register is 64 bits and holds 8 config bytes (entries 0-7).
// A write mask narrower than reg_t silently zeroes bytes 4-7.
//
// Pass: readback == written value -> j . (exit 0)
// Fail: readback != written value -> semihosting SYS_EXIT (exit 2)

.section .text
.globl _start
_start:
    // 0x18 per byte = NAPOT mode, no RWX, no lock; survives 0x9f byte mask
    li   t0, 0x1818181818181818
    csrw pmpcfg0, t0
    csrr t1, pmpcfg0
    bne  t0, t1, fail

    j    .              // pass: ISS detects j . and exits 0

fail:
    li   a0, 0x18
    .option push
    .option norvc
    slli zero, zero, 0x1f
    ebreak
    srai zero, zero, 7
    .option pop
    j    .
