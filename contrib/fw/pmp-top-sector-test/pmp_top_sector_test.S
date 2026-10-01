// PMP sector iteration must terminate when an access touches the top sector of
// the address space.
//
// pmp_check walked sectors with an inclusive address bound:
//   for(cur_addr = first; cur_addr <= last_sector; cur_addr += 4)
// When last_sector is the top 4-byte sector, the increment past it wraps
// cur_addr to 0, which is again <= last_sector, so the loop never ends. The
// simulator hangs with no trap and no log.
//
// The walk only runs for entries that are enabled, so entry 0 is programmed as
// an NA4 region over guard_word, well away from the top. It is unlocked, and
// guard_word lives in .data so no instruction fetch can overlap it; the entry
// exists only to make the walk run. The load from the top sector then matches
// no entry and must fall through to the M-mode default, which allows it. The
// load then reaches the simulator's default memory, which returns arbitrary
// data for unpopulated pages without faulting, so the address does not need to
// be backed. The loaded value is not checked.
//
// Buggy: the sector walk for entry 0 never terminates -> hang (CI timeout).
// Fixed: no entry matches, the M-mode load completes -> PASS.
//
// Pass: the load completes without a trap -> j . (exit 0)
// Fail: any trap, CSR or load -> semihosting SYS_EXIT (exit 2)

#include "pmp_test_common.h"

.section .text
.globl _start
_start:
    // Any trap means either PMP is unavailable or the load faulted -> FAIL
    la   t0, unexpected_trap
    csrw mtvec, t0

    // pmpaddr0 = guard_word >> 2: NA4 over one word nowhere near the top
    la   t1, guard_word
    srli t1, t1, 2
    csrw pmpaddr0, t1

    // pmpcfg0 byte 0 = 0x10 = NA4, unlocked, no R/W/X
    li   t1, 0x10
    csrw pmpcfg0, t1

    li   t0, -4         // 0xFFFF_FFFF_FFFF_FFFC, the top 4-byte sector
    lw   t1, 0(t0)
    j    .              // the load completed -> PASS

trap_entry unexpected_trap
    semihosting_fail    // CSR or load trapped -> FAIL

.section .data
    .align 2
guard_word:
    .word 0
