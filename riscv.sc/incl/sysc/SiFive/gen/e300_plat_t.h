#ifndef _E300_PLAT_MAP_H_
#define _E300_PLAT_MAP_H_
// need double braces, see
// https://stackoverflow.com/questions/6893700/how-to-construct-stdarray-object-with-initializer-list#6894191
const std::array<scc::target_memory_map_entry<32>, 8> e300_plat_map = {{
    {&i_clint, 0x2000000, 0xc000},
    {&i_plic, 0xc000000, 0x200008},
    {&i_aon, 0x10000000, 0x150},
    {&i_prci, 0x10008000, 0x14},
    {&i_gpio, 0x10012000, 0x44},
    {&i_uart0, 0x10013000, 0x1c},
    {&i_uart1, 0x10023000, 0x1c},
    {&i_spi, 0x10014000, 0x78},
}};

#endif /* _E300_PLAT_MAP_H_ */
