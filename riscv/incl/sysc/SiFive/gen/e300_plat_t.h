#ifndef _E300_PLAT_MAP_H_
#define _E300_PLAT_MAP_H_
// need double braces, see https://stackoverflow.com/questions/6893700/how-to-construct-stdarray-object-with-initializer-list#6894191
const std::array<sysc::target_memory_map_entry<32>, 4> e300_plat_map = {{
    {&i_plic, 0xc000000, 0x1000},
    {&i_gpio, 0x10012000, 0x1000},
    {&i_uart, 0x10013000, 0x1000},
    {&i_spi, 0x10014000, 0x1000},
}};

#endif /* _E300_PLAT_MAP_H_ */
