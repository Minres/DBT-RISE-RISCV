#ifndef _E300_PLAT_T_MAP_H_
#define _E300_PLAT_T_MAP_H_
// need double braces, see
// https://stackoverflow.com/questions/6893700/how-to-construct-stdarray-object-with-initializer-list#6894191
const std::array<scc::target_memory_map_entry<32>, 13> e300_plat_t_map = {{
    {i_clint->socket, 0x2000000, 0xc000},
    {i_plic->socket, 0xc000000, 0x200008},
    {i_aon->socket, 0x10000000, 0x150},
    {i_prci->socket, 0x10008000, 0x14},
    {i_gpio0->socket, 0x10012000, 0x44},
    {i_uart0->socket, 0x10013000, 0x1c},
    {i_qspi0->socket, 0x10014000, 0x78},
    {i_pwm0->socket, 0x10015000, 0x30},
    {i_uart1->socket, 0x10023000, 0x1c},
    {i_qspi1->socket, 0x10024000, 0x78},
    {i_pwm1->socket, 0x10025000, 0x30},
    {i_qspi2->socket, 0x10034000, 0x78},
    {i_pwm2->socket, 0x10035000, 0x30},
}};

#endif /* _E300_PLAT_T_MAP_H_ */
