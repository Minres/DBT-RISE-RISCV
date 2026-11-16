#include <array>
#include <cstdint>
#include <fstream>
#include <iss/instruction_decoder.h>
#include <limits>
#include <vector>

enum class opcode_e {
    C__LBU = 116,
    C__LHU = 117,
    C__LH = 118,
    C__SB = 119,
    C__SH = 120,

};
struct instruction_descriptor {
    uint32_t length;
    uint32_t value;
    uint32_t mask;
    opcode_e op;
};

static constexpr std::array<instruction_descriptor, 2> instr_descr = {{//{16, 0b1000000000000000, 0b1111110000000011, opcode_e::C__LBU},
                                                                       {16, 0b1000010000000000, 0b1111110001000011, opcode_e::C__LHU},
                                                                       {16, 0b1000010001000000, 0b1111110001000011, opcode_e::C__LH}}};

int main(int argc, char* argv[]) {
    iss::decoder<uint64_t, 2> instr_decoder(nonstd::span{[]() {
        std::vector<iss::generic_instruction_descriptor> g_instr_descr;
        g_instr_descr.reserve(instr_descr.size());
        for(uint32_t i = 0; i < instr_descr.size(); ++i)
            g_instr_descr.emplace_back(iss::generic_instruction_descriptor{instr_descr[i].value, instr_descr[i].mask, i});
        return g_instr_descr;
    }()});

    std::ofstream json_out{"idecode_tree.json"};
    json_out << instr_decoder.print_tree_as_json();

    auto inst_index1 = instr_decoder.decode_instr_cached(0, 0x8440);
    auto inst_index2 = instr_decoder.decode_instr_cached(0, 0x86b0);

    return inst_index1 == inst_index2 ? 0 : 1;
}
