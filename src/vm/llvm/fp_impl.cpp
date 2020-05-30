////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, MINRES Technologies GmbH
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Contributors:
//       eyck@minres.com - initial API and implementation
////////////////////////////////////////////////////////////////////////////////

#include <iss/iss.h>
#include <iss/llvm/vm_base.h>

extern "C" {
#include <softfloat.h>
#include "internals.h"
#include "specialize.h"
}

#include <limits>

namespace iss {
namespace llvm {
namespace fp_impl {

using namespace std;
using namespace ::llvm;

#define INT_TYPE(L)   Type::getIntNTy(mod->getContext(), L)
#define FLOAT_TYPE    Type::getFloatTy(mod->getContext())
#define DOUBLE_TYPE   Type::getDoubleTy(mod->getContext())
#define VOID_TYPE     Type::getVoidTy(mod->getContext())
#define THIS_PTR_TYPE Type::getIntNPtrTy(mod->getContext(), 8)
#define FDECLL(NAME, RET, ...)                                                                                         \
    Function *NAME##_func = CurrentModule->getFunction(#NAME);                                                         \
    if (!NAME##_func) {                                                                                                \
        std::vector<Type *> NAME##_args{__VA_ARGS__};                                                                  \
        FunctionType *NAME##_type = FunctionType::get(RET, NAME##_args, false);                                        \
        NAME##_func = Function::Create(NAME##_type, GlobalValue::ExternalLinkage, #NAME, CurrentModule);               \
        NAME##_func->setCallingConv(CallingConv::C);                                                                   \
    }

#define FDECL(NAME, RET, ...)                                                                                          \
    std::vector<Type *> NAME##_args{__VA_ARGS__};                                                                      \
    FunctionType *NAME##_type = FunctionType::get(RET, NAME##_args, false);                                      \
    mod->getOrInsertFunction(#NAME, NAME##_type);


void add_fp_functions_2_module(Module *mod, uint32_t flen, uint32_t xlen) {
    if(flen){
        FDECL(fget_flags, INT_TYPE(32));
        FDECL(fadd_s,     INT_TYPE(32), INT_TYPE(32), INT_TYPE(32), INT_TYPE(8));
        FDECL(fsub_s,     INT_TYPE(32), INT_TYPE(32), INT_TYPE(32), INT_TYPE(8));
        FDECL(fmul_s,     INT_TYPE(32), INT_TYPE(32), INT_TYPE(32), INT_TYPE(8));
        FDECL(fdiv_s,     INT_TYPE(32), INT_TYPE(32), INT_TYPE(32), INT_TYPE(8));
        FDECL(fsqrt_s,    INT_TYPE(32), INT_TYPE(32), INT_TYPE(8));
        FDECL(fcmp_s,     INT_TYPE(32), INT_TYPE(32), INT_TYPE(32), INT_TYPE(32));
        FDECL(fcvt_s,     INT_TYPE(32), INT_TYPE(32), INT_TYPE(32), INT_TYPE(8));
        FDECL(fmadd_s,    INT_TYPE(32), INT_TYPE(32), INT_TYPE(32), INT_TYPE(32), INT_TYPE(32), INT_TYPE(8));
        FDECL(fsel_s,     INT_TYPE(32), INT_TYPE(32), INT_TYPE(32), INT_TYPE(32));
        FDECL(fclass_s,   INT_TYPE(32), INT_TYPE(32));
        FDECL(fcvt_32_64,     INT_TYPE(64), INT_TYPE(32), INT_TYPE(32), INT_TYPE(8));
        FDECL(fcvt_64_32,     INT_TYPE(32), INT_TYPE(64), INT_TYPE(32), INT_TYPE(8));
        if(flen>32){
            FDECL(fconv_d2f,  INT_TYPE(32), INT_TYPE(64), INT_TYPE(8));
            FDECL(fconv_f2d,  INT_TYPE(64), INT_TYPE(32), INT_TYPE(8));
            FDECL(fadd_d,     INT_TYPE(64), INT_TYPE(64), INT_TYPE(64), INT_TYPE(8));
            FDECL(fsub_d,     INT_TYPE(64), INT_TYPE(64), INT_TYPE(64), INT_TYPE(8));
            FDECL(fmul_d,     INT_TYPE(64), INT_TYPE(64), INT_TYPE(64), INT_TYPE(8));
            FDECL(fdiv_d,     INT_TYPE(64), INT_TYPE(64), INT_TYPE(64), INT_TYPE(8));
            FDECL(fsqrt_d,    INT_TYPE(64), INT_TYPE(64), INT_TYPE(8));
            FDECL(fcmp_d,     INT_TYPE(64), INT_TYPE(64), INT_TYPE(64), INT_TYPE(32));
            FDECL(fcvt_d,     INT_TYPE(64), INT_TYPE(64), INT_TYPE(32), INT_TYPE(8));
            FDECL(fmadd_d,    INT_TYPE(64), INT_TYPE(64), INT_TYPE(64), INT_TYPE(64), INT_TYPE(32), INT_TYPE(8));
            FDECL(fsel_d,     INT_TYPE(64), INT_TYPE(64), INT_TYPE(64), INT_TYPE(32));
            FDECL(fclass_d,   INT_TYPE(64), INT_TYPE(64));
            FDECL(unbox_s,      INT_TYPE(32), INT_TYPE(64));

        }
    }
}

}
}
}
