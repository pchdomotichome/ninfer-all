#include "core/weight.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        // Public numerical cases straddle each registered Q4 implementation interval. They make
        // no assertion about the private route selected for any T.
        // 8, 16 and 17 straddle the small-T dispatch's exact-width boundary as well as the
        // registered intervals: a width that fills its tile runs an instantiation with no runtime
        // column count, and the widths either side of it do not (q4_linear_swiglu_gemv.cu).
        constexpr std::array<std::int32_t, 40> kTokenCases{
            1,   2,   3,   4,   5,   6,   7,   8,   9,   16,  17,  24,  25,  31,
            32,  33,  40,  41,  48,  49,  96,  128, 129, 152, 153, 168, 169, 176,
            192, 224, 225, 256, 257, 384, 385, 512, 513, 640, 641, 1024,
        };
        const int failures =
            run_profile("LinearSwiGLU Q4_A16",
                        {QType::Q4_G64_FP16, 34816, 5120, 17408, 1401U, ActivationCompute::A16},
                        kTokenCases, std::array<std::int32_t, 7>{7, 25, 49, 128, 152, 168, 1024}) +
            // The 4096-hidden dense model. Its intermediate is 12288, so every width in this kernel
            // -- the gate/up row offset, the output stride and the launch grid -- differs from the
            // 5120-hidden model's, and a constant left at the other model's value reads the wrong
            // half of the parent and writes outside the output.
            run_profile("LinearSwiGLU Q4_A16 9B",
                        {QType::Q4_G64_FP16, 24576, 4096, 12288, 1402U, ActivationCompute::A16},
                        kTokenCases, std::array<std::int32_t, 7>{7, 25, 49, 128, 152, 168, 1024});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU Q4_A16 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU Q4_A16 test failed: " << error.what() << '\n';
        return 1;
    }
}
