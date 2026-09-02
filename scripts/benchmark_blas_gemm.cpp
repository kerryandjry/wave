#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#define CHECK_HIP(expr) check_hip((expr), #expr)
#define CHECK_ROCBLAS(expr) check_rocblas((expr), #expr)
#define CHECK_HIPBLAS(expr) check_hipblas((expr), #expr)

static void check_hip(hipError_t status, const char *expr) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(expr) + ": " +
                             hipGetErrorString(status));
}

static void check_rocblas(rocblas_status status, const char *expr) {
  if (status != rocblas_status_success)
    throw std::runtime_error(std::string(expr) + ": rocBLAS status " +
                             std::to_string(status));
}

static void check_hipblas(hipblasStatus_t status, const char *expr) {
  if (status != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(expr) + ": hipBLAS status " +
                             std::to_string(status));
}

__global__ void fill_half(__half *values, size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count)
    values[index] = __float2half(1.0f);
}

struct Buffers {
  void *a = nullptr;
  void *b = nullptr;
  void *c = nullptr;

  Buffers(int m, int n, int k) {
    CHECK_HIP(hipMalloc(&a, static_cast<size_t>(m) * k * sizeof(__half)));
    CHECK_HIP(hipMalloc(&b, static_cast<size_t>(n) * k * sizeof(__half)));
    CHECK_HIP(hipMalloc(&c, static_cast<size_t>(m) * n * sizeof(float)));
    const size_t a_count = static_cast<size_t>(m) * k;
    const size_t b_count = static_cast<size_t>(n) * k;
    fill_half<<<(a_count + 255) / 256, 256>>>(static_cast<__half *>(a),
                                              a_count);
    fill_half<<<(b_count + 255) / 256, 256>>>(static_cast<__half *>(b),
                                              b_count);
    CHECK_HIP(hipMemset(c, 0, static_cast<size_t>(m) * n * sizeof(float)));
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
  }

  ~Buffers() {
    if (a)
      (void)hipFree(a);
    if (b)
      (void)hipFree(b);
    if (c)
      (void)hipFree(c);
  }
};

template <typename Launch>
static float time_launches(Launch launch, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i)
    launch();
  CHECK_HIP(hipDeviceSynchronize());

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  CHECK_HIP(hipEventCreate(&start));
  CHECK_HIP(hipEventCreate(&stop));
  CHECK_HIP(hipEventRecord(start));
  for (int i = 0; i < iterations; ++i)
    launch();
  CHECK_HIP(hipEventRecord(stop));
  CHECK_HIP(hipEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CHECK_HIP(hipEventElapsedTime(&elapsed_ms, start, stop));
  CHECK_HIP(hipEventDestroy(start));
  CHECK_HIP(hipEventDestroy(stop));
  return elapsed_ms / iterations;
}

static float benchmark_rocblas(const Buffers &buffers, int m, int n, int k,
                               int warmup, int iterations) {
  rocblas_handle handle = nullptr;
  CHECK_ROCBLAS(rocblas_create_handle(&handle));
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Row-major C = A * B^T is column-major C^T = B * A^T.
  auto launch = [&]() {
    CHECK_ROCBLAS(rocblas_gemm_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none, n, m, k,
        &alpha, buffers.b, rocblas_datatype_f16_r, k, buffers.a,
        rocblas_datatype_f16_r, k, &beta, buffers.c, rocblas_datatype_f32_r, n,
        buffers.c, rocblas_datatype_f32_r, n, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, 0));
  };
  const float time_ms = time_launches(launch, warmup, iterations);
  CHECK_ROCBLAS(rocblas_destroy_handle(handle));
  return time_ms;
}

static float benchmark_hipblaslt(const Buffers &buffers, int m, int n, int k,
                                 int warmup, int iterations) {
  hipblasLtHandle_t handle = nullptr;
  hipblasLtMatmulDesc_t operation = nullptr;
  hipblasLtMatrixLayout_t a_layout = nullptr;
  hipblasLtMatrixLayout_t b_layout = nullptr;
  hipblasLtMatrixLayout_t c_layout = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  void *workspace = nullptr;
  constexpr size_t workspace_size = 32ULL * 1024 * 1024;

  CHECK_HIPBLAS(hipblasLtCreate(&handle));
  CHECK_HIPBLAS(
      hipblasLtMatmulDescCreate(&operation, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  const hipblasOperation_t trans_a = HIPBLAS_OP_T;
  const hipblasOperation_t trans_b = HIPBLAS_OP_N;
  CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
      operation, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
  CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(
      operation, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

  // The same column-major reinterpretation used by the rocBLAS path.
  CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&a_layout, HIP_R_16F, k, n, k));
  CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&b_layout, HIP_R_16F, k, m, k));
  CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&c_layout, HIP_R_32F, n, m, n));
  CHECK_HIPBLAS(hipblasLtMatmulPreferenceCreate(&preference));
  CHECK_HIPBLAS(hipblasLtMatmulPreferenceSetAttribute(
      preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size,
      sizeof(workspace_size)));
  CHECK_HIP(hipMalloc(&workspace, workspace_size));

  hipblasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  CHECK_HIPBLAS(hipblasLtMatmulAlgoGetHeuristic(
      handle, operation, a_layout, b_layout, c_layout, c_layout, preference, 1,
      &heuristic, &returned));
  if (returned == 0 || heuristic.state != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("hipBLASLt found no GEMM algorithm");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto launch = [&]() {
    CHECK_HIPBLAS(hipblasLtMatmul(
        handle, operation, &alpha, buffers.b, a_layout, buffers.a, b_layout,
        &beta, buffers.c, c_layout, buffers.c, c_layout, &heuristic.algo,
        workspace, workspace_size, nullptr));
  };
  const float time_ms = time_launches(launch, warmup, iterations);

  CHECK_HIP(hipFree(workspace));
  CHECK_HIPBLAS(hipblasLtMatmulPreferenceDestroy(preference));
  CHECK_HIPBLAS(hipblasLtMatrixLayoutDestroy(c_layout));
  CHECK_HIPBLAS(hipblasLtMatrixLayoutDestroy(b_layout));
  CHECK_HIPBLAS(hipblasLtMatrixLayoutDestroy(a_layout));
  CHECK_HIPBLAS(hipblasLtMatmulDescDestroy(operation));
  CHECK_HIPBLAS(hipblasLtDestroy(handle));
  return time_ms;
}

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "usage: " << argv[0]
              << " <rocblas|hipblaslt> <M> <N> <K> <warmup> <iterations>\n";
    return 2;
  }

  try {
    const std::string backend = argv[1];
    const int m = std::stoi(argv[2]);
    const int n = std::stoi(argv[3]);
    const int k = std::stoi(argv[4]);
    const int warmup = std::stoi(argv[5]);
    const int iterations = std::stoi(argv[6]);
    Buffers buffers(m, n, k);

    float time_ms = 0.0f;
    if (backend == "rocblas")
      time_ms = benchmark_rocblas(buffers, m, n, k, warmup, iterations);
    else if (backend == "hipblaslt")
      time_ms = benchmark_hipblaslt(buffers, m, n, k, warmup, iterations);
    else
      throw std::runtime_error("unknown backend: " + backend);

    float first = 0.0f;
    float last = 0.0f;
    CHECK_HIP(
        hipMemcpy(&first, buffers.c, sizeof(first), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(&last,
                        static_cast<const float *>(buffers.c) +
                            static_cast<size_t>(m) * n - 1,
                        sizeof(last), hipMemcpyDeviceToHost));
    if (std::abs(first - k) > 1.0f || std::abs(last - k) > 1.0f)
      throw std::runtime_error("GEMM correctness sample failed");

    const double flops = 2.0 * m * static_cast<double>(n) * k;
    const double tflops = flops / (time_ms / 1000.0) / 1.0e12;
    std::cout << std::fixed << std::setprecision(8) << "{\"backend\":\""
              << backend << "\",\"M\":" << m << ",\"N\":" << n << ",\"K\":" << k
              << ",\"warmup\":" << warmup << ",\"iterations\":" << iterations
              << ",\"time_ms\":" << time_ms << ",\"tflops\":" << tflops
              << "}\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
