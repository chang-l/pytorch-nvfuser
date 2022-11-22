
// Based on NVFuserTest.FusionBiasGeluBwd_CUDA

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmarks/cpp/nvfuser/utils.h>

using namespace torch::jit::fuser::cuda;

static void setupFusion(Fusion* fusion) {
  FusionGuard fg(fusion);

  // set up input tensor views
  auto t0 = makeContigTensor(2); // nDim = 2
  fusion->addInput(t0);
  // scaling tensor
  auto t1 = makeContigTensor(2);
  fusion->addInput(t1);
  auto t_idx = makeContigTensor(1, DataType::Int);
  fusion->addInput(t_idx);

  auto t2 = index_select(t0, 0, t_idx); // select at dim=0
  auto t3 = mul(t1, t2);
  auto t4 = add(t3, IrBuilder::create<Double>(17.0));

  // Save float output for validation
  fusion->addOutput(t4);
}

static std::vector<c10::IValue> setupInputs() {
  at::manual_seed(0);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  int nElem = 1023;
  int nElem_select = nElem + 115;
  int nFeat = 128;
  std::vector<int64_t> input_shape{nElem, nFeat};
  std::vector<int64_t> select_shape{nElem_select, nFeat};
  auto at_input = at::randn(input_shape, options);
  auto at_select = at::randn(select_shape, options);
  auto indx_options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto at_index = at::randint(nElem, {nElem_select}, indx_options);
  return {at_select, at_input, at_index};
}

//------------------------------------------------------------------------------

static void IndexSelect_SetupFusion(benchmark::State& benchmark_state) {
  for (auto _ : benchmark_state) {
    Fusion fusion;
    setupFusion(&fusion);
  }
}

BENCHMARK(IndexSelect_SetupFusion)->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------

static void IndexSelect_AutoSchedule(benchmark::State& benchmark_state) {
  for (auto _ : benchmark_state) {
    // Setup (not included in the measurement)
    benchmark_state.PauseTiming();
    Fusion fusion;
    setupFusion(&fusion);
    std::vector<c10::IValue> inputs = setupInputs();
    benchmark_state.ResumeTiming();

    // Auto-schedule
    schedulePointwise(&fusion, c10::ArrayRef<c10::IValue>(inputs));
  }
}

BENCHMARK(IndexSelect_AutoSchedule)->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------

static void IndexSelect_Lower(benchmark::State& benchmark_state) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  std::vector<c10::IValue> inputs = setupInputs();

  schedulePointwise(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  for (auto _ : benchmark_state) {
    GpuLower gpu_lower(&fusion);
  }
}

BENCHMARK(IndexSelect_Lower)->Unit(benchmark::kMillisecond);

//------------------------------------------------------------------------------

static void IndexSelect_Compile(benchmark::State& benchmark_state) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  std::vector<c10::IValue> inputs = setupInputs();

  auto lparams = schedulePointwise(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  for (auto _ : benchmark_state) {
    FusionExecutor executor;
    executor.compileFusion(&fusion, c10::ArrayRef<c10::IValue>(inputs), lparams);
  }
}

BENCHMARK(IndexSelect_Compile)->Unit(benchmark::kMillisecond);

//------------------------------------------------------------------------------

static void IndexSelect_RunFusion(benchmark::State& benchmark_state) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  std::vector<c10::IValue> inputs = setupInputs();

  auto lparams = schedulePointwise(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  FusionExecutor executor;
  executor.compileFusion(&fusion, c10::ArrayRef<c10::IValue>(inputs), lparams);

  C10_CUDA_CHECK(cudaDeviceSynchronize());

  at::Tensor output = at::empty_like(inputs[0].toTensor());

  for (auto _ : benchmark_state) {
    executor.runFusion(c10::ArrayRef<c10::IValue>(inputs), {output}, lparams);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    clearL2Cache();
  }
}

BENCHMARK(IndexSelect_RunFusion)->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------

// static void IndexSelect_RunFusion_GpuOnly(benchmark::State& benchmark_state) {
//   Fusion fusion;

//   // setup fusion
//   setupFusion(&fusion);

//   // inputs
//   std::vector<c10::IValue> inputs = setupInputs();

//   // outputs
//   std::vector<at::Tensor> outputs;

//   auto lparams = schedulePointwise(&fusion, c10::ArrayRef<c10::IValue>(inputs));

//   FusionExecutor executor;
//   executor.setMeasureKernelTimeFlag(true);
//   executor.compileFusion(&fusion);

//   C10_CUDA_CHECK(cudaDeviceSynchronize());

//   for (auto _ : benchmark_state) {
//     outputs = executor.runFusion(c10::ArrayRef<c10::IValue>(inputs), lparams);
//     benchmark_state.SetIterationTime(executor.kernelTimeMs() / 1000.0);
//     clearL2Cache();
//   }
// }

// BENCHMARK(IndexSelect_RunFusion_GpuOnly)
//     ->Unit(benchmark::kMicrosecond)
//     ->UseManualTime();

// //------------------------------------------------------------------------------

// static void IndexSelect_RunFusion_CpuOnly(benchmark::State& benchmark_state) {
//   Fusion fusion;

//   // setup fusion
//   setupFusion(&fusion);

//   // inputs
//   std::vector<c10::IValue> inputs = setupInputs();

//   // outputs
//   std::vector<at::Tensor> outputs;

//   auto lparams = schedulePointwise(&fusion, c10::ArrayRef<c10::IValue>(inputs));

//   FusionExecutor executor;
//   executor.setExecuteKernelFlag(false);
//   executor.compileFusion(&fusion);

//   for (auto _ : benchmark_state) {
//     outputs = executor.runFusion(c10::ArrayRef<c10::IValue>(inputs), lparams);
//   }
// }

// BENCHMARK(IndexSelect_RunFusion_CpuOnly)->Unit(benchmark::kMicrosecond);
