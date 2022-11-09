#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <cstdlib>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
#include <torch/csrc/jit/codegen/cuda/grouped_reduction.h>
#include <torch/csrc/jit/codegen/cuda/inlining.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_graphviz.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_magic_zero.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/reduction_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/test/test_gpu_validator.h>
#include <torch/csrc/jit/codegen/cuda/test/test_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <thread>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;
using namespace at::indexing;

// sh build.sh;
// build/bin/test_jit
// --gtest_filter='NVFuserTest*FusionIndexSelectExplicitBroadcast_CUDA*'
// build/bin/test_jit --gtest_filter='NVFuserTest*FusionIndexSelect_CUDA*'
// build/bin/test_jit --gtest_filter='NVFuserTest*FusionIndexSelect3DTv_CUDA*'
// build/bin/test_jit --gtest_filter='NVFuserTest*FusionIndexSelectCanSch_CUDA*'
// build/bin/test_jit --gtest_filter='NVFuserTest*FusionIndexSelect_Sum_CUDA*'
// build/bin/test_jit --gtest_filter='NVFuserTest*FusionIndexSelect1DSch_CUDA*'
// build/bin/test_jit
// --gtest_filter='NVFuserTest*FusionIndexSelectIdxTvFuseable_CUDA*'
// build/bin/test_jit
// --gtest_filter='NVFuserTest*FusionIndexSelectDim1InRank2_CUDA*'
// build/bin/test_jit
// --gtest_filter='NVFuserTest*FusionIndexSelectDim2InRank3_CUDA*'
TEST_F(NVFuserTest, FusionIndexSelectExplicitBroadcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 2;
  int nElem = 69;
  int nElem_select = nElem + 27;
  int nFeat = 66;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);

  // Register your inputs
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);

  TensorView* tv_idx_bc = broadcast(tv_idx, {false, true});
  TensorView* tv_sel = index_select(tv0, 0, tv_idx_bc);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
  // Register your outputs
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nElem, nFeat}, options); // lookup
  at::Tensor input1 =
      at::randn({nElem_select, nFeat}, options); // output&elemwise
  std::vector<int64_t> storage(nElem_select);
  for (int i = 0; i < nElem_select; ++i) {
    storage[i] = std::rand() % nElem;
  }
  auto opts = torch::TensorOptions().dtype(torch::kLong);
  auto input_idx_cpu =
      torch::from_blob(storage.data(), {int64_t(storage.size())}, opts).clone();
  auto input_idx = input_idx_cpu.to(torch::kCUDA);
  at::Tensor output = at::empty_like(input1);

  std::vector<IValue> aten_inputs = {input1, input0, input_idx};
  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  fe.runFusion(aten_inputs, {output}, lparams);

  auto tv0_ref = at::index_select(input0, 0, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_ref = tv2_ref + 17.0;

  TORCH_CHECK(output_ref.allclose(output));
}

TEST_F(NVFuserTest, FusionIndexSelect_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 2;
  int nElem = 1023;
  int nElem_select = nElem + 115;
  int nFeat = 128;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);

  // Register your inputs
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);

  TensorView* tv_sel = index_select(tv0, 0, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
  // Register your outputs
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nElem, nFeat}, options); // lookup
  at::Tensor input1 =
      at::randn({nElem_select, nFeat}, options); // output&elemwise
  std::vector<int64_t> storage(nElem_select);
  for (int i = 0; i < nElem_select; ++i) {
    storage[i] = std::rand() % nElem;
  }
  auto opts = torch::TensorOptions().dtype(torch::kLong);
  auto input_idx_cpu =
      torch::from_blob(storage.data(), {int64_t(storage.size())}, opts).clone();
  auto input_idx = input_idx_cpu.to(torch::kCUDA);
  at::Tensor output = at::empty_like(input1);

  std::vector<IValue> aten_inputs = {input1, input0, input_idx};
  auto lparams = schedulePointwise(&fusion, aten_inputs);
  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  fe.runFusion(aten_inputs, {output}, lparams);

  auto tv0_ref = at::index_select(input0, 0, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_ref = tv2_ref + 17.0;

  TORCH_CHECK(output_ref.allclose(output));
}

// Test 1D schedule
// If (n_elems * 2 > device_multiprocessor_count * kThreadX), just use 1D
// scheduler or use 2D scheduler
TEST_F(NVFuserTest, FusionIndexSelect1DSch_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 2;
  int nElem = 13;
  int nElem_select = nElem + 1;
  int nFeat = 7;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  // Register your inputs
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);

  TensorView* tv_sel = index_select(tv0, 0, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
  // Register your outputs
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nElem, nFeat}, options); // lookup
  at::Tensor input1 =
      at::randn({nElem_select, nFeat}, options); // output&elemwise
  std::vector<int64_t> storage(nElem_select);
  for (int i = 0; i < nElem_select; ++i) {
    storage[i] = std::rand() % nElem;
  }
  auto opts = torch::TensorOptions().dtype(torch::kLong);
  auto input_idx_cpu =
      torch::from_blob(storage.data(), {int64_t(storage.size())}, opts).clone();
  auto input_idx = input_idx_cpu.to(torch::kCUDA);
  at::Tensor output = at::empty_like(input1);

  std::vector<IValue> aten_inputs = {input1, input0, input_idx};
  auto lparams = schedulePointwise(&fusion, aten_inputs);
  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  fe.runFusion(aten_inputs, {output}, lparams);

  auto tv0_ref = at::index_select(input0, 0, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_ref = tv2_ref + 17.0;

  TORCH_CHECK(output_ref.allclose(output));
}

TEST_F(NVFuserTest, FusionIndexSelect3DTv_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 3;
  int nElem = 89;
  int nElem_select = nElem + 35;
  int nFeat0 = 255;
  int nFeat1 = 63;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);

  // Register your inputs
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);

  TensorView* tv_sel = index_select(tv0, 0, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(27.0), tv2);
  // Register your outputs
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nElem, nFeat0, nFeat1}, options); // lookup
  at::Tensor input1 =
      at::randn({nElem_select, nFeat0, nFeat1}, options); // output&elemwise
  std::vector<int64_t> storage(nElem_select);
  for (int i = 0; i < nElem_select; ++i) {
    storage[i] = std::rand() % nElem;
  }
  auto opts = torch::TensorOptions().dtype(torch::kLong);
  auto input_idx_cpu =
      torch::from_blob(storage.data(), {int64_t(storage.size())}, opts).clone();
  auto input_idx = input_idx_cpu.to(torch::kCUDA);
  at::Tensor output = at::empty_like(input1);

  std::vector<IValue> aten_inputs = {input1, input0, input_idx};
  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  fe.runFusion(aten_inputs, {output}, lparams);

  auto tv0_ref = at::index_select(input0, 0, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_ref = tv2_ref + 27.0;

  TORCH_CHECK(output_ref.allclose(output));
}

TEST_F(NVFuserTest, FusionIndexSelectCanSch_CUDA) {
  // dimensionality of the problem
  int nDims = 2;
  int nElem = 31;
  int nElem_select = nElem + 15;
  int nFeat = 64;

  // Negative Case I
  // lookup tv of index select cannot become conumser of other OP
  // Set up your input tensor views
  Fusion fusion_fail;
  FusionGuard fg(&fusion_fail);
  TensorView* tv_pre = makeContigTensor(nDims);
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  // Register your inputs
  fusion_fail.addInput(tv_pre);
  fusion_fail.addInput(tv1);
  fusion_fail.addInput(tv0);
  fusion_fail.addInput(tv_idx);
  TensorView* tv_t = mul(tv0, tv_pre);
  TensorView* tv_sel = index_select(tv_t, 0, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
  // Register your outputs
  fusion_fail.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nElem, nFeat}, options); // lookup
  at::Tensor input_pre = at::rand_like(input0);

  at::Tensor input1 =
      at::randn({nElem_select, nFeat}, options); // output&elemwise
  std::vector<int64_t> storage(nElem_select);
  for (int i = 0; i < nElem_select; ++i) {
    storage[i] = std::rand() % nElem;
  }
  auto opts = torch::TensorOptions().dtype(torch::kLong);
  auto input_idx_cpu =
      torch::from_blob(storage.data(), {int64_t(storage.size())}, opts).clone();
  auto input_idx = input_idx_cpu.to(torch::kCUDA);
  at::Tensor output = at::empty_like(input1);
  std::vector<IValue> aten_inputs = {input_pre, input1, input0, input_idx};

  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info(&fusion_fail, aten_inputs, true);
  auto sch_fail = SchedulerEntry::canSchedule(
      ScheduleHeuristic::PointWise, &fusion_fail, runtime_info);

  // Negative Case II
  // lookup tv of index select cannot become conumser of other OP
  // Set up your input tensor views
  Fusion fusion_sum_fail;
  FusionGuard fg_sum(&fusion_sum_fail);
  TensorView* tv_sum_pre = makeContigTensor(nDims);
  TensorView* tv_sum_0 = makeContigTensor(nDims);
  TensorView* tv_sum_1 = makeContigTensor(nDims);
  TensorView* tv_sum_idx = makeContigTensor(1, DataType::Int);
  // Register your inputs
  fusion_sum_fail.addInput(tv_sum_pre);
  fusion_sum_fail.addInput(tv_sum_1);
  fusion_sum_fail.addInput(tv_sum_0);
  fusion_sum_fail.addInput(tv_sum_idx);
  TensorView* tv_sum_t = mul(tv_sum_0, tv_sum_pre);
  TensorView* tv_sum_sel = index_select(tv_sum_t, 0, tv_sum_idx);
  TensorView* tv_sum_2 = mul(tv_sum_1, tv_sum_sel);
  TensorView* tv_sum_add = add(IrBuilder::create<Double>(17.0), tv_sum_2);
  auto tv_sum_3 = sum(tv_sum_add, {1});
  // Register your outputs
  fusion_sum_fail.addOutput(tv_sum_3);
  std::vector<IValue> aten_sum_inputs = {input_pre, input1, input0, input_idx};
  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_sum_info(
      &fusion_sum_fail, aten_sum_inputs, true);
  auto sch_sum_fail = SchedulerEntry::canSchedule(
      ScheduleHeuristic::Reduction, &fusion_sum_fail, runtime_sum_info);

  // Positive  Case I
  Fusion fusion_pass;
  FusionGuard fg_p(&fusion_pass);
  TensorView* tv0_p = makeContigTensor(nDims);
  TensorView* tv1_p = makeContigTensor(nDims);
  TensorView* tv_idx_p = makeContigTensor(1, DataType::Int);
  // Register your inputs
  fusion_pass.addInput(tv1_p);
  fusion_pass.addInput(tv0_p);
  fusion_pass.addInput(tv_idx_p);
  TensorView* tv_sel_p = index_select(tv0_p, 0, tv_idx_p);
  TensorView* tv2_p = mul(tv1_p, tv_sel_p);
  TensorView* tv3_p = add(IrBuilder::create<Double>(17.0), tv2_p);
  // Register your outputs
  fusion_pass.addOutput(tv3_p);
  // Schedule through magic scheduler
  std::vector<IValue> aten_inputs_pass = {input1, input0, input_idx};
  SchedulerRuntimeInfo runtime_info_pass(&fusion_pass, aten_inputs_pass, true);
  auto sch_pass = SchedulerEntry::canSchedule(
      ScheduleHeuristic::PointWise, &fusion_pass, runtime_info_pass);

  TORCH_CHECK(sch_pass == true && sch_fail == false && sch_sum_fail == false);
}

TEST_F(NVFuserTest, FusionIndexSelect_Sum_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 2;
  int nElem = 1023;
  int nElem_select = nElem + 115;
  int nFeat = 128;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  // Register your inputs
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  TensorView* tv_sel = index_select(tv0, 0, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv_add = add(IrBuilder::create<Double>(17.0), tv2);
  auto tv3 = sum(tv_add, {1});
  // Register your outputs
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nElem, nFeat}, options); // lookup
  at::Tensor input1 =
      at::randn({nElem_select, nFeat}, options); // output&elemwise
  std::vector<int64_t> storage(nElem_select);
  for (int i = 0; i < nElem_select; ++i) {
    storage[i] = std::rand() % nElem;
  }
  auto opts = torch::TensorOptions().dtype(torch::kLong);
  auto input_idx_cpu =
      torch::from_blob(storage.data(), {int64_t(storage.size())}, opts).clone();
  auto input_idx = input_idx_cpu.to(torch::kCUDA);
  at::Tensor output = at::empty_like(at::randn({nElem_select}, options));

  std::vector<IValue> aten_inputs = {input1, input0, input_idx};
  auto reduction_params = getReductionHeuristics(&fusion, aten_inputs);
  scheduleReduction(&fusion, *reduction_params);
  auto lparams = reduction_params->lparams;
  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  fe.runFusion(aten_inputs, {output}, lparams);

  auto tv0_ref = at::index_select(input0, 0, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_add = tv2_ref + 17.0;
  at::Tensor output_ref = output_add.sum({1});

  TORCH_CHECK(output_ref.allclose(output));
}

TEST_F(NVFuserTest, FusionIndexSelectIdxTvFuseable_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 2;
  int nElem = 23;
  int nElem_select = nElem + 15;
  int nFeat = 32;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  TensorView* tv_idx_pre = makeContigTensor(1, DataType::Int);
  // Register your inputs
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  fusion.addInput(tv_idx_pre);

  TensorView* tv_idx_ret = add(tv_idx, tv_idx_pre);
  TensorView* tv_sel = index_select(tv0, 0, tv_idx_ret);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
  // Register your outputs
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nElem, nFeat}, options); // lookup
  at::Tensor input1 =
      at::randn({nElem_select, nFeat}, options); // output&elemwise
  std::vector<int64_t> storage(nElem_select);
  std::vector<int64_t> storage_zero(nElem_select);
  for (int i = 0; i < nElem_select; ++i) {
    storage[i] = std::rand() % nElem;
    storage_zero[i] = 0;
  }
  auto opts = torch::TensorOptions().dtype(torch::kLong);
  auto input_idx_cpu =
      torch::from_blob(storage.data(), {int64_t(storage.size())}, opts).clone();
  auto input_idx_pre_cpu =
      torch::from_blob(
          storage_zero.data(), {int64_t(storage_zero.size())}, opts)
          .clone();

  auto input_idx = input_idx_cpu.to(torch::kCUDA);
  auto input_idx_pre = input_idx_pre_cpu.to(torch::kCUDA);
  at::Tensor output = at::empty_like(input1);

  std::vector<IValue> aten_inputs = {input1, input0, input_idx, input_idx_pre};
  auto lparams = schedulePointwise(&fusion, aten_inputs);
  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  fe.runFusion(aten_inputs, {output}, lparams);

  auto tv0_ref = at::index_select(input0, 0, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_ref = tv2_ref + 17.0;

  TORCH_CHECK(output_ref.allclose(output));
}

TEST_F(NVFuserTest, FusionIndexSelectDim1InRank2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 2;
  int nElem = 4;
  int nElem_select = nElem - 2;
  int nFeat = 3;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);

  // Register your inputs
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);

  TensorView* tv_sel = index_select(tv0, 1, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
  // Register your outputs
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nFeat, nElem}, options); // lookup
  at::Tensor input1 =
      at::randn({nFeat, nElem_select}, options); // output&elemwise
  std::vector<int64_t> storage(nElem_select);
  for (int i = 0; i < nElem_select; ++i) {
    storage[i] = std::rand() % nElem;
  }
  auto opts = torch::TensorOptions().dtype(torch::kLong);
  auto input_idx_cpu =
      torch::from_blob(storage.data(), {int64_t(storage.size())}, opts).clone();
  auto input_idx = input_idx_cpu.to(torch::kCUDA);
  at::Tensor output = at::empty_like(input1);

  std::vector<IValue> aten_inputs = {input1, input0, input_idx};
  auto lparams = schedulePointwise(&fusion, aten_inputs);
  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  fe.runFusion(aten_inputs, {output}, lparams);

  auto tv0_ref = at::index_select(input0, 1, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_ref = tv2_ref + 17.0;

  TORCH_CHECK(output_ref.allclose(output));
}

TEST_F(NVFuserTest, FusionIndexSelectDim2InRank3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 3;
  int nElem = 4;
  int nElem_select = nElem - 2;
  int nFeat0 = 5;
  int nFeat1 = 7;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);

  // Register your inputs
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);

  TensorView* tv_sel = index_select(tv0, 2, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
  // Register your outputs
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nFeat0, nFeat1, nElem}, options); // lookup
  at::Tensor input1 =
      at::randn({nFeat0, nFeat1, nElem_select}, options); // output&elemwise
  std::vector<int64_t> storage(nElem_select);
  for (int i = 0; i < nElem_select; ++i) {
    storage[i] = std::rand() % nElem;
  }
  auto opts = torch::TensorOptions().dtype(torch::kLong);
  auto input_idx_cpu =
      torch::from_blob(storage.data(), {int64_t(storage.size())}, opts).clone();
  auto input_idx = input_idx_cpu.to(torch::kCUDA);
  at::Tensor output = at::empty_like(input1);

  std::vector<IValue> aten_inputs = {input1, input0, input_idx};
  auto lparams = schedulePointwise(&fusion, aten_inputs);
  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  fe.runFusion(aten_inputs, {output}, lparams);

  auto tv0_ref = at::index_select(input0, 2, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_ref = tv2_ref + 17.0;

  TORCH_CHECK(output_ref.allclose(output));
}

TEST_F(NVFuserTest, FusionIndexSelectDim1InRank3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 3;
  int nElem = 4;
  int nElem_select = nElem - 2;
  int nFeat0 = 5;
  int nFeat1 = 7;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);

  // Register your inputs
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);

  TensorView* tv_sel = index_select(tv0, 1, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
  // Register your outputs
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({nFeat0, nElem, nFeat1}, options); // lookup
  at::Tensor input1 =
      at::randn({nFeat0, nElem_select, nFeat1}, options); // output&elemwise
  std::vector<int64_t> storage(nElem_select);
  for (int i = 0; i < nElem_select; ++i) {
    storage[i] = std::rand() % nElem;
  }
  auto opts = torch::TensorOptions().dtype(torch::kLong);
  auto input_idx_cpu =
      torch::from_blob(storage.data(), {int64_t(storage.size())}, opts).clone();
  auto input_idx = input_idx_cpu.to(torch::kCUDA);
  at::Tensor output = at::empty_like(input1);

  std::vector<IValue> aten_inputs = {input1, input0, input_idx};
  auto lparams = schedulePointwise(&fusion, aten_inputs);
  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  fe.runFusion(aten_inputs, {output}, lparams);

  auto tv0_ref = at::index_select(input0, 1, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_ref = tv2_ref + 17.0;

  TORCH_CHECK(output_ref.allclose(output));
}

TEST_F(NVFuserTest, FusionIndexSelectDim2InRank4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 4;
  int nElem = 4;
  int nElem_select = nElem + 15;
  int nFeat0 = 5;
  int nFeat1 = 7;
  int nFeat2 = 25;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);

  // Register your inputs
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);

  TensorView* tv_sel = index_select(tv0, 1, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Double>(17.0), tv2);
  // Register your outputs
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 =
      at::randn({nFeat0, nElem, nFeat1, nFeat2}, options); // lookup
  at::Tensor input1 = at::randn(
      {nFeat0, nElem_select, nFeat1, nFeat2}, options); // output&elemwise
  std::vector<int64_t> storage(nElem_select);
  for (int i = 0; i < nElem_select; ++i) {
    storage[i] = std::rand() % nElem;
  }
  auto opts = torch::TensorOptions().dtype(torch::kLong);
  auto input_idx_cpu =
      torch::from_blob(storage.data(), {int64_t(storage.size())}, opts).clone();
  auto input_idx = input_idx_cpu.to(torch::kCUDA);
  at::Tensor output = at::empty_like(input1);

  std::vector<IValue> aten_inputs = {input1, input0, input_idx};
  auto lparams = schedulePointwise(&fusion, aten_inputs);
  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  fe.runFusion(aten_inputs, {output}, lparams);

  auto tv0_ref = at::index_select(input0, 1, input_idx);
  at::Tensor tv2_ref = tv0_ref * input1;
  at::Tensor output_ref = tv2_ref + 17.0;

  TORCH_CHECK(output_ref.allclose(output));
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
