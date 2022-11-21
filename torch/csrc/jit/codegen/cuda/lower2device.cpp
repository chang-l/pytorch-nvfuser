#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_alias_memory.h>
#include <torch/csrc/jit/codegen/cuda/lower_allocation.h>
#include <torch/csrc/jit/codegen/cuda/lower_divisible_split.h>
#include <torch/csrc/jit/codegen/cuda/lower_double_buffer.h>
#include <torch/csrc/jit/codegen/cuda/lower_expr_sort.h>
#include <torch/csrc/jit/codegen/cuda/lower_fusion_simplifier.h>
#include <torch/csrc/jit/codegen/cuda/lower_index.h>
#include <torch/csrc/jit/codegen/cuda/lower_insert_syncs.h>
#include <torch/csrc/jit/codegen/cuda/lower_instrument.h>
#include <torch/csrc/jit/codegen/cuda/lower_loops.h>
#include <torch/csrc/jit/codegen/cuda/lower_magic_zero.h>
#include <torch/csrc/jit/codegen/cuda/lower_misaligned_vectorization.h>
#include <torch/csrc/jit/codegen/cuda/lower_predicate.h>
#include <torch/csrc/jit/codegen/cuda/lower_replace_size.h>
#include <torch/csrc/jit/codegen/cuda/lower_shift.h>
#include <torch/csrc/jit/codegen/cuda/lower_unroll.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_validation.h>
#include <torch/csrc/jit/codegen/cuda/lower_warp_reduce.h>

#include <list>
#include <unordered_map>
#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

thread_local GpuLower* active_gpu_lower = nullptr; // NOLINT
namespace {

class KIRCleaner : public OptOutDispatch {
 public:
  //! Remove nop IR nodes
  static std::vector<Expr*> cleanUp(const std::vector<Expr*>& loop_nests) {
    KIRCleaner cleaner;
    std::vector<Expr*> out_loop_nests;
    for (auto loop_nest : loop_nests) {
      cleaner.handle(loop_nest);
      // No need to keep the loop nest if it's determined to be nop
      if (!cleaner.is_nop_) {
        out_loop_nests.push_back(loop_nest);
      }
    }
    return out_loop_nests;
  }

 private:
  using OptOutDispatch::handle;
  void handle(Expr* expr) final {
    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      OptOutDispatch::handle(expr);
    } else {
      // Any non-scoping expr is not considered nop
      is_nop_ = false;
    }
  }

  void handle(kir::ForLoop* fl) final {
    auto exprs = fl->body().exprs();
    fl->body().clear();
    for (auto expr : exprs) {
      handle(expr);
      // Add the expr to the loop body only when the expr is not nop
      if (!is_nop_) {
        fl->body().push_back(expr);
      }
    }
    // The loop is nop when no expr exists in the body
    is_nop_ = fl->body().empty();
  }

  void handle(kir::IfThenElse* ite) final {
    const auto conditional = ite->predicate()->value();

    // Visit the then block
    auto then_exprs = ite->thenBody().exprs();
    ite->thenBody().clear();
    if (!conditional->isConst() || conditional->value().value()) {
      for (auto expr : then_exprs) {
        handle(expr);
        if (!is_nop_) {
          ite->thenBody().push_back(expr);
        }
      }
    }

    const bool then_nop = ite->thenBody().empty();

    // Visit the else block
    auto else_exprs = ite->elseBody().exprs();
    ite->elseBody().clear();
    if (!conditional->isConst() || !conditional->value().value()) {
      for (auto expr : else_exprs) {
        handle(expr);
        if (!is_nop_) {
          ite->elseBody().push_back(expr);
        }
      }
    }

    const bool else_nop = ite->elseBody().empty();

    // If the then block is nop but the else is not, invert the
    // conditional and move the exprs in the else block to the then
    // block.
    if (then_nop && !else_nop) {
      Bool* pred = ite->predicate()->value();
      Bool* not_pred = SimplifyingIrBuilder::notExpr(pred)->as<Bool>();
      ite->predicate()->setValue(not_pred);
      for (auto expr : ite->elseBody().exprs()) {
        ite->thenBody().push_back(expr);
      }
      ite->elseBody().clear();
    }

    // This IfThenElse is nop if both the then and else blocks are nop
    is_nop_ = then_nop && else_nop;
  }

 private:
  //! True if the last visited expr is nop
  bool is_nop_ = false;
};

} // namespace

void GpuLower::collectPaddedParallelDims() {
  bool can_be_single_warp = true;

  auto warp_size = at::cuda::warp_size();

  auto used_vals = fusion_->usedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    for (auto id : tv->domain()->domain()) {
      if (tv->definition()) {
        // TODO: Support GroupedReductionOp
        if (auto reduction = dynamic_cast<ReductionOp*>(tv->definition())) {
          if (ir_utils::getMaybeWarpReductionDim(
                  reduction->out(), reduction->in())
                  .has_value()) {
            warp_pad_info_.has_warp_reduction = true;
          }
        }
      }

      // Check ifi TIDx is padded in this kernel
      if (id->hasPaddingToMultipleOfWarp()) {
        TORCH_INTERNAL_ASSERT(
            id->getParallelType() == ParallelType::TIDx,
            "Padded types supported only on TIDx");
        warp_pad_info_.is_tidx_padded = true;
      }

      // Check all possible bindings of TIDx to see
      //  if TIDx will eventually be bound to a single warp.
      if (id->getParallelType() == ParallelType::TIDx) {
        auto size_after_padding = id->getMaybeSizeAfterPadding();
        bool padding_to_single_warp = size_after_padding.has_value() &&
            size_after_padding.value() == warp_size;

        if (id->extent()->isConstInt() &&
            id->extent()->evaluateInt() > warp_size &&
            !padding_to_single_warp) {
          // If we see any other TIDx binding that's larger than
          //  a warp or unknown, we shouldn't lower warp reduce
          //  to a single warp type.
          can_be_single_warp = false;
          warp_pad_info_.is_tidx_single_warp = false;
        } else if (can_be_single_warp) {
          if (padding_to_single_warp ||
              (id->extent()->isConstInt() &&
               id->extent()->evaluateInt() == warp_size)) {
            warp_pad_info_.is_tidx_single_warp = true;
          }
        }
      }
    }
  }
}

void assignRNGOffset(Fusion* fusion) {
  int counter = 0;
  for (auto expr : fusion->exprs()) {
    if (expr->isA<RNGOp>()) {
      auto rop = expr->as<RNGOp>();
      rop->setRNGOffset(counter++);
    }
  }
}

// Dump expr string if enable lower_verbose
void dumpExprsIfEnabled(const std::vector<Expr*>& exprs, std::string msg_pre) {
  if (isDebugDumpEnabled(DebugDumpOption::LowerVerbose)) {
    std::cout << msg_pre << ":" << std::endl;
    for (auto exp : exprs) {
      std::cout << exp->toString() << std::endl;
    }
  }
}

void GpuLower::lower(Fusion* fusion, DataType index_type) {
  FUSER_PERF_SCOPE("GpuLower::lower");
  TORCH_INTERNAL_ASSERT(fusion != nullptr);
  TORCH_INTERNAL_ASSERT(
      active_gpu_lower == nullptr, "Nested lowering passes are not supported");

  struct LowerGuard {
    LowerGuard(GpuLower* gpu_lower) {
      active_gpu_lower = gpu_lower;
    }
    ~LowerGuard() {
      active_gpu_lower = nullptr;
    }
  } lower_guard(this);
  // Copy fusion into a new kernel for processing
  kernel_ = std::make_unique<kir::Kernel>(fusion, index_type);
  // Alias the fusion kernel caries around as a view of itself.
  fusion_ = kernel_.get();

  // Convert tensor views of DataType::Index type to either Int or Int32
  for (auto tv : ir_utils::allTvs(fusion_)) {
    if (tv->dtype() == DataType::Index) {
      tv->resolveIndexDtype();
    }
  }
  assignRNGOffset(fusion_);

  FusionGuard fg(fusion_);
  // prepare for lowering
  dumpExprsIfEnabled(fusion_->exprs(), "Before validateIr");
  validateIr(fusion_);

  // Checks if any TIDx dim is marked as padded to a warp. Also checks if we can
  // determine the padding is explicitly a single warp.
  dumpExprsIfEnabled(fusion_->exprs(), "Before collectPaddedParallelDims");
  collectPaddedParallelDims();

  // Replaces integers that are tensor sizes by named scalars as "T0.size[0]"
  dumpExprsIfEnabled(fusion_->exprs(), "Before replaceSymbolicSizes");
  replaceSymbolicSizes(fusion_);

  // Build what's refered to as the compute at map. This map contains the
  // mappings of all iteration domains across the fusion. There are three types
  // of mappings Permissive, Exact, and Loop, see compute_at_map.h/cpp for more
  // information.
  compute_at_map_ = std::make_shared<ComputeAtMap>(fusion_);

  resolveComputeWith(fusion_);

  if (isDebugDumpEnabled(DebugDumpOption::ComputeAtMap)) {
    std::cout << compute_at_map_->toString() << std::endl;
  }
  dumpExprsIfEnabled(fusion_->exprs(), "Before validateAndPropagatePType");
  compute_at_map_->validateAndPropagatePType();

  // Uses compute_at_map, find all splits that are enforced to be divisible
  dumpExprsIfEnabled(fusion_->exprs(), "Before getAllDivisibleSplits");
  divisible_splits_ = getAllDivisibleSplits(fusion_, compute_at_map_.get());

  // Used in parallel dimension map
  dumpExprsIfEnabled(fusion_->exprs(), "Before build parallelDimensionMap");
  concretized_broadcast_domains_ =
      std::make_shared<const ConcretizedBroadcastDomains>(fusion_);

  parallelDimensionMap().build(fusion_);
  if (isDebugDumpEnabled(DebugDumpOption::ParallelDimensions)) {
    std::cout << "Parallel dimension map:" << std::endl;
    std::cout << parallel_dimension_map_.toString() << std::endl;
  }

  // Validate mma data format and compatibility if any on the fusion.
  dumpExprsIfEnabled(fusion_->exprs(), "Before validateMma");
  validateMma(fusion_);

  // Validate swizzle usage on the fusion schedule.
  dumpExprsIfEnabled(fusion_->exprs(), "Before validateSwizzle");
  validateSwizzle(fusion_);

  // Compute thread predicates. Depends on parallel_dimension_map_
  dumpExprsIfEnabled(fusion_->exprs(), "Before build thread_pred_map_");
  thread_pred_map_.build(fusion_);

  // Fuse cetain patterns of reductions, such as a grid reduction
  // followed by a grid broadcast. Only depends on parallelization and
  // thread predicate map.
  dumpExprsIfEnabled(fusion_->exprs(), "Before fuseReductionsAndBroadcasts");
  fuseReductionsAndBroadcasts(fusion_);

  // Scan the whole fusion and build mappings about halo extensions of
  // all IterDomains
  dumpExprsIfEnabled(fusion_->exprs(), "Before build HaloInfo");
  halo_info_ = std::make_shared<HaloInfo>(fusion_, compute_at_map_);

  // Want to run this after parallel map and halo info map are
  // created. vectorized_accesses_ and vectorized_set_info_ are filled.
  dumpExprsIfEnabled(
      fusion_->exprs(), "Before validateAndCollectVectorizeInfo");
  validateAndCollectVectorizeInfo(fusion_);

  // Depends on ComputeAtMap and HaloInfo.
  dumpExprsIfEnabled(
      fusion_->exprs(), "Before validateAndConvertIterDomainGrouping");
  validateAndConvertIterDomainGrouping(fusion_);

  // Assumes all grouped reductions are convered to
  // GroupedReductionOp, which is done by
  // validateAndConvertIterDomainGrouping
  dumpExprsIfEnabled(fusion_->exprs(), "Before validateGroupedReductions");
  validateGroupedReductions(fusion_);

  // Depends on thread_pred_map_, validates parallelization collects which
  // tensor views need WAR or RAW syncs
  dumpExprsIfEnabled(fusion_->exprs(), "Before SyncMap");
  sync_map_ = std::make_shared<const SyncMap>(fusion_);
  if (isDebugDumpEnabled(DebugDumpOption::SyncMap)) {
    std::cout << sync_map_->toString() << std::endl;
  }

  dumpExprsIfEnabled(fusion_->exprs(), "Before build partialSplitMap");
  partialSplitMap().build(fusion_);

  dumpExprsIfEnabled(fusion_->exprs(), "Before validatePartialSplit");
  validatePartialSplit(fusion_);

  dumpExprsIfEnabled(fusion_->exprs(), "Before build nonDivisibleSplitInfo");
  nonDivisibleSplitInfo().build(fusion_);

  // Detects all exprssions that don't need predicates. Depends on
  // nonDivisibleSplitInfo.
  dumpExprsIfEnabled(fusion_->exprs(), "Before build predicateElimination");
  predicateElimination().build(fusion_);

  dumpExprsIfEnabled(fusion_->exprs(), "Before build doubleBufferInfo");
  doubleBufferInfo().build(fusion_);

  dumpExprsIfEnabled(fusion_->exprs(), "Before allocateIndexVariables()");
  compute_at_map_->allocateIndexVariables();
  // Run our passes keeping the lowered expressions and forwarding
  // them

  // Reorder expressions for loop-nest generation respecting computeAt
  // relationships
  const auto exprs_sorted = reorderExprsForComputeAt();

  // Generate loop-nests and place each expression at its
  // corresponding loop
  dumpExprsIfEnabled(exprs_sorted, "Before LoopNestGenerator::loweredExprs");
  const auto exprs_lowered = LoopNestGenerator::loweredExprs(exprs_sorted);

  // Replace squeezes, Transpose, Shift, Gather, and View ops with
  // unary ops since they're not separately processed in lowering.
  dumpExprsIfEnabled(exprs_lowered, "Before unarySetOpInserter");
  const auto exprs_unary_replaced = unarySetOpInserter(exprs_lowered);

  // Insert allocations
  dumpExprsIfEnabled(exprs_unary_replaced, "Before insertAllocations");
  const auto exprs_alloced = insertAllocations(exprs_unary_replaced);

  // Insert read after write smem syncs
  dumpExprsIfEnabled(exprs_alloced, "Before insertRawThreadSynchronization");
  const auto exprs_raw_sync = insertRawThreadSynchronization(exprs_alloced);

  // Reuse memory locations
  dumpExprsIfEnabled(exprs_raw_sync, "Before reuseMemoryAllocations");
  const auto exprs_reuse_mem = reuseMemoryAllocations(exprs_raw_sync);

  // Insert SyncThreads at end of for-loop to avoid WAR race condition
  dumpExprsIfEnabled(exprs_reuse_mem, "Before insertWarThreadSynchronization");
  const auto exprs_war_sync = insertWarThreadSynchronization(exprs_reuse_mem);

  dumpExprsIfEnabled(exprs_war_sync, "Before DoubleBufferPass");
  const auto exprs_double_buffered = DoubleBufferPass::run(exprs_war_sync);

  // This pass inserts predicates as well as branches in the code. Up until now
  // the code is explicitly single shot for loop based. Need to be careful in
  // later passes when doing any kind of insertions in loop nest structure as
  // insertions could be on if then or else instead of directly on a for loop.
  dumpExprsIfEnabled(exprs_double_buffered, "Before UnrollPass");
  const auto exprs_unrolled_loops =
      UnrollPass::runPass(fusion_, exprs_double_buffered);

  dumpExprsIfEnabled(
      exprs_unrolled_loops, "Before processMisalignedVectorization");
  const auto exprs_unrolled_mv_loops =
      processMisalignedVectorization(exprs_unrolled_loops);

  dumpExprsIfEnabled(exprs_unrolled_mv_loops, "Before IndexLowering");
  const auto exprs_indexed_loops =
      IndexLowering::getIndexedExprs(exprs_unrolled_mv_loops);

  // TODO: It seems this type of optimization would be far easier to implement
  // on fusion ir than kernel ir. We should likely refactor this to at least run
  // before allocation insertion.
  dumpExprsIfEnabled(exprs_indexed_loops, "Before fuseWarpReduce");
  const auto exprs_with_fused_broadcast = fuseWarpReduce(exprs_indexed_loops);

  dumpExprsIfEnabled(
      exprs_with_fused_broadcast, "Before generateConditionalFromPredicate");
  const auto exprs_conditional_loops =
      generateConditionalFromPredicate(exprs_with_fused_broadcast);

  dumpExprsIfEnabled(exprs_conditional_loops, "Before allocateCommonIndices");
  const auto exprs_common_index_allocated =
      allocateCommonIndices(exprs_conditional_loops);

  // Insert fake zero updates to make sure nvrtc doesn't blow out register use
  // on index and predicate reuse
  dumpExprsIfEnabled(exprs_common_index_allocated, "Before insertMagicZero");
  const auto exprs_register_adjusted =
      insertMagicZero(exprs_common_index_allocated);

  dumpExprsIfEnabled(exprs_register_adjusted, "Before KIRCleaner");
  const auto exprs_cleaned_up_loops =
      KIRCleaner::cleanUp(exprs_register_adjusted);

  dumpExprsIfEnabled(exprs_cleaned_up_loops, "Before instrumentKernel");
  const auto exprs_instrumented = instrumentKernel(exprs_cleaned_up_loops);

  // We now have the lowered expressions, finalize the kernel IR. This function
  // will also copy over some relevant information for code generation from
  // GpuLower.
  kernel_->finalize(exprs_instrumented);
}

kir::Kernel* GpuLower::kernel() const {
  TORCH_CHECK(kernel_);
  return kernel_.get();
}

GpuLower* GpuLower::current() {
  TORCH_INTERNAL_ASSERT(
      active_gpu_lower != nullptr, "No active GpuLower available");
  return active_gpu_lower;
}

bool GpuLower::hasCurrent() {
  return active_gpu_lower != nullptr;
}

void GpuLower::propagateExprInfo(const Expr* old_expr, const Expr* new_expr) {
  pred_elimination_.propagateRemovalInfo(old_expr, new_expr);
  if (old_expr->isA<kir::Allocate>()) {
    auto alloc_info_it =
        localAllocationInfoMap().find(old_expr->as<kir::Allocate>());
    if (alloc_info_it != localAllocationInfoMap().end()) {
      auto alloc_info =
          std::make_unique<LocalAllocationInfo>(*(alloc_info_it->second));
      localAllocationInfoMap().emplace(
          new_expr->as<kir::Allocate>(), std::move(alloc_info));
    }
  }
}

bool GpuLower::resolveComputeWith(Fusion* fusion) {
  std::vector<Expr*> exprs_sorted;

  bool updated = false;
  for (auto val : fusion->usedMathVals()) {
    auto tv = dynamic_cast<TensorView*>(val);
    if (tv == nullptr) {
      continue;
    }
    if (tv->hasComputeWith()) {
      if (exprs_sorted.empty()) {
        exprs_sorted = reorderExprsForComputeAt();
      }
      if (tv->resolveComputeWith(exprs_sorted)) {
        updated = true;
        compute_at_map_->updateComputeWith(tv);
      }
    }
  }

  return updated;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
