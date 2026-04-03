from util.metric import eval_depth, MetricTracker
from util.train import to_cuda
from util.util import run_model_on_sample, get_focal_stack_and_fd_list
from util.vis import log_images
import torch
import random
from util.log import wandb_log_scalars
from tqdm import tqdm
import torch.nn.functional as F


@torch.no_grad()
def measure_model_efficiency(model, config, val_loader):
    """Measures peak GPU memory and FLOPs on one validation sample.

    Uses a forward pre-hook on the underlying model to capture the exact inputs
    that reach model.forward(), so FLOPs are measured only over the neural network
    (not over focal-stack generation). Peak GPU memory covers the full inference
    pipeline including focal-stack generation.

    Prints a summary table and returns a dict with the raw numbers.
    """
    underlying_model = model.module

    # ------------------------------------------------------------------
    # 1. Hook: capture the positional args passed to the underlying model
    # ------------------------------------------------------------------
    captured_inputs = []

    def _capture_pre_hook(module, args):
        if not captured_inputs:
            captured_inputs.append(
                tuple(a.detach() if isinstance(a, torch.Tensor) else a for a in args)
            )

    hook_handle = underlying_model.register_forward_pre_hook(_capture_pre_hook)

    # ------------------------------------------------------------------
    # 2. Run one sample — same branching logic as collect_metrics
    # ------------------------------------------------------------------
    sample = next(iter(val_loader))
    sample = to_cuda(sample)
    rgb        = sample['rgb']
    depth      = sample['gt']
    valid_mask = sample['valid_mask']
    K          = sample['K']
    focal_stack       = sample.get('focal_stack', None)
    fd_list_input     = sample.get('fd_list', None)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    focal_stack, fd_list, _ = get_focal_stack_and_fd_list(
        rgb=rgb, depth=depth, K=K, depth_valid_mask=valid_mask, config=config, dataset_sampled_from=config['val_loader_config']['dataset_name'], training=False,
        dataset_focal_stack=focal_stack, dataset_fd_list=fd_list_input)
    
    pd = run_model_on_sample(model=model, focal_stack=focal_stack, fd_list=fd_list, evaluating_model_trained_with_canonical_depth=config['training_with_canonical_depth'], K=K)

    torch.cuda.synchronize()
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    hook_handle.remove()

    # ------------------------------------------------------------------
    # 3. Parameter counts
    # ------------------------------------------------------------------
    total_params     = sum(p.numel() for p in underlying_model.parameters())
    trainable_params = sum(p.numel() for p in underlying_model.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # 4. FLOPs via fvcore using the captured inputs
    # ------------------------------------------------------------------
    total_flops = None
    if captured_inputs:
        try:
            from fvcore.nn import FlopCountAnalysis
            flop_analysis = FlopCountAnalysis(underlying_model, captured_inputs[0])
            flop_analysis.unsupported_ops_warnings(False)
            flop_analysis.uncalled_modules_warnings(False)
            total_flops = flop_analysis.total()
        except Exception as e:
            print(f"Warning: FLOPs measurement failed: {e}")
    else:
        print("Warning: No inputs were captured by the forward hook; skipping FLOPs measurement.")

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    SEP = "=" * 58
    print(f"\n{SEP}")
    print("  MODEL EFFICIENCY METRICS")
    print(SEP)
    print(f"  Model type        : FOSSA")
    print(f"  Input RGB shape   : {list(rgb.shape)}")
    if captured_inputs and any(isinstance(a, torch.Tensor) and a.dim() == 5 for a in captured_inputs[0]):
        fs = next(a for a in captured_inputs[0] if isinstance(a, torch.Tensor) and a.dim() == 5)
        print(f"  Focal stack shape : {list(fs.shape)}")
    print(f"  Total params      : {total_params:>15,}  ({total_params / 1e6:.2f} M)")
    print(f"  Trainable params  : {trainable_params:>15,}  ({trainable_params / 1e6:.2f} M)")
    param_mem_mb = total_params * 4 / (1024 ** 2)
    print(f"  Param memory      : {param_mem_mb:>12.1f} MB  ({param_mem_mb / 1024:.3f} GB, fp32)")
    print(f"  Peak GPU memory   : {peak_memory_mb:>12.1f} MB  ({peak_memory_mb / 1024:.3f} GB)")
    if total_flops is not None:
        print(f"  FLOPs             : {total_flops:>15,}  ({total_flops / 1e9:.2f} GFLOPs)")
    else:
        print(f"  FLOPs             : N/A")
    print(f"{SEP}\n")

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_memory_mb': param_mem_mb,
        'peak_memory_mb': peak_memory_mb,
        'flops': total_flops,
    }


@torch.no_grad()
def collect_metrics(val_loader, model, config, metric_tracker):
    pbar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc="Validating",
        dynamic_ncols=True,
        leave=True
    )
    for i, sample in pbar:
        sample = to_cuda(sample)
        # rgb shape: torch.Size([1, C, H, W]), depth shape: torch.Size([1, 1, H, W]), valid_mask shape: torch.Size([1, H, W]), K shape: torch.Size([1, 3, 3])
        rgb, depth, valid_mask, K = sample['rgb'], sample['gt'], sample['valid_mask'], sample['K']
        focal_stack = sample.get('focal_stack', None)
        fd_list_input = sample.get('fd_list', None)

        focal_stack, fd_list, _ = get_focal_stack_and_fd_list(
            rgb=rgb, depth=depth, K=K, depth_valid_mask=valid_mask, config=config, dataset_sampled_from=config['val_loader_config']['dataset_name'], training=False,
            dataset_focal_stack=focal_stack, dataset_fd_list=fd_list_input)
        

        pd = run_model_on_sample(model=model, focal_stack=focal_stack, fd_list=fd_list, evaluating_model_trained_with_canonical_depth=config['training_with_canonical_depth'], K=K)

        pd = pd.squeeze(0).squeeze(0)
        depth = depth.squeeze(0).squeeze(0)
        valid_mask = valid_mask.squeeze(0)


        metrics = eval_depth(pd, depth, valid_mask, i, eval_in_disparity_space=config['val_loader_config']['eval_in_disparity_space'])
        metric_tracker.update(metrics)

@torch.no_grad()
def validate(model, config, val_loader, val_subset, step, first_epoch=False):
    model.eval()
    if not config['logging_turned_off']:
        log_images(model, config, val_subset, 'val', step, first_epoch=first_epoch)

    metric_tracker = MetricTracker()
    collect_metrics(val_loader, model, config, metric_tracker)
    results = metric_tracker.get_metrics()
    wandb_log_scalars(results, step, 'val')
    return results
    # return {}
