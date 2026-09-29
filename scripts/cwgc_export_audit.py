"""Separate EMA averaging and missing BGFC context without fitting a model.

Training views are the default. Explicit test views are exploratory pilot
diagnostics. Report float-image metrics, not official exported PNG metrics.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from arguments import ModelParams, OptimizationParams, PipelineParams
from saved_settings import saved_namespace
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.loss_utils import ssim


@torch.no_grad()
def audit(checkpoint, output, split, views):
    parser = argparse.ArgumentParser()
    groups = [ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)]
    namespace = parser.parse_args([])
    for filename in ('cfg_args', 'optimization_args'):
        vars(namespace).update(saved_namespace(checkpoint.parent / filename))
    namespace.model_path = str(checkpoint.parent.resolve())
    data, options, pipe = [group.extract(namespace) for group in groups]
    state = torch.load(checkpoint, weights_only=False)
    if len(state) < 4 or int(state[1]) != 30000:
        raise ValueError('Expected a complete locally produced 30000-step checkpoint')
    keys = ('use_bgfc', 'use_at_gom', 'bgfc_hidden_dim', 'bgfc_gate_init_bias',
            'bgfc_thermal_grayscale_context', 'bgfc_rgb_luma_transfer_only',
            'use_render_calibration', 'use_color_refinement', 'color_refinement_hidden_dim',
            'color_refinement_max_residual')
    model = GaussianModel(data.sh_degree, **{key: getattr(data, key) for key in keys})
    scene = Scene(data, model, load_iteration=30000, shuffle=False)
    model.restore(state[0], options)
    model.set_color_refinement_runtime_enabled(True)
    context = model.get_cmo_states()
    empty_context = {key: torch.zeros_like(value) for key, value in context.items()}
    cameras = scene.getTestCameras() if split == 'test' else scene.getTrainSamplingCameras(paired_only=True)
    selected = torch.linspace(0, len(cameras)-1, min(views, len(cameras))).long().tolist()
    background = torch.full((3,), float(data.white_background), device='cuda')
    rows = []
    for parameter_state in ('raw', 'ema'):
        if parameter_state == 'ema':
            ema = state[3].get('ema_export_state')
            if not ema:
                raise ValueError('Checkpoint contains no EMA state')
            modules = {}
            for key, value in ema.items():
                if key.startswith('attr:'):
                    getattr(model, key.split(':', 1)[1]).copy_(value)
                elif key.startswith('module:'):
                    _, module_name, item = key.split(':', 2)
                    modules.setdefault(module_name, {})[item] = value
            for module_name, updates in modules.items():
                module = getattr(model, module_name)
                merged = dict(module.state_dict())
                merged.update(updates)
                module.load_state_dict(merged)
        for retained in (True, False):
            model.cmo_states = context if retained else empty_context
            records = []
            for index in selected:
                camera = cameras[index]
                package = render(camera, model, pipe, background)
                record = {'view': camera.image_name}
                for modality, target in (('color', camera.original_image), ('thermal', camera.original_thermal)):
                    prediction = package['render_'+modality].clamp(0, 1)
                    target = target.cuda().clamp(0, 1)
                    mse = (prediction-target).square().mean()
                    record[modality+'_PSNR'] = float(-10*mse.log10())
                    channel_mse = (prediction-target).square().flatten(1).mean(1)
                    record[modality+'_legacy_channel_mean_PSNR'] = float((-10*channel_mse.log10()).mean())
                    record[modality+'_SSIM'] = float(ssim(prediction, target))
                records.append(record)
            rows.append({'parameter_state': parameter_state, 'retain_context': retained,
                         'metrics': {key: statistics.mean(row[key] for row in records)
                                     for key in records[0] if key != 'view'},
                         'records': records})
    result = {'checkpoint': str(checkpoint.resolve()), 'iteration': int(state[1]),
              'parameter_updates': 0, 'split': split, 'train_views_only': split == 'train',
              'float_image_metrics': True, 'variants': rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps([{key: value for key, value in row.items() if key != 'records'} for row in rows]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'test'), default='train')
    parser.add_argument('--views', type=int, default=16)
    args = parser.parse_args()
    audit(args.checkpoint, args.output, args.split, args.views)
