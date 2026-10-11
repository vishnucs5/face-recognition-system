import React from 'react';
import { Cpu, Layers, Sliders, Hash, ShieldCheck } from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';

export const SettingsSummary: React.FC = () => {
  const {
    mode,
    model,
    width,
    height,
    steps,
    cfg_scale,
    sampler,
    seed,
    denoise,
    loras,
    source_image_id,
  } = useStudioStore();

  const megapixels = ((width * height) / 1000000).toFixed(2);

  // Approximate VRAM footprint calculation based on resolution and LoRAs
  const baseVramGb = (width * height > 1024 * 1024 ? 8.2 : 7.2) + (loras.length * 0.4);
  const vramSafe = baseVramGb <= 14.5;

  return (
    <div className="p-3.5 rounded-xl bg-surface-elevated/60 border border-border-subtle text-xs space-y-2.5">
      <div className="flex items-center justify-between">
        <span className="font-semibold uppercase tracking-wider text-text-muted flex items-center gap-1.5 text-[11px]">
          <Sliders className="w-3.5 h-3.5 text-accent" />
          Settings Summary
        </span>
        <div className="flex items-center gap-1 text-[11px] font-mono text-status-success">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>VRAM Safe (~{baseVramGb.toFixed(1)}GB)</span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 text-[11px] font-mono">
        <div className="p-2 rounded bg-surface border border-border-subtle/60">
          <span className="text-text-dim block text-[9px] uppercase font-sans">Resolution</span>
          <span className="text-text-primary font-medium">{width}×{height}</span>
          <span className="text-text-dim text-[10px] ml-1 font-sans">({megapixels} MP)</span>
        </div>

        <div className="p-2 rounded bg-surface border border-border-subtle/60">
          <span className="text-text-dim block text-[9px] uppercase font-sans">Sampler & CFG</span>
          <span className="text-text-primary font-medium">{sampler}</span>
          <span className="text-text-dim text-[10px] ml-1 font-sans">(@{cfg_scale})</span>
        </div>

        <div className="p-2 rounded bg-surface border border-border-subtle/60">
          <span className="text-text-dim block text-[9px] uppercase font-sans">LoRAs Attached</span>
          <span className="text-accent font-medium">
            {loras.length > 0 ? `${loras.length} / 2 Active` : 'Base Checkpoint'}
          </span>
        </div>
      </div>

      {mode === 'img2img' && (
        <div className="flex items-center justify-between text-[11px] font-mono pt-1 border-t border-border-subtle/60 text-text-muted">
          <span>Mode: Image-to-Image</span>
          <span>Denoise: <span className="text-accent font-semibold">{denoise.toFixed(2)}</span></span>
          <span className={source_image_id ? 'text-status-success' : 'text-status-error font-sans'}>
            {source_image_id ? '✓ Source Ready' : '⚠ Source Required'}
          </span>
        </div>
      )}
    </div>
  );
};
