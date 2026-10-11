import React, { useEffect, useState } from 'react';
import {
  X,
  Check,
  AlertTriangle,
  ArrowRight,
  Bookmark,
  Cpu,
  Layers,
  Sparkles,
} from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';
import { PresetDiffReview, PresetModel } from '../types';

export const PresetReviewModal: React.FC = () => {
  const {
    diffModalPreset,
    setDiffModalPreset,
    applyPresetToComposer,
    prompt,
    negative_prompt,
    model,
    width,
    height,
    steps,
    cfg_scale,
    sampler,
    seed,
    denoise,
    loras,
    vramPolicy,
  } = useStudioStore();

  const [diffData, setDiffData] = useState<PresetDiffReview | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!diffModalPreset) {
      setDiffData(null);
      return;
    }

    const fetchDiff = async () => {
      setLoading(true);
      try {
        const currentForm = {
          prompt,
          negative_prompt,
          model,
          width,
          height,
          steps,
          cfg_scale,
          sampler,
          seed,
          denoise,
          loras,
          vram_profile: vramPolicy,
        };

        const res = await fetch('/api/presets/diff', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            preset_id: diffModalPreset.id,
            current_form: currentForm,
          }),
        });

        if (res.ok) {
          const data: PresetDiffReview = await res.json();
          setDiffData(data);
        }
      } catch (err) {
        console.error('Failed to calculate preset diff:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchDiff();
  }, [diffModalPreset]);

  if (!diffModalPreset) return null;

  const handleConfirm = () => {
    applyPresetToComposer(diffModalPreset);
    setDiffModalPreset(null);
  };

  const missingDeps = diffData?.dependencies
    ? Object.entries(diffData.dependencies).filter(([, available]) => !available)
    : [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="w-full max-w-2xl bg-surface border border-border-subtle rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="p-5 border-b border-border-subtle flex items-center justify-between bg-surface-elevated">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-accent/20 border border-accent/40 text-accent">
              <Bookmark className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h2 className="text-base font-semibold text-text-primary">
                  Review Preset: {diffModalPreset.name}
                </h2>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-accent/15 text-accent font-medium">
                  v{diffModalPreset.version}
                </span>
                {diffModalPreset.is_built_in && (
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-surface border border-border-subtle text-text-dim">
                    Built-in
                  </span>
                )}
              </div>
              <p className="text-xs text-text-muted mt-0.5">{diffModalPreset.description}</p>
            </div>
          </div>
          <button
            onClick={() => setDiffModalPreset(null)}
            className="p-1.5 rounded-lg text-text-dim hover:text-text-primary hover:bg-surface transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 overflow-y-auto space-y-5 text-xs flex-1">
          {/* Missing Dependencies Warning */}
          {missingDeps.length > 0 && (
            <div className="p-3.5 rounded-xl bg-status-warning/15 border border-status-warning/30 space-y-1.5">
              <div className="flex items-center gap-1.5 text-status-warning font-semibold">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>Missing Model Dependencies</span>
              </div>
              <p className="text-text-muted text-[11px]">
                The following checkpoints or LoRAs are not currently installed in ComfyUI models folder. The preset will apply, but generation will require the file:
              </p>
              <div className="flex flex-wrap gap-1.5 pt-1">
                {missingDeps.map(([name]) => (
                  <span
                    key={name}
                    className="font-mono text-[10px] px-2 py-0.5 rounded bg-surface border border-status-warning/40 text-text-primary"
                  >
                    {name}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Diff Table */}
          {loading ? (
            <div className="py-12 flex flex-col items-center justify-center space-y-2 text-text-muted">
              <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
              <span className="font-mono text-[11px]">Computing parameter comparison...</span>
            </div>
          ) : diffData && Object.keys(diffData.changes).length > 0 ? (
            <div className="space-y-3">
              <span className="font-semibold uppercase tracking-wider text-text-muted text-[10px]">
                Parameter Changes ({Object.keys(diffData.changes).length})
              </span>
              <div className="rounded-xl border border-border-subtle overflow-hidden bg-surface-elevated">
                <div className="grid grid-cols-3 p-2.5 bg-surface text-[10px] font-mono uppercase text-text-dim border-b border-border-subtle font-semibold">
                  <span>Parameter</span>
                  <span>Current Setting</span>
                  <span>Preset Target</span>
                </div>
                <div className="divide-y divide-border-subtle/60 max-h-64 overflow-y-auto font-mono text-[11px]">
                  {Object.entries(diffData.changes).map(([key, change]) => {
                    const currentStr = typeof change.current === 'object' ? JSON.stringify(change.current) : String(change.current ?? 'None');
                    const presetStr = typeof change.preset === 'object' ? JSON.stringify(change.preset) : String(change.preset ?? 'None');

                    return (
                      <div key={key} className="grid grid-cols-3 p-2.5 items-center hover:bg-surface/50 transition-colors">
                        <span className="font-sans font-medium text-text-muted capitalize">
                          {key.replace(/_/g, ' ')}
                        </span>
                        <span className="text-text-dim truncate pr-2" title={currentStr}>
                          {currentStr}
                        </span>
                        <div className="flex items-center space-x-1.5 text-accent font-semibold truncate" title={presetStr}>
                          <ArrowRight className="w-3 h-3 shrink-0 text-text-dim" />
                          <span className="truncate">{presetStr}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ) : (
            <div className="p-4 rounded-xl bg-surface-elevated border border-border-subtle text-center text-text-muted font-mono text-[11px]">
              Current workspace parameters already match this preset.
            </div>
          )}

          {/* VRAM & Hardware Profile Estimate */}
          {diffData && (
            <div className="p-3 rounded-xl bg-surface-elevated border border-border-subtle flex items-center justify-between text-xs font-mono">
              <span className="text-text-muted flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-accent" />
                Estimated Peak VRAM for Preset:
              </span>
              <span className="font-semibold text-text-primary">
                {(diffData.estimated_vram_mb / 1024).toFixed(1)} GB VRAM
              </span>
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="p-4 border-t border-border-subtle bg-surface-elevated flex items-center justify-end space-x-3">
          <button
            onClick={() => setDiffModalPreset(null)}
            className="px-4 py-2 rounded-xl text-xs font-medium text-text-muted hover:text-text-primary hover:bg-surface transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            className="px-5 py-2 rounded-xl text-xs font-semibold bg-accent text-background hover:bg-accent-hover transition-colors shadow-md shadow-accent/10 flex items-center gap-1.5"
          >
            <Check className="w-3.5 h-3.5" />
            <span>Confirm & Apply Preset</span>
          </button>
        </div>
      </div>
    </div>
  );
};
