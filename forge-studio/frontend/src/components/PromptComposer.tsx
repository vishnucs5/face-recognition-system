import React from 'react';
import { Sparkles, Trash2, EyeOff, Tag } from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';

export const PromptComposer: React.FC = () => {
  const {
    prompt,
    setPrompt,
    negative_prompt,
    setNegativePrompt,
    setCopilotOpen,
    loras,
    loraVault,
  } = useStudioStore();

  const negativePresets = [
    {
      label: 'Standard Baseline',
      text: 'blurry, low quality, distorted, deformed, watermark, oversaturated',
    },
    {
      label: 'Photorealism',
      text: 'cgi, 3d render, plastic skin, cartoon, anime, illustration, oversaturated, blurry, bad anatomy',
    },
    {
      label: 'Illustration / Anime',
      text: 'photorealistic, real photo, 3d, noisy, grainy, distorted hands, bad eyes',
    },
  ];

  const promptLimit = 4000;
  const isPromptTooLong = prompt.length > promptLimit;

  return (
    <div className="space-y-4">
      {/* Positive Prompt */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-xs font-semibold uppercase tracking-wider text-text-muted flex items-center gap-1.5">
            Prompt
          </label>
          <div className="flex items-center space-x-2">
            <span
              className={`text-[10px] font-mono ${
                isPromptTooLong ? 'text-status-error font-bold' : 'text-text-dim'
              }`}
            >
              {prompt.length} / {promptLimit}
            </span>
            <button
              onClick={() => setCopilotOpen(true)}
              className="text-xs text-accent hover:underline flex items-center gap-1 font-medium"
            >
              <Sparkles className="w-3 h-3" />
              Copilot Suggest
            </button>
            {prompt && (
              <button
                onClick={() => setPrompt('')}
                className="text-xs text-text-dim hover:text-status-error transition-colors p-1 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                title="Clear prompt"
                aria-label="Clear positive prompt text"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            )}
          </div>
        </div>

        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe your subject, lighting, optics, and aesthetic composition..."
          rows={4}
          maxLength={4500}
          className={`w-full bg-surface-elevated text-text-primary text-sm rounded-xl p-3.5 border transition-all resize-none placeholder:text-text-dim focus:outline-none focus:ring-1 ${
            isPromptTooLong
              ? 'border-status-error focus:border-status-error focus:ring-status-error'
              : 'border-border-subtle focus:border-accent focus:ring-accent'
          }`}
        />

        {/* LoRA Trigger Words Insertion Chips */}
        {loras.length > 0 && (() => {
          const triggers = loras.flatMap((l) => {
            const v = loraVault.find((item) => item.filename === l.name || item.name === l.name);
            return v?.trigger_words || [];
          }).filter((tw, idx, arr) => arr.indexOf(tw) === idx && tw.trim() !== '');

          if (triggers.length === 0) return null;

          return (
            <div className="flex items-center gap-1.5 flex-wrap pt-1">
              <span className="text-[10px] uppercase font-mono text-text-dim flex items-center gap-1 shrink-0">
                <Tag className="w-3 h-3 text-accent" />
                LoRA Triggers:
              </span>
              {triggers.map((tw) => {
                const inPrompt = prompt.toLowerCase().includes(tw.toLowerCase());
                return (
                  <button
                    key={tw}
                    type="button"
                    onClick={() => {
                      if (!inPrompt) {
                        setPrompt(prompt ? `${prompt.trim()}, ${tw}` : tw);
                      }
                    }}
                    disabled={inPrompt}
                    aria-label={`Insert ${tw} into prompt`}
                    className={`text-[10px] px-2 py-0.5 rounded-md border font-mono transition-colors ${
                      inPrompt
                        ? 'bg-surface border-border-subtle text-text-dim opacity-60 cursor-default'
                        : 'bg-accent/15 border-accent/30 text-accent font-medium hover:bg-accent/25 hover:border-accent/50 cursor-pointer'
                    }`}
                  >
                    {inPrompt ? `✓ ${tw}` : `+ ${tw}`}
                  </button>
                );
              })}
            </div>
          );
        })()}
      </div>

      {/* Negative Prompt */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-xs font-semibold uppercase tracking-wider text-text-muted flex items-center gap-1.5">
            <EyeOff className="w-3.5 h-3.5 text-text-dim" />
            Negative Prompt
          </label>
          <div className="flex items-center space-x-2">
            <span className="text-[10px] font-mono text-text-dim">
              {negative_prompt.length} / {promptLimit}
            </span>
            {negative_prompt && (
              <button
                onClick={() => setNegativePrompt('')}
                className="text-xs text-text-dim hover:text-status-error transition-colors p-1 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                title="Clear negative prompt"
                aria-label="Clear negative prompt text"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            )}
          </div>
        </div>

        <textarea
          value={negative_prompt}
          onChange={(e) => setNegativePrompt(e.target.value)}
          placeholder="Unwanted artifacts (e.g. blurry, distorted, bad hands, oversaturated)..."
          rows={2}
          maxLength={4500}
          className="w-full bg-surface-elevated text-text-primary text-sm rounded-xl p-3 border border-border-subtle focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent transition-all resize-none placeholder:text-text-dim"
        />

        {/* Contradiction Warning (e.g. Asking for photo but negative prompt excludes photo) */}
        {(() => {
          const isPhotoPrompt = /\b(photo|photograph|photorealistic|dslr|portrait|camera|kodak|hyper-realistic)\b/i.test(prompt);
          const hasPhotoNegative = /\b(photo|photorealistic|real photo)\b/i.test(negative_prompt);
          if (isPhotoPrompt && hasPhotoNegative) {
            return (
              <div className="p-2 rounded-lg bg-status-warning/15 border border-status-warning/30 text-status-warning text-xs flex items-center justify-between gap-2">
                <span className="truncate">
                  ⚠️ Negative prompt bans "real photo / photorealistic", removing photographic textures.
                </span>
                <button
                  type="button"
                  onClick={() => setNegativePrompt(negativePresets[1].text)}
                  className="px-2 py-0.5 rounded bg-status-warning/20 hover:bg-status-warning/30 text-status-warning font-semibold text-[11px] whitespace-nowrap transition-colors border border-status-warning/40 shrink-0"
                >
                  Fix for Photorealism
                </button>
              </div>
            );
          }
          return null;
        })()}

        {/* Quick Negative Presets */}
        <div className="flex items-center gap-1.5 pt-1 overflow-x-auto pb-1">
          <span className="text-[10px] uppercase font-mono text-text-dim shrink-0 flex items-center gap-1">
            <Tag className="w-3 h-3" />
            Presets:
          </span>
          {negativePresets.map((preset) => {
            const isSelected = negative_prompt === preset.text;
            return (
              <button
                key={preset.label}
                type="button"
                onClick={() => setNegativePrompt(preset.text)}
                aria-label={`Apply ${preset.label} negative prompt preset`}
                className={`text-[10px] px-2 py-0.5 rounded-md border shrink-0 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                  isSelected
                    ? 'bg-accent/20 border-accent/40 text-accent font-medium'
                    : 'bg-surface-elevated border-border-subtle text-text-muted hover:text-text-primary hover:border-border-strong'
                }`}
              >
                {preset.label}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
