import React, { useState } from 'react';
import {
  Bot,
  X,
  Sparkles,
  Send,
  Check,
  ArrowRight,
  Lightbulb,
  Sliders,
  RotateCcw,
  ShieldAlert,
  ChevronRight,
  AlertTriangle,
  EyeOff,
  FileText,
  Paintbrush,
  Target,
  Zap,
  Shield,
  HelpCircle,
  Bookmark,
  Clock,
} from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';
import { CopilotActionResponse, CopilotResponse, CopilotSuggestion } from '../types';

interface ChatMessage {
  id: string;
  sender: 'user' | 'copilot';
  text: string;
  response?: CopilotResponse;
  actionResponse?: CopilotActionResponse;
}

export const CopilotDrawer: React.FC = () => {
  const {
    copilotOpen,
    setCopilotOpen,
    prompt,
    setPrompt,
    negative_prompt,
    setNegativePrompt,
    width,
    height,
    setDimensions,
    steps,
    setSteps,
    cfg_scale,
    setCfgScale,
    sampler,
    setSampler,
    denoise,
    setDenoise,
    setControlNetType,
    setControlNetStrength,
    model,
    loras,
    systemStatus,
    presets,
    applyPresetToComposer,
  } = useStudioStore();

  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [appliedFields, setAppliedFields] = useState<
    Record<string, { prompt?: boolean; negative?: boolean; params?: boolean; all?: boolean }>
  >({});

  if (!copilotOpen) return null;

  const ollamaOnline = systemStatus?.ollama_online ?? false;
  const activeModel = systemStatus?.ollama_model || 'qwen3-coder:30b';

  const handleSend = async (
    task:
      | 'improve_prompt'
      | 'variations'
      | 'configure_generation'
      | 'explain'
      | 'negative_prompt'
      | 'inpaint_advice'
      | 'controlnet_advice',
    customText?: string
  ) => {
    const textToSend = customText || inputMessage.trim() || prompt;
    if (!textToSend) return;

    const userMsgId = String(Date.now());
    const userMsg: ChatMessage = {
      id: userMsgId,
      sender: 'user',
      text: textToSend,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputMessage('');
    setLoading(true);

    try {
      const res = await fetch('/api/copilot/prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task,
          message: textToSend,
          current_context: {
            current_prompt: prompt,
            current_negative: negative_prompt,
            current_steps: steps,
            current_cfg: cfg_scale,
            current_sampler: sampler,
            model,
            active_loras: loras.map((l) => l.name),
          },
        }),
      });

      if (res.ok) {
        const data: CopilotResponse = await res.json();
        const copilotMsg: ChatMessage = {
          id: String(Date.now() + 1),
          sender: 'copilot',
          text: data.reply,
          response: data,
        };
        setMessages((prev) => [...prev, copilotMsg]);
      } else {
        const err = await res.json();
        setMessages((prev) => [
          ...prev,
          {
            id: String(Date.now() + 1),
            sender: 'copilot',
            text: `Error from Copilot server: ${err.detail || 'Service unavailable'}.`,
          },
        ]);
      }
    } catch (err) {
      console.error('Copilot request failed:', err);
      setMessages((prev) => [
        ...prev,
        {
          id: String(Date.now() + 1),
          sender: 'copilot',
          text: 'Network error communicating with Ollama service (120s timeout protection).',
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleStructuredAction = async (
    actionType: 'fast_generation' | 'quality' | 'vram_safe_resolution' | 'explain_slowness' | 'suggest_preset'
  ) => {
    const userMsgId = String(Date.now());
    const actionTitles = {
      fast_generation: 'Optimize for fast generation',
      quality: 'Optimize for maximum quality',
      vram_safe_resolution: 'Suggest VRAM-safe resolution',
      explain_slowness: 'Explain potential job slowness',
      suggest_preset: 'Suggest suitable preset',
    };

    setMessages((prev) => [
      ...prev,
      { id: userMsgId, sender: 'user', text: actionTitles[actionType] },
    ]);
    setLoading(true);

    try {
      const res = await fetch('/api/copilot/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action_type: actionType,
          current_form: {
            prompt,
            negative_prompt,
            model,
            width,
            height,
            steps,
            cfg_scale,
            sampler,
            denoise,
            loras,
          },
        }),
      });

      if (res.ok) {
        const actionData: CopilotActionResponse = await res.json();
        setMessages((prev) => [
          ...prev,
          {
            id: String(Date.now() + 1),
            sender: 'copilot',
            text: actionData.explanation,
            actionResponse: actionData,
          },
        ]);
      } else {
        const err = await res.json();
        setMessages((prev) => [
          ...prev,
          {
            id: String(Date.now() + 1),
            sender: 'copilot',
            text: `Action failed: ${err.detail || 'Service unavailable'}`,
          },
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: String(Date.now() + 1),
          sender: 'copilot',
          text: 'Network error communicating with Copilot service.',
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleApplyActionChanges = (msgId: string, actionResp: CopilotActionResponse) => {
    const changes = actionResp.suggested_changes;
    if (changes.prompt) setPrompt(changes.prompt);
    if (changes.negative_prompt) setNegativePrompt(changes.negative_prompt);
    if (changes.width && changes.height) setDimensions(changes.width, changes.height);
    if (changes.steps) setSteps(changes.steps);
    if (changes.cfg_scale) setCfgScale(changes.cfg_scale);
    if (changes.sampler) setSampler(changes.sampler);
    if (changes.denoise !== undefined) setDenoise(changes.denoise);

    if (actionResp.preset_id) {
      const p = presets.find((x) => x.id === actionResp.preset_id);
      if (p) applyPresetToComposer(p);
    }

    setAppliedFields((prev) => ({
      ...prev,
      [msgId]: { ...prev[msgId], all: true },
    }));
  };

  const handleApplyField = (
    msgId: string,
    field: 'prompt' | 'negative' | 'params' | 'all',
    s: CopilotSuggestion
  ) => {
    if (field === 'prompt' && s.prompt) {
      setPrompt(s.prompt);
    } else if (field === 'negative' && s.negative_prompt) {
      setNegativePrompt(s.negative_prompt);
    } else if (field === 'params') {
      if (s.width && s.height) setDimensions(s.width, s.height);
      if (s.steps) setSteps(s.steps);
      if (s.cfg_scale) setCfgScale(s.cfg_scale);
      if (s.sampler) setSampler(s.sampler);
      if (s.denoise !== undefined) setDenoise(s.denoise);
      if (s.controlnet_type) setControlNetType(s.controlnet_type);
      if (s.controlnet_strength !== undefined) setControlNetStrength(s.controlnet_strength);
    } else if (field === 'all') {
      if (s.prompt) setPrompt(s.prompt);
      if (s.negative_prompt) setNegativePrompt(s.negative_prompt);
      if (s.width && s.height) setDimensions(s.width, s.height);
      if (s.steps) setSteps(s.steps);
      if (s.cfg_scale) setCfgScale(s.cfg_scale);
      if (s.sampler) setSampler(s.sampler);
      if (s.denoise !== undefined) setDenoise(s.denoise);
      if (s.controlnet_type) setControlNetType(s.controlnet_type);
      if (s.controlnet_strength !== undefined) setControlNetStrength(s.controlnet_strength);
    }

    setAppliedFields((prev) => ({
      ...prev,
      [msgId]: {
        ...prev[msgId],
        [field]: true,
        ...(field === 'all' ? { prompt: true, negative: true, params: true, all: true } : {}),
      },
    }));
  };

  return (
    <div className="fixed inset-y-0 right-0 w-full sm:w-[440px] bg-surface border-l border-border-strong shadow-2xl z-50 flex flex-col animate-in slide-in-from-right duration-200 select-none">
      {/* Header */}
      <div className="px-5 py-4 border-b border-border-subtle flex items-center justify-between bg-surface-elevated/40 shrink-0">
        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded-lg bg-accent/20 border border-accent/40 flex items-center justify-center text-accent">
            <Bot className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-text-primary">Studio Copilot</h2>
            <p className="text-[10px] text-text-muted flex items-center gap-1.5 font-mono">
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  ollamaOnline ? 'bg-status-success animate-pulse' : 'bg-status-warning'
                }`}
              />
              {ollamaOnline ? `${activeModel} (Online)` : 'Ollama Offline (Heuristics)'}
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-1">
          {messages.length > 0 && (
            <button
              onClick={() => setMessages([])}
              className="p-1.5 text-text-dim hover:text-text-primary rounded-lg transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              title="Clear conversation"
              aria-label="Clear Copilot conversation history"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          )}
          <button
            onClick={() => setCopilotOpen(false)}
            className="text-text-dim hover:text-text-primary p-1.5 rounded-lg transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            title="Close drawer"
            aria-label="Close Studio Copilot drawer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Advisory Banner */}
      <div className="px-4 py-2 bg-surface-elevated/70 border-b border-border-subtle text-[11px] text-text-muted flex items-center gap-2">
        <ShieldAlert className="w-3.5 h-3.5 text-accent shrink-0" />
        <span>Strictly review-only. Changes apply only when you confirm.</span>
      </div>

      {/* Structured Action Chips Strip */}
      <div className="px-3 py-2 bg-surface border-b border-border-subtle flex items-center space-x-1.5 overflow-x-auto scrollbar-none shrink-0">
        {[
          { id: 'fast_generation', label: 'Fast Speed', icon: Zap },
          { id: 'quality', label: 'Max Quality', icon: Sparkles },
          { id: 'vram_safe_resolution', label: 'Safe Res', icon: Shield },
          { id: 'explain_slowness', label: 'Explain Delay', icon: HelpCircle },
          { id: 'suggest_preset', label: 'Suggest Preset', icon: Bookmark },
        ].map((act) => {
          const Icon = act.icon;
          return (
            <button
              key={act.id}
              onClick={() => handleStructuredAction(act.id as any)}
              disabled={loading}
              className="px-2.5 py-1 rounded-lg bg-surface-elevated hover:bg-surface border border-border-subtle text-[10px] font-mono text-text-muted hover:text-accent transition-colors flex items-center gap-1 shrink-0 disabled:opacity-50"
            >
              <Icon className="w-3 h-3 text-accent" />
              <span>{act.label}</span>
            </button>
          );
        })}
      </div>

      {/* Ollama Offline Warning Banner */}
      {!ollamaOnline && (
        <div className="mx-4 mt-3 p-2.5 rounded-xl bg-status-warning/15 border border-status-warning/30 text-status-warning text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>Ollama is offline. Rule-based art director active.</span>
        </div>
      )}

      {/* Main Conversation Thread */}
      <div className="flex-1 p-4 overflow-y-auto space-y-4 text-xs">
        {messages.length === 0 ? (
          <div className="space-y-4 py-2">
            <div className="p-4 rounded-xl bg-surface-elevated/80 border border-border-subtle space-y-2">
              <div className="flex items-center gap-2 text-accent font-medium">
                <Lightbulb className="w-4 h-4" />
                <span>Interactive AI Art Director</span>
              </div>
              <p className="text-text-muted leading-relaxed">
                Copilot provides advisory guidance for your RTX 5070 Ti 16 GB workstation. All suggestions display diff cards and require your explicit review before applying.
              </p>
            </div>

            {/* Quick Task Triggers */}
            <div className="space-y-2">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-text-dim block font-mono">
                Advisory Actions
              </span>
              <div className="space-y-1.5">
                <button
                  onClick={() => handleStructuredAction('fast_generation')}
                  disabled={loading}
                  className="w-full text-left p-3 rounded-xl bg-surface-elevated hover:bg-surface-hover border border-border-subtle text-text-primary font-medium flex items-center justify-between transition-colors shadow-sm"
                >
                  <div className="space-y-0.5">
                    <span className="block text-xs font-semibold">Optimize for Fast Speed</span>
                    <span className="text-[10px] text-text-dim block">Tuned for rapid 16-step draft iteration (~10s)</span>
                  </div>
                  <Zap className="w-4 h-4 text-accent" />
                </button>

                <button
                  onClick={() => handleStructuredAction('quality')}
                  disabled={loading}
                  className="w-full text-left p-3 rounded-xl bg-surface-elevated hover:bg-surface-hover border border-border-subtle text-text-primary font-medium flex items-center justify-between transition-colors shadow-sm"
                >
                  <div className="space-y-0.5">
                    <span className="block text-xs font-semibold">Optimize for Maximum Quality</span>
                    <span className="text-[10px] text-text-dim block">28-32 steps with photographic prompts</span>
                  </div>
                  <Sparkles className="w-4 h-4 text-accent" />
                </button>

                <button
                  onClick={() => handleStructuredAction('vram_safe_resolution')}
                  disabled={loading}
                  className="w-full text-left p-3 rounded-xl bg-surface-elevated hover:bg-surface-hover border border-border-subtle text-text-primary font-medium flex items-center justify-between transition-colors shadow-sm"
                >
                  <div className="space-y-0.5">
                    <span className="block text-xs font-semibold">Suggest VRAM-Safe Resolution</span>
                    <span className="text-[10px] text-text-dim block">Evaluates 16 GB budget against model baseline</span>
                  </div>
                  <Shield className="w-4 h-4 text-accent" />
                </button>

                <button
                  onClick={() => handleStructuredAction('explain_slowness')}
                  disabled={loading}
                  className="w-full text-left p-3 rounded-xl bg-surface-elevated hover:bg-surface-hover border border-border-subtle text-text-primary font-medium flex items-center justify-between transition-colors shadow-sm"
                >
                  <div className="space-y-0.5">
                    <span className="block text-xs font-semibold">Explain Potential Job Slowness</span>
                    <span className="text-[10px] text-text-dim block">Inspects steps, resolution, and LoRA overhead</span>
                  </div>
                  <HelpCircle className="w-4 h-4 text-accent" />
                </button>

                <button
                  onClick={() => handleStructuredAction('suggest_preset')}
                  disabled={loading}
                  className="w-full text-left p-3 rounded-xl bg-surface-elevated hover:bg-surface-hover border border-border-subtle text-text-primary font-medium flex items-center justify-between transition-colors shadow-sm"
                >
                  <div className="space-y-0.5">
                    <span className="block text-xs font-semibold">Suggest Matching Preset</span>
                    <span className="text-[10px] text-text-dim block">Matches prompt subject against 7 studio presets</span>
                  </div>
                  <Bookmark className="w-4 h-4 text-accent" />
                </button>
              </div>
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`space-y-2 ${msg.sender === 'user' ? 'text-right' : 'text-left'}`}
            >
              {msg.sender === 'user' ? (
                <div className="inline-block bg-accent/15 border border-accent/30 text-text-primary p-3 rounded-xl text-left max-w-[85%] font-medium">
                  {msg.text}
                </div>
              ) : (
                <div className="p-4 rounded-xl bg-surface-elevated border border-border-subtle space-y-3 text-left">
                  <div className="text-text-primary whitespace-pre-wrap leading-relaxed text-[11px] font-mono">
                    {msg.text}
                  </div>

                  {/* Structured Action Review Card */}
                  {msg.actionResponse && (
                    <div className="p-3.5 rounded-xl bg-surface border border-accent/40 space-y-3 font-mono text-[11px]">
                      <div className="flex items-center justify-between border-b border-border-subtle pb-2">
                        <span className="text-accent font-semibold flex items-center gap-1.5 uppercase text-[10px]">
                          <Zap className="w-3.5 h-3.5" />
                          Advisory Suggestion
                        </span>
                        {msg.actionResponse.preset_name && (
                          <span className="px-2 py-0.5 rounded bg-accent/15 text-accent text-[10px]">
                            Preset: {msg.actionResponse.preset_name}
                          </span>
                        )}
                      </div>

                      {/* Parameter changes grid */}
                      {Object.keys(msg.actionResponse.suggested_changes).length > 0 && (
                        <div className="space-y-1.5">
                          <span className="text-[10px] uppercase font-sans text-text-dim block">
                            Suggested Parameter Modifications:
                          </span>
                          <div className="grid grid-cols-2 gap-1.5">
                            {Object.entries(msg.actionResponse.suggested_changes).map(([k, val]) => (
                              <div key={k} className="p-1.5 rounded bg-surface-elevated text-[10px]">
                                <span className="text-text-dim block capitalize">{k.replace('_', ' ')}</span>
                                <span className="text-accent font-semibold">{String(val)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Metrics Footer */}
                      <div className="flex items-center justify-between text-[10px] text-text-dim pt-1 border-t border-border-subtle">
                        <span>Ollama 120s timeout safe</span>
                        {msg.actionResponse.metrics?.duration_ms && (
                          <span>{msg.actionResponse.metrics.duration_ms} ms latency</span>
                        )}
                      </div>

                      {/* Explicit Apply Changes Button */}
                      <button
                        onClick={() => handleApplyActionChanges(msg.id, msg.actionResponse!)}
                        disabled={appliedFields[msg.id]?.all}
                        className={`w-full py-2 px-3 rounded-lg font-semibold text-xs flex items-center justify-center space-x-1.5 transition-all ${
                          appliedFields[msg.id]?.all
                            ? 'bg-status-success/20 text-status-success border border-status-success/40'
                            : 'bg-accent text-background hover:bg-accent-hover shadow-sm'
                        }`}
                      >
                        {appliedFields[msg.id]?.all ? (
                          <>
                            <Check className="w-4 h-4" />
                            <span>Changes Applied to Composer</span>
                          </>
                        ) : (
                          <>
                            <Sparkles className="w-4 h-4" />
                            <span>Apply Suggested Changes</span>
                          </>
                        )}
                      </button>
                    </div>
                  )}

                  {/* Standard variations list */}
                  {msg.response?.variations && msg.response.variations.length > 0 && (
                    <div className="space-y-1.5 pt-2 border-t border-border-subtle">
                      <span className="text-[10px] font-semibold uppercase text-accent font-mono block">
                        Select Variation to Insert:
                      </span>
                      {msg.response.variations.map((v, i) => (
                        <button
                          key={i}
                          onClick={() => setPrompt(v)}
                          className="w-full text-left p-2.5 rounded-lg bg-surface hover:bg-surface-hover border border-border-subtle text-[11px] text-text-muted hover:text-text-primary transition-colors flex items-center justify-between group"
                        >
                          <span className="line-clamp-2">{v}</span>
                          <ChevronRight className="w-3.5 h-3.5 text-accent opacity-0 group-hover:opacity-100 transition-opacity shrink-0 ml-1" />
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Legacy proposed settings card */}
                  {msg.response?.suggested_settings && (
                    <div className="p-3 rounded-xl bg-surface border border-accent/30 space-y-2.5">
                      <div className="flex items-center justify-between text-[11px] font-semibold text-accent font-mono">
                        <span className="flex items-center gap-1.5">
                          <Sliders className="w-3 h-3" />
                          Proposed Recipe Changes
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-1.5 text-[10px] font-mono">
                        {msg.response.suggested_settings.steps && (
                          <div className="p-1.5 rounded bg-surface-elevated">
                            <span className="text-text-dim block">Steps</span>
                            <span className="text-text-primary">{steps} → {msg.response.suggested_settings.steps}</span>
                          </div>
                        )}
                        {msg.response.suggested_settings.cfg_scale && (
                          <div className="p-1.5 rounded bg-surface-elevated">
                            <span className="text-text-dim block">CFG</span>
                            <span className="text-text-primary">{cfg_scale} → {msg.response.suggested_settings.cfg_scale}</span>
                          </div>
                        )}
                        {msg.response.suggested_settings.sampler && (
                          <div className="p-1.5 rounded bg-surface-elevated">
                            <span className="text-text-dim block">Sampler</span>
                            <span className="text-text-primary">{sampler} → {msg.response.suggested_settings.sampler}</span>
                          </div>
                        )}
                        {msg.response.suggested_settings.width && (
                          <div className="p-1.5 rounded bg-surface-elevated">
                            <span className="text-text-dim block">Resolution</span>
                            <span className="text-text-primary">{msg.response.suggested_settings.width}×{msg.response.suggested_settings.height}</span>
                          </div>
                        )}
                      </div>

                      {/* Granular Field Apply Actions */}
                      <div className="flex flex-col gap-1.5 pt-1">
                        <button
                          onClick={() => handleApplyField(msg.id, 'all', msg.response!.suggested_settings!)}
                          disabled={appliedFields[msg.id]?.all}
                          className={`w-full py-2 px-3 rounded-lg font-semibold text-xs flex items-center justify-center space-x-1.5 transition-all mt-1 ${
                            appliedFields[msg.id]?.all
                              ? 'bg-status-success/20 text-status-success border border-status-success/40'
                              : 'bg-accent text-background hover:bg-accent-hover shadow-sm'
                          }`}
                        >
                          {appliedFields[msg.id]?.all ? (
                            <>
                              <Check className="w-4 h-4" />
                              <span>All Recipe Changes Applied</span>
                            </>
                          ) : (
                            <>
                              <Sparkles className="w-4 h-4" />
                              <span>Apply All Recipe Changes</span>
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))
        )}

        {loading && (
          <div className="p-4 rounded-xl bg-surface-elevated text-center text-text-muted space-y-2 border border-border-subtle">
            <div className="w-4 h-4 border-2 border-accent border-t-transparent rounded-full animate-spin mx-auto" />
            <p className="text-xs">Consulting Studio Copilot...</p>
          </div>
        )}
      </div>

      {/* Input Message Footer */}
      <div className="p-4 border-t border-border-subtle bg-surface-elevated/40 shrink-0">
        <div className="relative">
          <input
            type="text"
            placeholder="Ask Copilot for art direction, styles, or settings..."
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend('configure_generation');
              }
            }}
            className="w-full bg-surface text-text-primary text-xs rounded-xl pl-3.5 pr-10 py-3 border border-border-subtle focus:border-accent focus:outline-none placeholder:text-text-dim"
          />
          <button
            onClick={() => handleSend('configure_generation')}
            disabled={loading || (!inputMessage.trim() && !prompt.trim())}
            aria-label="Send message to Studio Copilot"
            className="absolute right-2 top-2 p-1.5 rounded-lg bg-accent text-background hover:bg-accent-hover disabled:opacity-40 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            title="Send query"
          >
            <Send className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};
