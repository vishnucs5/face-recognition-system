import React, { useEffect, useRef } from 'react';
import { LeftRail } from './components/LeftRail';
import { Header } from './components/Header';
import { ErrorBanner } from './components/ErrorBanner';
import { PromptComposer } from './components/PromptComposer';
import { GenerationSettings } from './components/GenerationSettings';
import { RecentGenerationsStrip } from './components/RecentGenerationsStrip';
import { PreviewPanel } from './components/PreviewPanel';
import { CopilotDrawer } from './components/CopilotDrawer';
import { GalleryView } from './components/GalleryView';
import { SettingsView } from './components/SettingsView';
import { LorasView } from './components/LorasView';
import { TrainingView } from './components/TrainingView';
import { PresetsView } from './components/PresetsView';
import { PresetReviewModal } from './components/PresetReviewModal';
import { KeyboardShortcutsModal } from './components/KeyboardShortcutsModal';
import { useStudioStore } from './store/useStudioStore';
import { GenerationProgressUpdate, GenerationResponse } from './types';

export const App: React.FC = () => {
  const {
    activeNav,
    setActiveNav,
    mode,
    setMode,
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
    source_image_id,
    mask_image_id,
    controlnet_type,
    controlnet_model,
    controlnet_strength,
    controlnet_start_percent,
    controlnet_end_percent,
    controlnet_preprocessor,
    startGeneration,
    setProgress,
    finishGeneration,
    cancelActiveGeneration,
    currentJobId,
    setSystemStatus,
    telemetry,
    setTelemetry,
    priority,
    selectedPreset,
    loadPresets,
    setAvailableModels,
    setAvailableLoras,
    setLoraVault,
    setRecentGenerations,
    setErrorMessage,
    mobileView,
    setMobileView,
    copilotOpen,
    setCopilotOpen,
    shortcutsModalOpen,
    setShortcutsModalOpen,
    randomizeSeed,
    isGenerating,
    isSubmitting,
    setIsSubmitting,
  } = useStudioStore();

  const wsRef = useRef<WebSocket | null>(null);

  // Initial data loading & telemetry polling
  useEffect(() => {
    loadPresets();

    const fetchSystemData = async () => {
      // Pause polling when user is not viewing this browser tab
      if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
        return;
      }

      try {
        const [statusRes, modelsRes, lorasRes, gensRes, telemetryRes] = await Promise.all([
          fetch('/api/status'),
          fetch('/api/models'),
          fetch('/api/loras'),
          fetch('/api/generations?limit=12'),
          fetch('/api/system/telemetry'),
        ]);

        if (statusRes.ok) setSystemStatus(await statusRes.json());
        if (modelsRes.ok) setAvailableModels(await modelsRes.json());
        if (lorasRes.ok) {
          const lorasData = await lorasRes.json();
          setAvailableLoras(lorasData);
          setLoraVault(lorasData);
        }
        if (gensRes.ok) setRecentGenerations(await gensRes.json());
        if (telemetryRes.ok) {
          setTelemetry(await telemetryRes.json());
        } else {
          setTelemetry(null);
        }
      } catch (err) {
        console.warn('Backend connection polling note:', err);
      }
    };

    fetchSystemData();
    const interval = setInterval(fetchSystemData, 5000);

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        fetchSystemData();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      clearInterval(interval);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  // Fallback polling for in-flight generations (guarantees terminal state sync if WS drops)
  useEffect(() => {
    if (!isGenerating || !currentJobId) return;

    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/generations/${currentJobId}`);
        if (!res.ok) return;
        const gen: GenerationResponse = await res.json();
        if (gen.status === 'completed' || gen.status === 'failed' || gen.status === 'cancelled') {
          if (wsRef.current) wsRef.current.close();
          finishGeneration(gen);
        }
      } catch (err) {
        console.warn('Fallback polling notice:', err);
      }
    }, 2000);

    return () => clearInterval(pollInterval);
  }, [isGenerating, currentJobId, finishGeneration]);

  // Global keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const activeElement = document.activeElement;
      const isInput =
        activeElement instanceof HTMLInputElement ||
        activeElement instanceof HTMLTextAreaElement ||
        (activeElement as HTMLElement)?.isContentEditable;

      // Escape: Close modal, drawer, or cancel job
      if (e.key === 'Escape') {
        if (shortcutsModalOpen) {
          e.preventDefault();
          setShortcutsModalOpen(false);
          return;
        }
        if (copilotOpen) {
          e.preventDefault();
          setCopilotOpen(false);
          return;
        }
        if (isGenerating && currentJobId) {
          e.preventDefault();
          handleCancel();
          return;
        }
      }

      // '?' key when not in an input field: Open shortcuts reference
      if (e.key === '?' && !isInput && !e.ctrlKey && !e.metaKey && !e.altKey) {
        e.preventDefault();
        setShortcutsModalOpen(true);
        return;
      }

      // Mod key combinations (Ctrl / Cmd)
      if (e.ctrlKey || e.metaKey) {
        // Ctrl+Enter: Execute generation
        if (e.key === 'Enter') {
          e.preventDefault();
          if (!isGenerating && !isSubmitting) {
            handleGenerate();
          }
          return;
        }

        // Ctrl+K: Toggle Copilot drawer
        if (e.key.toLowerCase() === 'k') {
          e.preventDefault();
          setCopilotOpen(!copilotOpen);
          return;
        }

        // Ctrl+G: Toggle between Studio and Gallery
        if (e.key.toLowerCase() === 'g') {
          e.preventDefault();
          setActiveNav(activeNav === 'gallery' ? 'generate' : 'gallery');
          return;
        }

        // Ctrl+D: Randomize seed
        if (e.key.toLowerCase() === 'd') {
          e.preventDefault();
          randomizeSeed();
          return;
        }

        // Ctrl+1: Text to Image
        if (e.key === '1') {
          e.preventDefault();
          setMode('txt2img');
          setActiveNav('generate');
          return;
        }

        // Ctrl+2: Image to Image
        if (e.key === '2') {
          e.preventDefault();
          setMode('img2img');
          setActiveNav('generate');
          return;
        }

        // Ctrl+3: Inpainting
        if (e.key === '3') {
          e.preventDefault();
          setMode('inpaint');
          setActiveNav('generate');
          return;
        }

        // Ctrl+4: ControlNet
        if (e.key === '4') {
          e.preventDefault();
          setMode('controlnet');
          setActiveNav('generate');
          return;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
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
    source_image_id,
    mask_image_id,
    controlnet_type,
    controlnet_model,
    controlnet_strength,
    mode,
    copilotOpen,
    shortcutsModalOpen,
    activeNav,
    isGenerating,
    currentJobId,
  ]);

  const handleGenerate = async () => {
    if (isGenerating || isSubmitting) return;
    if (!prompt.trim()) {
      setErrorMessage('Please enter a prompt before generating.');
      return;
    }
    if (mode === 'img2img' && !source_image_id) {
      setErrorMessage('Image-to-Image mode requires an uploaded source image.');
      return;
    }
    if (mode === 'inpaint') {
      if (!source_image_id || !mask_image_id) {
        setErrorMessage('Inpainting requires both a source image and a painted mask.');
        return;
      }
    }
    if (mode === 'controlnet') {
      if (!source_image_id) {
        setErrorMessage('ControlNet mode requires an input guide image.');
        return;
      }
      if (!controlnet_model) {
        setErrorMessage(
          'ControlNet model unavailable — place a controlnet model (e.g. control_v11p_sd15_canny.safetensors) in ComfyUI/models/controlnet/ to enable ControlNet generation.'
        );
        return;
      }
    }

    setIsSubmitting(true);
    try {
      const payload = {
        mode,
        prompt: prompt.trim(),
        negative_prompt: negative_prompt.trim(),
        model,
        width,
        height,
        steps,
        cfg_scale,
        sampler,
        seed: seed ?? undefined,
        denoise,
        loras,
        source_image_id: (mode === 'img2img' || mode === 'inpaint' || mode === 'controlnet') ? source_image_id : undefined,
        mask_image_id: mode === 'inpaint' ? mask_image_id : undefined,
        controlnet_type: mode === 'controlnet' ? controlnet_type : undefined,
        controlnet_model: mode === 'controlnet' ? controlnet_model : undefined,
        controlnet_strength: mode === 'controlnet' ? controlnet_strength : undefined,
        controlnet_start_percent: mode === 'controlnet' ? controlnet_start_percent : undefined,
        controlnet_end_percent: mode === 'controlnet' ? controlnet_end_percent : undefined,
        controlnet_preprocessor: mode === 'controlnet' ? controlnet_preprocessor : undefined,
        priority: priority ?? 2,
        client_request_id: typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : undefined,
        preset_name: selectedPreset?.name || undefined,
        preset_version: selectedPreset?.version || undefined,
      };

      const res = await fetch('/api/generations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        setErrorMessage(err.detail || 'Failed to submit generation.');
        setIsSubmitting(false);
        return;
      }

      const initialGen: GenerationResponse = await res.json();
      startGeneration(initialGen);
      setMobileView('preview'); // On mobile/tablet, switch to preview so progress is immediately visible

      // Connect to WebSocket progress feed
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.host}/api/ws/generations/${initialGen.id}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onmessage = async (event) => {
        try {
          const update: GenerationProgressUpdate = JSON.parse(event.data);
          setProgress(update);

          if (update.status === 'completed' || update.status === 'failed' || update.status === 'cancelled') {
            ws.close();
            // Fetch final record from database
            const finalRes = await fetch(`/api/generations/${initialGen.id}`);
            if (finalRes.ok) {
              const fullRecord: GenerationResponse = await finalRes.json();
              finishGeneration(fullRecord);
            }
          }
        } catch (e) {
          console.error('Error parsing WS message:', e);
        }
      };

      ws.onerror = (e) => console.error('WebSocket error:', e);
    } catch (err: any) {
      console.error('Generation initiation error:', err);
      setErrorMessage(err?.message || 'Error communicating with generation server.');
      setIsSubmitting(false);
    }
  };

  const handleCancel = async (jobId?: string) => {
    const targetId = jobId || currentJobId;
    if (!targetId) return;
    try {
      await fetch(`/api/generations/${targetId}/cancel`, { method: 'POST' });
      cancelActiveGeneration(targetId);
      if (wsRef.current) wsRef.current.close();
    } catch (err) {
      console.error('Cancellation error:', err);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-text-primary">
      {/* 1. Left Rail Navigation (desktop permanent, collapsible on compact) */}
      <LeftRail />

      {/* 2. Main Center-Workspace */}
      {activeNav === 'gallery' ? (
        <GalleryView />
      ) : activeNav === 'presets' ? (
        <PresetsView />
      ) : activeNav === 'loras' ? (
        <LorasView />
      ) : activeNav === 'training' ? (
        <TrainingView />
      ) : activeNav === 'settings' ? (
        <SettingsView />
      ) : (
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <Header />

          {/* Error & Engine Notices Banner */}
          <ErrorBanner />

          {/* Controls & Workspace Split */}
          <div className="flex-1 flex overflow-hidden">
            {/* Scrollable Generation Form (Desktop: 50% left column; Mobile: visible when mobileView === 'controls') */}
            <main
              className={`w-full lg:w-1/2 p-6 overflow-y-auto border-r border-border-subtle space-y-6 ${
                mobileView === 'controls' ? 'block' : 'hidden lg:block'
              }`}
            >
              <PromptComposer />
              <GenerationSettings onGenerate={handleGenerate} onCancel={handleCancel} />
              <div className="pt-2 border-t border-border-subtle">
                <RecentGenerationsStrip />
              </div>
            </main>

            {/* Right Preview Panel (Desktop: 50% right column; Mobile: visible when mobileView === 'preview') */}
            <aside
              className={`w-full lg:w-1/2 overflow-hidden ${
                mobileView === 'preview' ? 'block' : 'hidden lg:block'
              }`}
            >
              <PreviewPanel onRetry={handleGenerate} />
            </aside>
          </div>
        </div>
      )}

      {/* 3. Studio Copilot Slide-over Drawer */}
      <CopilotDrawer />

      {/* 4. Keyboard Shortcuts Reference Dialog */}
      <KeyboardShortcutsModal />

      {/* 5. Preset Review Diff Modal */}
      <PresetReviewModal />
    </div>
  );
};
