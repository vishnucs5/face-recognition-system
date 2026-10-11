import React from 'react';
import {
  History,
  Sparkles,
  RotateCcw,
  AlertTriangle,
  Clock,
  Trash2,
  Ban,
  Pause,
  Play,
  X,
  ArrowUp,
  ArrowDown,
  Square,
  Zap,
} from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';

export const RecentGenerationsStrip: React.FC = () => {
  const {
    recentGenerations,
    activeGeneration,
    reuseSettings,
    selectGeneration,
    emptyQueue,
    isGenerating,
    currentJobId,
    isQueuePaused,
    pauseQueue,
    resumeQueue,
    cancelQueuedJobsOnly,
    interruptActiveJob,
    reorderQueue,
  } = useStudioStore();

  const queuedJobs = recentGenerations.filter(
    (g) => g.status === 'queued' || g.status === 'waiting_for_gpu'
  );
  const hasQueued = queuedJobs.length > 0;
  const runningJob = recentGenerations.find((g) => g.status === 'running' || g.id === currentJobId);

  const handleCancelSingle = async (e: React.MouseEvent, jobId: string) => {
    e.stopPropagation();
    try {
      await fetch(`/api/generations/${jobId}/cancel`, { method: 'POST' });
      const gensRes = await fetch('/api/generations?limit=12');
      if (gensRes.ok) useStudioStore.getState().setRecentGenerations(await gensRes.json());
    } catch (err) {
      console.error('Failed to cancel job:', err);
    }
  };

  const handlePriorityChange = async (e: React.MouseEvent, jobId: string, newPriority: number) => {
    e.stopPropagation();
    await reorderQueue(jobId, newPriority);
  };

  if (recentGenerations.length === 0) {
    return (
      <div className="p-3 rounded-xl bg-surface-elevated/40 border border-border-subtle flex items-center justify-between text-xs text-text-dim">
        <span className="flex items-center gap-1.5">
          <History className="w-3.5 h-3.5" />
          Queue & Recent History
        </span>
        <span className="text-[11px] font-mono">0 generations in current session</span>
      </div>
    );
  }

  return (
    <div className="space-y-2.5">
      {/* Queue Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center space-x-2">
          <span className="font-semibold uppercase tracking-wider text-text-muted flex items-center gap-1.5">
            <History className="w-3.5 h-3.5 text-accent" />
            Queue ({queuedJobs.length} queued, {recentGenerations.length} total)
          </span>
          {isQueuePaused && (
            <span className="px-1.5 py-0.5 rounded bg-status-warning/20 text-status-warning font-mono text-[10px] font-semibold border border-status-warning/30 animate-pulse">
              PAUSED
            </span>
          )}
        </div>

        <div className="flex items-center space-x-1.5">
          {/* Pause / Resume Button */}
          <button
            onClick={() => (isQueuePaused ? resumeQueue() : pauseQueue())}
            className={`px-2 py-1 rounded-lg text-[10px] font-mono transition-colors flex items-center gap-1 border ${
              isQueuePaused
                ? 'bg-status-success/20 text-status-success border-status-success/40 hover:bg-status-success/30'
                : 'bg-surface-elevated text-text-muted border-border-subtle hover:text-text-primary'
            }`}
            title={isQueuePaused ? 'Resume queue dispatch' : 'Pause queue dispatch'}
          >
            {isQueuePaused ? <Play className="w-3 h-3 fill-current" /> : <Pause className="w-3 h-3 fill-current" />}
            <span>{isQueuePaused ? 'Resume Queue' : 'Pause Queue'}</span>
          </button>

          {/* Interrupt Active Job */}
          {runningJob && (
            <button
              onClick={() => interruptActiveJob()}
              className="px-2 py-1 rounded-lg text-[10px] font-mono bg-status-error/20 text-status-error border border-status-error/40 hover:bg-status-error/30 transition-colors flex items-center gap-1"
              title="Interrupt active running job on GPU"
            >
              <Square className="w-3 h-3 fill-current" />
              <span>Interrupt Active</span>
            </button>
          )}

          {/* Cancel All Queued */}
          {hasQueued && (
            <button
              onClick={() => cancelQueuedJobsOnly()}
              className="px-2 py-1 rounded-lg text-[10px] font-mono bg-surface-elevated hover:bg-status-error/20 text-text-muted hover:text-status-error border border-border-subtle transition-colors flex items-center gap-1"
              title="Cancel only queued jobs (preserves completed history)"
            >
              <Ban className="w-3 h-3" />
              <span>Cancel Queued ({queuedJobs.length})</span>
            </button>
          )}

          {/* Clear local non-running history */}
          <button
            onClick={emptyQueue}
            className="text-[10px] text-text-dim hover:text-text-primary p-1 rounded hover:bg-surface-elevated transition-colors"
            title="Clear canceled/failed records from list"
            aria-label="Clean queue display"
          >
            <Trash2 className="w-3 h-3" />
          </button>
        </div>
      </div>

      {/* Generations Carousel Strip */}
      <div className="flex space-x-2.5 overflow-x-auto pb-1.5 scrollbar-thin">
        {recentGenerations.slice(0, 14).map((gen) => {
          const isActive = activeGeneration?.id === gen.id;
          const isJobRunning = (isGenerating && currentJobId === gen.id) || gen.status === 'running';
          const isQueued = gen.status === 'queued' || gen.status === 'waiting_for_gpu';

          return (
            <div
              key={gen.id}
              onClick={() => selectGeneration(gen)}
              className={`relative shrink-0 w-24 h-24 rounded-lg overflow-hidden border cursor-pointer group bg-surface-elevated transition-all ${
                isActive
                  ? 'border-accent ring-1 ring-accent shadow-md shadow-accent/10'
                  : gen.status === 'failed'
                  ? 'border-status-error/60 bg-status-error/5 hover:border-status-error'
                  : gen.status === 'cancelled'
                  ? 'border-border-subtle opacity-75 hover:opacity-100'
                  : isJobRunning
                  ? 'border-accent/80 ring-1 ring-accent/50 animate-pulse'
                  : isQueued
                  ? 'border-border-subtle hover:border-accent/50'
                  : 'border-border-subtle hover:border-border-strong'
              }`}
            >
              {gen.image_url ? (
                <img
                  src={gen.image_url}
                  alt={gen.prompt}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                />
              ) : gen.status === 'failed' ? (
                <div className="w-full h-full flex flex-col items-center justify-center text-[10px] text-status-error p-1 text-center font-mono">
                  <AlertTriangle className="w-4 h-4 text-status-error mb-1" />
                  <span className="text-[9px] font-semibold">Failed</span>
                  {(gen.retry_count ?? 0) > 0 && <span className="text-[8px] text-text-dim">{gen.retry_count} retries</span>}
                </div>
              ) : gen.status === 'cancelled' ? (
                <div className="w-full h-full flex flex-col items-center justify-center text-[10px] text-text-dim p-1 text-center font-mono">
                  <Ban className="w-4 h-4 text-text-dim mb-1" />
                  <span className="text-[9px]">Canceled</span>
                </div>
              ) : isJobRunning ? (
                <div className="w-full h-full flex flex-col items-center justify-center text-[10px] text-accent p-1 text-center font-mono">
                  <Sparkles className="w-4 h-4 text-accent mb-1 animate-spin" />
                  <span className="text-[9px] font-semibold">Running</span>
                </div>
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center text-[10px] text-text-muted p-1 text-center font-mono space-y-0.5">
                  <Clock className="w-4 h-4 text-accent mb-0.5" />
                  <span className="text-[9px] font-medium">
                    {gen.status === 'waiting_for_gpu' ? 'Waiting GPU' : 'Queued'}
                  </span>
                  {gen.estimated_vram_mb && (
                    <span className="text-[8px] text-text-dim">
                      {(gen.estimated_vram_mb / 1024).toFixed(1)} GB
                    </span>
                  )}
                </div>
              )}

              {/* Status & Priority Badge */}
              <div className="absolute top-1 left-1 flex items-center space-x-1">
                <span
                  className={`px-1 py-0.2 rounded text-[8px] font-mono uppercase tracking-wider backdrop-blur-sm border ${
                    gen.status === 'completed'
                      ? 'bg-background/80 text-text-muted border-border-subtle'
                      : gen.status === 'failed'
                      ? 'bg-status-error/80 text-background border-status-error font-bold'
                      : gen.status === 'cancelled'
                      ? 'bg-background/80 text-text-dim border-border-subtle'
                      : isJobRunning
                      ? 'bg-accent text-background font-bold border-accent animate-pulse'
                      : 'bg-background/80 text-accent font-semibold border-border-subtle'
                  }`}
                >
                  {gen.status === 'completed' ? gen.mode : isQueued ? (gen.priority === 1 ? 'P1' : gen.priority === 3 ? 'P3' : 'P2') : gen.status}
                </span>
              </div>

              {/* Queued Action Overlay: Priority Reordering & Cancel */}
              {isQueued && (
                <div className="absolute inset-0 bg-background/90 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center space-y-1 p-1">
                  <div className="flex items-center space-x-1">
                    <button
                      onClick={(e) => handlePriorityChange(e, gen.id, 1)}
                      className={`p-1 rounded ${gen.priority === 1 ? 'bg-accent text-background' : 'bg-surface text-text-muted hover:text-text-primary'}`}
                      title="Set High Priority (P1)"
                    >
                      <ArrowUp className="w-3 h-3" />
                    </button>
                    <button
                      onClick={(e) => handlePriorityChange(e, gen.id, 3)}
                      className={`p-1 rounded ${gen.priority === 3 ? 'bg-accent text-background' : 'bg-surface text-text-muted hover:text-text-primary'}`}
                      title="Set Low Priority (P3)"
                    >
                      <ArrowDown className="w-3 h-3" />
                    </button>
                    <button
                      onClick={(e) => handleCancelSingle(e, gen.id)}
                      className="p-1 rounded bg-surface hover:bg-status-error/20 text-status-error transition-colors"
                      title="Cancel this queued job"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                  <span className="text-[8px] font-mono text-text-dim">
                    {gen.priority === 1 ? 'High' : gen.priority === 3 ? 'Low' : 'Normal'}
                  </span>
                </div>
              )}

              {/* Completed Hover Action Overlay: Reuse Settings */}
              {gen.status === 'completed' && (
                <div className="absolute inset-0 bg-background/80 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center space-x-1 p-1">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      reuseSettings(gen);
                    }}
                    className="p-1.5 rounded-lg bg-surface border border-border-subtle text-accent hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                    title="Reuse parameters"
                    aria-label="Reuse parameters from this generation"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}

              {/* Seed pill (for completed images) */}
              {gen.image_url && (
                <div className="absolute bottom-1 right-1 px-1 py-0.2 rounded bg-background/80 text-[8px] font-mono text-text-dim">
                  {gen.seed ? String(gen.seed).slice(0, 4) : 'R'}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
