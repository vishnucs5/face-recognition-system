import React, { useEffect, useRef, useState } from 'react';
import {
  Bookmark,
  Sparkles,
  Plus,
  Download,
  Upload,
  Copy,
  Trash2,
  Edit2,
  Check,
  X,
  AlertCircle,
  Cpu,
  Layers,
  Sliders,
  Maximize2,
  CheckCircle2,
} from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';
import { PresetModel } from '../types';

export const PresetsView: React.FC = () => {
  const {
    presets,
    loadPresets,
    setDiffModalPreset,
    setErrorMessage,
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
    controlnet_type,
    controlnet_model,
    controlnet_strength,
    mode,
    vramPolicy,
  } = useStudioStore();

  const [activeFilter, setActiveFilter] = useState<'all' | 'built_in' | 'user'>('all');
  const [saveModalOpen, setSaveModalOpen] = useState(false);
  const [newPresetName, setNewPresetName] = useState('');
  const [newPresetDesc, setNewPresetDesc] = useState('');
  const [editModalPreset, setEditModalPreset] = useState<PresetModel | null>(null);
  const [editName, setEditName] = useState('');
  const [editDesc, setEditDesc] = useState('');
  const [importStatus, setImportStatus] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadPresets();
  }, []);

  const filteredPresets = presets.filter((p) => {
    if (activeFilter === 'built_in') return p.is_built_in;
    if (activeFilter === 'user') return !p.is_built_in;
    return true;
  });

  const handleSaveCurrentAsPreset = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newPresetName.trim()) return;

    try {
      const payload = {
        name: newPresetName.trim(),
        description: newPresetDesc.trim(),
        mode,
        base_model_family: model.toLowerCase().includes('sd_xl') || model.toLowerCase().includes('sdxl') ? 'sdxl' : 'sd15',
        checkpoint_name: model,
        prompt_template: prompt,
        negative_prompt,
        width,
        height,
        steps,
        cfg_scale,
        sampler,
        scheduler: 'karras',
        seed_policy: seed !== null ? 'fixed' : 'random',
        fixed_seed: seed,
        denoise,
        loras,
        controlnet_settings: controlnet_model ? { type: controlnet_type, model: controlnet_model, strength: controlnet_strength } : {},
        vram_profile: vramPolicy,
      };

      const res = await fetch('/api/presets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        setSaveModalOpen(false);
        setNewPresetName('');
        setNewPresetDesc('');
        await loadPresets();
      } else {
        const err = await res.json();
        setErrorMessage(err.detail || 'Failed to save preset.');
      }
    } catch (err) {
      console.error('Error saving preset:', err);
    }
  };

  const handleDuplicate = async (presetId: string) => {
    try {
      const res = await fetch(`/api/presets/${presetId}/duplicate`, { method: 'POST' });
      if (res.ok) {
        await loadPresets();
      }
    } catch (err) {
      console.error('Error duplicating preset:', err);
    }
  };

  const handleDelete = async (presetId: string) => {
    try {
      const res = await fetch(`/api/presets/${presetId}`, { method: 'DELETE' });
      if (res.ok) {
        await loadPresets();
      }
    } catch (err) {
      console.error('Error deleting preset:', err);
    }
  };

  const handleUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editModalPreset || !editName.trim()) return;

    try {
      const res = await fetch(`/api/presets/${editModalPreset.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: editName.trim(),
          description: editDesc.trim(),
        }),
      });

      if (res.ok) {
        setEditModalPreset(null);
        await loadPresets();
      }
    } catch (err) {
      console.error('Error updating preset:', err);
    }
  };

  const handleExport = async () => {
    try {
      const res = await fetch('/api/presets/export');
      if (res.ok) {
        const data = await res.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `forge_presets_${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }
    } catch (err) {
      console.error('Error exporting presets:', err);
    }
  };

  const handleImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const res = await fetch('/api/presets/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ json_content: text }),
      });

      if (res.ok) {
        const data = await res.json();
        setImportStatus(`Imported ${data.imported_count} presets successfully.`);
        await loadPresets();
        setTimeout(() => setImportStatus(null), 4000);
      } else {
        const err = await res.json();
        setErrorMessage(err.detail || 'Failed to import presets. Invalid or unsafe schema.');
      }
    } catch (err) {
      setErrorMessage('Failed to read presets JSON file.');
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  return (
    <div className="flex-1 p-6 md:p-10 overflow-y-auto space-y-6 max-w-6xl">
      {/* Header and Actions */}
      <div className="space-y-3 border-b border-border-subtle pb-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-text-primary flex items-center gap-2.5">
              <Bookmark className="w-6 h-6 text-accent" />
              Workflow Presets
            </h1>
            <p className="text-xs text-text-muted mt-1">
              Tested studio parameter templates for portraits, landscapes, product shots, inpainting, and ControlNet.
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setSaveModalOpen(true)}
              className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-accent text-background hover:bg-accent-hover transition-colors flex items-center gap-1.5 shadow-md shadow-accent/10"
            >
              <Plus className="w-4 h-4" />
              <span>Save Current as Preset</span>
            </button>
            <button
              onClick={handleExport}
              className="px-3 py-2 rounded-xl text-xs font-medium bg-surface-elevated hover:bg-surface border border-border-subtle text-text-primary transition-colors flex items-center gap-1.5"
              title="Export user presets as JSON"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export</span>
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-3 py-2 rounded-xl text-xs font-medium bg-surface-elevated hover:bg-surface border border-border-subtle text-text-primary transition-colors flex items-center gap-1.5"
              title="Import presets from JSON"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>Import</span>
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImportFile}
              accept=".json"
              className="hidden"
            />
          </div>
        </div>

        {importStatus && (
          <div className="p-3 rounded-xl bg-status-success/15 border border-status-success/30 text-xs text-status-success flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4" />
            <span>{importStatus}</span>
          </div>
        )}

        {/* Filter Tabs */}
        <div className="flex items-center space-x-1.5 pt-2">
          {[
            { id: 'all', label: `All (${presets.length})` },
            { id: 'built_in', label: 'Factory Built-in (7)' },
            { id: 'user', label: `My Presets (${presets.filter((p) => !p.is_built_in).length})` },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveFilter(tab.id as any)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
                activeFilter === tab.id
                  ? 'bg-surface-elevated text-accent font-semibold border border-border-subtle'
                  : 'text-text-muted hover:text-text-primary hover:bg-surface'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Preset Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredPresets.map((p) => (
          <div
            key={p.id}
            className="p-5 rounded-2xl bg-surface border border-border-subtle hover:border-border-strong transition-all flex flex-col justify-between space-y-4 group"
          >
            <div className="space-y-3">
              {/* Card Top: Title, Badges */}
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="flex items-center space-x-2">
                    <h3 className="font-semibold text-text-primary text-sm group-hover:text-accent transition-colors">
                      {p.name}
                    </h3>
                    {p.is_built_in ? (
                      <span className="text-[9px] font-mono uppercase px-1.5 py-0.2 rounded bg-surface-elevated text-text-dim border border-border-subtle">
                        Built-in
                      </span>
                    ) : (
                      <span className="text-[9px] font-mono uppercase px-1.5 py-0.2 rounded bg-accent/15 text-accent border border-accent/30 font-semibold">
                        User
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-text-muted mt-1 line-clamp-2">{p.description}</p>
                </div>
              </div>

              {/* Parameters Grid */}
              <div className="grid grid-cols-2 gap-2 text-[11px] font-mono bg-surface-elevated p-3 rounded-xl border border-border-subtle">
                <div>
                  <span className="text-text-dim block text-[10px] uppercase font-sans">Resolution</span>
                  <span className="text-text-primary font-semibold">{p.width} × {p.height}</span>
                </div>
                <div>
                  <span className="text-text-dim block text-[10px] uppercase font-sans">Steps & CFG</span>
                  <span className="text-text-primary">{p.steps} steps / {p.cfg_scale}</span>
                </div>
                <div>
                  <span className="text-text-dim block text-[10px] uppercase font-sans">Sampler</span>
                  <span className="text-text-primary capitalize truncate block">{p.sampler}</span>
                </div>
                <div>
                  <span className="text-text-dim block text-[10px] uppercase font-sans">Profile</span>
                  <span className="text-accent font-semibold capitalize">{p.vram_profile}</span>
                </div>
              </div>

              {/* Prompt Template Snippet */}
              <div className="p-2.5 rounded-lg bg-surface-elevated/60 border border-border-subtle/60 text-[11px] text-text-muted font-mono line-clamp-2">
                "{p.prompt_template}"
              </div>
            </div>

            {/* Card Actions */}
            <div className="pt-3 border-t border-border-subtle/80 flex items-center justify-between">
              <div className="flex items-center space-x-1">
                <button
                  onClick={() => handleDuplicate(p.id)}
                  className="p-1.5 rounded-lg text-text-dim hover:text-text-primary hover:bg-surface-elevated transition-colors"
                  title="Duplicate as new preset"
                >
                  <Copy className="w-3.5 h-3.5" />
                </button>
                {!p.is_built_in && (
                  <>
                    <button
                      onClick={() => {
                        setEditModalPreset(p);
                        setEditName(p.name);
                        setEditDesc(p.description);
                      }}
                      className="p-1.5 rounded-lg text-text-dim hover:text-text-primary hover:bg-surface-elevated transition-colors"
                      title="Edit preset details"
                    >
                      <Edit2 className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => handleDelete(p.id)}
                      className="p-1.5 rounded-lg text-text-dim hover:text-status-error hover:bg-surface-elevated transition-colors"
                      title="Delete user preset"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </>
                )}
              </div>

              <button
                onClick={() => setDiffModalPreset(p)}
                className="px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-accent/20 hover:bg-accent text-accent hover:text-background transition-all border border-accent/40 shadow-sm flex items-center gap-1.5"
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>Apply Preset</span>
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Save Current as Preset Modal */}
      {saveModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm">
          <form
            onSubmit={handleSaveCurrentAsPreset}
            className="w-full max-w-md bg-surface border border-border-subtle rounded-2xl p-6 space-y-4 shadow-2xl"
          >
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-text-primary">Save Workspace as Preset</h2>
              <button
                type="button"
                onClick={() => setSaveModalOpen(false)}
                className="text-text-dim hover:text-text-primary"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <p className="text-xs text-text-muted">
              Snapshot current prompt, sampler, resolution, and VRAM profile into a reusable user preset.
            </p>

            <div className="space-y-3 text-xs">
              <div>
                <label className="text-text-muted font-medium block mb-1">Preset Name</label>
                <input
                  type="text"
                  required
                  value={newPresetName}
                  onChange={(e) => setNewPresetName(e.target.value)}
                  placeholder="e.g. Moody Cyberpunk Night"
                  className="w-full bg-surface-elevated text-text-primary rounded-lg p-2.5 border border-border-subtle focus:border-accent focus:outline-none"
                />
              </div>

              <div>
                <label className="text-text-muted font-medium block mb-1">Description</label>
                <textarea
                  rows={2}
                  value={newPresetDesc}
                  onChange={(e) => setNewPresetDesc(e.target.value)}
                  placeholder="Describe optimal use-cases or visual style"
                  className="w-full bg-surface-elevated text-text-primary rounded-lg p-2.5 border border-border-subtle focus:border-accent focus:outline-none"
                />
              </div>
            </div>

            <div className="flex justify-end space-x-2 pt-2">
              <button
                type="button"
                onClick={() => setSaveModalOpen(false)}
                className="px-3 py-1.5 rounded-lg text-xs text-text-muted hover:bg-surface-elevated"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-accent text-background hover:bg-accent-hover"
              >
                Save Preset
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Edit User Preset Modal */}
      {editModalPreset && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm">
          <form
            onSubmit={handleUpdate}
            className="w-full max-w-md bg-surface border border-border-subtle rounded-2xl p-6 space-y-4 shadow-2xl"
          >
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-text-primary">Edit Preset Details</h2>
              <button
                type="button"
                onClick={() => setEditModalPreset(null)}
                className="text-text-dim hover:text-text-primary"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <label className="text-text-muted font-medium block mb-1">Preset Name</label>
                <input
                  type="text"
                  required
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  className="w-full bg-surface-elevated text-text-primary rounded-lg p-2.5 border border-border-subtle focus:border-accent focus:outline-none"
                />
              </div>

              <div>
                <label className="text-text-muted font-medium block mb-1">Description</label>
                <textarea
                  rows={2}
                  value={editDesc}
                  onChange={(e) => setEditDesc(e.target.value)}
                  className="w-full bg-surface-elevated text-text-primary rounded-lg p-2.5 border border-border-subtle focus:border-accent focus:outline-none"
                />
              </div>
            </div>

            <div className="flex justify-end space-x-2 pt-2">
              <button
                type="button"
                onClick={() => setEditModalPreset(null)}
                className="px-3 py-1.5 rounded-lg text-xs text-text-muted hover:bg-surface-elevated"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-accent text-background hover:bg-accent-hover"
              >
                Save Changes
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
};
