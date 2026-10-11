import React from 'react';
import { X, Keyboard, Command, Sparkles, Navigation, Sliders } from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';

export const KeyboardShortcutsModal: React.FC = () => {
  const { shortcutsModalOpen, setShortcutsModalOpen } = useStudioStore();

  if (!shortcutsModalOpen) return null;

  const isMac = typeof window !== 'undefined' && /Mac|iPod|iPhone|iPad/.test(navigator.platform);
  const modKey = isMac ? '⌘' : 'Ctrl';

  const shortcutGroups = [
    {
      title: 'Generation & Controls',
      icon: <Sparkles className="w-4 h-4 text-accent" />,
      items: [
        { keys: [`${modKey}`, 'Enter'], description: 'Submit & execute generation job' },
        { keys: [`${modKey}`, 'D'], description: 'Randomize seed (dice roll)' },
        { keys: ['Esc'], description: 'Cancel active generation or close open modal' },
      ],
    },
    {
      title: 'Navigation & Modes',
      icon: <Navigation className="w-4 h-4 text-accent" />,
      items: [
        { keys: [`${modKey}`, '1'], description: 'Switch to Text to Image mode' },
        { keys: [`${modKey}`, '2'], description: 'Switch to Image to Image mode' },
        { keys: [`${modKey}`, 'G'], description: 'Toggle between Studio & Gallery' },
      ],
    },
    {
      title: 'Copilot & Overlays',
      icon: <Sliders className="w-4 h-4 text-accent" />,
      items: [
        { keys: [`${modKey}`, 'K'], description: 'Toggle Studio Copilot assistant drawer' },
        { keys: ['?'], description: 'Open this keyboard shortcuts reference' },
        { keys: ['Esc'], description: 'Close any active overlay or drawer' },
      ],
    },
  ];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-fade-in"
      role="dialog"
      aria-modal="true"
      aria-labelledby="shortcuts-dialog-title"
      onClick={() => setShortcutsModalOpen(false)}
    >
      <div
        className="w-full max-w-lg bg-surface border border-border-subtle rounded-2xl shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-border-subtle flex items-center justify-between bg-surface-elevated/40">
          <div className="flex items-center space-x-2.5">
            <div className="p-1.5 rounded-lg bg-accent/20 border border-accent/40 text-accent">
              <Keyboard className="w-4 h-4" />
            </div>
            <div>
              <h2 id="shortcuts-dialog-title" className="text-sm font-semibold text-text-primary tracking-tight">
                Keyboard Shortcuts
              </h2>
              <p className="text-[11px] text-text-muted">Accelerate your workflow with workstation hotkeys</p>
            </div>
          </div>
          <button
            onClick={() => setShortcutsModalOpen(false)}
            className="p-1.5 rounded-lg text-text-muted hover:text-text-primary hover:bg-surface-elevated transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            aria-label="Close keyboard shortcuts dialog"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-5 max-h-[70vh] overflow-y-auto">
          {shortcutGroups.map((group, idx) => (
            <div key={idx} className="space-y-2.5">
              <div className="flex items-center space-x-2 text-xs font-semibold text-text-primary uppercase tracking-wider font-mono">
                {group.icon}
                <span>{group.title}</span>
              </div>
              <div className="space-y-1.5">
                {group.items.map((item, itemIdx) => (
                  <div
                    key={itemIdx}
                    className="flex items-center justify-between p-2 rounded-lg bg-surface-elevated/60 border border-border-subtle/50 hover:border-border-subtle transition-colors"
                  >
                    <span className="text-xs text-text-muted">{item.description}</span>
                    <div className="flex items-center space-x-1">
                      {item.keys.map((k, kIdx) => (
                        <kbd
                          key={kIdx}
                          className="px-2 py-0.5 text-[11px] font-mono font-semibold text-accent bg-surface border border-accent/30 rounded shadow-xs"
                        >
                          {k}
                        </kbd>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-border-subtle bg-surface-elevated/20 flex items-center justify-between text-[11px] text-text-dim">
          <span>Press <kbd className="font-mono text-text-muted font-medium">Esc</kbd> anytime to dismiss overlays</span>
          <button
            onClick={() => setShortcutsModalOpen(false)}
            className="px-3 py-1 rounded-md text-xs font-medium bg-surface-elevated hover:bg-surface text-text-primary border border-border-subtle transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  );
};
