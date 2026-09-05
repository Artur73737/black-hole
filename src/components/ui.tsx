import type { ReactNode } from 'react';

export function Slider({ label, value, min, max, step, onChange, fmt, log }: {
  label: string; value: number; min: number; max: number; step?: number;
  onChange: (v: number) => void; fmt?: (v: number) => string; log?: boolean;
}) {
  const toSlider = (v: number) => (log ? (Math.log(v) - Math.log(min)) / (Math.log(max) - Math.log(min)) : (v - min) / (max - min));
  const fromSlider = (s: number) => (log ? Math.exp(Math.log(min) + s * (Math.log(max) - Math.log(min))) : min + s * (max - min));
  return (
    <label className="block">
      <div className="flex justify-between text-[11px] tracking-wide text-zinc-400 mb-0.5">
        <span>{label}</span>
        <span className="text-amber-200/90 font-mono">{fmt ? fmt(value) : value.toFixed(2)}</span>
      </div>
      <input
        type="range" min={0} max={1} step={log ? 0.001 : (step ?? (max - min) / 1000) / (max - min)}
        value={toSlider(value)}
        onChange={(e) => onChange(fromSlider(parseFloat(e.target.value)))}
        className="w-full h-1 accent-amber-400 cursor-pointer"
      />
    </label>
  );
}

export function Toggle({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button onClick={() => onChange(!value)} className="flex items-center justify-between w-full text-[11px] tracking-wide text-zinc-300 py-0.5">
      <span>{label}</span>
      <span className={`w-8 h-4 rounded-full relative transition ${value ? 'bg-amber-400/80' : 'bg-zinc-700'}`}>
        <span className={`absolute top-0.5 w-3 h-3 rounded-full bg-black transition ${value ? 'left-4.5 translate-x-[1px]' : 'left-0.5'}`} />
      </span>
    </button>
  );
}

export function Section({ title, children, right }: { title: string; children: ReactNode; right?: ReactNode }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-[10px] uppercase tracking-[0.2em] text-amber-300/80">{title}</h3>
        {right}
      </div>
      {children}
    </div>
  );
}

export function Row({ k, v, title }: { k: string; v: string; title?: string }) {
  return (
    <div className="flex justify-between gap-3 text-[11px] leading-5" title={title}>
      <span className="text-zinc-400">{k}</span>
      <span className="font-mono text-zinc-100 text-right">{v}</span>
    </div>
  );
}
