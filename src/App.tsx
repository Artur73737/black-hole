import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import skyUrl from './assets/mw4k.jpg';
import { BlackHoleRenderer, type RenderParams } from './gl/renderer';
import { FreeCamera } from './gl/camera';
import { buildBlackbodyLUT, blackbodyCSS } from './physics/blackbody';
import { derive, kerrMetric, keplerOmega, zamoOrbitalSpeed, diskPeakTemperature, sci, fmtTime, fmtLen, C, G, KB, AU } from './physics/blackhole';
import { Slider, Toggle, Section, Row } from './components/ui';

type Quality = 'bassa' | 'media' | 'alta' | 'ultra';
const QUALITY: Record<Quality, { resScale: number; maxSteps: number; stepScale: number }> = {
  bassa: { resScale: 0.5, maxSteps: 280, stepScale: 0.03 },
  media: { resScale: 0.7, maxSteps: 450, stepScale: 0.022 },
  alta: { resScale: 1.0, maxSteps: 650, stepScale: 0.017 },
  ultra: { resScale: 1.0, maxSteps: 1000, stepScale: 0.011 },
};

interface Settings {
  massSolar: number;
  spin: number;
  rout: number;
  innerFade: number;
  tempPeak: number;
  beaming: number;
  dopplerColor: number;
  diskOpacity: number;
  diskTurb: number;
  tempSlope: number;
  haze: number;
  skyBrightness: number;
  skyTilt: number;
  skyYaw: number;
  exposure: number;
  bloom: number;
  fov: number;
  timeWarp: number;
  quality: Quality;
  mdotEdd: number;
}

const DEFAULTS: Settings = {
  massSolar: 1e8,
  spin: 0.6,
  rout: 20,
  innerFade: 0.72,
  tempPeak: 6600,
  beaming: 0.6,
  dopplerColor: 0.6,
  diskOpacity: 2.2,
  diskTurb: 0.85,
  tempSlope: 0.62,
  haze: 0.35,
  skyBrightness: 0.35,
  skyTilt: 1.15,
  skyYaw: 0.6,
  exposure: 1.25,
  bloom: 0.14,
  fov: 62,
  timeWarp: 120,
  quality: 'media',
  mdotEdd: 0.1,
};

const PRESETS: Record<string, Partial<Settings>> = {
  'Interstellar (Thorne)': { spin: 0.6, beaming: 0.6, dopplerColor: 0.6, tempPeak: 6600, innerFade: 0.72, rout: 20, tempSlope: 0.62, haze: 0.35, exposure: 1.25 },
  'Interstellar (Nolan, no beaming)': { spin: 0.6, beaming: 0.1, dopplerColor: 0.05, tempPeak: 5600, innerFade: 0.6, rout: 22, tempSlope: 0.45, haze: 0.4, exposure: 1.6 },
  'Fisico puro (NT, g⁴)': { spin: 0.6, beaming: 1, dopplerColor: 1, tempPeak: 7000, innerFade: 0.8, rout: 20, tempSlope: 1, haze: 0.2, exposure: 1.4 },
  'Gargantua a=0.999': { spin: 0.999, beaming: 0.7, dopplerColor: 0.7, tempPeak: 7500, innerFade: 0.8, rout: 16, tempSlope: 0.62, haze: 0.35 },
  'Schwarzschild': { spin: 0, beaming: 0.7, dopplerColor: 0.7, tempPeak: 6500, innerFade: 0.75, rout: 24, tempSlope: 0.62, haze: 0.35 },
};

interface Hud {
  fps: number; r: number; alpha: number; omega: number; vEsc: number; tidal: number; accum: number; theta: number; vOrb: number; speed: number;
}

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<BlackHoleRenderer | null>(null);
  const cameraRef = useRef<FreeCamera | null>(null);
  const urlOpts = useMemo(() => {
    const sp = new URLSearchParams(window.location.search);
    const q = sp.get('q') as Quality | null;
    return { bare: sp.get('bare') === '1', quality: q && q in QUALITY ? q : null, preset: sp.get('preset') };
  }, []);
  const [settings, setSettings] = useState<Settings>(() => ({
    ...DEFAULTS,
    ...(urlOpts.preset && PRESETS[urlOpts.preset] ? PRESETS[urlOpts.preset] : {}),
    ...(urlOpts.quality ? { quality: urlOpts.quality } : {}),
  }));
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [errMsg, setErrMsg] = useState('');
  const [showPanel, setShowPanel] = useState(!urlOpts.bare);
  const [bare] = useState(urlOpts.bare);
  const [showPhysics, setShowPhysics] = useState(false);
  const [locked, setLocked] = useState(false);
  const [hud, setHud] = useState<Hud>({ fps: 0, r: 26, alpha: 1, omega: 0, vEsc: 0, tidal: 0, accum: 0, theta: 90, vOrb: 0, speed: 1 });
  const lut = useMemo(() => buildBlackbodyLUT(), []);

  const bh = useMemo(() => derive({ massSolar: settings.massSolar, spin: settings.spin }), [settings.massSolar, settings.spin]);
  const bhRef = useRef(bh);
  bhRef.current = bh;

  const set = useCallback(<K extends keyof Settings>(k: K, v: Settings[K]) => setSettings((s) => ({ ...s, [k]: v })), []);

  // ---------- init renderer ----------
  useEffect(() => {
    const canvas = canvasRef.current!;
    let cancelled = false;
    let raf = 0;
    let renderer: BlackHoleRenderer | null = null;
    let cam: FreeCamera | null = null;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      try {
        renderer = new BlackHoleRenderer(canvas, img, lut);
        cam = new FreeCamera(canvas);
        rendererRef.current = renderer;
        cameraRef.current = cam;
        setStatus('ready');
      } catch (e) {
        setErrMsg((e as Error).message);
        setStatus('error');
        return;
      }
      let last = performance.now();
      let fpsAcc = 0, fpsN = 0, hudT = 0;
      const loop = (now: number) => {
        if (cancelled || !renderer || !cam) return;
        const dt = Math.min((now - last) / 1000, 0.1);
        last = now;
        const s = settingsRef.current;
        const b = bhRef.current;
        const q = QUALITY[s.quality];
        const rh = b.rPlus / b.rg;
        const moving = cam.update(dt, rh);
        const params: RenderParams = {
          spin: s.spin, rh, rin: b.rIscoPro, rout: Math.max(s.rout, b.rIscoPro + 1), innerFade: s.innerFade, tempPeak: s.tempPeak,
          beaming: s.beaming, dopplerColor: s.dopplerColor, diskOpacity: s.diskOpacity, diskTurb: s.diskTurb, tempSlope: s.tempSlope, haze: s.haze,
          skyBrightness: s.skyBrightness, skyTilt: s.skyTilt, skyYaw: s.skyYaw, maxSteps: q.maxSteps, stepScale: q.stepScale,
          exposure: s.exposure, bloom: s.bloom, resScale: q.resScale, fov: s.fov,
        };
        // tempo simulato in unità GM/c^3: dt reale × warp / t_M
        renderer.render(params, cam.state(), dt, s.timeWarp / b.tM, moving);
        fpsAcc += dt; fpsN++; hudT += dt;
        if (hudT > 0.2) {
          const p = cam.pos;
          const r = Math.hypot(p[0], p[1], p[2]);
          const th = Math.acos(Math.max(-1, Math.min(1, p[2] / r)));
          const m = kerrMetric(r, th, s.spin);
          setHud({
            fps: fpsN / fpsAcc, r, alpha: m.alpha, omega: m.omega / b.tM, vEsc: Math.sqrt(2 / r), tidal: (2 * G * b.M * 2) / Math.pow(r * b.rg, 3),
            accum: renderer.accumulated, theta: (th * 180) / Math.PI, vOrb: r > b.rIscoPro ? zamoOrbitalSpeed(r, s.spin) : NaN, speed: cam.speedMul,
          });
          fpsAcc = 0; fpsN = 0; hudT = 0;
        }
        raf = requestAnimationFrame(loop);
      };
      raf = requestAnimationFrame(loop);
    };
    img.onerror = () => { setErrMsg('Impossibile caricare la texture del cielo'); setStatus('error'); };
    img.src = skyUrl;
    const onLock = () => setLocked(document.pointerLockElement === canvas);
    document.addEventListener('pointerlockchange', onLock);
    const onKey = (e: KeyboardEvent) => {
      if (e.code === 'KeyH') setShowPanel((v) => !v);
      if (e.code === 'KeyP') setShowPhysics((v) => !v);
      if (e.code === 'KeyR' && cameraRef.current) { cameraRef.current.pos = [0, -26, 1.9]; cameraRef.current.lookAtOrigin(); }
      if (e.code === 'KeyF' && cameraRef.current) cameraRef.current.lookAtOrigin();
    };
    window.addEventListener('keydown', onKey);
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      document.removeEventListener('pointerlockchange', onLock);
      window.removeEventListener('keydown', onKey);
      cam?.dispose();
      renderer?.dispose();
      rendererRef.current = null;
      cameraRef.current = null;
    };
  }, [lut]);

  const requestLock = () => { canvasRef.current?.requestPointerLock?.(); };

  const applyPreset = (name: string) => setSettings((s) => ({ ...s, ...PRESETS[name] }));

  const Tphys = diskPeakTemperature(bh.rIscoPro, bh.M, settings.mdotEdd * bh.Mdot_Edd);
  const tauRatio = 1 / Math.max(hud.alpha, 1e-6);

  return (
    <div ref={wrapRef} className="fixed inset-0 bg-black text-zinc-200 select-none overflow-hidden font-sans">
      <canvas ref={canvasRef} onClick={requestLock} className="absolute inset-0 w-full h-full cursor-crosshair" />

      {/* crosshair */}
      {locked && <div className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-1 h-1 rounded-full bg-white/50" />}

      {status === 'loading' && (
        <div className="absolute inset-0 grid place-items-center bg-black">
          <div className="text-center space-y-3">
            <div className="text-2xl tracking-[0.5em] text-amber-200 font-light">GARGANTUA</div>
            <div className="text-xs text-zinc-500 tracking-widest">integrazione delle geodetiche nulle di Kerr…</div>
            <div className="w-48 h-px bg-zinc-800 mx-auto overflow-hidden"><div className="h-full w-1/3 bg-amber-300 animate-[slide_1.2s_linear_infinite]" /></div>
          </div>
        </div>
      )}
      {status === 'error' && (
        <div className="absolute inset-0 grid place-items-center bg-black">
          <div className="max-w-md text-center space-y-2">
            <div className="text-red-400">Errore di inizializzazione</div>
            <div className="text-xs text-zinc-400 font-mono">{errMsg}</div>
            <div className="text-xs text-zinc-500">Serve un browser con WebGL2 e render target in virgola mobile (Chrome/Edge/Firefox desktop).</div>
          </div>
        </div>
      )}

      {/* Titolo + comandi */}
      <div className={`absolute top-4 left-4 space-y-1 pointer-events-none ${bare ? 'hidden' : ''}`}>
        <h1 className="text-lg tracking-[0.45em] text-amber-100 font-light drop-shadow">GARGANTUA</h1>
        <p className="text-[10px] tracking-[0.2em] text-zinc-400 uppercase">Ray tracer geodetico di Kerr · M = {sci(settings.massSolar, 2)} M☉ · a/M = {settings.spin.toFixed(3)}</p>
        {!locked && status === 'ready' && (
          <p className="text-[11px] text-zinc-300/90 mt-2 bg-black/40 backdrop-blur px-2 py-1 rounded inline-block pointer-events-auto">
            Clicca sulla scena per prendere il controllo · <span className="text-amber-200">WASD</span> muovi · <span className="text-amber-200">SPAZIO</span> sali · <span className="text-amber-200">CTRL/C</span> scendi · <span className="text-amber-200">SHIFT</span> turbo · <span className="text-amber-200">rotella</span> velocità · <span className="text-amber-200">F</span> guarda il buco nero · <span className="text-amber-200">R</span> reset · <span className="text-amber-200">H</span> pannello · <span className="text-amber-200">P</span> fisica
          </p>
        )}
      </div>

      {/* HUD navigazione */}
      {status === 'ready' && !bare && (
        <div className="absolute bottom-4 left-4 bg-black/45 backdrop-blur-md border border-white/10 rounded-lg px-3 py-2 w-[290px] text-[11px] pointer-events-none">
          <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono">
            <span className="text-zinc-500">r</span><span>{hud.r.toFixed(2)} r_g = {fmtLen(hud.r * bh.rg)} = {(hud.r / 2).toFixed(2)} r_s</span>
            <span className="text-zinc-500">θ</span><span>{hud.theta.toFixed(1)}° {hud.r < bh.rErgoEq / bh.rg && Math.abs(hud.theta - 90) < 30 ? <span className="text-amber-300">· ERGOSFERA</span> : ''}</span>
            <span className="text-zinc-500">dτ/dt</span><span>{hud.alpha.toFixed(5)} <span className="text-zinc-500">(1 h qui = {tauRatio < 1e4 ? (tauRatio).toFixed(3) + ' h' : fmtTime(3600 * tauRatio)} lontano)</span></span>
            <span className="text-zinc-500">ω_LT</span><span>{sci(hud.omega, 3)} rad/s <span className="text-zinc-500">(frame dragging)</span></span>
            <span className="text-zinc-500">v_fuga</span><span>{(hud.vEsc).toFixed(3)} c</span>
            <span className="text-zinc-500">v_orb</span><span>{isNaN(hud.vOrb) ? 'nessuna orbita stabile' : hud.vOrb.toFixed(3) + ' c'}</span>
            <span className="text-zinc-500">marea</span><span>{sci(hud.tidal, 2)} m/s² su 2 m</span>
            <span className="text-zinc-500">fps</span><span>{hud.fps.toFixed(0)} · campioni/pixel {hud.accum} · vel ×{hud.speed.toFixed(2)}</span>
          </div>
        </div>
      )}

      {/* Pannello controlli */}
      {status === 'ready' && showPanel && (
        <div className="absolute top-4 right-4 bottom-4 w-[300px] overflow-y-auto bg-black/55 backdrop-blur-md border border-white/10 rounded-lg p-4 space-y-5 text-zinc-200 [scrollbar-width:thin]">
          <Section title="Preset">
            <div className="grid grid-cols-2 gap-1.5">
              {Object.keys(PRESETS).map((n) => (
                <button key={n} onClick={() => applyPreset(n)} className="text-[10px] px-2 py-1.5 rounded border border-white/10 bg-white/5 hover:bg-amber-400/20 hover:border-amber-300/40 transition text-left leading-tight">{n}</button>
              ))}
            </div>
          </Section>

          <Section title="Buco nero (Kerr)">
            <Slider label="Massa M (M☉)" value={settings.massSolar} min={10} max={1e10} log onChange={(v) => set('massSolar', v)} fmt={(v) => sci(v, 2)} />
            <Slider label="Spin a/M" value={settings.spin} min={0} max={0.998} step={0.001} onChange={(v) => set('spin', v)} fmt={(v) => v.toFixed(3)} />
            <div className="text-[10px] text-zinc-500 leading-4">
              r₊ = {(bh.rPlus / bh.rg).toFixed(3)} r_g · ISCO = {bh.rIscoPro.toFixed(3)} r_g · orbita fotonica = {bh.rPhPro.toFixed(3)} r_g · η_NT = {(bh.eta_NT * 100).toFixed(1)}%
            </div>
          </Section>

          <Section title="Disco di accrescimento">
            <Slider label="Raggio esterno (r_g)" value={settings.rout} min={6} max={60} step={0.5} onChange={(v) => set('rout', v)} fmt={(v) => v.toFixed(1)} />
            <Slider label="T di picco (K) — Planck" value={settings.tempPeak} min={2000} max={40000} log onChange={(v) => set('tempPeak', v)} fmt={(v) => v.toFixed(0) + ' K'} />
            <div className="flex items-center gap-2 text-[10px] text-zinc-400">
              <span className="inline-block w-4 h-4 rounded-full border border-white/20" style={{ background: blackbodyCSS(settings.tempPeak, lut) }} />
              <span>Novikov–Thorne con Ṁ = {settings.mdotEdd.toFixed(2)} Ṁ_Edd → T_max = {sci(Tphys, 3)} K</span>
              <button className="ml-auto px-1.5 py-0.5 rounded bg-white/10 hover:bg-amber-400/20" onClick={() => set('tempPeak', Math.min(40000, Math.max(2000, Tphys)))}>usa</button>
            </div>
            <Slider label="Ṁ / Ṁ_Eddington" value={settings.mdotEdd} min={0.001} max={1} log onChange={(v) => set('mdotEdd', v)} fmt={(v) => v.toFixed(3)} />
            <Slider label="r_in profilo T (× ISCO)" value={settings.innerFade} min={0.3} max={0.999} step={0.001} onChange={(v) => set('innerFade', v)} fmt={(v) => v.toFixed(2)} />
            <Slider label="Pendenza radiale T (1 = NT esatto r^-3/4)" value={settings.tempSlope} min={0.2} max={1.2} step={0.01} onChange={(v) => set('tempSlope', v)} />
            <Slider label="Alone volumetrico (H/r ≈ 0.05)" value={settings.haze} min={0} max={1.5} step={0.01} onChange={(v) => set('haze', v)} />
            <Slider label="Profondità ottica τ₀" value={settings.diskOpacity} min={0.2} max={8} step={0.05} onChange={(v) => set('diskOpacity', v)} />
            <Slider label="Turbolenza MHD (contrasto)" value={settings.diskTurb} min={0} max={1} step={0.01} onChange={(v) => set('diskTurb', v)} />
            <Slider label="Beaming relativistico (g⁴)" value={settings.beaming} min={0} max={1} step={0.01} onChange={(v) => set('beaming', v)} fmt={(v) => (v === 1 ? 'fisico' : v === 0 ? 'off (film)' : v.toFixed(2))} />
            <Slider label="Shift Doppler+gravitazionale (colore)" value={settings.dopplerColor} min={0} max={1} step={0.01} onChange={(v) => set('dopplerColor', v)} fmt={(v) => (v === 1 ? 'fisico' : v === 0 ? 'off (film)' : v.toFixed(2))} />
            <Slider label="Dilatazione temporale (warp)" value={settings.timeWarp} min={0} max={3000} step={1} onChange={(v) => set('timeWarp', v)} fmt={(v) => (v === 0 ? 'fermo' : '×' + v.toFixed(0))} />
          </Section>

          <Section title="Cielo (Via Lattea, ESO/S. Brunier)">
            <Slider label="Luminosità stelle" value={settings.skyBrightness} min={0} max={2} step={0.01} onChange={(v) => set('skyBrightness', v)} />
            <Slider label="Inclinazione piano galattico" value={settings.skyTilt} min={0} max={Math.PI} step={0.01} onChange={(v) => set('skyTilt', v)} fmt={(v) => ((v * 180) / Math.PI).toFixed(0) + '°'} />
            <Slider label="Rotazione" value={settings.skyYaw} min={0} max={2 * Math.PI} step={0.01} onChange={(v) => set('skyYaw', v)} fmt={(v) => ((v * 180) / Math.PI).toFixed(0) + '°'} />
          </Section>

          <Section title="Camera & rendering">
            <Slider label="Campo visivo" value={settings.fov} min={20} max={120} step={1} onChange={(v) => set('fov', v)} fmt={(v) => v.toFixed(0) + '°'} />
            <Slider label="Esposizione" value={settings.exposure} min={0.05} max={8} log onChange={(v) => set('exposure', v)} />
            <Slider label="Bloom / glare" value={settings.bloom} min={0} max={0.6} step={0.01} onChange={(v) => set('bloom', v)} />
            <div className="text-[11px] text-zinc-400 mb-1">Qualità (risoluzione · passi RK4)</div>
            <div className="grid grid-cols-4 gap-1">
              {(Object.keys(QUALITY) as Quality[]).map((q) => (
                <button key={q} onClick={() => set('quality', q)} className={`text-[10px] py-1 rounded border transition ${settings.quality === q ? 'bg-amber-400/25 border-amber-300/50 text-amber-100' : 'border-white/10 bg-white/5 hover:bg-white/10'}`}>{q}</button>
              ))}
            </div>
            <div className="text-[10px] text-zinc-500">{QUALITY[settings.quality].maxSteps} passi · {Math.round(QUALITY[settings.quality].resScale * 100)}% risoluzione · supersampling progressivo da fermo</div>
            <Toggle label="Pannello fisica (P)" value={showPhysics} onChange={setShowPhysics} />
          </Section>
          <p className="text-[9px] text-zinc-600 leading-4">Panorama Via Lattea: ESO / S. Brunier (CC BY 4.0). Metrica di Kerr in coordinate di Boyer–Lindquist; equazioni di Hamilton integrate con RK4 adattivo; tetrade ZAMO; profilo Novikov–Thorne; emissione di Planck integrata sulle CMF CIE 1931; trasporto radiativo I = B(T)(1−e^−τ); invarianza di Liouville I_ν/ν³.</p>
        </div>
      )}

      {/* Pannello fisica */}
      {status === 'ready' && showPhysics && (
        <div className={`absolute bottom-4 ${showPanel ? 'right-[316px]' : 'right-4'} w-[340px] max-h-[70vh] overflow-y-auto bg-black/60 backdrop-blur-md border border-white/10 rounded-lg p-4 space-y-4 [scrollbar-width:thin]`}>
          <div className="flex items-center justify-between">
            <h2 className="text-[10px] uppercase tracking-[0.25em] text-amber-300">Fisica del buco nero</h2>
            <button className="text-zinc-500 hover:text-white text-xs" onClick={() => setShowPhysics(false)}>✕</button>
          </div>
          <Section title="Geometria (no-hair: M, J, Q=0)">
            <Row k="r_g = GM/c²" v={fmtLen(bh.rg)} />
            <Row k="r_s = 2GM/c²" v={fmtLen(bh.rs)} />
            <Row k="Orizzonte r₊" v={`${(bh.rPlus / bh.rg).toFixed(4)} r_g`} title="r± = GM/c² ± √((GM/c²)² − a²)" />
            <Row k="Orizzonte di Cauchy r₋" v={`${(bh.rMinus / bh.rg).toFixed(4)} r_g`} />
            <Row k="Ergosfera (equatore)" v={`${(bh.rErgoEq / bh.rg).toFixed(2)} r_g`} title="r_ergo(θ) = GM/c² (1+√(1−a²cos²θ))" />
            <Row k="Orbita fotonica (pro/retro)" v={`${bh.rPhPro.toFixed(3)} / ${bh.rPhRetro.toFixed(3)} r_g`} />
            <Row k="ISCO (pro/retro)" v={`${bh.rIscoPro.toFixed(3)} / ${bh.rIscoRetro.toFixed(3)} r_g`} title="Bardeen–Press–Teukolsky 1972" />
            <Row k="Ombra (Schwarzschild)" v={`b = 3√3 r_g = ${fmtLen(bh.bShadow * bh.rg)}`} />
            <Row k="J = a M c" v={`${sci(bh.J)} kg m²/s`} />
            <Row k="Ω_H orizzonte" v={`${sci(bh.omegaH)} rad/s`} title="Ω_H = a c/(r₊²+a²)" />
            <Row k="Area A = 4π(r₊²+a²)" v={`${sci(bh.area)} m²`} />
            <Row k="Unità di tempo GM/c³" v={fmtTime(bh.tM)} />
          </Section>
          <Section title="Termodinamica (Bardeen–Carter–Hawking)">
            <Row k="κ gravità di superficie" v={`${sci(bh.kappa)} m/s²`} title="κ = c²(r₊−r₋)/(2(r₊²+a²)) – legge zero: costante sull'orizzonte" />
            <Row k="T_Hawking = ħκ/2πck_B" v={`${sci(bh.T_H)} K`} />
            <Row k="S_BH = k_B c³A/4ħG" v={`${sci(bh.S_BH)} J/K = ${sci(bh.S_BH_kB)} k_B`} />
            <Row k="M_irr" v={`${sci(bh.M_irr / bh.M * 100, 3)}% di M`} title="M_irr = √(A c⁴/16πG²)" />
            <Row k="Energia estraibile (Penrose)" v={`${(bh.extractable * 100).toFixed(2)}% Mc²`} />
            <Row k="Potenza Hawking" v={`${sci(bh.P_H)} W`} title="P = ħc⁶/(15360πG²M²)" />
            <Row k="Tempo di evaporazione" v={fmtTime(bh.t_evap)} title="t = 5120πG²M³/(ħc⁴)" />
            <Row k="Seconda legge" v="δA ≥ 0" />
          </Section>
          <Section title="Disco (Novikov–Thorne / Shakura–Sunyaev)">
            <Row k="η_NT = 1 − E_ISCO" v={`${(bh.eta_NT * 100).toFixed(2)}%`} />
            <Row k="L_Eddington" v={`${sci(bh.L_Edd)} W`} title="L_Edd = 4πGMm_p c/σ_T" />
            <Row k="Ṁ_Edd = L_Edd/ηc²" v={`${sci(bh.Mdot_Edd / 1.98847e30 * 3.15576e7, 3)} M☉/anno`} />
            <Row k="L = η Ṁ c²" v={`${sci(bh.eta_NT * settings.mdotEdd * bh.Mdot_Edd * C * C)} W`} />
            <Row k="T_max NT (fisica)" v={`${sci(Tphys, 3)} K`} title="F(r) = 3GMṀ/(8πr³)(1−√(r_in/r)), σT⁴ = F" />
            <Row k="λ_max Wien (T picco)" v={`${(2.897771955e-3 / settings.tempPeak * 1e9).toFixed(0)} nm`} />
            <Row k="v_orb @ ISCO (ZAMO)" v={`${bh.vIsco.toFixed(3)} c`} />
            <Row k="Ω_K @ ISCO" v={`${sci(keplerOmega(bh.rIscoPro, settings.spin) / bh.tM)} rad/s · P = ${fmtTime(bh.T_isco_period)}`} title="Ω_K = 1/(r^{3/2}+a)" />
            <Row k="k_B T_picco" v={`${sci(KB * settings.tempPeak / 1.602e-19, 3)} eV`} />
          </Section>
          <Section title="Ringdown (modi quasi-normali l=m=2)">
            <Row k="f_QNM" v={`${sci(bh.f_QNM)} Hz`} title="Echeverria 1989: f ≈ [1−0.63(1−a)^0.3] c³/(2πGM)" />
            <Row k="Q · τ" v={`${bh.Q_QNM.toFixed(2)} · ${fmtTime(bh.tau_QNM)}`} />
          </Section>
          <Section title="Osservatore (qui)">
            <Row k="Lapse α = dτ/dt (ZAMO)" v={hud.alpha.toFixed(6)} title="α = √(ΔΣ/A)" />
            <Row k="Redshift 1+z verso ∞" v={(1 / hud.alpha).toFixed(4)} />
            <Row k="ω frame dragging" v={`${sci(hud.omega)} rad/s`} title="ω = −g_tφ/g_φφ = 2ar/A" />
            <Row k="Distanza" v={`${(hud.r * bh.rg / AU).toFixed(3)} UA`} />
            <Row k="Ritardo Shapiro (∅)" v={`${sci(2 * G * bh.M / (C ** 3) * Math.log(4 * hud.r * hud.r), 3)} s`} title="Δt ≈ 2GM/c³ ln(4 r_E r_R / b²), r_R = r_E" />
          </Section>
        </div>
      )}

      {!showPanel && status === 'ready' && !bare && (
        <button onClick={() => setShowPanel(true)} className="absolute top-4 right-4 text-[10px] tracking-widest px-2 py-1 rounded bg-black/50 border border-white/10 hover:bg-white/10">PANNELLO (H)</button>
      )}
    </div>
  );
}
