/**
 * Fisica del buco nero di Kerr – tutte le grandezze in SI.
 * Le formule seguono Bardeen–Press–Teukolsky (1972), Novikov–Thorne (1973),
 * Bardeen–Carter–Hawking (1973), Hawking (1974), Echeverria (1989).
 */

// ---------- Costanti fisiche (CODATA 2018) ----------
export const C = 299792458; // m/s
export const G = 6.6743e-11; // m^3 kg^-1 s^-2
export const MSUN = 1.98847e30; // kg
export const KB = 1.380649e-23; // J/K
export const H = 6.62607015e-34; // J s
export const HBAR = H / (2 * Math.PI);
export const SIGMA_SB = 5.670374419e-8; // W m^-2 K^-4
export const M_E = 9.1093837015e-31; // kg
export const M_P = 1.67262192369e-27; // kg
export const E_CHARGE = 1.602176634e-19; // C
export const EPS0 = 8.8541878128e-12;
export const MU0 = 1.25663706212e-6;
export const ALPHA_FS = 7.2973525693e-3;
export const SIGMA_T = 6.6524587321e-29; // m^2 (Thomson)
export const L_PLANCK = Math.sqrt(HBAR * G / (C * C * C)); // m
export const WIEN_B = 2.897771955e-3; // m K
export const YEAR = 3.15576e7; // s
export const AU = 1.495978707e11; // m

export interface KerrParams {
  massSolar: number; // M / M_sun
  spin: number; // a* = a c / (G M)  in [0,1)
}

export interface KerrDerived {
  M: number; // kg
  rg: number; // GM/c^2 (m)
  rs: number; // 2GM/c^2 (m)
  tM: number; // GM/c^3 (s) – unità di tempo geometrica
  a: number; // a = J/(Mc) (m)
  J: number; // momento angolare (kg m^2/s)
  rPlus: number; // orizzonte esterno (m)
  rMinus: number; // orizzonte di Cauchy (m)
  rErgoEq: number; // ergosfera equatoriale (m)
  rIscoPro: number; // in unità di rg
  rIscoRetro: number;
  rPhPro: number; // orbita fotonica prograda (rg)
  rPhRetro: number;
  bShadow: number; // raggio apparente ombra (rg), Schwarzschild 3√3
  omegaH: number; // rad/s
  kappa: number; // gravità di superficie (m/s^2)
  T_H: number; // K
  area: number; // m^2
  S_BH: number; // J/K
  S_BH_kB: number; // in unità di k_B
  M_irr: number; // kg
  extractable: number; // frazione energia estraibile
  P_H: number; // W (potenza Hawking)
  t_evap: number; // s
  L_Edd: number; // W
  eta_NT: number; // efficienza Novikov–Thorne
  Mdot_Edd: number; // kg/s
  E_isco: number;
  vIsco: number; // velocità orbitale a ISCO (frazione di c)
  omegaIsco: number; // rad/s
  T_isco_period: number; // s
  f_QNM: number; // Hz  (modo quasi-normale l=m=2)
  Q_QNM: number;
  tau_QNM: number; // s
}

/** ISCO di Bardeen–Press–Teukolsky in unità di rg. sign=+1 progrado, -1 retrogrado */
export function isco(a: number, sign: 1 | -1 = 1): number {
  const z1 = 1 + Math.cbrt(1 - a * a) * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
  const z2 = Math.sqrt(3 * a * a + z1 * z1);
  return 3 + z2 - sign * Math.sqrt((3 - z1) * (3 + z1 + 2 * z2));
}

/** Raggio dell'orbita fotonica circolare equatoriale (rg) */
export function photonOrbit(a: number, sign: 1 | -1 = 1): number {
  return 2 * (1 + Math.cos((2 / 3) * Math.acos(-sign * a)));
}

/** Energia specifica di un'orbita circolare prograda in Kerr (unità c^2) */
export function circularEnergy(r: number, a: number): number {
  const r32 = Math.pow(r, 1.5);
  return (1 - 2 / r + a / r32) / Math.sqrt(1 - 3 / r + (2 * a) / r32);
}

/** Frequenza kepleriana relativistica Ω = 1/(r^{3/2}+a) (unità c^3/GM) */
export function keplerOmega(r: number, a: number): number {
  return 1 / (Math.pow(r, 1.5) + a);
}

export function derive(p: KerrParams): KerrDerived {
  const M = p.massSolar * MSUN;
  const a_ = Math.min(Math.max(p.spin, 0), 0.9999);
  const rg = (G * M) / (C * C);
  const rs = 2 * rg;
  const tM = (G * M) / (C * C * C);
  const a = a_ * rg;
  const J = a * M * C;
  const root = Math.sqrt(Math.max(1 - a_ * a_, 0));
  const rPlus = rg * (1 + root);
  const rMinus = rg * (1 - root);
  const rErgoEq = 2 * rg;
  const rIscoPro = isco(a_, 1);
  const rIscoRetro = isco(a_, -1);
  const rPhPro = photonOrbit(a_, 1);
  const rPhRetro = photonOrbit(a_, -1);
  const bShadow = 3 * Math.sqrt(3);
  const omegaH = (a * C) / (rPlus * rPlus + a * a); // rad/s
  const kappa = (C * C * (rPlus - rMinus)) / (2 * (rPlus * rPlus + a * a));
  const T_H = (HBAR * kappa) / (2 * Math.PI * C * KB);
  const area = 4 * Math.PI * (rPlus * rPlus + a * a);
  const S_BH = (KB * C * C * C * area) / (4 * HBAR * G);
  const S_BH_kB = S_BH / KB;
  const M_irr = Math.sqrt((area * C * C * C * C) / (16 * Math.PI * G * G));
  const extractable = 1 - M_irr / M;
  const P_H = (HBAR * Math.pow(C, 6)) / (15360 * Math.PI * G * G * M * M);
  const t_evap = (5120 * Math.PI * G * G * M * M * M) / (HBAR * Math.pow(C, 4));
  const L_Edd = (4 * Math.PI * G * M * M_P * C) / SIGMA_T;
  const E_isco = circularEnergy(rIscoPro, a_);
  const eta_NT = 1 - E_isco;
  const Mdot_Edd = L_Edd / (eta_NT * C * C);
  const omegaIscoGeom = keplerOmega(rIscoPro, a_);
  const omegaIsco = omegaIscoGeom / tM;
  // velocità misurata da un ZAMO all'ISCO
  const vIsco = zamoOrbitalSpeed(rIscoPro, a_);
  const T_isco_period = (2 * Math.PI) / omegaIsco;
  // Modo quasi-normale fondamentale l=m=2 (fit di Echeverria 1989)
  const f_QNM = ((1 - 0.63 * Math.pow(1 - a_, 0.3)) / (2 * Math.PI)) / tM;
  const Q_QNM = 2 * Math.pow(1 - a_, -0.45);
  const tau_QNM = Q_QNM / (Math.PI * f_QNM);
  return {
    M, rg, rs, tM, a, J, rPlus, rMinus, rErgoEq, rIscoPro, rIscoRetro, rPhPro, rPhRetro,
    bShadow, omegaH, kappa, T_H, area, S_BH, S_BH_kB, M_irr, extractable, P_H, t_evap,
    L_Edd, eta_NT, Mdot_Edd, E_isco, vIsco, omegaIsco, T_isco_period, f_QNM, Q_QNM, tau_QNM,
  };
}

/** Componenti metriche di Kerr (Boyer–Lindquist, G=c=M=1) */
export function kerrMetric(r: number, theta: number, a: number) {
  const s = Math.sin(theta), c = Math.cos(theta);
  const r2 = r * r, a2 = a * a;
  const Sigma = r2 + a2 * c * c;
  const Delta = r2 - 2 * r + a2;
  const A = (r2 + a2) * (r2 + a2) - a2 * Delta * s * s;
  const gtt = -(1 - (2 * r) / Sigma);
  const gtp = (-2 * a * r * s * s) / Sigma;
  const gpp = (A * s * s) / Sigma;
  const grr = Sigma / Delta;
  const gthth = Sigma;
  const omega = (2 * a * r) / A; // frame dragging
  const alpha = Math.sqrt(Math.max((Delta * Sigma) / A, 0)); // lapse
  return { Sigma, Delta, A, gtt, gtp, gpp, grr, gthth, omega, alpha };
}

/** Velocità orbitale kepleriana misurata da un ZAMO (frazione di c) */
export function zamoOrbitalSpeed(r: number, a: number): number {
  const m = kerrMetric(r, Math.PI / 2, a);
  const Om = keplerOmega(r, a);
  const v = (Math.sqrt(m.gpp) * (Om - m.omega)) / m.alpha;
  return Math.min(Math.abs(v), 0.999999);
}

/**
 * Temperatura efficace Novikov–Thorne (approssimazione newtoniana del flusso)
 * F(r) = 3GM Ṁ / (8π r^3) (1 - sqrt(r_in/r)),  σT^4 = F
 */
export function diskTemperature(rOverRg: number, rInOverRg: number, M: number, Mdot: number): number {
  const rg = (G * M) / (C * C);
  const r = rOverRg * rg;
  const rin = rInOverRg * rg;
  if (r <= rin) return 0;
  const F = ((3 * G * M * Mdot) / (8 * Math.PI * r * r * r)) * (1 - Math.sqrt(rin / r));
  return Math.pow(F / SIGMA_SB, 0.25);
}

export function diskPeakTemperature(rInOverRg: number, M: number, Mdot: number): number {
  // il massimo del profilo NT newtoniano è a r = (49/36) r_in
  return diskTemperature((49 / 36) * rInOverRg, rInOverRg, M, Mdot);
}

// ---------- formattazione ----------
export function sci(x: number, digits = 3): string {
  if (!isFinite(x)) return '∞';
  if (x === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(x)));
  if (e > -3 && e < 5) return x.toPrecision(digits + 1).replace(/\.?0+$/, '');
  const m = x / Math.pow(10, e);
  return `${m.toFixed(digits - 1)}×10^${e}`;
}

export function fmtTime(s: number): string {
  if (s < 60) return `${sci(s)} s`;
  if (s < 3600) return `${(s / 60).toFixed(2)} min`;
  if (s < 86400) return `${(s / 3600).toFixed(2)} h`;
  if (s < YEAR) return `${(s / 86400).toFixed(2)} giorni`;
  return `${sci(s / YEAR)} anni`;
}

export function fmtLen(m: number): string {
  if (m > 0.05 * AU) return `${(m / AU).toFixed(3)} UA`;
  if (m > 1e3) return `${sci(m / 1e3)} km`;
  return `${sci(m)} m`;
}
