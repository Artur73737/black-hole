/**
 * LUT corpo nero: B_λ(T) (legge di Planck) integrata sulle funzioni
 * colorimetriche CIE 1931 (fit multi-lobo di Wyman, Sloan & Shirley 2013),
 * poi XYZ → sRGB lineare. Il risultato è una RADIANZA (non normalizzata):
 * la luminanza cresce con T come prevede Planck, quindi il fattore g^4
 * del beaming emerge naturalmente campionando la LUT a T_obs = g·T_em.
 */
import { C, H, KB } from './blackhole';

export const BB_TMIN = 800; // K
export const BB_TMAX = 60000; // K
export const BB_SIZE = 1024;
export const BB_TREF = 6500; // luminanza unitaria a 6500 K

function g(x: number, mu: number, s1: number, s2: number) {
  const t = (x - mu) / (x < mu ? s1 : s2);
  return Math.exp(-0.5 * t * t);
}
function cieX(l: number) {
  return 1.056 * g(l, 599.8, 37.9, 31.0) + 0.362 * g(l, 442.0, 16.0, 26.7) - 0.065 * g(l, 501.1, 20.4, 26.2);
}
function cieY(l: number) {
  return 0.821 * g(l, 568.8, 46.9, 40.5) + 0.286 * g(l, 530.9, 16.3, 31.1);
}
function cieZ(l: number) {
  return 1.217 * g(l, 437.0, 11.8, 36.0) + 0.681 * g(l, 459.0, 26.0, 13.8);
}

/** Radianza spettrale di Planck B_λ(T) [W sr^-1 m^-3] */
export function planck(lambda: number, T: number): number {
  const x = (H * C) / (lambda * KB * T);
  if (x > 700) return 0;
  return (2 * H * C * C) / Math.pow(lambda, 5) / Math.expm1(x);
}

function xyzOf(T: number): [number, number, number] {
  let X = 0, Y = 0, Z = 0;
  for (let l = 380; l <= 780; l += 2) {
    const B = planck(l * 1e-9, T);
    X += B * cieX(l);
    Y += B * cieY(l);
    Z += B * cieZ(l);
  }
  return [X, Y, Z];
}

function xyzToLinearSRGB([X, Y, Z]: [number, number, number]): [number, number, number] {
  return [
    3.2406 * X - 1.5372 * Y - 0.4986 * Z,
    -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
    0.0557 * X - 0.204 * Y + 1.057 * Z,
  ];
}

export function buildBlackbodyLUT(): Float32Array {
  const data = new Float32Array(BB_SIZE * 4);
  const refY = xyzOf(BB_TREF)[1];
  const lmin = Math.log(BB_TMIN), lmax = Math.log(BB_TMAX);
  for (let i = 0; i < BB_SIZE; i++) {
    const T = Math.exp(lmin + ((lmax - lmin) * i) / (BB_SIZE - 1));
    const xyz = xyzOf(T);
    const rgb = xyzToLinearSRGB([xyz[0] / refY, xyz[1] / refY, xyz[2] / refY]);
    // gestione fuori-gamut: preserva la luminanza, desatura verso il bianco
    const Y = xyz[1] / refY;
    let [r, gg, b] = rgb;
    const mn = Math.min(r, gg, b);
    if (mn < 0) {
      const lum = 0.2126 * r + 0.7152 * gg + 0.0722 * b;
      const k = lum / (lum - mn);
      r = lum + (r - lum) * k;
      gg = lum + (gg - lum) * k;
      b = lum + (b - lum) * k;
    }
    data[i * 4] = Math.max(r, 0);
    data[i * 4 + 1] = Math.max(gg, 0);
    data[i * 4 + 2] = Math.max(b, 0);
    data[i * 4 + 3] = Y;
  }
  return data;
}

/** Colore (sRGB gamma, normalizzato) per anteprime UI */
export function blackbodyCSS(T: number, lut: Float32Array): string {
  const lmin = Math.log(BB_TMIN), lmax = Math.log(BB_TMAX);
  const f = Math.min(Math.max((Math.log(T) - lmin) / (lmax - lmin), 0), 1);
  const i = Math.round(f * (BB_SIZE - 1));
  let r = lut[i * 4], g = lut[i * 4 + 1], b = lut[i * 4 + 2];
  const m = Math.max(r, g, b, 1e-9);
  r /= m; g /= m; b /= m;
  const gam = (v: number) => Math.round(255 * Math.pow(v, 1 / 2.2));
  return `rgb(${gam(r)},${gam(g)},${gam(b)})`;
}
