import type { CameraState } from './renderer';

type V3 = [number, number, number];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3): V3 => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

/**
 * Camera libera. L'asse z è l'asse di rotazione del buco nero.
 * Le distanze sono in unità di r_g = GM/c².
 */
export class FreeCamera {
  pos: V3 = [0, -26, 1.9];
  yaw = Math.PI / 2; // guarda verso +y (il buco nero all'origine)
  pitch = -0.07;
  speedMul = 1;
  keys = new Set<string>();
  moving = false;
  private el: HTMLElement;
  private onMove = (e: MouseEvent) => {
    if (document.pointerLockElement !== this.el) return;
    this.yaw -= e.movementX * 0.0022;
    this.pitch -= e.movementY * 0.0022;
    this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch));
    this.moving = true;
  };
  private onKey = (e: KeyboardEvent) => {
    const tag = (e.target as HTMLElement)?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (e.type === 'keydown') this.keys.add(e.code); else this.keys.delete(e.code);
    if (['Space', 'KeyW', 'KeyA', 'KeyS', 'KeyD'].includes(e.code)) e.preventDefault();
  };
  private onWheel = (e: WheelEvent) => {
    this.speedMul *= Math.exp(-e.deltaY * 0.001);
    this.speedMul = Math.max(0.02, Math.min(50, this.speedMul));
  };
  private onBlur = () => this.keys.clear();

  constructor(el: HTMLElement) {
    this.el = el;
    window.addEventListener('mousemove', this.onMove);
    window.addEventListener('keydown', this.onKey);
    window.addEventListener('keyup', this.onKey);
    window.addEventListener('blur', this.onBlur);
    el.addEventListener('wheel', this.onWheel, { passive: true });
  }

  dispose() {
    window.removeEventListener('mousemove', this.onMove);
    window.removeEventListener('keydown', this.onKey);
    window.removeEventListener('keyup', this.onKey);
    window.removeEventListener('blur', this.onBlur);
    this.el.removeEventListener('wheel', this.onWheel);
  }

  basis() {
    const cp = Math.cos(this.pitch);
    const forward: V3 = [cp * Math.cos(this.yaw), cp * Math.sin(this.yaw), Math.sin(this.pitch)];
    const worldUp: V3 = [0, 0, 1];
    const right = norm(cross(forward, worldUp));
    const up = norm(cross(right, forward));
    return { forward, right, up };
  }

  /** aggiorna posizione; rHorizon in r_g, per non entrare nell'orizzonte */
  update(dt: number, rHorizon: number) {
    const { forward, right } = this.basis();
    const r = Math.hypot(...this.pos);
    // velocità proporzionale alla distanza dall'orizzonte: manovrabile sia vicino che lontano
    const base = Math.max(r - rHorizon, 0.05) * 0.55 * this.speedMul * (this.keys.has('ShiftLeft') || this.keys.has('ShiftRight') ? 3 : 1);
    let v: V3 = [0, 0, 0];
    const add = (d: V3, s: number) => { v = [v[0] + d[0] * s, v[1] + d[1] * s, v[2] + d[2] * s]; };
    if (this.keys.has('KeyW') || this.keys.has('ArrowUp')) add(forward, 1);
    if (this.keys.has('KeyS') || this.keys.has('ArrowDown')) add(forward, -1);
    if (this.keys.has('KeyD') || this.keys.has('ArrowRight')) add(right, 1);
    if (this.keys.has('KeyA') || this.keys.has('ArrowLeft')) add(right, -1);
    if (this.keys.has('Space')) add([0, 0, 1], 1);
    if (this.keys.has('ControlLeft') || this.keys.has('KeyC') || this.keys.has('KeyQ')) add([0, 0, 1], -1);
    const moved = v[0] !== 0 || v[1] !== 0 || v[2] !== 0;
    if (moved) {
      const vn = norm(v);
      this.pos = [this.pos[0] + vn[0] * base * dt, this.pos[1] + vn[1] * base * dt, this.pos[2] + vn[2] * base * dt];
      // vincolo: resta fuori dall'orizzonte
      const rn = Math.hypot(...this.pos);
      const rmin = rHorizon * 1.04 + 0.02;
      if (rn < rmin) this.pos = [this.pos[0] / rn * rmin, this.pos[1] / rn * rmin, this.pos[2] / rn * rmin];
      const rmax = 900;
      if (rn > rmax) this.pos = [this.pos[0] / rn * rmax, this.pos[1] / rn * rmax, this.pos[2] / rn * rmax];
    }
    const wasMoving = this.moving || moved;
    this.moving = false;
    return wasMoving;
  }

  state(): CameraState {
    const b = this.basis();
    return { pos: [...this.pos], right: b.right, up: b.up, forward: b.forward };
  }

  lookAtOrigin() {
    const d = norm([-this.pos[0], -this.pos[1], -this.pos[2]]);
    this.yaw = Math.atan2(d[1], d[0]);
    this.pitch = Math.asin(d[2]);
  }
}
