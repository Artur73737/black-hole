import { VERT, TRACE_FRAG, DOWNSAMPLE_FRAG, UPSAMPLE_FRAG, COMPOSITE_FRAG } from './shaders';
import { BB_SIZE } from '../physics/blackbody';

export interface RenderParams {
  spin: number;
  rh: number;
  rin: number;
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
  maxSteps: number;
  stepScale: number;
  exposure: number;
  bloom: number;
  resScale: number;
  fov: number; // gradi
}

export interface CameraState {
  pos: [number, number, number];
  right: [number, number, number];
  up: [number, number, number];
  forward: [number, number, number];
}

type Prog = { prog: WebGLProgram; u: Record<string, WebGLUniformLocation | null> };

const BLOOM_LEVELS = 6;

export class BlackHoleRenderer {
  private gl: WebGL2RenderingContext;
  private canvas: HTMLCanvasElement;
  private trace!: Prog;
  private down!: Prog;
  private upP!: Prog;
  private comp!: Prog;
  private vao!: WebGLVertexArrayObject;
  private skyTex!: WebGLTexture;
  private bbTex!: WebGLTexture;
  private accum: { tex: WebGLTexture; fb: WebGLFramebuffer }[] = [];
  private bloomTex: { tex: WebGLTexture; fb: WebGLFramebuffer; w: number; h: number }[] = [];
  private bloomUp: { tex: WebGLTexture; fb: WebGLFramebuffer; w: number; h: number }[] = [];
  private ping = 0;
  private rw = 0;
  private rh = 0;
  private frame = 0;
  private lastKey = '';
  private halton = 0;
  public simTime = 0;
  public accumulated = 0;

  constructor(canvas: HTMLCanvasElement, skyImage: HTMLImageElement, bbLUT: Float32Array) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, powerPreference: 'high-performance', preserveDrawingBuffer: false });
    if (!gl) throw new Error('WebGL2 non disponibile');
    this.gl = gl;
    if (!gl.getExtension('EXT_color_buffer_float') && !gl.getExtension('EXT_color_buffer_half_float')) {
      throw new Error('Float render targets non supportati');
    }
    gl.getExtension('OES_texture_float_linear');

    this.trace = this.program(VERT, TRACE_FRAG, [
      'uResolution', 'uCamPos', 'uCamBasis', 'uTanHalfFov', 'uSpin', 'uRh', 'uTime', 'uRin', 'uRout', 'uInnerFade', 'uTempPeak',
      'uBeaming', 'uDopplerColor', 'uDiskOpacity', 'uDiskTurb', 'uTempSlope', 'uHaze', 'uSkyBrightness', 'uSkyTilt', 'uSkyYaw', 'uJitter', 'uMaxSteps',
      'uStepScale', 'uFrame', 'uSky', 'uBB', 'uPrev',
    ]);
    this.down = this.program(VERT, DOWNSAMPLE_FRAG, ['uTex', 'uTexel', 'uFirst']);
    this.upP = this.program(VERT, UPSAMPLE_FRAG, ['uTex', 'uPrevLevel', 'uTexel', 'uRadius']);
    this.comp = this.program(VERT, COMPOSITE_FRAG, ['uScene', 'uBloom', 'uExposure', 'uBloomStrength', 'uTime', 'uResolution']);

    this.vao = gl.createVertexArray()!;

    // cielo: sRGB con mipmap (decodifica gamma corretta in hardware)
    this.skyTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.skyTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.SRGB8_ALPHA8, gl.RGBA, gl.UNSIGNED_BYTE, skyImage);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const aniso = gl.getExtension('EXT_texture_filter_anisotropic');
    if (aniso) gl.texParameterf(gl.TEXTURE_2D, aniso.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(8, gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT)));

    // LUT corpo nero
    this.bbTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.bbTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, BB_SIZE, 1, 0, gl.RGBA, gl.FLOAT, bbLUT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  private program(vs: string, fs: string, uniforms: string[]): Prog {
    const gl = this.gl;
    const compile = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(s);
        throw new Error('Shader error: ' + log);
      }
      return s;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('Link error: ' + gl.getProgramInfoLog(p));
    const u: Record<string, WebGLUniformLocation | null> = {};
    for (const n of uniforms) u[n] = gl.getUniformLocation(p, n);
    return { prog: p, u };
  }

  private makeTarget(w: number, h: number, linear: boolean) {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, linear ? gl.LINEAR : gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, linear ? gl.LINEAR : gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fb = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { tex, fb, w, h };
  }

  private ensureTargets(w: number, h: number) {
    if (w === this.rw && h === this.rh) return;
    const gl = this.gl;
    for (const t of [...this.accum, ...this.bloomTex, ...this.bloomUp]) {
      gl.deleteTexture(t.tex);
      gl.deleteFramebuffer(t.fb);
    }
    this.rw = w; this.rh = h;
    this.accum = [this.makeTarget(w, h, true), this.makeTarget(w, h, true)];
    this.bloomTex = []; this.bloomUp = [];
    let bw = w, bh = h;
    for (let i = 0; i < BLOOM_LEVELS; i++) {
      bw = Math.max(1, Math.floor(bw / 2)); bh = Math.max(1, Math.floor(bh / 2));
      this.bloomTex.push(this.makeTarget(bw, bh, true));
      this.bloomUp.push(this.makeTarget(bw, bh, true));
    }
    this.frame = 0;
  }

  resetAccumulation() { this.frame = 0; }

  private haltonNext(): [number, number] {
    const rad = (i: number, b: number) => { let f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); } return r; };
    this.halton = (this.halton + 1) % 1024;
    return [rad(this.halton, 2) - 0.5, rad(this.halton, 3) - 0.5];
  }

  render(p: RenderParams, cam: CameraState, dt: number, timeWarp: number, moving: boolean) {
    const gl = this.gl;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const ch = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== cw || this.canvas.height !== ch) { this.canvas.width = cw; this.canvas.height = ch; }
    const rw = Math.max(64, Math.floor(cw * p.resScale));
    const rh = Math.max(64, Math.floor(ch * p.resScale));
    this.ensureTargets(rw, rh);

    this.simTime += dt * timeWarp;
    const key = JSON.stringify([p.spin, p.rin, p.rout, p.innerFade, p.tempPeak, p.beaming, p.dopplerColor, p.diskOpacity, p.diskTurb, p.tempSlope, p.haze, p.skyBrightness, p.skyTilt, p.skyYaw, p.maxSteps, p.stepScale, p.fov, cam.pos, cam.forward, cam.up]);
    if (key !== this.lastKey || moving) this.frame = 0;
    this.lastKey = key;
    // con il disco in rotazione: media mobile corta (AA + motion blur); da fermo: accumulo progressivo
    const cap = timeWarp !== 0 ? 5 : 4096;
    const frameW = Math.min(this.frame, cap);

    const jitter = this.frame === 0 && moving ? [0, 0] : this.haltonNext();

    // ---- pass 1: ray tracing + accumulo ----
    const src = this.accum[this.ping];
    const dst = this.accum[1 - this.ping];
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fb);
    gl.viewport(0, 0, rw, rh);
    gl.useProgram(this.trace.prog);
    gl.bindVertexArray(this.vao);
    const u = this.trace.u;
    gl.uniform2f(u.uResolution, rw, rh);
    gl.uniform3fv(u.uCamPos, cam.pos);
    gl.uniformMatrix3fv(u.uCamBasis, false, new Float32Array([...cam.right, ...cam.up, ...cam.forward]));
    gl.uniform1f(u.uTanHalfFov, Math.tan((p.fov * Math.PI) / 360));
    gl.uniform1f(u.uSpin, p.spin);
    gl.uniform1f(u.uRh, p.rh);
    gl.uniform1f(u.uTime, this.simTime);
    gl.uniform1f(u.uRin, p.rin);
    gl.uniform1f(u.uRout, p.rout);
    gl.uniform1f(u.uInnerFade, p.innerFade);
    gl.uniform1f(u.uTempPeak, p.tempPeak);
    gl.uniform1f(u.uBeaming, p.beaming);
    gl.uniform1f(u.uDopplerColor, p.dopplerColor);
    gl.uniform1f(u.uDiskOpacity, p.diskOpacity);
    gl.uniform1f(u.uDiskTurb, p.diskTurb);
    gl.uniform1f(u.uTempSlope, p.tempSlope);
    gl.uniform1f(u.uHaze, p.haze);
    gl.uniform1f(u.uSkyBrightness, p.skyBrightness);
    gl.uniform1f(u.uSkyTilt, p.skyTilt);
    gl.uniform1f(u.uSkyYaw, p.skyYaw);
    gl.uniform2f(u.uJitter, jitter[0], jitter[1]);
    gl.uniform1i(u.uMaxSteps, p.maxSteps);
    gl.uniform1f(u.uStepScale, p.stepScale);
    gl.uniform1f(u.uFrame, frameW);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.skyTex); gl.uniform1i(u.uSky, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.bbTex); gl.uniform1i(u.uBB, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, src.tex); gl.uniform1i(u.uPrev, 2);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.ping = 1 - this.ping;
    this.frame++;
    this.accumulated = this.frame;

    // ---- pass 2: bloom downsample ----
    gl.useProgram(this.down.prog);
    let prevTex = dst.tex; let pw = rw, ph = rh;
    for (let i = 0; i < BLOOM_LEVELS; i++) {
      const t = this.bloomTex[i];
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb);
      gl.viewport(0, 0, t.w, t.h);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, prevTex);
      gl.uniform1i(this.down.u.uTex, 0);
      gl.uniform2f(this.down.u.uTexel, 1 / pw, 1 / ph);
      gl.uniform1f(this.down.u.uFirst, i === 0 ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      prevTex = t.tex; pw = t.w; ph = t.h;
    }
    // ---- pass 3: bloom upsample (additivo) ----
    gl.useProgram(this.upP.prog);
    let upPrev = this.bloomTex[BLOOM_LEVELS - 1];
    for (let i = BLOOM_LEVELS - 2; i >= 0; i--) {
      const dstL = this.bloomUp[i];
      gl.bindFramebuffer(gl.FRAMEBUFFER, dstL.fb);
      gl.viewport(0, 0, dstL.w, dstL.h);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, upPrev.tex);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.bloomTex[i].tex);
      gl.uniform1i(this.upP.u.uTex, 0);
      gl.uniform1i(this.upP.u.uPrevLevel, 1);
      gl.uniform2f(this.upP.u.uTexel, 1 / upPrev.w, 1 / upPrev.h);
      gl.uniform1f(this.upP.u.uRadius, 1.0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      upPrev = dstL;
    }

    // ---- pass 4: composite + tonemap ----
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, cw, ch);
    gl.useProgram(this.comp.prog);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, dst.tex);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, upPrev.tex);
    gl.uniform1i(this.comp.u.uScene, 0);
    gl.uniform1i(this.comp.u.uBloom, 1);
    gl.uniform1f(this.comp.u.uExposure, p.exposure);
    gl.uniform1f(this.comp.u.uBloomStrength, p.bloom);
    gl.uniform1f(this.comp.u.uTime, performance.now() / 1000);
    gl.uniform2f(this.comp.u.uResolution, cw, ch);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  dispose() {
    const gl = this.gl;
    for (const t of [...this.accum, ...this.bloomTex, ...this.bloomUp]) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fb); }
    gl.deleteTexture(this.skyTex); gl.deleteTexture(this.bbTex);
    gl.deleteProgram(this.trace.prog); gl.deleteProgram(this.down.prog); gl.deleteProgram(this.upP.prog); gl.deleteProgram(this.comp.prog);
  }
}
