export const VERT = `#version 300 es
precision highp float;
out vec2 vUv;
void main(){
  // fullscreen triangle
  vec2 p = vec2((gl_VertexID<<1)&2, gl_VertexID&2);
  vUv = p;
  gl_Position = vec4(p*2.0-1.0, 0.0, 1.0);
}`;

/**
 * RAY TRACER GEODETICO DI KERR (Boyer–Lindquist, G=c=M=1)
 * ---------------------------------------------------------
 *  H = (1/2Σ)[ Δ p_r² + p_θ² − P²/Δ + W²/sin²θ ],  P=(r²+a²)p_t + a p_φ,  W=p_φ + a sin²θ p_t
 *  dx^μ/dλ = ∂H/∂p_μ ,  dp_μ/dλ = −∂H/∂x^μ   (p_t, p_φ conservati: E, L_z)
 *  Integrazione RK4 con passo adattivo; raggi tracciati all'indietro dalla camera (tetrade ZAMO).
 *  Disco: profilo Novikov–Thorne, emissione di Planck (LUT CIE), fattore g = E_oss/E_em,
 *  trasporto radiativo I = B(T)(1−e^{−τ}) con accumulo di trasmittanza (immagini multiple).
 */
export const TRACE_FRAG = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

in vec2 vUv;
out vec4 fragColor;

uniform vec2  uResolution;
uniform vec3  uCamPos;
uniform mat3  uCamBasis;   // colonne: right, up, forward
uniform float uTanHalfFov;
uniform float uSpin;
uniform float uRh;         // r_+ orizzonte
uniform float uTime;       // in unità GM/c^3
uniform float uRin;
uniform float uRout;
uniform float uInnerFade;  // r_in del profilo di temperatura (frazione di uRin)
uniform float uTempPeak;   // K
uniform float uBeaming;    // 1 = fisico (g^4), 0 = disattivato
uniform float uDopplerColor; // 1 = fisico, 0 = solo intensità
uniform float uDiskOpacity;
uniform float uDiskTurb;
uniform float uTempSlope;  // 1 = profilo NT esatto
uniform float uHaze;       // intensità alone volumetrico
uniform float uSkyBrightness;
uniform float uSkyTilt;
uniform float uSkyYaw;
uniform vec2  uJitter;
uniform int   uMaxSteps;
uniform float uStepScale;
uniform float uFrame;      // numero frame accumulati
uniform sampler2D uSky;
uniform sampler2D uBB;
uniform sampler2D uPrev;

const float PI = 3.14159265358979;
const float BB_LTMIN = 6.684611727667927;  // ln(800)
const float BB_LTMAX = 11.002099841204238; // ln(60000)

float a;   // spin
float pt, pph; // costanti del moto (E, L_z) del raggio "inverso"

// ---------- rumore ----------
vec3 hash3(vec3 p){
  p = vec3(dot(p,vec3(127.1,311.7, 74.7)), dot(p,vec3(269.5,183.3,246.1)), dot(p,vec3(113.5,271.9,124.6)));
  return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}
float gnoise(vec3 p){
  vec3 i = floor(p); vec3 f = fract(p);
  vec3 u = f*f*(3.0-2.0*f);
  return mix(mix(mix(dot(hash3(i+vec3(0,0,0)),f-vec3(0,0,0)), dot(hash3(i+vec3(1,0,0)),f-vec3(1,0,0)),u.x),
                 mix(dot(hash3(i+vec3(0,1,0)),f-vec3(0,1,0)), dot(hash3(i+vec3(1,1,0)),f-vec3(1,1,0)),u.x),u.y),
             mix(mix(dot(hash3(i+vec3(0,0,1)),f-vec3(0,0,1)), dot(hash3(i+vec3(1,0,1)),f-vec3(1,0,1)),u.x),
                 mix(dot(hash3(i+vec3(0,1,1)),f-vec3(0,1,1)), dot(hash3(i+vec3(1,1,1)),f-vec3(1,1,1)),u.x),u.y),u.z);
}
float fbm(vec3 p, int oct){
  float v = 0.0, amp = 0.5;
  mat3 m = mat3(0.8,0.6,0.0, -0.6,0.8,0.0, 0.0,0.0,1.0);
  for(int i=0;i<6;i++){
    if(i>=oct) break;
    v += amp*gnoise(p);
    p = m*p*2.03 + vec3(1.7,9.2,3.1);
    amp *= 0.5;
  }
  return v;
}

// ---------- corpo nero ----------
vec3 blackbody(float T){
  float f = (log(max(T,1.0)) - BB_LTMIN)/(BB_LTMAX - BB_LTMIN);
  vec4 s = texture(uBB, vec2(clamp(f,0.0,1.0), 0.5));
  // sotto 800 K: estrapola spegnendo
  if(f < 0.0) s.rgb *= exp(f*12.0);
  return s.rgb;
}

// ---------- metrica ----------
struct Metric { float Sigma, Delta, A, gtt, gtp, gpp, omega, alpha; };
Metric kerr(float r, float th){
  Metric m;
  float s = sin(th), c = cos(th);
  float r2=r*r, a2=a*a, s2=s*s;
  m.Sigma = r2 + a2*c*c;
  m.Delta = r2 - 2.0*r + a2;
  m.A = (r2+a2)*(r2+a2) - a2*m.Delta*s2;
  m.gtt = -(1.0 - 2.0*r/m.Sigma);
  m.gtp = -2.0*a*r*s2/m.Sigma;
  m.gpp = m.A*s2/m.Sigma;
  m.omega = 2.0*a*r/m.A;                 // frame dragging (Lense–Thirring)
  m.alpha = sqrt(max(m.Delta*m.Sigma/m.A, 1e-12)); // lapse
  return m;
}

// ---------- equazioni di Hamilton ----------
// stato: x = (r, θ, φ), p = (p_r, p_θ)
void rhs(in vec3 x, in vec2 p, out vec3 dx, out vec2 dp){
  float r = x.x, th = x.y;
  float s = sin(th), c = cos(th);
  float s2 = max(s*s, 1e-7);
  float r2 = r*r, a2 = a*a;
  float Sig = r2 + a2*c*c;
  float Del = r2 - 2.0*r + a2;
  float P = (r2 + a2)*pt + a*pph;
  float W = pph + a*s2*pt;
  float pr = p.x, pth = p.y;
  float F = Del*pr*pr + pth*pth - P*P/Del + W*W/s2;
  float dDel = 2.0*r - 2.0;
  float dP = 2.0*r*pt;
  float dF_r = dDel*pr*pr - (2.0*P*dP*Del - P*P*dDel)/(Del*Del);
  float dW = 2.0*a*s*c*pt;
  float dF_th = 2.0*W*dW/s2 - 2.0*W*W*c/(s2*s);
  float dSig_r = 2.0*r;
  float dSig_th = -2.0*a2*s*c;
  float inv2S = 0.5/Sig;
  dx.x = Del*pr/Sig;
  dx.y = pth/Sig;
  dx.z = (-a*P/Del + W/s2)/Sig;
  dp.x = -(dF_r*inv2S - F*dSig_r*inv2S/Sig);
  dp.y = -(dF_th*inv2S - F*dSig_th*inv2S/Sig);
}

void rk4(inout vec3 x, inout vec2 p, float h){
  vec3 k1x,k2x,k3x,k4x; vec2 k1p,k2p,k3p,k4p;
  rhs(x, p, k1x, k1p);
  rhs(x + 0.5*h*k1x, p + 0.5*h*k1p, k2x, k2p);
  rhs(x + 0.5*h*k2x, p + 0.5*h*k2p, k3x, k3p);
  rhs(x + h*k3x, p + h*k3p, k4x, k4p);
  x += h*(k1x + 2.0*k2x + 2.0*k3x + k4x)/6.0;
  p += h*(k1p + 2.0*k2p + 2.0*k3p + k4p)/6.0;
}

// ---------- cielo (Via Lattea) ----------
vec3 skyColor(vec3 d, float lodBias){
  // orientamento del piano galattico
  float cy = cos(uSkyYaw), sy = sin(uSkyYaw);
  d = vec3(cy*d.x - sy*d.y, sy*d.x + cy*d.y, d.z);
  float ct = cos(uSkyTilt), st = sin(uSkyTilt);
  d = vec3(d.x, ct*d.y - st*d.z, st*d.y + ct*d.z);
  float lon = atan(d.y, d.x);
  float lat = asin(clamp(d.z, -1.0, 1.0));
  vec2 uv = vec2(0.5 - lon/(2.0*PI), 0.5 - lat/PI);
  // LOD dal footprint angolare del pixel (texel/pixel) + compressione da lensing
  float texPerPix = (4096.0/(2.0*PI)) * (2.0*uTanHalfFov/uResolution.y);
  float lod = log2(max(texPerPix, 1e-3)) + lodBias;
  return textureLod(uSky, uv, lod).rgb;
}

// ---------- disco ----------
float diskPattern(float r, float phi, float t){
  float Om = 1.0/(pow(r,1.5) + a);           // Ω_K di Kerr (Bardeen)
  float ph = phi - Om*t;
  float u = log(r);
  vec3 p1 = vec3(cos(ph), sin(ph), 0.0)*2.2 + vec3(0.0,0.0,u*7.0);
  float n1 = fbm(p1, 5);
  vec3 p2 = vec3(cos(ph), sin(ph), 0.0)*9.0 + vec3(0.0,0.0,u*26.0) + 4.0;
  float n2 = fbm(p2, 4);
  // filamenti fini allungati lungo φ (shear kepleriano)
  float streaks = fbm(vec3(cos(ph), sin(ph), 0.0)*1.2 + vec3(0.0,0.0,u*60.0) + 11.0, 3);
  float d = 0.55 + 0.9*n1 + 0.45*n2 + 0.35*streaks;
  return max(d, 0.0);
}
float diskDensity(float r, float phi){
  // flow-map temporale: due fasi sfalsate per limitare lo shear
  float T = 90.0;
  float w = fract(uTime/T);
  float t1 = w*T, t2 = fract(uTime/T + 0.5)*T;
  float k = abs(2.0*w - 1.0);
  float d1 = diskPattern(r, phi, t1);
  float d2 = diskPattern(r, phi + 2.3, t2);
  return mix(d1, d2, k);
}

// T_eff Novikov–Thorne: σT⁴ = 3GMṀ/(8πr³)(1−√(r_in/r)); uTempSlope=1 → profilo esatto
float diskTemp(float r){
  float rin_T = uRin*uInnerFade;
  float f = pow(r,-3.0)*(1.0 - sqrt(rin_T/r));
  float fmax = 0.056652/(rin_T*rin_T*rin_T);
  float x = max(f/fmax, 0.0);
  return uTempPeak*pow(x, 0.25*uTempSlope);
}
float diskRadial(float r){
  float radial = smoothstep(uRout, uRout*0.55, r) * smoothstep(uRin, uRin*1.06, r);
  return radial*pow(uRin/r, 0.6);
}
// fattore g per un emettitore su orbita circolare kepleriana a raggio r
float diskG(float r){
  Metric me = kerr(r, 0.5*PI);
  float Om = 1.0/(pow(r,1.5) + a);
  float ut = 1.0/sqrt(max(-(me.gtt + 2.0*me.gtp*Om + me.gpp*Om*Om), 1e-6));
  float Eem = (pt + Om*pph)*ut;
  return 1.0/max(Eem, 1e-4);
}
vec3 shiftedPlanck(float Tem, float g){
  float Tobs = Tem*pow(g, uDopplerColor);
  return blackbody(Tobs)*pow(g, 4.0*(uBeaming - uDopplerColor));
}

vec3 diskEmission(float r, float phi, float pth, float g, out float alpha){
  float Tem = diskTemp(r);
  float dens = diskDensity(r, phi);
  float emis = mix(1.0, dens, uDiskTurb);
  float radial = diskRadial(r);
  // inclinazione locale del raggio rispetto al piano: spessore attraversato ∝ 1/|cos i|
  Metric m = kerr(r, 0.5*PI);
  float Eloc = (pt + m.omega*pph)/m.alpha;
  float cosi = clamp(abs(pth/sqrt(m.Sigma))/max(Eloc,1e-4), 0.04, 1.0);
  float tau = uDiskOpacity*radial*(0.35 + dens*1.2)/cosi;
  alpha = 1.0 - exp(-tau);
  return shiftedPlanck(Tem, g)*emis*alpha;
}

// alone volumetrico: atmosfera del disco con equilibrio idrostatico verticale ρ ∝ exp(−z²/2H²), H = c_s/Ω_K
vec3 diskHaze(float r, float z, float phi, float h){
  float Hs = 0.045*r + 0.02;
  float dens = exp(-z*z/(2.0*Hs*Hs));
  if(dens < 0.02) return vec3(0.0);
  float radial = diskRadial(r);
  if(radial < 1e-3) return vec3(0.0);
  // texture economica (2 ottave) co-rotante
  float Om = 1.0/(pow(r,1.5) + a);
  float ph = phi - Om*mod(uTime, 90.0);
  float tex = 0.75 + 0.7*fbm(vec3(cos(ph), sin(ph), 0.0)*2.2 + vec3(0.0,0.0,log(r)*7.0), 2);
  float g = diskG(r);
  float e = uHaze*dens*radial*tex*h/(Hs*2.5066);
  return shiftedPlanck(diskTemp(r), g)*e;
}

void main(){
  a = uSpin;
  vec2 frag = gl_FragCoord.xy + uJitter;
  vec2 ndc = (frag/uResolution)*2.0 - 1.0;
  float aspect = uResolution.x/uResolution.y;
  vec3 d = normalize(uCamBasis*vec3(ndc.x*aspect*uTanHalfFov, ndc.y*uTanHalfFov, 1.0));

  // camera in Boyer–Lindquist
  vec3 cp = uCamPos;
  float rc = length(cp);
  float thc = acos(clamp(cp.z/rc, -1.0, 1.0));
  float phc = atan(cp.y, cp.x);
  float sth = sin(thc), cth = cos(thc), sph = sin(phc), cph = cos(phc);
  vec3 er = vec3(sth*cph, sth*sph, cth);
  vec3 eth = vec3(cth*cph, cth*sph, -sth);
  vec3 eph = vec3(-sph, cph, 0.0);
  vec3 n = vec3(dot(d,er), dot(d,eth), dot(d,eph));

  // tetrade ZAMO -> 4-impulso (invertito) del fotone, E_loc = 1
  Metric mc = kerr(rc, thc);
  float sthc = max(abs(sth), 1e-4);
  float p_t_up = -1.0/mc.alpha;
  float p_ph_up = -mc.omega/mc.alpha + n.z*sqrt(mc.Sigma/mc.A)/sthc;
  float p_r_up = n.x*sqrt(mc.Delta/mc.Sigma);
  float p_th_up = n.y/sqrt(mc.Sigma);
  pt  = mc.gtt*p_t_up + mc.gtp*p_ph_up;
  pph = mc.gtp*p_t_up + mc.gpp*p_ph_up;
  vec3 x = vec3(rc, thc, phc);
  vec2 p = vec2(mc.Sigma/mc.Delta*p_r_up, mc.Sigma*p_th_up);

  vec3 col = vec3(0.0);
  float trans = 1.0;
  float rEsc = max(220.0, rc*1.6 + 40.0);
  float rStop = uRh*1.012 + 0.002;
  bool captured = false;
  bool escaped = false;
  float bend = 0.0;
  vec3 dirOut = d;

  for(int i=0; i<2000; i++){
    if(i >= uMaxSteps) break;
    vec3 x0 = x; vec2 p0 = p;
    float r = x.x;
    // passo adattivo: piccolo vicino all'orizzonte/sfera fotonica, grande lontano
    float h = uStepScale*r*(1.0 + r/28.0)*sqrt(max(1.0 - uRh/r, 0.004));
    // raffinamento vicino al piano del disco
    float zabs = abs(r*cos(x.y));
    if(r < uRout*1.2 && zabs < 0.6*h) h = max(h*0.35, 0.008);
    // vicino all'asse polare la barriera centrifuga L²/sin²θ è ripida: passo ridotto
    float sN = abs(sin(x.y));
    h *= clamp(sN*5.0 + 0.02, 0.02, 1.0);
    rk4(x, p, h);

    // attraversamento dei poli
    if(x.y < 0.0){ x.y = -x.y; p.y = -p.y; x.z += PI; }
    else if(x.y > PI){ x.y = 2.0*PI - x.y; p.y = -p.y; x.z += PI; }

    // attraversamento del piano equatoriale -> disco
    float c0 = cos(x0.y), c1 = cos(x.y);
    if(c0*c1 < 0.0){
      float f = c0/(c0 - c1);
      float rcr = mix(x0.x, x.x, f);
      if(rcr > uRin && rcr < uRout){
        float phcr = mix(x0.z, x.z, f);
        float pthcr = mix(p0.y, p.y, f);
        // emettitore su orbita circolare kepleriana: u^μ = u^t(1,0,0,Ω); g = E_oss/E_em (E_oss = 1)
        float g = diskG(rcr);
        float alpha;
        vec3 e = diskEmission(rcr, phcr, pthcr, g, alpha);
        col += trans*e;
        trans *= (1.0 - alpha);
        if(trans < 0.004) break;
      }
    }

    // alone volumetrico sopra/sotto il piano
    if(uHaze > 0.0){
      float rr = x.x; float cz = cos(x.y);
      float rcyl = rr*sqrt(max(1.0 - cz*cz, 0.0));
      if(rcyl > uRin && rcyl < uRout && abs(rr*cz) < 0.25*rcyl){
        col += trans*diskHaze(rcyl, rr*cz, x.z, h);
      }
    }

    if(x.x < rStop){ captured = true; break; }
    if(x.x > rEsc){ escaped = true; break; }
  }

  if(!captured && trans > 0.004){
    // direzione asintotica dal 4-impulso (componenti contravarianti)
    vec3 dx; vec2 dp; rhs(x, p, dx, dp);
    float r = x.x, th = x.y, ph = x.z;
    float s = sin(th), c = cos(th);
    vec3 er2 = vec3(s*cos(ph), s*sin(ph), c);
    vec3 eth2 = vec3(c*cos(ph), c*sin(ph), -s);
    vec3 eph2 = vec3(-sin(ph), cos(ph), 0.0);
    vec3 v = dx.x*er2 + r*dx.y*eth2 + r*s*dx.z*eph2;
    if(!escaped) v = er2; // fallback (troppi passi): direzione radiale
    dirOut = normalize(v);
    bend = acos(clamp(dot(dirOut, d), -1.0, 1.0));
    float gsky = 1.0/max(pt, 1e-3);           // blueshift del cielo per osservatore vicino
    float lodBias = clamp(bend/PI, 0.0, 2.5)*1.6;
    vec3 sky = skyColor(dirOut, lodBias)*uSkyBrightness*clamp(pow(gsky,4.0), 0.0, 40.0);
    col += trans*sky;
  }

  // accumulo temporale (supersampling progressivo)
  vec3 prev = texelFetch(uPrev, ivec2(gl_FragCoord.xy), 0).rgb;
  float w = 1.0/(uFrame + 1.0);
  vec3 outc = mix(prev, col, w);
  fragColor = vec4(outc, 1.0);
}`;

export const DOWNSAMPLE_FRAG = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 fragColor;
uniform sampler2D uTex; uniform vec2 uTexel; uniform float uFirst;
void main(){
  // 13-tap (Jimenez) downsample, robusto contro il fireflies del disco
  vec3 a = texture(uTex, vUv + uTexel*vec2(-2,-2)).rgb;
  vec3 b = texture(uTex, vUv + uTexel*vec2( 0,-2)).rgb;
  vec3 c = texture(uTex, vUv + uTexel*vec2( 2,-2)).rgb;
  vec3 d = texture(uTex, vUv + uTexel*vec2(-2, 0)).rgb;
  vec3 e = texture(uTex, vUv).rgb;
  vec3 f = texture(uTex, vUv + uTexel*vec2( 2, 0)).rgb;
  vec3 g = texture(uTex, vUv + uTexel*vec2(-2, 2)).rgb;
  vec3 h = texture(uTex, vUv + uTexel*vec2( 0, 2)).rgb;
  vec3 i = texture(uTex, vUv + uTexel*vec2( 2, 2)).rgb;
  vec3 j = texture(uTex, vUv + uTexel*vec2(-1,-1)).rgb;
  vec3 k = texture(uTex, vUv + uTexel*vec2( 1,-1)).rgb;
  vec3 l = texture(uTex, vUv + uTexel*vec2(-1, 1)).rgb;
  vec3 m = texture(uTex, vUv + uTexel*vec2( 1, 1)).rgb;
  vec3 col = e*0.125 + (a+c+g+i)*0.03125 + (b+d+f+h)*0.0625 + (j+k+l+m)*0.125;
  if(uFirst > 0.5){
    // soft-knee: il bloom parte solo dalle regioni luminose
    float lum = dot(col, vec3(0.2126,0.7152,0.0722));
    float knee = smoothstep(0.0, 1.5, lum);
    col *= knee;
  }
  fragColor = vec4(col, 1.0);
}`;

export const UPSAMPLE_FRAG = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 fragColor;
uniform sampler2D uTex; uniform sampler2D uPrevLevel; uniform vec2 uTexel; uniform float uRadius;
void main(){
  vec2 t = uTexel*uRadius;
  vec3 s = texture(uTex, vUv + vec2(-t.x,-t.y)).rgb*1.0 + texture(uTex, vUv + vec2(0,-t.y)).rgb*2.0 + texture(uTex, vUv + vec2(t.x,-t.y)).rgb*1.0
         + texture(uTex, vUv + vec2(-t.x,0)).rgb*2.0 + texture(uTex, vUv).rgb*4.0 + texture(uTex, vUv + vec2(t.x,0)).rgb*2.0
         + texture(uTex, vUv + vec2(-t.x,t.y)).rgb*1.0 + texture(uTex, vUv + vec2(0,t.y)).rgb*2.0 + texture(uTex, vUv + vec2(t.x,t.y)).rgb*1.0;
  s /= 16.0;
  fragColor = vec4(s + texture(uPrevLevel, vUv).rgb, 1.0);
}`;

export const COMPOSITE_FRAG = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 fragColor;
uniform sampler2D uScene; uniform sampler2D uBloom;
uniform float uExposure; uniform float uBloomStrength; uniform float uTime; uniform vec2 uResolution;

vec3 acesFitted(vec3 v){
  const mat3 IN = mat3(0.59719,0.07600,0.02840, 0.35458,0.90834,0.13383, 0.04823,0.01566,0.83777);
  const mat3 OUT = mat3(1.60475,-0.10208,-0.00327, -0.53108,1.10813,-0.07276, -0.07367,-0.00605,1.07602);
  v = IN*v;
  vec3 a = v*(v+0.0245786) - 0.000090537;
  vec3 b = v*(0.983729*v+0.4329510) + 0.238081;
  v = a/b;
  return clamp(OUT*v, 0.0, 1.0);
}
float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233)))*43758.5453); }

void main(){
  vec3 hdr = texture(uScene, vUv).rgb;
  vec3 bloom = texture(uBloom, vUv).rgb;
  // glare anisotropo leggero (diffrazione) + bloom gaussiano
  vec3 c = hdr + bloom*uBloomStrength;
  c *= uExposure;
  // vignettatura ottica lieve
  vec2 q = vUv*2.0-1.0;
  c *= 1.0 - 0.18*dot(q,q)*dot(q,q);
  vec3 mapped = acesFitted(c);
  mapped = pow(mapped, vec3(1.0/2.2));
  // dithering per evitare banding nel cielo scuro
  mapped += (hash(gl_FragCoord.xy + fract(uTime)) - 0.5)/255.0;
  fragColor = vec4(mapped, 1.0);
}`;
