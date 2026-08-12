"use client";

import { useEffect, useRef } from "react";

/**
 * Layered ridgelines at dusk — the hero backdrop.
 *
 * Drawn rather than photographed so it scales cleanly, carries no licensing,
 * and stays abstract enough to read as a firm rather than a travel brochure.
 * Each ridge is a summed sine, so the silhouette is smooth and never repeats
 * visibly across the width.
 */
const LAYERS = [
  { amp: 0.05, y: 0.47, freq: 1.1, phase: 0.0, tone: 0.06 },
  { amp: 0.07, y: 0.6, freq: 0.8, phase: 2.1, tone: 0.26 },
  { amp: 0.06, y: 0.72, freq: 1.4, phase: 4.4, tone: 0.5 },
  { amp: 0.05, y: 0.82, freq: 1.0, phase: 1.2, tone: 0.72 },
  { amp: 0.045, y: 0.95, freq: 1.7, phase: 3.3, tone: 1.0 },
];

export default function Ridgeline() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let w = 0;
    let h = 0;

    const mix = (a: number[], b: number[], t: number) =>
      `rgb(${Math.round(a[0] + (b[0] - a[0]) * t)},${Math.round(
        a[1] + (b[1] - a[1]) * t,
      )},${Math.round(a[2] + (b[2] - a[2]) * t)})`;

    // Far ridges wash out toward the sky; near ridges go almost black.
    const far = [74, 88, 76];
    const near = [17, 22, 18];

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      w = rect.width;
      h = rect.height;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const ridgeY = (
      x: number,
      { amp, y, freq, phase }: (typeof LAYERS)[number],
    ) => {
      const u = (x / w) * Math.PI * 2 * freq;
      const n =
        Math.sin(u + phase) * 0.6 +
        Math.sin(u * 2.3 + phase * 1.7) * 0.26 +
        Math.sin(u * 4.1 + phase * 0.6) * 0.14;
      return h * y - n * h * amp;
    };

    const draw = () => {
      ctx.clearRect(0, 0, w, h);

      // Sky: a low warm glow sitting just above the horizon.
      const sky = ctx.createLinearGradient(0, 0, 0, h);
      sky.addColorStop(0, "#171d18");
      sky.addColorStop(0.55, "#243026");
      sky.addColorStop(0.78, "#4a4632");
      sky.addColorStop(1, "#6b5730");
      ctx.fillStyle = sky;
      ctx.fillRect(0, 0, w, h);

      // Sun sits ABOVE the farthest ridge (y = 0.44h) or the ridges paint over
      // it and the sky reads as an empty gradient.
      const cx = w * 0.74;
      const cy = h * 0.32;
      const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, h * 0.7);
      glow.addColorStop(0, "rgba(222,172,84,0.55)");
      glow.addColorStop(0.3, "rgba(200,146,62,0.2)");
      glow.addColorStop(1, "rgba(196,142,60,0)");
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, w, h);

      ctx.beginPath();
      ctx.arc(cx, cy, h * 0.062, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(233,192,118,0.92)";
      ctx.fill();

      // Ridges, far to near.
      for (const layer of LAYERS) {
        ctx.beginPath();
        ctx.moveTo(0, ridgeY(0, layer));
        for (let x = 1; x <= w; x += 2) ctx.lineTo(x, ridgeY(x, layer));
        ctx.lineTo(w, h);
        ctx.lineTo(0, h);
        ctx.closePath();
        ctx.fillStyle = mix(far, near, layer.tone);
        ctx.fill();
      }
    };

    resize();
    draw();

    const onResize = () => {
      resize();
      draw();
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  return (
    <canvas
      ref={ref}
      className="absolute inset-0 h-full w-full"
      aria-hidden="true"
    />
  );
}
