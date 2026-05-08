import { useState, useRef, useEffect, useCallback } from "react";

const API_BASE = "http://localhost:8000";

const KNOWN_MODELS = [
  {
    id: "unet",
    label: "U-Net",
    description: "Task-specific encoder-decoder. Best Dice (13.84%).",
    dice: 13.84,
    iou: 7.19,
    params: "~31M",
  },
];

const S = {
  root: {
    fontFamily: "'IBM Plex Sans', 'Helvetica Neue', Helvetica, sans-serif",
    fontWeight: 300,
    fontSize: 14,
    background: "#f7f7f5",
    color: "#0f0f0e",
    minHeight: "100vh",
  },
  letterhead: {
    background: "#fff",
    borderBottom: "2px solid #0f0f0e",
    padding: "1.6rem 2.4rem 1.2rem",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-end",
    gap: "2rem",
  },
  institution: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "0.58rem",
    letterSpacing: "0.2em",
    textTransform: "uppercase",
    color: "#7a7a76",
    marginBottom: "0.3rem",
  },
  reportTitle: {
    fontFamily: "'IBM Plex Serif', Georgia, serif",
    fontSize: "1.55rem",
    fontWeight: 400,
    fontStyle: "italic",
    letterSpacing: "-0.01em",
    lineHeight: 1.15,
  },
  reportMeta: {
    textAlign: "right",
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "0.62rem",
    color: "#7a7a76",
    lineHeight: 1.85,
  },
  infoBar: {
    background: "#0f0f0e",
    color: "#fff",
    padding: "0.65rem 2.4rem",
    display: "flex",
    gap: "2.5rem",
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "0.62rem",
    letterSpacing: "0.08em",
    flexWrap: "wrap",
  },
  infoItemLabel: { color: "rgba(255,255,255,0.4)", marginRight: "0.45rem" },
  body: {
    maxWidth: 1060,
    margin: "0 auto",
    padding: "2.5rem 2.4rem",
    display: "grid",
    gridTemplateColumns: "1fr 280px",
    gap: "2.5rem",
    alignItems: "start",
  },
  sectionLabel: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "0.58rem",
    letterSpacing: "0.2em",
    textTransform: "uppercase",
    color: "#7a7a76",
    borderTop: "1.5px solid #0f0f0e",
    paddingTop: "0.45rem",
    marginBottom: "1.1rem",
    display: "flex",
    justifyContent: "space-between",
  },
  uploadCard: {
    background: "#fff",
    border: "1px solid #d8d8d4",
    marginBottom: "1.5rem",
    overflow: "hidden",
  },
  uploadZone: (dragging) => ({
    border: `1.5px dashed ${dragging ? "#0f0f0e" : "#d8d8d4"}`,
    background: dragging ? "#fafaf8" : "#f7f7f5",
    padding: "2rem 1.5rem",
    textAlign: "center",
    cursor: "pointer",
    transition: "all 0.15s",
    margin: "1rem",
  }),
  uploadIcon: { fontSize: "1.6rem", marginBottom: "0.5rem" },
  uploadText: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "0.65rem",
    color: "#7a7a76",
    lineHeight: 1.8,
  },
  previewWrap: {
    margin: "1rem",
    border: "1px solid #d8d8d4",
    position: "relative",
    background: "#000",
  },
  previewImg: { width: "100%", display: "block" },
  changeBtn: {
    position: "absolute",
    top: 6,
    right: 6,
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "0.55rem",
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    background: "rgba(0,0,0,0.7)",
    color: "#fff",
    border: "none",
    padding: "0.25rem 0.55rem",
    cursor: "pointer",
  },
  runBtn: (disabled) => ({
    width: "calc(100% - 2rem)",
    padding: "0.7rem",
    background: disabled ? "#d8d8d4" : "#0f0f0e",
    color: disabled ? "#aaa" : "#fff",
    border: "none",
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "0.65rem",
    letterSpacing: "0.15em",
    textTransform: "uppercase",
    cursor: disabled ? "not-allowed" : "pointer",
    transition: "background 0.15s",
    margin: "1rem",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "0.5rem",
  }),
  outputStrip: {
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: 1,
    background: "#d8d8d4",
    border: "1px solid #d8d8d4",
    marginBottom: 1,
  },
  outputPane: { background: "#fff" },
  outputPaneHead: {
    padding: "0.32rem 0.55rem",
    borderBottom: "1px solid #ebebea",
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "0.52rem",
    letterSpacing: "0.1em",
    color: "#b0b0aa",
    textTransform: "uppercase",
  },
  outputImgWrap: {
    aspectRatio: "1",
    background: "#080808",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
    position: "relative",
  },
  outputImg: { width: "100%", height: "100%", objectFit: "cover", display: "block" },
  placeholder: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "0.52rem",
    color: "#333",
    textAlign: "center",
    lineHeight: 1.8,
    padding: "0 1rem",
  },
  spinnerOverlay: {
    position: "absolute",
    inset: 0,
    background: "rgba(8,8,8,0.75)",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: "0.5rem",
  },
  metricsRow: {
    display: "grid",
    gridTemplateColumns: "repeat(4, 1fr)",
    gap: 1,
    background: "#d8d8d4",
    border: "1px solid #d8d8d4",
    marginBottom: 1,
  },
  metricCell: { background: "#fff", padding: "0.65rem 0.7rem" },
  metricKey: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "0.52rem",
    letterSpacing: "0.1em",
    color: "#b0b0aa",
    textTransform: "uppercase",
    marginBottom: "0.15rem",
  },
  metricVal: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "1.05rem",
    fontWeight: 500,
    color: "#0f0f0e",
  },
  metricSub: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "0.5rem",
    color: "#b0b0aa",
    marginTop: 2,
  },
  errBox: {
    borderLeft: "3px solid #c0392b",
    padding: "0.9rem 1rem",
    marginTop: "1rem",
    background: "#fdf0ef",
  },
  errLabel: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "0.55rem",
    letterSpacing: "0.15em",
    textTransform: "uppercase",
    color: "#7a7a76",
    marginBottom: "0.3rem",
  },
  errText: {
    fontFamily: "'IBM Plex Serif', Georgia, serif",
    fontSize: "0.85rem",
    fontStyle: "italic",
    lineHeight: 1.75,
    color: "#3a3a38",
  },
  sidebar: { position: "sticky", top: "1.5rem" },
  sideCard: {
    background: "#fff",
    border: "1px solid #d8d8d4",
    marginBottom: "1.2rem",
    overflow: "hidden",
  },
  sideCardHead: {
    background: "#0f0f0e",
    color: "#fff",
    padding: "0.5rem 0.8rem",
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "0.58rem",
    letterSpacing: "0.15em",
    textTransform: "uppercase",
  },
  sideCardBody: { padding: "0.8rem" },
  statRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "baseline",
    padding: "0.4rem 0",
    borderBottom: "1px solid #ebebea",
  },
  statKey: { fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.62rem", color: "#7a7a76" },
  statVal: { fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.78rem", fontWeight: 500 },
  pill: (state) => ({
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "0.52rem",
    padding: "0.18rem 0.55rem",
    border: `1px solid ${state === "ready" ? "#1a6640" : state === "running" ? "#1a4a8a" : "#d8d8d4"}`,
    color: state === "ready" ? "#1a6640" : state === "running" ? "#1a4a8a" : "#b0b0aa",
    background: state === "ready" ? "#eef6f1" : state === "running" ? "#eef3fa" : "transparent",
    display: "inline-flex",
    alignItems: "center",
    gap: "0.35rem",
    borderRadius: 1,
  }),
  footer: {
    borderTop: "2px solid #0f0f0e",
    padding: "0.9rem 2.4rem",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    background: "#fff",
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "0.58rem",
    color: "#7a7a76",
  },
  ring: {
    width: 26,
    height: 26,
    border: "1.5px solid rgba(255,255,255,0.15)",
    borderTopColor: "#fff",
    borderRadius: "50%",
    animation: "spin 0.75s linear infinite",
  },
};

if (typeof document !== "undefined" && !document.getElementById("__ts_kf")) {
  const style = document.createElement("style");
  style.id = "__ts_kf";
  style.textContent = `
    @keyframes spin { to { transform: rotate(360deg); } }
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;1,300&family=IBM+Plex+Serif:ital,wght@0,300;0,400;1,300;1,400&family=IBM+Plex+Sans:wght@300;400&display=swap');
  `;
  document.head.appendChild(style);
}

function Spinner({ label }) {
  return (
    <div style={S.spinnerOverlay}>
      <div style={S.ring} />
      <div style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: "0.52rem", color: "rgba(255,255,255,0.4)", letterSpacing: "0.1em" }}>
        {label}
      </div>
    </div>
  );
}

export default function SegmentationApp({ apiBase = API_BASE }) {
  const [imageFile, setImageFile]     = useState(null);
  const [imageSrc, setImageSrc]       = useState(null);
  const [dragging, setDragging]       = useState(false);
  const [status, setStatus]           = useState("idle");
  const [result, setResult]           = useState(null);
  const [loadingStep, setLoadingStep] = useState("");
  const [backendOnline, setBackend]   = useState(null);

  const fileInputRef     = useRef(null);
  const maskCanvasRef    = useRef(null);
  const overlayCanvasRef = useRef(null);

  const activeModel = KNOWN_MODELS[0];

  useEffect(() => {
    fetch(`${apiBase}/api/health`, { signal: AbortSignal.timeout(3000) })
      .then(r => r.ok ? setBackend(true) : setBackend(false))
      .catch(() => setBackend(false));
  }, [apiBase]);

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) return;
    setImageFile(file);
    setResult(null);
    const reader = new FileReader();
    reader.onload = e => { setImageSrc(e.target.result); setStatus("ready"); };
    reader.readAsDataURL(file);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const clearImage = () => {
    setImageFile(null); setImageSrc(null); setResult(null); setStatus("idle");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const runInference = useCallback(async () => {
    if (!imageFile || status === "running") return;
    setStatus("running");
    setResult(null);

    const steps = ["ENCODING…", "SEGMENTING…", "POSTPROCESSING…", "COMPUTING STATS…"];
    let si = 0;
    setLoadingStep(steps[0]);
    const interval = setInterval(() => setLoadingStep(steps[++si % steps.length]), 650);

    try {
      const form = new FormData();
      form.append("image", imageFile);
      form.append("model", "unet");

      const res = await fetch(`${apiBase}/api/segment`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      clearInterval(interval);
      setResult(data);
      setStatus("done");

      [{ ref: maskCanvasRef, src: data.mask_b64 }, { ref: overlayCanvasRef, src: data.overlay_b64 }].forEach(({ ref, src }) => {
        if (!ref.current || !src) return;
        const img = new Image();
        img.onload = () => {
          const c = ref.current;
          if (!c) return;
          c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
        };
        img.src = `data:image/png;base64,${src}`;
      });

    } catch (err) {
      clearInterval(interval);
      setStatus("error");
      console.error(err);
    }
  }, [imageFile, status, apiBase]);

  const canRun = imageSrc && status !== "running";

  return (
    <div style={S.root}>

      <div style={S.letterhead}>
        <div>
          <div style={S.institution}>Simon Fraser University · CMPT 340 · Medical Image Analysis</div>
          <div style={S.reportTitle}>Brain Tumour Segmentation<br />Interactive Inference Report</div>
        </div>
        <div style={S.reportMeta}>
          <div><strong>Dataset</strong> &nbsp; BraTS2020</div>
          <div><strong>Modality</strong> &nbsp; MRI · Axial</div>
          <div><strong>Backend</strong> &nbsp; {backendOnline === null ? "checking…" : backendOnline ? "online" : "offline"}</div>
        </div>
      </div>

      <div style={S.infoBar}>
        <span><span style={S.infoItemLabel}>STUDY</span>Glioma Segmentation</span>
        <span><span style={S.infoItemLabel}>MODEL</span>U-Net</span>
        <span><span style={S.infoItemLabel}>METRIC</span>Mask Coverage · Confidence · Tumour Pixels · Inference Time</span>
        {result && <span><span style={S.infoItemLabel}>INFERENCE</span>{result.elapsed_ms} ms</span>}
      </div>

      <div style={S.body}>
        <main>

          <div style={S.sectionLabel}>
            <span>1. &nbsp; Image Input</span>
            <div style={S.pill(status === "done" ? "ready" : status === "running" ? "running" : "idle")}>
              <span style={{ width: 5, height: 5, borderRadius: "50%", background: "currentColor", display: "inline-block" }} />
              {status === "idle" ? "Awaiting input" : status === "ready" ? "Ready" : status === "running" ? "Running" : status === "done" ? "Complete" : "Error"}
            </div>
          </div>

          <div style={S.uploadCard}>
            {!imageSrc ? (
              <div
                style={S.uploadZone(dragging)}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={e => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
              >
                <div style={S.uploadIcon}>⬜</div>
                <div style={S.uploadText}>
                  <strong style={{ color: "#0f0f0e" }}>Drop MRI slice here</strong><br />
                  or click to browse · PNG / JPG
                </div>
                <input ref={fileInputRef} type="file" accept="image/*" style={{ display: "none" }} onChange={e => handleFile(e.target.files[0])} />
              </div>
            ) : (
              <div style={S.previewWrap}>
                <img src={imageSrc} alt="MRI input" style={S.previewImg} />
                <button style={S.changeBtn} onClick={clearImage}>CHANGE</button>
              </div>
            )}

            <button style={S.runBtn(!canRun)} onClick={runInference} disabled={!canRun}>
              {status === "running" ? "PROCESSING…" : "▶ RUN SEGMENTATION"}
            </button>
          </div>

          <div style={S.sectionLabel}>
            <span>2. &nbsp; Segmentation Output</span>
          </div>

          <div style={S.outputStrip}>
            <div style={S.outputPane}>
              <div style={S.outputPaneHead}>Input slice</div>
              <div style={S.outputImgWrap}>
                {imageSrc
                  ? <img src={imageSrc} alt="input" style={S.outputImg} />
                  : <div style={S.placeholder}>Upload an image<br />to begin</div>}
              </div>
            </div>

            <div style={S.outputPane}>
              <div style={S.outputPaneHead}>Predicted mask</div>
              <div style={{ ...S.outputImgWrap, position: "relative" }}>
                {status === "running" && <Spinner label={loadingStep} />}
                {status !== "running" && !result && <div style={S.placeholder}>Mask will appear<br />after inference</div>}
                <canvas ref={maskCanvasRef} width={300} height={300} style={{ width: "100%", height: "100%", display: result ? "block" : "none" }} />
              </div>
            </div>

            <div style={S.outputPane}>
              <div style={S.outputPaneHead}>Overlay</div>
              <div style={{ ...S.outputImgWrap, position: "relative" }}>
                {status === "running" && <Spinner label="…" />}
                {status !== "running" && !result && <div style={S.placeholder}>Overlay will appear<br />after inference</div>}
                <canvas ref={overlayCanvasRef} width={300} height={300} style={{ width: "100%", height: "100%", display: result ? "block" : "none" }} />
              </div>
            </div>
          </div>

          {result && result.pred_stats && (
            <div style={S.metricsRow}>
              {[
                { key: "Mask Coverage",  val: `${result.pred_stats.mask_coverage}%`,               sub: "of image area" },
                { key: "Confidence",     val: `${result.pred_stats.estimated_confidence}%`,        sub: "prediction score" },
                { key: "Tumour Pixels",  val: result.pred_stats.predicted_pixels.toLocaleString(), sub: "pixels detected" },
                { key: "Inference Time", val: `${result.elapsed_ms} ms`,                           sub: "end-to-end" },
              ].map(({ key, val, sub }) => (
                <div key={key} style={S.metricCell}>
                  <div style={S.metricKey}>{key}</div>
                  <div style={S.metricVal}>{val}</div>
                  <div style={S.metricSub}>{sub}</div>
                </div>
              ))}
            </div>
          )}

          {status === "error" && (
            <div style={S.errBox}>
              <div style={S.errLabel}>Error</div>
              <div style={S.errText}>Inference failed. Check that the backend is running on {apiBase}.</div>
            </div>
          )}

        </main>

        <aside style={S.sidebar}>

          <div style={S.sideCard}>
            <div style={S.sideCardHead}>Model</div>
            <div style={S.sideCardBody}>
              <div style={{ fontFamily: "'IBM Plex Serif',Georgia,serif", fontSize: "0.82rem", fontStyle: "italic", color: "#3a3a38", marginBottom: "0.8rem", lineHeight: 1.6 }}>
                {activeModel.description}
              </div>
              {[
                ["Dice (reported)", `${activeModel.dice}%`],
                ["IoU (reported)",  `${activeModel.iou}%`],
                ["Parameters",      activeModel.params],
              ].map(([k, v]) => (
                <div key={k} style={S.statRow}>
                  <span style={S.statKey}>{k}</span>
                  <span style={S.statVal}>{v}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={S.sideCard}>
            <div style={S.sideCardHead}>Study Parameters</div>
            <div style={S.sideCardBody}>
              {[
                ["Dataset",      "BraTS2020"],
                ["Train slices", "Full BraTS2020"],
                ["Val slices",   "Full BraTS2020"],
                ["Test slices",  "Full BraTS2020"],
                ["Modality",     "MRI"],
                ["Tumour type",  "Glioma"],
              ].map(([k, v]) => (
                <div key={k} style={S.statRow}>
                  <span style={S.statKey}>{k}</span>
                  <span style={S.statVal}>{v}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={S.sideCard}>
            <div style={S.sideCardHead}>Key Finding</div>
            <div style={S.sideCardBody}>
              <p style={{ fontFamily: "'IBM Plex Serif',Georgia,serif", fontStyle: "italic", fontSize: "0.82rem", color: "#3a3a38", lineHeight: 1.7 }}>
                "Scale does not substitute for domain specificity. U-Net's inductive bias aligns with MRI regularities — SAM's does not."
              </p>
            </div>
          </div>

          {result && (
            <div style={S.sideCard}>
              <div style={S.sideCardHead}>Last Run</div>
              <div style={S.sideCardBody}>
                {[
                  ["Model",      "U-Net"],
                  ["Coverage",   `${result.pred_stats?.mask_coverage ?? "—"}%`],
                  ["Confidence", `${result.pred_stats?.estimated_confidence ?? "—"}%`],
                  ["Time",       `${result.elapsed_ms} ms`],
                  ["Source",     "Live backend"],
                ].map(([k, v]) => (
                  <div key={k} style={S.statRow}>
                    <span style={S.statKey}>{k}</span>
                    <span style={S.statVal}>{v}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

        </aside>
      </div>

      <div style={S.footer}>
        <div>Simon Fraser University · CMPT 340 · 2025</div>
        <div>BraTS2020 · U-Net</div>
      </div>
    </div>
  );
}
