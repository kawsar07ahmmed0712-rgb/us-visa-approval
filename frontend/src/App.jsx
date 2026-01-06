import { useEffect, useMemo, useRef, useState } from "react";

/* =========================
   Utilities
   ========================= */

function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

function toNum(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

function safeArray(v, fallback = []) {
  return Array.isArray(v) && v.length ? v : fallback;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function highlightJson(obj) {
  const json = escapeHtml(JSON.stringify(obj, null, 2));
  let out = json.replace(/"([^"]+)"(?=\s*:)/g, '<span class="json-key">"$1"</span>');
  out = out.replace(/:\s*"([^"]*)"/g, ': <span class="json-str">"$1"</span>');
  out = out.replace(/\b-?\d+(\.\d+)?\b/g, '<span class="json-num">$&</span>');
  out = out.replace(/\b(true|false|null)\b/g, '<span class="json-bool">$1</span>');
  return out;
}

function formatPct01(x) {
  if (!Number.isFinite(x)) return "—";
  return `${Math.round(x * 100)}%`;
}

function timeNow() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/* =========================
   UI Primitives
   ========================= */

function Badge({ tone = "neutral", children }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

function Button({ variant = "default", disabled, onClick, children, type = "button" }) {
  return (
    <button type={type} className={`btn btn-${variant}`} disabled={disabled} onClick={onClick}>
      {children}
    </button>
  );
}

function Card({ title, subtitle, right, children }) {
  return (
    <section className="card">
      <header className="card-head">
        <div className="card-head-left">
          <div className="card-title">{title}</div>
          {subtitle ? <div className="card-subtitle">{subtitle}</div> : null}
        </div>
        {right ? <div className="card-head-right">{right}</div> : null}
      </header>
      <div className="card-body">{children}</div>
    </section>
  );
}

function Field({ label, hint, error, children }) {
  return (
    <div className="field">
      <div className="field-top">
        <div className="field-label">{label}</div>
        {hint ? <div className="field-hint">{hint}</div> : null}
      </div>
      {children}
      {error ? <div className="field-error">{error}</div> : null}
    </div>
  );
}

function Progress({ value01 }) {
  const pct = clamp(Math.round((Number(value01) || 0) * 100), 0, 100);
  return (
    <div className="progress" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
      <div className="progress-fill" style={{ width: `${pct}%` }} />
    </div>
  );
}

function Segmented({ value, onChange, options }) {
  return (
    <div className="seg">
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          className={`seg-item ${value === opt.value ? "seg-item-active" : ""}`}
          onClick={() => onChange(opt.value)}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

function Toast({ show, text }) {
  return (
    <div className={`toast ${show ? "toast-show" : ""}`} aria-live="polite" aria-atomic="true">
      {text}
    </div>
  );
}

/* =========================
   App
   ========================= */

export default function App() {
  const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";
  const STORAGE_KEY = "visa_demo_form_prod_final_v1";

  const historyRef = useRef([]);

  const FALLBACK = useMemo(
    () => ({
      continents: ["Asia", "Africa", "Europe", "North America", "South America", "Oceania"],
      regions: ["Northeast", "South", "Midwest", "West"],
      wageUnits: ["Hour", "Week", "Month", "Year"],
      education: ["High School", "Bachelor's", "Master's", "Doctorate"],
      yn: ["Y", "N"],
    }),
    []
  );

  const defaultForm = useMemo(
    () => ({
      continent: "Asia",
      region_of_employment: "Northeast",
      unit_of_wage: "Year",
      education_of_employee: "Bachelor's",
      has_job_experience: "Y",
      requires_job_training: "N",
      full_time_position: "Y",
      no_of_employees: 50,
      prevailing_wage: 70000,
      yr_of_estab: 2005,
    }),
    []
  );

  const [form, setForm] = useState(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return defaultForm;
      return { ...defaultForm, ...JSON.parse(raw) };
    } catch {
      return defaultForm;
    }
  });

  const [api, setApi] = useState({
    ok: false,
    threshold: null,
    label_pos: null,
    label_neg: null,
    lastCheckedAt: null,
  });

  const [meta, setMeta] = useState({
    ok: false,
    rows: null,
    source: null,
    categorical_options: null,
    numeric_ranges: null,
    lastLoadedAt: null,
  });

  const [fieldErrors, setFieldErrors] = useState({});
  const [busy, setBusy] = useState(false);

  const [result, setResult] = useState(null);
  const [recent, setRecent] = useState([]);

  const [rightTab, setRightTab] = useState("presets"); // presets | payload | history | meta
  const [toast, setToast] = useState({ show: false, text: "" });

  // Full-viewport background + prevent horizontal overflow
  useEffect(() => {
    document.documentElement.style.background = "#020617";
    document.body.style.background = "#020617";
    document.body.style.overflowX = "hidden";
    const root = document.getElementById("root");
    if (root) {
      root.style.background = "#020617";
      root.style.minHeight = "100vh";
    }
  }, []);

  // Autosave
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(form));
    } catch {
      // ignore
    }
  }, [form]);

  // Load /health
  useEffect(() => {
    const ac = new AbortController();
    let alive = true;

    async function check() {
      try {
        const res = await fetch(`${API_URL}/health`, { signal: ac.signal });
        const data = await res.json().catch(() => null);
        if (!alive) return;

        if (data && typeof data === "object") {
          setApi({
            ok: Boolean(data.ok && res.ok),
            threshold: typeof data.threshold === "number" ? data.threshold : null,
            label_pos: data.label_pos ?? null,
            label_neg: data.label_neg ?? null,
            lastCheckedAt: timeNow(),
          });
        } else {
          setApi({
            ok: false,
            threshold: null,
            label_pos: null,
            label_neg: null,
            lastCheckedAt: timeNow(),
          });
        }
      } catch {
        if (!alive) return;
        setApi({
          ok: false,
          threshold: null,
          label_pos: null,
          label_neg: null,
          lastCheckedAt: timeNow(),
        });
      }
    }

    check();
    const t = setInterval(check, 10000);

    return () => {
      alive = false;
      ac.abort();
      clearInterval(t);
    };
  }, [API_URL]);

  // Load /meta
  useEffect(() => {
    const ac = new AbortController();
    let alive = true;

    async function loadMeta() {
      try {
        const res = await fetch(`${API_URL}/meta`, { signal: ac.signal });
        const data = await res.json().catch(() => null);
        if (!alive) return;

        if (res.ok && data && data.ok) {
          setMeta({
            ok: true,
            rows: data.rows ?? null,
            source: data.source ?? null,
            categorical_options: data.categorical_options ?? null,
            numeric_ranges: data.numeric_ranges ?? null,
            lastLoadedAt: timeNow(),
          });
        } else {
          setMeta((m) => ({ ...m, ok: false, lastLoadedAt: timeNow() }));
        }
      } catch {
        if (!alive) return;
        setMeta((m) => ({ ...m, ok: false, lastLoadedAt: timeNow() }));
      }
    }

    loadMeta();
    return () => {
      alive = false;
      ac.abort();
    };
  }, [API_URL]);

  const OPTIONS = useMemo(() => {
    const cat = meta.categorical_options || {};
    return {
      continents: safeArray(cat.continent, FALLBACK.continents),
      regions: safeArray(cat.region_of_employment, FALLBACK.regions),
      wageUnits: safeArray(cat.unit_of_wage, FALLBACK.wageUnits),
      education: safeArray(cat.education_of_employee, FALLBACK.education),
      yn_hasExp: safeArray(cat.has_job_experience, FALLBACK.yn),
      yn_training: safeArray(cat.requires_job_training, FALLBACK.yn),
      yn_fullTime: safeArray(cat.full_time_position, FALLBACK.yn),
    };
  }, [meta, FALLBACK]);

  // Keep select values valid after meta loads
  useEffect(() => {
    function ensureValue(key, list) {
      const v = String(form[key] ?? "");
      if (Array.isArray(list) && list.length && !list.includes(v)) return list[0];
      return null;
    }

    const patch = {};
    const p1 = ensureValue("continent", OPTIONS.continents);
    const p2 = ensureValue("region_of_employment", OPTIONS.regions);
    const p3 = ensureValue("unit_of_wage", OPTIONS.wageUnits);
    const p4 = ensureValue("education_of_employee", OPTIONS.education);
    const p5 = ensureValue("has_job_experience", OPTIONS.yn_hasExp);
    const p6 = ensureValue("requires_job_training", OPTIONS.yn_training);
    const p7 = ensureValue("full_time_position", OPTIONS.yn_fullTime);

    if (p1) patch.continent = p1;
    if (p2) patch.region_of_employment = p2;
    if (p3) patch.unit_of_wage = p3;
    if (p4) patch.education_of_employee = p4;
    if (p5) patch.has_job_experience = p5;
    if (p6) patch.requires_job_training = p6;
    if (p7) patch.full_time_position = p7;

    if (Object.keys(patch).length) setForm((s) => ({ ...s, ...patch }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    OPTIONS.continents,
    OPTIONS.regions,
    OPTIONS.wageUnits,
    OPTIONS.education,
    OPTIONS.yn_hasExp,
    OPTIONS.yn_training,
    OPTIONS.yn_fullTime,
  ]);

  const ranges = meta.numeric_ranges || {};
  const rangeHint = (key, fallbackText) => {
    const r = ranges?.[key];
    if (!r || typeof r !== "object") return fallbackText;
    const min = Number.isFinite(r.min) ? Math.round(r.min) : null;
    const max = Number.isFinite(r.max) ? Math.round(r.max) : null;
    if (min === null || max === null) return fallbackText;
    return `${min} — ${max}`;
  };

  const wageHint = rangeHint("prevailing_wage", "> 0");
  const empHint = rangeHint("no_of_employees", "≥ 1");
  const estabHint = rangeHint("yr_of_estab", "1900 — current year");

  const payload = useMemo(() => {
    return {
      continent: String(form.continent).trim(),
      region_of_employment: String(form.region_of_employment).trim(),
      unit_of_wage: String(form.unit_of_wage).trim(),
      education_of_employee: String(form.education_of_employee).trim(),
      has_job_experience: String(form.has_job_experience).trim(),
      requires_job_training: String(form.requires_job_training).trim(),
      full_time_position: String(form.full_time_position).trim(),
      no_of_employees: Number(form.no_of_employees),
      prevailing_wage: Number(form.prevailing_wage),
      yr_of_estab: Number(form.yr_of_estab),
    };
  }, [form]);

  const onChange = (e) => {
    const { name, value } = e.target;
    setForm((p) => ({ ...p, [name]: value }));
    setFieldErrors((p) => ({ ...p, [name]: "" }));
  };

  const validate = () => {
    const errs = {};
    const nowYear = new Date().getFullYear();

    const emp = toNum(form.no_of_employees);
    const wage = toNum(form.prevailing_wage);
    const estab = toNum(form.yr_of_estab);

    if (!Number.isFinite(emp) || emp < 1) errs.no_of_employees = "Value must be ≥ 1";
    if (!Number.isFinite(wage) || wage <= 0) errs.prevailing_wage = "Value must be > 0";
    if (!Number.isFinite(estab) || estab < 1900 || estab > nowYear) errs.yr_of_estab = `Value must be between 1900 and ${nowYear}`;

    return errs;
  };

  const predict = async () => {
    const errs = validate();
    setFieldErrors(errs);
    if (Object.keys(errs).length) return;

    setBusy(true);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json().catch(() => null);

      if (!data) throw new Error(`Invalid response (HTTP ${res.status})`);
      if (data.ok === false) throw new Error(data.error || "Prediction failed");
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

      setResult(data);

      const item = {
        at: timeNow(),
        label: data.label,
        score: Number(data.score_pos_class),
      };
      historyRef.current = [item, ...historyRef.current].slice(0, 10);
      setRecent(historyRef.current);
      setRightTab("history");
    } catch (err) {
      setResult({
        ok: false,
        error: err?.message ? String(err.message) : "Prediction failed",
        threshold: api.threshold ?? 0.5,
      });
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    setForm(defaultForm);
    setFieldErrors({});
    setResult(null);
    setRecent([]);
    historyRef.current = [];
    setRightTab("presets");
  };

  const copyPayload = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
      setToast({ show: true, text: "Copied" });
      setTimeout(() => setToast({ show: false, text: "" }), 1000);
    } catch {
      setToast({ show: true, text: "Copy failed" });
      setTimeout(() => setToast({ show: false, text: "" }), 1000);
    }
  };

  const apiTone = api.ok ? "ok" : "bad";
  const metaTone = meta.ok ? "ok" : "neutral";

  const thr = result?.threshold ?? api.threshold ?? 0.5;
  const score = result?.score_pos_class;
  const scoreOk = Number.isFinite(Number(score));
  const score01 = scoreOk ? Number(score) : 0;

  let decision = "—";
  if (result?.ok === false) decision = "ERROR";
  else if (result?.label) decision = String(result.label);
  else if (scoreOk) decision = Number(score) >= Number(thr) ? "CERTIFIED" : "DENIED";

  const decisionTone = decision === "CERTIFIED" ? "ok" : decision === "DENIED" || decision === "ERROR" ? "bad" : "neutral";

  const presets = useMemo(() => {
    const wageMin = ranges?.prevailing_wage?.min;
    const wageMax = ranges?.prevailing_wage?.max;
    const wageP50 = ranges?.prevailing_wage?.p50;

    const estabMin = ranges?.yr_of_estab?.min;
    const estabP50 = ranges?.yr_of_estab?.p50;

    return [
      {
        name: "Low wage / New company",
        apply: () =>
          setForm((p) => ({
            ...p,
            prevailing_wage: Number.isFinite(wageMin) ? Math.round(wageMin) : 25000,
            yr_of_estab: new Date().getFullYear() - 5,
            has_job_experience: OPTIONS.yn_hasExp?.[1] ?? "N",
            full_time_position: OPTIONS.yn_fullTime?.[1] ?? "N",
            requires_job_training: OPTIONS.yn_training?.[0] ?? "Y",
          })),
      },
      {
        name: "High wage / Experienced",
        apply: () =>
          setForm((p) => ({
            ...p,
            prevailing_wage: Number.isFinite(wageMax) ? Math.round(wageMax) : 120000,
            yr_of_estab: Number.isFinite(estabMin) ? Math.round(estabMin) : 1998,
            has_job_experience: OPTIONS.yn_hasExp?.[0] ?? "Y",
            full_time_position: OPTIONS.yn_fullTime?.[0] ?? "Y",
            requires_job_training: OPTIONS.yn_training?.[1] ?? "N",
          })),
      },
      {
        name: "Mid wage / Training required",
        apply: () =>
          setForm((p) => ({
            ...p,
            prevailing_wage: Number.isFinite(wageP50) ? Math.round(wageP50) : 65000,
            yr_of_estab: Number.isFinite(estabP50) ? Math.round(estabP50) : 2008,
            requires_job_training: OPTIONS.yn_training?.[0] ?? "Y",
            has_job_experience: OPTIONS.yn_hasExp?.[0] ?? "Y",
          })),
      },
    ];
  }, [OPTIONS, ranges]);

  return (
    <div className="shell">
      <style>{`
        :root{
          --bg0:#020617;
          --txt:#e5e7eb;
          --mut: rgba(229,231,235,0.74);
          --card: rgba(255,255,255,0.06);
          --bd: rgba(255,255,255,0.11);
          --bd2: rgba(255,255,255,0.16);
          --shadow: 0 18px 40px rgba(0,0,0,0.36);
        }
        *{ box-sizing:border-box; }
        html, body, #root{ width:100%; min-height:100%; background: var(--bg0); }
        body{ margin:0; overflow-x:hidden; color: var(--txt); }

        .shell{
          min-height:100vh;
          width:100%;
          overflow:hidden;
          color: var(--txt);
        }
        .shell::before{
          content:"";
          position:fixed;
          inset:0;
          z-index:-1;
          background:
            radial-gradient(1200px 700px at 8% 12%, rgba(59,130,246,0.35), transparent 55%),
            radial-gradient(900px 500px at 92% 18%, rgba(16,185,129,0.25), transparent 55%),
            radial-gradient(900px 520px at 70% 90%, rgba(236,72,153,0.10), transparent 60%),
            linear-gradient(135deg, rgba(30,58,138,0.20), rgba(15,23,42,0.18)),
            var(--bg0);
        }

        .container{
          width:100%;
          max-width:none;
          margin:0;
          padding: 18px clamp(12px, 2vw, 28px);
        }

        .topbar{
          display:flex;
          align-items:flex-start;
          justify-content:space-between;
          gap:12px;
        }
        .brand{
          display:flex;
          flex-direction:column;
          gap:8px;
        }
        .brand-row{
          display:flex;
          align-items:center;
          gap:10px;
          flex-wrap:wrap;
        }
        .brand h1{
          margin:0;
          font-size: 22px;
          font-weight: 1000;
          letter-spacing: 0.2px;
        }
        .meta-line{
          font-size:12px;
          color: var(--mut);
        }
        .meta-line code{
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
          font-size: 12px;
          background: rgba(255,255,255,0.06);
          border: 1px solid rgba(255,255,255,0.10);
          padding: 2px 6px;
          border-radius: 999px;
          color: rgba(229,231,235,0.90);
        }

        .badge{
          display:inline-flex;
          align-items:center;
          padding: 6px 10px;
          border-radius: 999px;
          font-size: 12px;
          font-weight: 1000;
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.06);
          line-height: 1;
        }
        .badge-ok{ background: rgba(16,185,129,0.16); border-color: rgba(16,185,129,0.38); color: #d1fae5; }
        .badge-bad{ background: rgba(239,68,68,0.16); border-color: rgba(239,68,68,0.38); color: #fee2e2; }
        .badge-neutral{ color: rgba(229,231,235,0.95); }

        .btn{
          padding: 10px 12px;
          border-radius: 12px;
          border: 1px solid var(--bd2);
          background: rgba(255,255,255,0.06);
          color: var(--txt);
          cursor:pointer;
          font-weight: 1000;
          transition: transform .08s ease, background .15s ease, border-color .15s ease;
        }
        .btn:hover{ background: rgba(255,255,255,0.08); }
        .btn:active{ transform: translateY(1px); }
        .btn:disabled{ opacity: 0.55; cursor:not-allowed; }

        .btn-primary{
          background: rgba(59,130,246,0.22);
          border-color: rgba(59,130,246,0.35);
        }
        .btn-primary:hover{
          background: rgba(59,130,246,0.28);
        }

        .grid{
          margin-top: 16px;
          display:grid;
          gap: 12px;
          grid-template-columns: minmax(520px, 1.25fr) minmax(360px, 0.85fr) minmax(380px, 0.90fr);
          align-items:start;
        }
        @media (max-width: 1150px){
          .grid{ grid-template-columns: 1fr; }
        }

        .card{
          border-radius: 18px;
          padding: 16px;
          background: var(--card);
          border: 1px solid var(--bd);
          box-shadow: var(--shadow);
          backdrop-filter: blur(10px);
        }
        .card-head{
          display:flex;
          justify-content:space-between;
          align-items:flex-start;
          gap: 12px;
        }
        .card-title{
          font-size: 18px;
          font-weight: 1000;
        }
        .card-subtitle{
          margin-top: 6px;
          font-size: 12px;
          color: var(--mut);
        }
        .card-body{
          margin-top: 14px;
        }

        .form{
          display:grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
        }
        @media (max-width: 720px){
          .form{ grid-template-columns: 1fr; }
        }

        .field{ display:grid; gap: 6px; }
        .field-top{ display:flex; justify-content:space-between; gap: 10px; align-items:baseline; }
        .field-label{ font-size: 12px; font-weight: 1000; }
        .field-hint{ font-size: 11px; color: var(--mut); }
        .field-error{ font-size: 12px; color: #fecaca; }

        select, input{
          width: 100%;
          padding: 10px 12px;
          border-radius: 12px;
          border: 1px solid rgba(255,255,255,0.14);
          background: rgba(15,23,42,0.55);
          color: var(--txt);
          outline: none;
        }
        select:focus, input:focus{
          border-color: rgba(96,165,250,0.45);
          box-shadow: 0 0 0 4px rgba(59,130,246,0.10);
        }

        .actions{
          grid-column: 1 / -1;
          display:flex;
          gap: 10px;
          align-items:center;
          justify-content:space-between;
          flex-wrap:wrap;
          margin-top: 4px;
        }
        .actions-left{
          display:flex;
          gap: 10px;
          align-items:center;
          flex-wrap:wrap;
        }

        .seg{
          display:inline-flex;
          gap: 6px;
          padding: 6px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.04);
          margin-bottom: 12px;
        }
        .seg-item{
          padding: 8px 10px;
          border-radius: 999px;
          border: 1px solid transparent;
          background: transparent;
          color: rgba(229,231,235,0.92);
          cursor:pointer;
          font-weight: 1000;
          font-size: 12px;
        }
        .seg-item:hover{
          background: rgba(255,255,255,0.06);
        }
        .seg-item-active{
          background: rgba(59,130,246,0.18);
          border-color: rgba(59,130,246,0.28);
        }

        .progress{
          height: 12px;
          border-radius: 999px;
          background: rgba(255,255,255,0.10);
          overflow:hidden;
        }
        .progress-fill{
          height: 12px;
          border-radius: 999px;
          background: linear-gradient(90deg, rgba(96,165,250,1), rgba(52,211,153,1));
        }

        .resultGrid{
          display:grid;
          gap: 12px;
        }
        .metricRow{
          display:flex;
          justify-content:space-between;
          align-items:baseline;
          gap: 10px;
        }
        .metricTitle{
          font-size: 13px;
          font-weight: 1000;
        }
        .metricValue{
          font-size: 12px;
          color: var(--mut);
        }

        .errorBox{
          padding: 10px 12px;
          border-radius: 14px;
          border: 1px solid rgba(239,68,68,0.35);
          background: rgba(239,68,68,0.10);
          color: #fee2e2;
          font-weight: 1000;
          font-size: 12px;
        }

        .jsonbox{
          background: rgba(2,6,23,0.78);
          border: 1px solid rgba(255,255,255,0.10);
          padding: 12px;
          border-radius: 14px;
          overflow-x: auto;
          line-height: 1.5;
          margin: 0;
        }
        .json-key{ color:#60a5fa; }
        .json-str{ color:#34d399; }
        .json-num{ color:#fb923c; }
        .json-bool{ color:#c084fc; }

        details summary{
          cursor:pointer;
          font-weight: 1000;
          font-size: 12px;
          color: rgba(229,231,235,0.92);
        }

        .toolButtons{
          display:grid;
          gap: 10px;
        }
        .toolBtn{
          text-align:left;
          padding: 10px 12px;
          border-radius: 14px;
          border: 1px solid rgba(255,255,255,0.14);
          background: rgba(255,255,255,0.05);
          color: rgba(229,231,235,0.95);
          cursor:pointer;
          font-weight: 1000;
        }
        .toolBtn:hover{
          background: rgba(255,255,255,0.08);
        }

        .miniList{
          display:grid;
          gap: 8px;
          margin-top: 12px;
        }
        .miniItem{
          display:flex;
          justify-content:space-between;
          gap: 12px;
          padding: 10px 12px;
          border-radius: 14px;
          border: 1px solid rgba(255,255,255,0.10);
          background: rgba(255,255,255,0.04);
          font-size: 12px;
        }
        .miniLeft{ display:grid; gap: 4px; }
        .miniLabel{ font-weight: 1000; }
        .miniTime{ color: var(--mut); }
        .miniRight{ text-align:right; display:grid; gap: 4px; }
        .miniScore{ font-weight: 1000; }
        .miniSub{ color: var(--mut); }

        .toast{
          position: fixed;
          right: 16px;
          bottom: 16px;
          z-index: 50;
          background: rgba(15,23,42,0.85);
          border: 1px solid rgba(255,255,255,0.14);
          color: rgba(229,231,235,0.95);
          padding: 10px 12px;
          border-radius: 14px;
          box-shadow: 0 20px 50px rgba(0,0,0,0.4);
          opacity: 0;
          transform: translateY(10px);
          pointer-events: none;
          transition: opacity .18s ease, transform .18s ease;
          font-weight: 1000;
          font-size: 12px;
        }
        .toast-show{
          opacity: 1;
          transform: translateY(0);
        }

        .footer{
          margin-top: 14px;
          font-size: 12px;
          color: rgba(229,231,235,0.55);
        }
      `}</style>

      <Toast show={toast.show} text={toast.text} />

      <div className="container">
        <div className="topbar">
          <div className="brand">
            <div className="brand-row">
              <h1>US Visa Approval Prediction</h1>
              <Badge tone={apiTone}>{api.ok ? "API Online" : "API Offline"}</Badge>
              <Badge tone={metaTone}>{meta.ok ? `Meta Ready${meta.rows ? ` (${meta.rows})` : ""}` : "Meta Unavailable"}</Badge>
            </div>
            <div className="meta-line">
              API: <code>{API_URL}</code>
            </div>
          </div>

          <div className="brand-row">
            <Button variant="default" onClick={reset}>
              Reset
            </Button>
          </div>
        </div>

        <div className="grid">
          {/* Form */}
          <Card title="Applicant Details" subtitle="Structured inputs with dataset-driven categories when available.">
            <div className="form">
              <Field label="Continent">
                <select name="continent" value={form.continent} onChange={onChange}>
                  {OPTIONS.continents.map((v) => (
                    <option key={v} value={v} style={{ color: "#111827" }}>
                      {v}
                    </option>
                  ))}
                </select>
              </Field>

              <Field label="Region of Employment">
                <select name="region_of_employment" value={form.region_of_employment} onChange={onChange}>
                  {OPTIONS.regions.map((v) => (
                    <option key={v} value={v} style={{ color: "#111827" }}>
                      {v}
                    </option>
                  ))}
                </select>
              </Field>

              <Field label="Unit of Wage">
                <select name="unit_of_wage" value={form.unit_of_wage} onChange={onChange}>
                  {OPTIONS.wageUnits.map((v) => (
                    <option key={v} value={v} style={{ color: "#111827" }}>
                      {v}
                    </option>
                  ))}
                </select>
              </Field>

              <Field label="Education">
                <select name="education_of_employee" value={form.education_of_employee} onChange={onChange}>
                  {OPTIONS.education.map((v) => (
                    <option key={v} value={v} style={{ color: "#111827" }}>
                      {v}
                    </option>
                  ))}
                </select>
              </Field>

              <Field label="Has Job Experience">
                <select name="has_job_experience" value={form.has_job_experience} onChange={onChange}>
                  {OPTIONS.yn_hasExp.map((v) => (
                    <option key={v} value={v} style={{ color: "#111827" }}>
                      {v}
                    </option>
                  ))}
                </select>
              </Field>

              <Field label="Requires Job Training">
                <select name="requires_job_training" value={form.requires_job_training} onChange={onChange}>
                  {OPTIONS.yn_training.map((v) => (
                    <option key={v} value={v} style={{ color: "#111827" }}>
                      {v}
                    </option>
                  ))}
                </select>
              </Field>

              <Field label="Full Time Position">
                <select name="full_time_position" value={form.full_time_position} onChange={onChange}>
                  {OPTIONS.yn_fullTime.map((v) => (
                    <option key={v} value={v} style={{ color: "#111827" }}>
                      {v}
                    </option>
                  ))}
                </select>
              </Field>

              <Field label="Number of Employees" hint={empHint} error={fieldErrors.no_of_employees}>
                <input name="no_of_employees" value={form.no_of_employees} onChange={onChange} type="number" />
              </Field>

              <Field label="Prevailing Wage" hint={wageHint} error={fieldErrors.prevailing_wage}>
                <input name="prevailing_wage" value={form.prevailing_wage} onChange={onChange} type="number" />
              </Field>

              <Field label="Year of Establishment" hint={estabHint} error={fieldErrors.yr_of_estab}>
                <input name="yr_of_estab" value={form.yr_of_estab} onChange={onChange} type="number" />
              </Field>

              <div className="actions">
                <div className="actions-left">
                  <Button variant="primary" disabled={busy || !api.ok} onClick={predict}>
                    {busy ? "Predicting…" : "Predict"}
                  </Button>
                  <Button variant="default" disabled={busy} onClick={copyPayload}>
                    Copy Payload
                  </Button>
                </div>

                <div className="brand-row">
                  <Badge tone="neutral">{api.lastCheckedAt ? `Health: ${api.lastCheckedAt}` : "Health: —"}</Badge>
                  <Badge tone="neutral">{meta.lastLoadedAt ? `Meta: ${meta.lastLoadedAt}` : "Meta: —"}</Badge>
                </div>
              </div>
            </div>
          </Card>

          {/* Result */}
          <Card title="Prediction Result" subtitle="Probability score and threshold decision." right={<Badge tone={decisionTone}>{decision}</Badge>}>
            <div className="resultGrid">
              {result?.ok === false || result?.error ? <div className="errorBox">{result.error || "Prediction failed"}</div> : null}

              <div className="metricRow">
                <div className="metricTitle">Score (positive class)</div>
                <div className="metricValue">{scoreOk ? `${Number(score).toFixed(4)} (${formatPct01(Number(score))})` : "—"}</div>
              </div>

              <div className="metricRow">
                <div className="metricTitle">Threshold</div>
                <div className="metricValue">{Number.isFinite(Number(thr)) ? Number(thr).toFixed(2) : "—"}</div>
              </div>

              <Progress value01={score01} />

              <details>
                <summary>Raw JSON</summary>
                <pre className="jsonbox" dangerouslySetInnerHTML={{ __html: highlightJson(result || {}) }} />
              </details>
            </div>
          </Card>

          {/* Workspace */}
          <Card title="Workspace" subtitle="Presets, payload, history, and meta details.">
            <Segmented
              value={rightTab}
              onChange={setRightTab}
              options={[
                { value: "presets", label: "Presets" },
                { value: "payload", label: "Payload" },
                { value: "history", label: "History" },
                { value: "meta", label: "Meta" },
              ]}
            />

            {rightTab === "presets" ? (
              <>
                <div className="toolButtons">
                  {presets.map((p) => (
                    <button key={p.name} className="toolBtn" onClick={p.apply} type="button">
                      {p.name}
                    </button>
                  ))}
                </div>

                <div className="miniList">
                  <div className="miniItem">
                    <div className="miniLeft">
                      <div className="miniLabel">Wage Range</div>
                      <div className="miniTime">{wageHint}</div>
                    </div>
                    <div className="miniRight">
                      <div className="miniScore">Employees</div>
                      <div className="miniSub">{empHint}</div>
                    </div>
                  </div>

                  <div className="miniItem">
                    <div className="miniLeft">
                      <div className="miniLabel">Establishment Year</div>
                      <div className="miniTime">{estabHint}</div>
                    </div>
                    <div className="miniRight">
                      <div className="miniScore">Threshold</div>
                      <div className="miniSub">{Number.isFinite(Number(api.threshold)) ? Number(api.threshold).toFixed(2) : "—"}</div>
                    </div>
                  </div>
                </div>
              </>
            ) : null}

            {rightTab === "payload" ? <pre className="jsonbox" dangerouslySetInnerHTML={{ __html: highlightJson(payload) }} /> : null}

            {rightTab === "history" ? (
              <div className="miniList">
                {recent.length ? (
                  recent.map((r, idx) => (
                    <div key={idx} className="miniItem">
                      <div className="miniLeft">
                        <div className="miniLabel">{r.label}</div>
                        <div className="miniTime">{r.at}</div>
                      </div>
                      <div className="miniRight">
                        <div className="miniScore">{formatPct01(r.score)}</div>
                        <div className="miniSub">score</div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="miniItem">
                    <div className="miniLeft">
                      <div className="miniLabel">No history</div>
                      <div className="miniTime">—</div>
                    </div>
                    <div className="miniRight">
                      <div className="miniScore">—</div>
                      <div className="miniSub">—</div>
                    </div>
                  </div>
                )}
              </div>
            ) : null}

            {rightTab === "meta" ? (
              <>
                <div className="miniList">
                  <div className="miniItem">
                    <div className="miniLeft">
                      <div className="miniLabel">Meta Source</div>
                      <div className="miniTime">{meta.source || "—"}</div>
                    </div>
                    <div className="miniRight">
                      <div className="miniScore">Rows</div>
                      <div className="miniSub">{meta.rows ?? "—"}</div>
                    </div>
                  </div>
                </div>

                <details>
                  <summary>Numeric Ranges</summary>
                  <pre className="jsonbox" dangerouslySetInnerHTML={{ __html: highlightJson(meta.numeric_ranges || {}) }} />
                </details>

                <details>
                  <summary>Categorical Options</summary>
                  <pre className="jsonbox" dangerouslySetInnerHTML={{ __html: highlightJson(meta.categorical_options || {}) }} />
                </details>
              </>
            ) : null}
          </Card>
        </div>

        <div className="footer">© Local Pipeline Demo</div>
      </div>
    </div>
  );
}
