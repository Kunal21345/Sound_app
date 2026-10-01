"use client"

import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react"
import type { ChangeEvent, DragEvent } from "react"
import { AudioLines, Check, ChevronRight, Download, FileAudio2, Loader2, Settings2, Sparkles, UploadCloud, X } from "lucide-react"
import { exportMaskedWav } from "@/lib/expression-mask"
import { Speaker } from "@/components/speaker"
import { RecordSound } from "@/components/record-sound"

const MAX_BYTES = 25 * 1024 * 1024

export default function Studio() {
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [script, setScript] = useState("")
  const [expression, setExpression] = useState("calm")
  const [energy, setEnergy] = useState(45)
  const [pace, setPace] = useState(100)
  const [variation, setVariation] = useState(30)
  const [downloading, setDownloading] = useState(false)
  const [generatedLabel, setGeneratedLabel] = useState("")
  const [autoPunctuation, setAutoPunctuation] = useState(true)
  const [clarity, setClarity] = useState(true)
  const mask = useMemo(() => ({ expression, energy, pace, variation, clarity }), [expression, energy, pace, variation, clarity])
  const [generated, setGenerated] = useState("")
  const [status, setStatus] = useState("")
  const [loading, setLoading] = useState(false)
  const [toast, setToast] = useState("")

  const [sourceUrl, setSourceUrl] = useState("")
  const [drawerOpen, setDrawerOpen] = useState(false)
  const drawerRef = useRef<HTMLDialogElement>(null)
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (!file) { setSourceUrl(""); return }
    const url = URL.createObjectURL(file)
    setSourceUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])
  useEffect(() => {
    if (!drawerOpen) return
    const previous = document.body.style.overflow
    document.body.style.overflow = "hidden"
    const media = window.matchMedia("(min-width: 901px)")
    const closeOnDesktop = () => { if (media.matches) drawerRef.current?.close() }
    media.addEventListener("change", closeOnDesktop)
    return () => {
      document.body.style.overflow = previous
      media.removeEventListener("change", closeOnDesktop)
    }
  }, [drawerOpen])
  useEffect(() => () => { if (toastTimer.current) clearTimeout(toastTimer.current) }, [])

  const notify = useCallback((message: string) => {
    setToast(message)
    if (toastTimer.current) clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setToast(""), 4000)
  }, [])
  const receiveFile = (next: File | undefined) => {
    if (!next) return
    if (!next.type.startsWith("audio/")) return notify("Choose an audio file to continue")
    if (next.size > MAX_BYTES) return notify("That file is larger than 25 MB")
    setFile(next)
  }
  const fileChanged = (event: ChangeEvent<HTMLInputElement>) => receiveFile(event.target.files?.[0])
  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault()
    setIsDragging(false)
    receiveFile(event.dataTransfer.files?.[0])
  }

  const generate = async () => {
    const text = script.trim()
    if (loading) return
    if (!file) return notify("Upload or record a voice before generating")
    if (!text) return notify("Add a script before generating")
    setLoading(true)
    setStatus("Creating your audio…")
    setGenerated("")
    try {
      const body = new FormData()
      body.append("reference", file)
      body.append("question", text)
      body.append("expression", "neutral")
      body.append("energy", "45")
      body.append("pace", "1")
      body.append("variation", "0.3")
      body.append("autopunct", String(autoPunctuation))
      body.append("clarity", "false")
      const response = await fetch("/api/ask", { method: "POST", body })
      let data: { error?: string; audio_url?: string }
      try {
        data = await response.json()
      } catch {
        throw new Error("Speech service is unavailable. Start the Flask server on port 5000 and try again.")
      }
      if (!response.ok) throw new Error(data.error || "Request failed")
      if (!data.audio_url) throw new Error("No generated audio was returned")
      setGeneratedLabel(file.name)
      setGenerated(data.audio_url)
      setStatus("Generated speech")
      notify("Your speech is ready")
    } catch (error) {
      const rawMessage = error instanceof Error ? error.message : "Generation failed"
      const message = rawMessage.includes("Failed to fetch")
        ? "Speech service is unavailable. Start the Flask server on port 5000 and try again."
        : rawMessage
      setStatus(message)
      notify(message)
    } finally {
      setLoading(false)
    }
  }

  const download = async () => {
    if (!generated || downloading) return
    setDownloading(true)
    try {
      const blob = await exportMaskedWav(generated, mask)
      const url = URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = url
      link.download = "generated-speech.wav"
      link.click()
      setTimeout(() => URL.revokeObjectURL(url), 60_000)
    } catch (error) {
      notify(error instanceof Error ? error.message : "Download failed")
    } finally { setDownloading(false) }
  }

  const controls = <>
    <div className="control-intro"><span className="mini-label">YOUR VOICE</span><div className="voice-choice"><span className="voice-avatar"><AudioLines size={19} /></span><div><strong>{file ? "Reference voice" : "No voice selected"}</strong><span title={file?.name}>{file?.name || "Upload or record to get started"}</span></div>{file && <Check size={16} />}</div></div>
    <div className="control-section"><div className="control-label"><span>Expression</span><span className="control-caption">Delivery style</span></div><div className="segmented" aria-label="Expression">{[["calm", "Calm"], ["neutral", "Neutral"], ["excited", "Bright"]].map(([value, label]) => <button key={value} type="button" aria-pressed={expression === value} className={expression === value ? "selected" : ""} onClick={() => setExpression(value)}>{label}</button>)}</div></div>
    <Range label="Energy" left="Gentle" right="Dynamic" value={energy} onChange={setEnergy} format={(n) => `${n}%`} />
    <Range label="Speaking pace" left="Slower" right="Faster" value={pace} min={70} max={130} onChange={setPace} format={(n) => `${(n / 100).toFixed(1)}×`} />
    <Range label="Variation" left="Consistent" right="Expressive" value={variation} onChange={setVariation} format={(n) => `${n}%`} />
    <div className="control-toggles"><Toggle title="Auto-punctuation" detail="Natural pauses between lines" checked={autoPunctuation} onChange={setAutoPunctuation} /><Toggle title="Enhance clarity" detail="Reduce low-frequency rumble" checked={clarity} onChange={setClarity} /></div>
    <p className="settings-note">Expression, energy, pace, variation, and clarity apply instantly to your audio. Auto-punctuation applies when generating.</p>
  </>

  const openControls = () => { drawerRef.current?.showModal(); setDrawerOpen(true) }
  return <div className="studio-shell">
    <div className="page-heading"><div><div className="eyebrow">CREATE WITH YOUR VOICE</div><h1>Text to speech<span>.</span></h1><p>A familiar voice. A new story. Make every word your own.</p></div><span className="workspace-badge"><span /> Voice studio</span></div>
    <div className="studio-grid"><main className="studio-main">
      <section className="surface reference-section" aria-labelledby="reference-title"><div className="section-heading"><div><span className="step-number">01</span><h2 id="reference-title">Your voice</h2></div><span className="section-note">Reference audio</span></div>
        {!file ? <label className={`upload-zone ${isDragging ? "is-dragging" : ""}`} onDragOver={(event) => { event.preventDefault(); setIsDragging(true) }} onDragLeave={() => setIsDragging(false)} onDrop={handleDrop}>
          <span className="upload-icon"><UploadCloud size={23} strokeWidth={1.5} /></span><strong>Drop your voice here <span>or browse files</span></strong><span className="upload-formats">WAV, MP3, M4A · Up to 25 MB</span><input type="file" accept="audio/*" aria-label="Upload reference audio" onChange={fileChanged} /><span className="upload-tip">Use a clear recording with at least 3 seconds of speech.</span>
        </label> : <div className="reference-preview"><div className="reference-meta"><span><FileAudio2 size={16} />{(file.size / 1024 / 1024).toFixed(2)} MB <span className="reference-ready"><Check size={12} /> Ready to use</span></span><button className="icon-button" onClick={() => setFile(null)} aria-label="Remove reference audio"><X size={17} /></button></div><Speaker key={sourceUrl} src={sourceUrl} title={file.name} subtitle="Your reference voice" waveformFile={file} onError={notify} /></div>}
        <RecordSound onUse={receiveFile} onError={notify} />
      </section>
      <section className="surface script-section" aria-labelledby="script-title"><div className="section-heading"><div><span className="step-number">02</span><h2 id="script-title">What should it say?</h2></div><button className="text-button" disabled={!script || loading} onClick={() => setScript("")}>Clear text</button></div>
        <textarea aria-labelledby="script-title" value={script} onChange={(event) => setScript(event.target.value)} onKeyDown={(event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { event.preventDefault(); void generate() } }} placeholder="Every great story starts with a few words…

Write or paste your script here, and bring it to life in your own voice." />
        <div className="editor-meta"><span>{script.length.toLocaleString()} <span>characters</span></span><span className="keyboard-hint">⌘ / Ctrl + Enter</span></div>
        <div className="editor-actions"><button className="settings-trigger" aria-label="Voice settings" onClick={openControls} aria-haspopup="dialog" aria-expanded={drawerOpen}><Settings2 size={17} /><span>Voice settings</span><ChevronRight size={15} /></button><span className="desktop-generation-note"><AudioLines size={15} /> Made with your reference voice</span><button className="generate-button" disabled={loading || !file || !script.trim()} onClick={generate}>{loading ? <Loader2 size={17} className="spin" /> : <Sparkles size={17} />}{loading ? "Generating…" : "Generate speech"}</button></div>
      </section>
      <section className={`surface output-section ${generated ? "has-result" : ""}`} aria-labelledby="output-title"><div className="section-heading"><div><span className="step-number">03</span><h2 id="output-title">Your audio</h2></div>{generated ? <button type="button" className="download-button" onClick={download} disabled={downloading}>{downloading ? <Loader2 size={16} className="spin" /> : <Download size={16} />}<span>{downloading ? "Saving…" : "Download"}</span><span className="format-tag">WAV</span></button> : <span className="section-note">Ready when you are</span>}</div>
        {generated ? <div className="output-player"><Speaker key={generated} src={generated} title="Generated speech" subtitle={`${expression === "excited" ? "Bright" : expression === "calm" ? "Calm" : "Neutral"} · ${generatedLabel}`} mask={mask} onError={notify} /><div className="output-caption"><span className="success-dot" /> Ready to listen, save, and share.</div></div> : <div className="output-empty"><span className={`empty-wave ${loading ? "is-loading" : ""}`} aria-hidden="true">{[10, 22, 35, 19, 48, 30, 42, 17, 32, 22, 10].map((height, index) => <i key={index} style={{ height, animationDelay: `${index * .08}s` }} />)}</span><strong>{loading ? "Bringing your words to life" : "A little text. A lot of possibility."}</strong><p>{loading ? "Your voice is taking shape. This can take a moment." : "Generate speech to preview and download it here."}</p></div>}
        {status && status !== "Generated speech" && !loading && <p className="generation-error" role="alert">{status}</p>}
      </section>
      <p className="workspace-footnote">Your voice, your expression. Created locally.</p>
    </main>
    <aside className="desktop-controls surface" aria-labelledby="desktop-settings-title"><div className="settings-heading"><h2 id="desktop-settings-title">Voice settings</h2><Settings2 size={18} /></div>{controls}</aside>
    </div>
    <dialog ref={drawerRef} className="settings-drawer" aria-labelledby="drawer-title" onClose={() => setDrawerOpen(false)} onClick={(event) => { const rect = event.currentTarget.getBoundingClientRect(); if (event.target === event.currentTarget && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) event.currentTarget.close() }}>
      <div className="drawer-heading"><div><span className="mini-label">MAKE IT YOURS</span><h2 id="drawer-title">Voice settings</h2></div><button className="icon-button" onClick={() => drawerRef.current?.close()} aria-label="Close voice settings"><X size={21} /></button></div><div className="drawer-content">{controls}</div><div className="drawer-footer"><button className="generate-button" onClick={() => drawerRef.current?.close()}><Check size={17} />Done</button></div>
    </dialog>
    <div className={`toast ${toast ? "show" : ""}`} role="status" aria-live="polite">{toast}</div>
  </div>
}

function Range({ label, left, right, value, min = 0, max = 100, onChange, format }: { label: string; left: string; right: string; value: number; min?: number; max?: number; onChange: (n: number) => void; format: (n: number) => string }) {
  const id = useId()
  return <div className="control-section"><div className="control-label"><label htmlFor={id}>{label}</label><output htmlFor={id}>{format(value)}</output></div><input id={id} className="setting-range" type="range" min={min} max={max} value={value} style={{ backgroundImage: `linear-gradient(to right, var(--foreground) ${(value - min) / (max - min) * 100}%, var(--border) ${(value - min) / (max - min) * 100}%)` }} onChange={(event) => onChange(Number(event.target.value))} /><div className="range-hints"><span>{left}</span><span>{right}</span></div></div>
}

function Toggle({ title, detail, checked, onChange }: { title: string; detail: string; checked: boolean; onChange: (v: boolean) => void }) {
  return <label className="toggle-row"><span><strong>{title}</strong><span>{detail}</span></span><input className="setting-switch" role="switch" type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /></label>
}
