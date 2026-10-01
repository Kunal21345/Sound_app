"use client"

import { useEffect, useRef, useState } from "react"
import { Mic, Square } from "lucide-react"
import { Speaker } from "@/components/speaker"

// PCM WAV works with the reference decoder without browser-specific codecs.
export function encodeWav(chunks: Float32Array[], sampleRate: number): Blob {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0)
  const buffer = new ArrayBuffer(44 + length * 2)
  const view = new DataView(buffer)
  const write = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i))
  }
  write(0, "RIFF"); view.setUint32(4, 36 + length * 2, true)
  write(8, "WAVE"); write(12, "fmt "); view.setUint32(16, 16, true)
  view.setUint16(20, 1, true); view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true); view.setUint16(34, 16, true)
  write(36, "data"); view.setUint32(40, length * 2, true)
  let offset = 44
  for (const chunk of chunks) for (const value of chunk) {
    const sample = Math.max(-1, Math.min(1, value))
    view.setInt16(offset, sample * (sample < 0 ? 32768 : 32767), true)
    offset += 2
  }
  return new Blob([buffer], { type: "audio/wav" })
}

export function RecordSound({ onUse, onError }: { onUse: (file: File) => void; onError: (message: string) => void }) {
  const [recording, setRecording] = useState(false)
  const [pending, setPending] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [clip, setClip] = useState<File | null>(null)
  const [url, setUrl] = useState("")
  const mounted = useRef(false)
  const starting = useRef(false)
  const session = useRef<{ context: AudioContext; stream: MediaStream; source: MediaStreamAudioSourceNode; processor: ScriptProcessorNode; chunks: Float32Array[]; timer: ReturnType<typeof setInterval> } | null>(null)

  const release = () => {
    const active = session.current
    if (!active) return
    session.current = null
    clearInterval(active.timer)
    active.processor.onaudioprocess = null
    active.processor.disconnect()
    active.source.disconnect()
    active.stream.getTracks().forEach(track => track.stop())
    void active.context.close()
  }
  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false; release() }
  }, [])
  useEffect(() => {
    if (!clip) { setUrl(""); return }
    const objectUrl = URL.createObjectURL(clip)
    setUrl(objectUrl)
    return () => URL.revokeObjectURL(objectUrl)
  }, [clip])

  const stop = () => {
    const active = session.current
    if (!active) return
    const duration = active.chunks.reduce((total, chunk) => total + chunk.length, 0) / active.context.sampleRate
    const wav = encodeWav(active.chunks, active.context.sampleRate)
    release()
    setRecording(false)
    if (duration < 3) { onError("Record at least 3 seconds of speech, then try again."); return }
    setClip(new File([wav], "recorded-reference.wav", { type: "audio/wav" }))
  }

  const start = async () => {
    if (starting.current || session.current) return
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) {
      onError("Microphone recording requires a supported browser on HTTPS or localhost.")
      return
    }
    starting.current = true
    setPending(true)
    let stream: MediaStream | null = null
    let context: AudioContext | null = null
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      if (!mounted.current) { stream.getTracks().forEach(track => track.stop()); return }
      context = new AudioContext()
      await context.resume()
      if (!mounted.current) { stream.getTracks().forEach(track => track.stop()); void context.close(); return }
      const source = context.createMediaStreamSource(stream)
      const processor = context.createScriptProcessor(4096, 1, 1)
      const chunks: Float32Array[] = []
      processor.onaudioprocess = event => { chunks.push(new Float32Array(event.inputBuffer.getChannelData(0))) }
      source.connect(processor)
      processor.connect(context.destination)
      const started = performance.now()
      const timer = setInterval(() => {
        setSeconds(Math.floor((performance.now() - started) / 1000))
        if (performance.now() - started >= 60000) stop()
      }, 250)
      session.current = { context, stream, source, processor, chunks, timer }
      setClip(null)
      setSeconds(0)
      setRecording(true)
    } catch (error) {
      stream?.getTracks().forEach(track => track.stop())
      if (context && context.state !== "closed") void context.close()
      if (mounted.current) onError(error instanceof DOMException && error.name === "NotAllowedError"
        ? "Allow microphone access in your browser to record your voice."
        : "Could not start recording. Check that your microphone is connected and available.")
    } finally {
      starting.current = false
      if (mounted.current) setPending(false)
    }
  }

  return <div className="record-sound">
    <div className="record-actions">
      <button type="button" className="record-button" disabled={pending} onClick={recording ? stop : () => void start()}>
        {recording ? <Square size={16} /> : <Mic size={16} />}
        {pending ? "Connecting microphone…" : recording ? "Stop recording" : clip ? "Record again" : "Record sound"}
      </button>
      <span role="status">{recording ? `Recording · ${seconds}s / 60s` : "Record 3–60 seconds of clear speech."}</span>
    </div>
    {clip && <div className="record-preview">
      <Speaker key={url} src={url} title="Your recording" subtitle="Preview before using" waveformFile={clip} onError={onError} />
      <div className="record-actions"><button type="button" className="record-button" onClick={() => { onUse(clip); setClip(null) }}>Use recording</button><button type="button" className="text-button" onClick={() => setClip(null)}>Discard</button></div>
    </div>}
  </div>
}
