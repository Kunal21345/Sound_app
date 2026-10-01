"use client"

import { useEffect, useRef, useState } from "react"
import { Pause, Play, Volume2 } from "lucide-react"
import { createExpressionMask, maskRate, type ExpressionMask } from "@/lib/expression-mask"

type SpeakerProps = {
  src: string
  title: string
  subtitle?: string
  waveformFile?: File
  onError?: (message: string) => void
  mask?: ExpressionMask
}

function timeLabel(seconds: number) {
  if (!Number.isFinite(seconds)) return "0:00"
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`
}

export function Speaker({ src, title, subtitle, waveformFile, onError, mask }: SpeakerProps) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const processingRef = useRef<{ context: AudioContext; graph: ReturnType<typeof createExpressionMask> } | null>(null)
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(70)
  const [peaks, setPeaks] = useState<number[]>([])
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!mask) return
    processingRef.current?.graph.update(mask)
    if (audioRef.current) {
      audioRef.current.preservesPitch = false
      audioRef.current.playbackRate = maskRate(mask)
    }
  }, [mask])

  useEffect(() => () => {
    const processing = processingRef.current
    if (processing) {
      processing.graph.dispose()
      void processing.context.close()
      processingRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!src) return
    let disposed = false
    const abort = new AbortController()
    setPeaks([])
    setBusy(true)
    async function analyze() {
      let context: AudioContext | undefined
      try {
        const buffer = waveformFile ? await waveformFile.arrayBuffer() : await fetch(src, { signal: abort.signal }).then(response => {
          if (!response.ok) throw new Error("Audio unavailable")
          return response.arrayBuffer()
        })
        if (disposed) return
        context = new AudioContext()
        const decoded = await context.decodeAudioData(buffer)
        const channel = decoded.getChannelData(0)
        const bins = 72
        const next = Array.from({ length: bins }, (_, bin) => {
          const start = Math.floor((bin / bins) * channel.length)
          const end = Math.min(channel.length, Math.max(start + 1, Math.floor(((bin + 1) / bins) * channel.length)))
          let sum = 0
          // Sample within each bin to keep long recordings responsive.
          const stride = Math.max(1, Math.floor((end - start) / 1000))
          let count = 0
          for (let i = start; i < end; i += stride) { sum += channel[i] * channel[i]; count += 1 }
          return Math.max(.08, Math.min(1, Math.sqrt(sum / Math.max(1, count)) * 3.8))
        })
        if (!disposed) setPeaks(next)
      } catch {
        if (!disposed) onError?.("Waveform unavailable. You can still play the audio.")
      } finally {
        if (context) await context.close()
        if (!disposed) setBusy(false)
      }
    }
    void analyze()
    return () => { disposed = true; abort.abort() }
  }, [src, waveformFile, onError])

  const toggle = async () => {
    const audio = audioRef.current
    if (!audio || !src) return
    if (audio.paused) {
      try {
        if (mask) {
          if (!processingRef.current) {
            const context = new AudioContext()
            const graph = createExpressionMask(context, context.createMediaElementSource(audio))
            processingRef.current = { context, graph }
          }
          processingRef.current.graph.update(mask, true)
          audio.preservesPitch = false
          audio.playbackRate = maskRate(mask)
          await processingRef.current.context.resume()
        }
        await audio.play()
      } catch { onError?.("The audio preview could not be played.") }
    } else audio.pause()
  }
  const progress = duration ? time / duration : 0
  return <div className="speaker" aria-label={`${title} audio player`}>
    <button className="speaker-play" type="button" disabled={!src} onClick={toggle} aria-label={playing ? "Pause audio" : "Play audio"}>
      {playing ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}
    </button>
    <div className="speaker-info"><div className="speaker-title" title={title}>{title}</div><div className="speaker-meta" title={subtitle}>{subtitle || "Audio preview"}</div></div>
    <div className="waveform-scrubber"><div className="wave-bars" aria-hidden="true">{(peaks.length ? peaks : Array(72).fill(.08)).map((peak, index) => <span key={index} className={progress > 0 && index / 72 <= progress ? "played" : ""} style={{ height: `${peak * 100}%` }} />)}</div>
      <input type="range" min="0" max={duration || 1} step=".01" value={Math.min(time, duration || 1)} disabled={!duration} aria-label="Seek audio" aria-valuetext={`${timeLabel(time)} of ${timeLabel(duration)}`} onChange={event => { const next = Number(event.target.value); if (audioRef.current) audioRef.current.currentTime = next; setTime(next) }} />
    </div>
    <div className="speaker-bottom"><span>{busy ? "Reading waveform…" : `${timeLabel(time)} / ${timeLabel(duration)}`}</span><label className="volume-control"><Volume2 size={13} /><input className="speaker-vol" type="range" min="0" max="100" value={volume} aria-label="Volume" onChange={event => { const next = Number(event.target.value); setVolume(next); if (audioRef.current) audioRef.current.volume = next / 100 }} /></label></div>
    <audio ref={audioRef} src={src || undefined} preload="metadata" onTimeUpdate={event => setTime(event.currentTarget.currentTime)} onLoadedMetadata={event => { setDuration(Number.isFinite(event.currentTarget.duration) ? event.currentTarget.duration : 0); event.currentTarget.volume = volume / 100 }} onEnded={() => setPlaying(false)} onPause={() => setPlaying(false)} onPlay={() => setPlaying(true)} onError={() => onError?.("The audio preview could not be loaded.")} />
  </div>
}
