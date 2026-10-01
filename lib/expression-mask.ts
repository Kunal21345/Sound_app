export type ExpressionMask = {
  expression: string
  energy: number
  pace: number
  variation: number
  clarity: boolean
}

export function maskRate(mask: ExpressionMask) {
  return mask.pace / 100
}

// The source stays untouched; both preview and export use this same graph.
export function createExpressionMask(context: BaseAudioContext, source: AudioNode) {
  const rumble = context.createBiquadFilter()
  rumble.type = "highpass"
  const tone = context.createBiquadFilter()
  tone.type = "highshelf"
  tone.frequency.value = 2200
  const gain = context.createGain()
  const compressor = context.createDynamicsCompressor()
  compressor.threshold.value = -12
  compressor.knee.value = 12
  compressor.ratio.value = 4
  const motion = context.createOscillator()
  const depth = context.createGain()
  motion.connect(depth).connect(gain.gain)
  source.connect(rumble).connect(tone).connect(gain).connect(compressor).connect(context.destination)
  motion.start()

  return {
    update(mask: ExpressionMask, instant = false) {
      const calm = mask.expression === "calm"
      const bright = mask.expression === "excited"
      const level = Math.pow(10, ((mask.energy - 45) * .075 + (calm ? -1.5 : bright ? 1 : 0)) / 20)
      const amount = mask.variation / 100 * (calm ? .025 : bright ? .08 : .045)
      const set = (param: AudioParam, value: number) => {
        param.cancelScheduledValues(context.currentTime)
        if (instant) param.setValueAtTime(value, context.currentTime)
        else param.setTargetAtTime(value, context.currentTime, .015)
      }
      set(rumble.frequency, mask.clarity ? 75 : 10)
      set(tone.gain, calm ? -3 : bright ? 3 : 0)
      set(gain.gain, level)
      set(depth.gain, level * amount)
      set(motion.frequency, calm ? 1.4 : bright ? 3.8 : 2.4)
    },
    dispose() {
      motion.stop()
      for (const node of [source, rumble, tone, gain, compressor, motion, depth]) node.disconnect()
    },
  }
}

export async function exportMaskedWav(src: string, mask: ExpressionMask) {
  const response = await fetch(src)
  if (!response.ok) throw new Error("The generated audio could not be downloaded.")
  const bytes = await response.arrayBuffer()
  const decoder = new AudioContext()
  let buffer: AudioBuffer
  try { buffer = await decoder.decodeAudioData(bytes) } finally { await decoder.close() }
  const context = new OfflineAudioContext(buffer.numberOfChannels, Math.ceil(buffer.length / maskRate(mask)), buffer.sampleRate)
  const source = context.createBufferSource()
  source.buffer = buffer
  source.playbackRate.value = maskRate(mask)
  const graph = createExpressionMask(context, source)
  graph.update(mask, true)
  source.start()
  const rendered = await context.startRendering()
  graph.dispose()
  const channels = rendered.numberOfChannels
  const dataSize = rendered.length * channels * 2
  const wav = new ArrayBuffer(44 + dataSize)
  const view = new DataView(wav)
  const write = (offset: number, text: string) => { for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i)) }
  write(0, "RIFF"); view.setUint32(4, 36 + dataSize, true); write(8, "WAVE")
  write(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true)
  view.setUint16(22, channels, true); view.setUint32(24, rendered.sampleRate, true)
  view.setUint32(28, rendered.sampleRate * channels * 2, true)
  view.setUint16(32, channels * 2, true); view.setUint16(34, 16, true)
  write(36, "data"); view.setUint32(40, dataSize, true)
  const samples = Array.from({ length: channels }, (_, i) => rendered.getChannelData(i))
  for (let frame = 0; frame < rendered.length; frame++) {
    for (let channel = 0; channel < channels; channel++) {
      const value = Math.max(-1, Math.min(1, samples[channel][frame]))
      view.setInt16(44 + (frame * channels + channel) * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true)
    }
  }
  return new Blob([wav], { type: "audio/wav" })
}
