class PCM16Processor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.buffer = [];
    this.phase = 0;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;

    let i = this.phase;
    while (i < input.length) {
      const s = Math.max(-1, Math.min(1, input[Math.floor(i)]));
      this.buffer.push(s < 0 ? s * 0x8000 : s * 0x7fff);
      i += this.ratio;
    }
    this.phase = i - input.length;

    if (this.buffer.length >= 1600) { // 100 ms at 16 kHz
      const chunk = new Int16Array(this.buffer.splice(0, 1600));
      this.port.postMessage(chunk.buffer, [chunk.buffer]);
    }
    return true;
  }
}
registerProcessor("pcm16-processor", PCM16Processor);