export function commonWaveformDisplayScale(waveforms) {
  const peakAmplitude = waveforms.reduce(
    (overallMaximum, waveform) =>
      waveform.points.reduce(
        (waveformMaximum, point) =>
          Math.max(
            waveformMaximum,
            Math.abs(point.minimum_amplitude * waveform.gain),
            Math.abs(point.maximum_amplitude * waveform.gain),
          ),
        overallMaximum,
      ),
    0.01,
  );
  return 1 / peakAmplitude;
}
