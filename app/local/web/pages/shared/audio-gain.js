export function createMediaElementGainController(mediaElements) {
  const configuredGains = new Map(
    mediaElements.map(({ id }) => [id, 1]),
  );
  const gainNodes = new Map();
  let audioContext = null;

  async function ensureConnected() {
    if (audioContext === null) {
      audioContext = new AudioContext();
      for (const { id, element } of mediaElements) {
        const sourceNode = audioContext.createMediaElementSource(element);
        const gainNode = audioContext.createGain();
        gainNode.gain.value = configuredGains.get(id);
        sourceNode.connect(gainNode);
        gainNode.connect(audioContext.destination);
        gainNodes.set(id, gainNode);
      }
    }
    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }
  }

  function setGain(id, gain) {
    configuredGains.set(id, gain);
    const gainNode = gainNodes.get(id);
    if (gainNode !== undefined) {
      gainNode.gain.value = gain;
    }
  }

  return { ensureConnected, setGain };
}
