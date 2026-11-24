// SPDX-License-Identifier: AGPL-3.0-only
export function createSession({ client, openCamera, readFrame, onState, onResult,
  schedule = callback => setTimeout(callback, 200), cancel = clearTimeout }) {
  let active;
  function stop(state = 'stopped', error) {
    const previous = active;
    active = undefined;
    if (previous) {
      previous.controller.abort();
      cancel(previous.timer);
      previous.camera?.stop();
    }
    client.stop();
    onState({ state, stage: previous?.stage, code: typeof error?.code === 'string' ? error.code : error?.name === 'Error' ? error.message : error?.name });
  }
  async function start() {
    if (active) return;
    const current = { controller: new AbortController(), stage: 'model' };
    active = current;
    const signal = current.controller.signal;
    onState({ state: 'loading' });
    try {
      await client.start({ signal, onProgress: progress => { if (active === current) onState({ state: 'loading', progress }); } });
      signal.throwIfAborted();
      current.stage = 'camera';
      onState({ state: 'camera' });
      const camera = await openCamera(signal);
      if (active !== current || signal.aborted) { camera.stop(); return; }
      current.camera = camera;
      current.stage = 'inference';
      onState({ state: 'running' });
      async function tick() {
        if (active !== current) return;
        try {
          const result = await client.infer(readFrame(), signal);
          if (active !== current) return;
          onResult(result);
          current.timer = schedule(tick);
        } catch (error) { if (active === current) stop('error', error); }
      }
      current.timer = schedule(tick);
    } catch (error) { if (active === current) stop('error', error); }
  }
  return { start, stop };
}
