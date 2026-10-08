export const HIDDEN_SESSION_STOP_DELAY_MS = 60_000;

export function shouldScheduleHiddenSessionStop(documentHidden, socketReadyState) {
  return documentHidden && (socketReadyState === 0 || socketReadyState === 1);
}
