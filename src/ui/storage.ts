// localStorage access, wrapped so a private-browsing / quota failure never
// breaks the app.

const PREFIX = 'musicvis:';

export function loadSetting<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    if (raw === null) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function saveSetting<T>(key: string, value: T): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    // ignore (quota exceeded, private mode, disabled storage, etc.)
  }
}
