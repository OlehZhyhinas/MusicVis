// Local persistence in IndexedDB. Every access is wrapped: if IndexedDB is
// unavailable (private mode, blocked storage) the app keeps working in memory.

const DB_NAME = 'musicvis-v2';
const DB_VERSION = 1;
const KV = 'kv';
const THUMBS = 'thumbs';

function req<T>(r: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

export class Store {
  private db: IDBDatabase | null = null;
  private memKv = new Map<string, unknown>();
  private memThumbs = new Map<string, string>();
  available = false;

  async open(): Promise<void> {
    try {
      if (typeof indexedDB === 'undefined') return;
      this.db = await new Promise<IDBDatabase>((resolve, reject) => {
        const r = indexedDB.open(DB_NAME, DB_VERSION);
        r.onupgradeneeded = () => {
          const db = r.result;
          if (!db.objectStoreNames.contains(KV)) db.createObjectStore(KV);
          if (!db.objectStoreNames.contains(THUMBS)) db.createObjectStore(THUMBS);
        };
        r.onsuccess = () => resolve(r.result);
        r.onerror = () => reject(r.error);
        r.onblocked = () => reject(new Error('blocked'));
      });
      this.available = true;
    } catch (err) {
      console.warn('[v2] IndexedDB unavailable, keeping the population in memory', err);
      this.db = null;
    }
  }

  async get<T>(key: string): Promise<T | undefined> {
    if (this.db) {
      try {
        return (await req(this.db.transaction(KV, 'readonly').objectStore(KV).get(key))) as T | undefined;
      } catch (err) {
        console.warn('[v2] read failed', err);
      }
    }
    return this.memKv.get(key) as T | undefined;
  }

  async set(key: string, value: unknown): Promise<void> {
    this.memKv.set(key, value);
    if (!this.db) return;
    try {
      await req(this.db.transaction(KV, 'readwrite').objectStore(KV).put(value, key));
    } catch (err) {
      console.warn('[v2] write failed', err);
    }
  }

  async getThumb(id: string): Promise<string | undefined> {
    const m = this.memThumbs.get(id);
    if (m) return m;
    if (!this.db) return undefined;
    try {
      const v = (await req(this.db.transaction(THUMBS, 'readonly').objectStore(THUMBS).get(id))) as string | undefined;
      if (v) this.memThumbs.set(id, v);
      return v;
    } catch {
      return undefined;
    }
  }

  async setThumb(id: string, url: string): Promise<void> {
    this.memThumbs.set(id, url);
    if (!this.db) return;
    try {
      await req(this.db.transaction(THUMBS, 'readwrite').objectStore(THUMBS).put(url, id));
    } catch (err) {
      console.warn('[v2] thumbnail write failed', err);
    }
  }

  async deleteThumbs(ids: string[]): Promise<void> {
    for (const id of ids) this.memThumbs.delete(id);
    if (!this.db || !ids.length) return;
    try {
      const st = this.db.transaction(THUMBS, 'readwrite').objectStore(THUMBS);
      await Promise.all(ids.map((id) => req(st.delete(id))));
    } catch (err) {
      console.warn('[v2] thumbnail delete failed', err);
    }
  }

  async clearThumbs(): Promise<void> {
    this.memThumbs.clear();
    if (!this.db) return;
    try {
      await req(this.db.transaction(THUMBS, 'readwrite').objectStore(THUMBS).clear());
    } catch (err) {
      console.warn('[v2] thumbnail clear failed', err);
    }
  }
}
