// Safe bridge between the renderer and native capabilities. contextIsolation is
// on, so the renderer only sees exactly what we expose here.
const { contextBridge, ipcRenderer, webUtils } = require('electron');

contextBridge.exposeInMainWorld('lumen', {
  pickFolder: () => ipcRenderer.invoke('pick-folder'),
  pickImage: () => ipcRenderer.invoke('pick-image'),
  openPath: (p) => ipcRenderer.invoke('open-path', p),
  revealPath: (p) => ipcRenderer.invoke('reveal-path', p),
  // Resolve a dropped File to its absolute path (for drag-to-search-by-image).
  getFilePath: (file) => webUtils.getPathForFile(file),
});
