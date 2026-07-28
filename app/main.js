// Electron main process: spawn the Python sidecar, then open a window pointed
// at it. The renderer is served BY the sidecar (static files at /), so the whole
// app is one origin — no CORS, no file:// quirks.
const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const { spawn } = require('child_process');
const net = require('net');
const http = require('http');
const path = require('path');
const process = require('process');

const REPO_ROOT = path.join(__dirname, '..');

let pyProc = null;
let serverPort = null;

function pythonBin() {
  // dev layout: the project venv. (Packaging will bundle its own runtime later.)
  return process.platform === 'win32'
    ? path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe') 
    : path.join(REPO_ROOT, '.venv', 'bin', 'python');
}

function findFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref(); // don't keep the node process alive just for this server
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port)); //close the server and resolve with the port
    });
  });
}

function startSidecar(port) {
  pyProc = spawn(pythonBin(), ['server.py', '--port', String(port)], {
    cwd: REPO_ROOT,
    stdio: ['ignore', 'pipe', 'pipe'], //index 0 for stdin, 1 for stdout, 2 for stderr
    //
  });
  pyProc.stdout.on('data', (d) => console.log('[sidecar]', d.toString().trim()));
  pyProc.stderr.on('data', (d) => console.error('[sidecar]', d.toString().trim()));
  pyProc.on('exit', (code) => console.log('[sidecar] exited', code));
}

function waitForHealth(port, timeoutMs = 40000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const attempt = () => {
      const req = http.get(
        { host: '127.0.0.1', port, path: '/api/health', timeout: 1000 },
        (res) => {
          res.resume();
          if (res.statusCode === 200) resolve();
          else retry();
        }
      );
      req.on('error', retry);
      req.on('timeout', () => { req.destroy(); retry(); });
    };
    const retry = () => {
      if (Date.now() - start > timeoutMs) reject(new Error('sidecar did not become healthy'));
      else setTimeout(attempt, 300);
    };
    attempt();
  });
}

function createWindow(port) {
  const win = new BrowserWindow({
    width: 1240,
    height: 840,
    minWidth: 820,
    minHeight: 560,
    title: 'Lumen',
    backgroundColor: '#0f1115',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadURL(`http://127.0.0.1:${port}/`);
}

app.whenReady().then(async () => {
  try {
    serverPort = await findFreePort();
    startSidecar(serverPort);
    await waitForHealth(serverPort);
    console.log(`Server ready at http://127.0.0.1:${serverPort}/`);
  } catch (e) {
    dialog.showErrorBox('Lumen failed to start', String(e));
  }
  createWindow(serverPort);

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow(serverPort);
  }); 
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('quit', () => {
  if (pyProc) pyProc.kill();
});

// ---- IPC: native things the renderer can't do itself ----
ipcMain.handle('pick-folder', async () => {
  const r = await dialog.showOpenDialog({ properties: ['openDirectory'] });
  return r.canceled ? null : r.filePaths[0];
});

ipcMain.handle('pick-image', async () => {
  const r = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [{ name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff', 'heic'] }],
  });
  return r.canceled ? null : r.filePaths[0];
});

ipcMain.handle('open-path', async (_e, p) => { await shell.openPath(p); });
ipcMain.handle('reveal-path', async (_e, p) => { shell.showItemInFolder(p); });
