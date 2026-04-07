import AppKit
import SwiftUI
import AVFoundation
import AVFAudio

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {

    let backendManager = BackendManager()
    private var logsWindow: NSWindow?

    // MARK: - NSApplicationDelegate

    func applicationDidFinishLaunching(_ notification: Notification) {
        configureMainWindow()
        requestMicrophoneAccess()
    }

    func applicationWillTerminate(_ notification: Notification) {
        backendManager.stop()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    // MARK: - Microphone permission
    // Use AVAudioApplication on macOS 14+ (the recommended API for subprocess attribution).
    // Fall back to AVCaptureDevice on macOS 13.

    private func requestMicrophoneAccess() {
        if #available(macOS 14.0, *) {
            switch AVAudioApplication.shared.recordPermission {
            case .granted:
                runAudioProbe()
            case .undetermined:
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    AVAudioApplication.requestRecordPermission { granted in
                        DispatchQueue.main.async {
                            if granted { self.runAudioProbe() }
                            else        { self.showMicDeniedAlert(); self.backendManager.start() }
                        }
                    }
                }
            case .denied:
                showMicDeniedAlert()
                backendManager.start()
            @unknown default:
                runAudioProbe()
            }
        } else {
            // macOS 13 fallback
            switch AVCaptureDevice.authorizationStatus(for: .audio) {
            case .authorized:
                runAudioProbe()
            case .notDetermined:
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    AVCaptureDevice.requestAccess(for: .audio) { granted in
                        DispatchQueue.main.async {
                            if granted { self.runAudioProbe() }
                            else        { self.showMicDeniedAlert(); self.backendManager.start() }
                        }
                    }
                }
            case .denied, .restricted:
                showMicDeniedAlert()
                backendManager.start()
            @unknown default:
                runAudioProbe()
            }
        }
    }

    // MARK: - Audio probe
    //
    // Runs a 1-second Python snippet that:
    //   1. Prints the selected input device name + sample rate
    //   2. Records 1 s and prints the RMS level
    //      → RMS ≈ 0  : device open but receiving silence (wrong device / TCC blocked)
    //      → RMS > 50 : real audio arriving — VAD should work
    //
    // This also warms up the Python binary's TCC attribution under Jarvis.app so
    // the main script inherits it without a second dialog.

    private func runAudioProbe() {
        let root   = backendManager.projectRoot
        let python = root + "/.venv311/bin/python"

        guard FileManager.default.fileExists(atPath: python) else {
            backendManager.appendLog("⚠️  Python not found — skipping audio probe\n\n")
            backendManager.start()
            return
        }

        let probe = #"""
import sys, time
try:
    import sounddevice as sd, numpy as np

    dev = sd.query_devices(kind='input')
    print(f"[probe] device : {dev['name']}", flush=True)
    print(f"[probe] hw rate: {int(dev['default_samplerate'])} Hz", flush=True)

    frames = []
    def _cb(data, n, t, s):
        frames.append(bytes(data))

    with sd.RawInputStream(samplerate=16000, blocksize=480,
                           dtype='int16', channels=1, callback=_cb):
        time.sleep(1.0)   # record for 1 second — speak now if you want to calibrate

    if frames:
        arr = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(arr ** 2)))
        status = "✅ audio OK" if rms > 50 else "⚠️  SILENT — wrong device or TCC blocked"
        print(f"[probe] RMS    : {rms:.1f}  {status}", flush=True)
    else:
        print("[probe] ⚠️  no frames captured", flush=True)

except Exception as e:
    print(f"[probe] ERROR: {e}", file=sys.stderr, flush=True)
"""#

        let proc = Process()
        proc.executableURL       = URL(fileURLWithPath: python)
        proc.arguments           = ["-c", probe]
        proc.currentDirectoryURL = URL(fileURLWithPath: root)

        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        proc.environment = env

        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError  = pipe

        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let str = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor [weak self] in self?.backendManager.appendLog(str) }
        }

        proc.terminationHandler = { [weak self] _ in
            pipe.fileHandleForReading.readabilityHandler = nil
            Task { @MainActor [weak self] in
                self?.backendManager.appendLog("\n")
                self?.backendManager.start()
            }
        }

        backendManager.appendLog("▶  Audio probe (1 s) — speak into the mic now to calibrate…\n")
        do {
            try proc.run()
        } catch {
            backendManager.appendLog("Probe launch failed: \(error)\n")
            backendManager.start()
        }
    }

    // MARK: - Mic-denied alert

    private func showMicDeniedAlert() {
        let alert = NSAlert()
        alert.alertStyle      = .warning
        alert.messageText     = "Microphone Access Needed"
        alert.informativeText =
            "Jarvis can't hear you because microphone access was denied.\n\n" +
            "Open System Settings → Privacy & Security → Microphone and switch on Jarvis.\n\n" +
            "If only 'JarvisApp' (old build, no icon) appears — disable it, then run:\n" +
            "  tccutil reset Microphone com.felix.jarvis\n" +
            "and relaunch Jarvis."
        alert.addButton(withTitle: "Open System Settings")
        alert.addButton(withTitle: "Ignore")
        if alert.runModal() == .alertFirstButtonReturn {
            NSWorkspace.shared.open(
                URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")!)
        }
    }

    // MARK: - Window configuration

    private func configureMainWindow() {
        guard let window = NSApp.windows.first else { return }
        window.minSize                     = NSSize(width: 400, height: 400)
        window.titleVisibility             = .hidden
        window.titlebarAppearsTransparent  = true
        window.styleMask.insert(.fullSizeContentView)
        window.isMovableByWindowBackground = true
        window.setFrameAutosaveName("JarvisMainWindow")
    }

    // MARK: - Logs window

    func showLogsWindow() {
        if let w = logsWindow { w.makeKeyAndOrderFront(nil); return }
        let controller = NSHostingController(
            rootView: LogsView().environmentObject(backendManager)
        )
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 720, height: 480),
            styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered, defer: false
        )
        window.title = "Jarvis — Backend Logs"
        window.contentViewController = controller
        window.center()
        window.makeKeyAndOrderFront(nil)
        logsWindow = window
    }
}
