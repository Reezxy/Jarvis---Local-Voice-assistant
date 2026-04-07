import AppKit
import SwiftUI
import AVFoundation

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

    private func requestMicrophoneAccess() {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {

        case .authorized:
            runAudioProbe()

        case .notDetermined:
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    DispatchQueue.main.async {
                        if granted {
                            self.runAudioProbe()
                        } else {
                            self.showMicDeniedAlert()
                            self.backendManager.start()
                        }
                    }
                }
            }

        case .denied, .restricted:
            backendManager.appendLog(
                "⚠️  Microphone access denied for com.felix.jarvis.\n" +
                "   → System Settings → Privacy & Security → Microphone → enable Jarvis\n" +
                "   → Or reset via Terminal: tccutil reset Microphone com.felix.jarvis\n\n"
            )
            showMicDeniedAlert()
            backendManager.start()

        @unknown default:
            backendManager.start()
        }
    }

    // MARK: - Audio probe
    //
    // Runs a tiny Python snippet that opens the microphone for 0.15 s and
    // immediately closes it.  Two purposes:
    //
    //  1. If macOS hasn't yet granted TCC access to the Python binary itself
    //     (separate from the Swift app), it will show the system prompt here —
    //     BEFORE the main script starts.
    //
    //  2. It verifies that sounddevice can actually open the device and records
    //     the result in the logs.
    //
    // Note: we do NOT keep AVAudioEngine running in Swift.  That would hold the
    // audio device at 44100 Hz and prevent Python/PortAudio from opening it at
    // 16000 Hz — causing silent captures.

    private func runAudioProbe() {
        let root      = backendManager.projectRoot
        let python    = root + "/.venv311/bin/python"

        guard FileManager.default.fileExists(atPath: python) else {
            backendManager.appendLog("⚠️  Python not found — skipping audio probe\n\n")
            backendManager.start()
            return
        }

        let probe = """
import sys
try:
    import sounddevice as sd, time
    dev = sd.query_devices(kind='input')
    print(f"[probe] input device: {dev['name']}  rate: {int(dev['default_samplerate'])} Hz", flush=True)
    with sd.RawInputStream(samplerate=16000, blocksize=480, dtype='int16', channels=1):
        time.sleep(0.15)
    print("[probe] audio OK", flush=True)
except Exception as e:
    print(f"[probe] ERROR: {e}", file=sys.stderr, flush=True)
"""

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

        backendManager.appendLog("▶  Running audio probe…\n")
        do {
            try proc.run()
        } catch {
            backendManager.appendLog("Audio probe launch failed: \(error)\n")
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
            "If Jarvis doesn't appear in the list, run this once in Terminal:\n" +
            "  tccutil reset Microphone com.felix.jarvis"
        alert.addButton(withTitle: "Open System Settings")
        alert.addButton(withTitle: "Ignore")

        if alert.runModal() == .alertFirstButtonReturn {
            let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")!
            NSWorkspace.shared.open(url)
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
        if let w = logsWindow {
            w.makeKeyAndOrderFront(nil)
            return
        }
        let controller = NSHostingController(
            rootView: LogsView().environmentObject(backendManager)
        )
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 720, height: 480),
            styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Jarvis — Backend Logs"
        window.contentViewController = controller
        window.center()
        window.makeKeyAndOrderFront(nil)
        logsWindow = window
    }
}
