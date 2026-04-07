import AppKit
import SwiftUI
import AVFoundation

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {

    let backendManager = BackendManager()
    private var logsWindow: NSWindow?

    /// Kept alive for the entire app lifetime.
    /// macOS attributes microphone TCC access to the responsible process (this app).
    /// Holding an active AVAudioEngine input session ensures the audio subsystem stays
    /// open for the whole process group — including the Python subprocess.
    private var micEngine: AVAudioEngine?

    // MARK: - NSApplicationDelegate

    func applicationDidFinishLaunching(_ notification: Notification) {
        configureMainWindow()
        requestMicrophoneAccess()
    }

    func applicationWillTerminate(_ notification: Notification) {
        micEngine?.stop()
        backendManager.stop()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    // MARK: - Microphone permission + audio session

    private func requestMicrophoneAccess() {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {

        case .authorized:
            startAudioSession()

        case .notDetermined:
            // Show the system permission dialog (0.5 s delay so the window is visible first)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    DispatchQueue.main.async {
                        if granted {
                            self.startAudioSession()
                        } else {
                            self.showMicDeniedAlert()
                            self.backendManager.start()
                        }
                    }
                }
            }

        case .denied, .restricted:
            // TCC already has a denied entry — dialog will NOT appear again.
            backendManager.appendLog(
                "⚠️  Microphone access denied for com.felix.jarvis.\n" +
                "   → System Settings → Privacy & Security → Microphone → enable Jarvis\n" +
                "   → Or reset via Terminal: tccutil reset Microphone com.felix.jarvis\n\n"
            )
            showMicDeniedAlert()
            backendManager.start()   // start anyway so the UI appears

        @unknown default:
            backendManager.start()
        }
    }

    /// Starts an AVAudioEngine input tap that runs for the app's lifetime.
    ///
    /// Why: PortAudio (used by Python's sounddevice) needs the parent process to hold
    /// an active CoreAudio input session so that the subprocess is covered by the same
    /// TCC grant. Without this the subprocess gets silence even when "Mic: On" in
    /// System Settings.
    private func startAudioSession() {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let format = inputNode.outputFormat(forBus: 0)

        // Silent tap — we discard all buffers; we only need the session to stay open.
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: format) { _, _ in }

        do {
            try engine.start()
            micEngine = engine    // strong reference keeps it alive
            backendManager.appendLog("🎙  Audio session active (\(Int(format.sampleRate)) Hz)\n\n")
        } catch {
            backendManager.appendLog("⚠️  Audio session failed: \(error.localizedDescription)\n" +
                                     "    Python may not be able to access the microphone.\n\n")
        }

        // Start backend regardless of whether the engine started
        backendManager.start()
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
