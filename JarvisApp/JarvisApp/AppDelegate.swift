import AppKit
import SwiftUI
import AVFoundation
import AVFAudio

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {

    let backendManager = BackendManager()
    private var logsWindow: NSWindow?
    /// Run the mic flow once; wait for a SwiftUI window so the system permission sheet can attach.
    private var didScheduleMicrophoneFlow = false
    /// Einmaliger Warmup, damit TCC / Mikrofon-Gerät wirklich angebunden wird (nicht nur requestAccess).
    private var didRunMicCaptureWarmup = false

    // MARK: - NSApplicationDelegate

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.activate(ignoringOtherApps: true)
        configureMainWindow()
    }

    func applicationDidBecomeActive(_ notification: Notification) {
        guard !didScheduleMicrophoneFlow else { return }
        didScheduleMicrophoneFlow = true
        waitForWindowThenRequestMicrophone(retry: 0)
    }

    private func waitForWindowThenRequestMicrophone(retry: Int) {
        if NSApp.windows.isEmpty, retry < 40 {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                self.waitForWindowThenRequestMicrophone(retry: retry + 1)
            }
            return
        }
        DispatchQueue.main.async {
            self.configureMainWindow()
            // Python sofort; Mikrofon kurz verzögert, damit das System-Sheet nicht hinter dem Fenster hängt.
            self.backendManager.start()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
                self.requestMicrophoneAccess()
                // Async-Callbacks von requestAccess können später feuern — nochmal prüfen.
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
                    self.runMicCaptureWarmupIfAuthorized(reason: "verzögert nach Berechtigungs-Callbacks")
                }
            }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        backendManager.stop()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    // MARK: - Microphone permission
    //
    // • AVCaptureDevice: klassischer Eintrag unter Datenschutz → Mikrofon.
    // • AVAudioApplication (macOS 14+): Zuordnung für Audio/Unterprozesse (Python + sounddevice).

    private func requestMicrophoneAccess() {
        NSApp.activate(ignoringOtherApps: true)
        let front = NSApp.keyWindow ?? NSApp.mainWindow ?? NSApp.windows.first
        front?.makeKeyAndOrderFront(nil)
        NSApp.requestUserAttention(.informationalRequest)

        let cap = AVCaptureDevice.authorizationStatus(for: .audio)
        backendManager.appendLog("[Mic] AVCapture (Mikrofon): \(Self.describeCaptureStatus(cap))\n")

        if #available(macOS 14.0, *) {
            let rec = AVAudioApplication.shared.recordPermission
            backendManager.appendLog("[Mic] AVAudioApplication (Aufnahme): \(Self.describeRecordPermission(rec))\n")
            if cap == .denied || cap == .restricted || rec == .denied {
                backendManager.appendLog("[Mic] Verweigert — kein System-Dialog mehr; Einstellungen oder tccutil reset.\n")
                presentMicDeniedNeedsSettingsAlert()
                return
            }
            if rec == .undetermined {
                backendManager.appendLog("[Mic] Fordere AVAudioApplication.requestRecordPermission an …\n")
                AVAudioApplication.requestRecordPermission { granted in
                    DispatchQueue.main.async {
                        self.backendManager.appendLog("[Mic] AVAudioApplication Ergebnis: \(granted ? "erlaubt" : "abgelehnt")\n")
                        self.runMicCaptureWarmupIfAuthorized(reason: "nach AVAudioApplication")
                    }
                }
            }
            if cap == .notDetermined {
                backendManager.appendLog("[Mic] Fordere AVCaptureDevice.requestAccess(audio) an …\n")
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    DispatchQueue.main.async {
                        self.backendManager.appendLog("[Mic] AVCapture Ergebnis: \(granted ? "erlaubt" : "abgelehnt")\n")
                        self.runMicCaptureWarmupIfAuthorized(reason: "nach AVCapture requestAccess")
                    }
                }
            } else if cap == .authorized {
                runMicCaptureWarmupIfAuthorized(reason: "AVCapture war schon authorized")
            }
            return
        }

        switch cap {
        case .authorized:
            runMicCaptureWarmupIfAuthorized(reason: "AVCapture war schon authorized")
        case .notDetermined:
            backendManager.appendLog("[Mic] Fordere AVCaptureDevice.requestAccess(audio) an …\n")
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                DispatchQueue.main.async {
                    self.backendManager.appendLog("[Mic] AVCapture Ergebnis: \(granted ? "erlaubt" : "abgelehnt")\n")
                    self.runMicCaptureWarmupIfAuthorized(reason: "nach AVCapture requestAccess")
                }
            }
        case .denied, .restricted:
            backendManager.appendLog("[Mic] Verweigert — kein System-Dialog mehr.\n")
            presentMicDeniedNeedsSettingsAlert()
        @unknown default:
            break
        }
    }

    /// Kurz die Capture-Pipeline öffnen — triggert TCC/Zuordnung zuverlässiger als nur `requestAccess`.
    private func runMicCaptureWarmupIfAuthorized(reason: String) {
        guard !didRunMicCaptureWarmup else { return }
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else { return }
        didRunMicCaptureWarmup = true
        backendManager.appendLog("[Mic] Warmup (\(reason)): starte AVCaptureSession (~0,6 s) …\n")

        DispatchQueue.global(qos: .userInitiated).async {
            let session = AVCaptureSession()
            session.beginConfiguration()
            if session.canSetSessionPreset(.medium) {
                session.sessionPreset = .medium
            }
            guard let device = AVCaptureDevice.default(for: .audio) else {
                DispatchQueue.main.async {
                    self.backendManager.appendLog("[Mic] Warmup: kein Standard-Audiogerät.\n")
                }
                return
            }
            do {
                let input = try AVCaptureDeviceInput(device: device)
                guard session.canAddInput(input) else {
                    DispatchQueue.main.async {
                        self.backendManager.appendLog("[Mic] Warmup: canAddInput == false.\n")
                    }
                    return
                }
                session.addInput(input)
            } catch {
                DispatchQueue.main.async {
                    self.backendManager.appendLog("[Mic] Warmup: \(error.localizedDescription)\n")
                }
                return
            }
            session.commitConfiguration()
            session.startRunning()
            Thread.sleep(forTimeInterval: 0.65)
            session.stopRunning()
            DispatchQueue.main.async {
                self.backendManager.appendLog("[Mic] Warmup beendet — prüfe Datenschutz → Mikrofon auf „Jarvis“.\n")
            }
        }
    }

    private static func describeCaptureStatus(_ s: AVAuthorizationStatus) -> String {
        switch s {
        case .notDetermined: return "notDetermined (System-Dialog möglich)"
        case .restricted:    return "restricted"
        case .denied:        return "denied"
        case .authorized:    return "authorized"
        @unknown default:    return "unknown"
        }
    }

    @available(macOS 14.0, *)
    private static func describeRecordPermission(_ p: AVAudioApplication.recordPermission) -> String {
        switch p {
        case .undetermined: return "undetermined (System-Dialog möglich)"
        case .denied:       return "denied"
        case .granted:      return "granted"
        @unknown default:   return "unknown"
        }
    }

    // MARK: - Reset microphone permission

    func resetMicrophonePermission() {
        let confirm = NSAlert()
        confirm.alertStyle      = .informational
        confirm.messageText     = "Reset Microphone Permission?"
        confirm.informativeText =
            "This will clear Jarvis's microphone entry from System Settings so macOS " +
            "asks you again on the next launch.\n\n" +
            "Jarvis will quit automatically — just relaunch it and click Allow."
        confirm.addButton(withTitle: "Reset & Quit")
        confirm.addButton(withTitle: "Cancel")
        guard confirm.runModal() == .alertFirstButtonReturn else { return }

        let bundleID = Bundle.main.bundleIdentifier ?? ""
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/tccutil")
        proc.arguments     = ["reset", "Microphone", bundleID]
        try? proc.run()
        proc.waitUntilExit()

        NSApp.terminate(nil)
    }

    /// No system sheet possible anymore after deny — only Settings or `tccutil` reset.
    private func presentMicDeniedNeedsSettingsAlert() {
        let alert = NSAlert()
        alert.alertStyle      = .warning
        alert.messageText     = "Microphone Access Denied"
        let bid = Bundle.main.bundleIdentifier ?? ""
        alert.informativeText =
            "macOS will not show the permission dialog again for Jarvis.\n\n" +
            "Turn the switch on for Jarvis under Privacy & Security → Microphone, " +
            "or use Jarvis → Reset Mic Permission…\n\n" +
            "Terminal:  tccutil reset Microphone \(bid)"
        alert.addButton(withTitle: "Open System Settings")
        alert.addButton(withTitle: "Continue without Microphone")
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
