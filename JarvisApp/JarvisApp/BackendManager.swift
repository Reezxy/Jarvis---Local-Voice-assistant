import Foundation
import Darwin

/// All persistent data lives here — survives app updates, works for any user.
private let kAppSupportDir: URL = {
    let fm   = FileManager.default
    let base = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
    let dir  = base.appendingPathComponent("Jarvis")
    try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}()

@MainActor
final class BackendManager: ObservableObject {

    // MARK: - Phase

    enum Phase: Equatable {
        case idle
        case setup      // first-time install running
        case starting   // backend booting, waiting for :3000
        case ready
        case failed(String)

        static func == (lhs: Phase, rhs: Phase) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle), (.setup, .setup), (.starting, .starting), (.ready, .ready):
                return true
            case (.failed(let a), .failed(let b)):
                return a == b
            default:
                return false
            }
        }

        var failureMessage: String? {
            if case .failed(let m) = self { return m }
            return nil
        }
    }

    // MARK: - Published

    @Published var phase:    Phase  = .idle
    @Published var setupLog: String = ""   // live output during first-time setup
    @Published var logs:     String = ""   // backend runtime logs
    @Published var lastSTT:  String = ""   // latest voice transcription

    // MARK: - Private

    private var process:       Process?
    private var stdoutBuffer = ""
    private var sttClearTask: Task<Void, Never>?

    // MARK: - Paths

    let appSupportDir = kAppSupportDir
    var venvDir:       URL { appSupportDir.appendingPathComponent(".venv") }
    var venvPython:    URL { venvDir.appendingPathComponent("bin/python") }
    var venvPip:       URL { venvDir.appendingPathComponent("bin/pip") }
    var setupSentinel: URL { appSupportDir.appendingPathComponent(".setup_complete") }
    var hfCacheDir:    URL { appSupportDir.appendingPathComponent("hf_cache") }

    /// Script bundled inside .app/Contents/Resources/
    var bundledScriptPath: String {
        Bundle.main.path(forResource: "chatbot_speech_to_speech", ofType: "py") ?? ""
    }
    /// Directory of bundled scripts (used as Python working dir so `import ws_server` works)
    var bundledResourcesDir: String {
        guard let p = Bundle.main.path(forResource: "chatbot_speech_to_speech", ofType: "py") else { return "" }
        return (p as NSString).deletingLastPathComponent
    }
    var requirementsPath: String {
        Bundle.main.path(forResource: "requirements_speech_to_speech", ofType: "txt") ?? ""
    }
    /// Pre-built frontend dist bundled in .app/Contents/Resources/
    var bundledDistDir: URL? {
        Bundle.main.url(forResource: "dist", withExtension: nil)
    }

    // MARK: - Entry point

    func start() {
        guard process == nil else { return }
        switch phase {
        case .setup, .starting, .ready: return
        default: break
        }
        phase = .idle

        let sentinelExists = FileManager.default.fileExists(atPath: setupSentinel.path)
        let venvExists     = FileManager.default.fileExists(atPath: venvPython.path)

        if sentinelExists && venvExists {
            startBackend()
        } else {
            Task { await self.runSetup() }
        }
    }

    // MARK: - First-time setup

    private func runSetup() async {
        phase    = .setup
        setupLog = ""

        // 1. Find Python
        appendSetup("🔍  Searching for Python 3...\n")
        guard let systemPython = findSystemPython() else {
            appendSetup("\n❌  Python 3 not found on this Mac.\n\n")
            appendSetup("Please install Python 3.11 or newer:\n")
            appendSetup("  https://www.python.org/downloads/\n\n")
            appendSetup("Then relaunch Jarvis.\n")
            phase = .failed("python_not_found")
            return
        }
        appendSetup("✅  Found: \(systemPython)\n\n")

        // 2. Create venv
        if !FileManager.default.fileExists(atPath: venvPython.path) {
            appendSetup("📦  Creating Python environment...\n")
            let code = await runCmd(systemPython, ["-m", "venv", venvDir.path])
            guard code == 0 else {
                phase = .failed("Failed to create Python environment (code \(code))")
                return
            }
            appendSetup("✅  Environment created.\n\n")
        } else {
            appendSetup("✅  Python environment already exists.\n\n")
        }

        // 3. Upgrade pip
        appendSetup("⬆️   Upgrading pip...\n")
        _ = await runCmd(venvPip.path, ["install", "--upgrade", "pip", "-q"])
        appendSetup("✅  Done.\n\n")

        // 4. Install requirements
        guard !requirementsPath.isEmpty else {
            phase = .failed("requirements_speech_to_speech.txt missing from app bundle")
            return
        }
        appendSetup("📥  Installing AI packages...\n")
        appendSetup("    (llama-cpp, faster-whisper, kokoro-onnx, sounddevice, …)\n")
        appendSetup("    This takes 2–5 minutes on first launch.\n\n")

        let reqCode = await runCmd(venvPip.path, ["install", "-r", requirementsPath, "-q"])
        guard reqCode == 0 else {
            phase = .failed("Package installation failed (code \(reqCode))")
            return
        }
        appendSetup("\n✅  All packages installed.\n\n")

        // 5. Write sentinel
        try? "done".write(toFile: setupSentinel.path, atomically: true, encoding: .utf8)
        appendSetup("🚀  Starting Jarvis...\n")
        startBackend()
    }

    private func findSystemPython() -> String? {
        let candidates = [
            "/opt/homebrew/bin/python3.11",
            "/opt/homebrew/bin/python3.12",
            "/opt/homebrew/bin/python3.13",
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3.11",
            "/usr/local/bin/python3.12",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ]
        let fm = FileManager.default
        return candidates.first { fm.fileExists(atPath: $0) }
    }

    // MARK: - Backend launch

    private func startBackend() {
        guard process == nil else { return }
        phase = .starting

        let pythonPath = venvPython.path
        let scriptPath = bundledScriptPath

        guard FileManager.default.fileExists(atPath: pythonPath) else {
            phase = .failed("Python not found at \(pythonPath)")
            return
        }
        guard !scriptPath.isEmpty, FileManager.default.fileExists(atPath: scriptPath) else {
            phase = .failed("chatbot_speech_to_speech.py missing from app bundle")
            return
        }

        let proc = Process()
        proc.executableURL       = URL(fileURLWithPath: pythonPath)
        proc.arguments           = [scriptPath]
        // Run from the bundled scripts dir so `import ws_server` works
        proc.currentDirectoryURL = URL(fileURLWithPath: bundledResourcesDir.isEmpty
                                       ? appSupportDir.path : bundledResourcesDir)

        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"]  = "1"
        env["JARVIS_DATA_DIR"]   = appSupportDir.path       // models, config override
        env["HF_HOME"]           = hfCacheDir.path           // Whisper + LLM cache
        if let dist = bundledDistDir {
            env["JARVIS_DIST_DIR"] = dist.path              // pre-built UI
        }
        // Prepend venv/bin to PATH so subprocesses find the right Python
        let venvBin = venvDir.appendingPathComponent("bin").path
        env["PATH"] = "\(venvBin):\(env["PATH"] ?? "/usr/bin:/bin")"
        proc.environment = env

        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError  = errPipe

        outPipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            guard !data.isEmpty, let s = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor [weak self] in self?.processStdout(s) }
        }
        errPipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            guard !data.isEmpty, let s = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor [weak self] in self?.appendLog(s) }
        }
        proc.terminationHandler = { [weak self] p in
            outPipe.fileHandleForReading.readabilityHandler = nil
            errPipe.fileHandleForReading.readabilityHandler = nil
            let code = p.terminationStatus
            Task { @MainActor [weak self] in
                self?.process = nil
                self?.appendLog("\n[Backend exited — code \(code)]\n")
                if case .ready = self?.phase { self?.phase = .idle }
            }
        }

        do {
            try proc.run()
            process = proc
            appendLog("▶  Backend started (PID \(proc.processIdentifier))\n")
            appendLog("   Data dir: \(appSupportDir.path)\n\n")
            pollForReady()
        } catch {
            phase = .failed("Launch failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Command runner (for setup steps)

    private func runCmd(_ executable: String, _ args: [String]) async -> Int32 {
        await withCheckedContinuation { cont in
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: executable)
            proc.arguments     = args

            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError  = pipe

            pipe.fileHandleForReading.readabilityHandler = { [weak self] h in
                let data = h.availableData
                guard !data.isEmpty, let s = String(data: data, encoding: .utf8) else { return }
                Task { @MainActor [weak self] in self?.setupLog += s }
            }
            proc.terminationHandler = { p in
                pipe.fileHandleForReading.readabilityHandler = nil
                cont.resume(returning: p.terminationStatus)
            }
            do { try proc.run() }
            catch {
                Task { @MainActor [weak self] in self?.setupLog += "Error: \(error)\n" }
                cont.resume(returning: -1)
            }
        }
    }

    // MARK: - Polling for readiness

    private func pollForReady() {
        Task { @MainActor in
            for _ in 0 ..< 1_200 {   // up to 10 minutes
                guard process != nil else { return }
                if await checkPort3000() {
                    phase = .ready
                    return
                }
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
            if phase == .starting {
                phase = .failed("Backend didn't respond on :3000 after 10 min.\nCheck Show Logs for details.")
            }
        }
    }

    private func checkPort3000() async -> Bool {
        guard let url = URL(string: "http://127.0.0.1:3000/api/status") else { return false }
        var req = URLRequest(url: url, timeoutInterval: 3)
        req.httpMethod = "GET"
        do {
            let (_, res) = try await URLSession.shared.data(for: req)
            return (res as? HTTPURLResponse)?.statusCode == 200
        } catch { return false }
    }

    // MARK: - Stop / Restart / Reset

    func stop() {
        guard let proc = process else { return }
        Darwin.kill(proc.processIdentifier, SIGINT)
        let p = proc
        DispatchQueue.global().asyncAfter(deadline: .now() + 4) {
            if p.isRunning { Darwin.kill(p.processIdentifier, SIGKILL) }
        }
        process = nil
        phase   = .idle
    }

    func restart() {
        stop()
        phase = .idle
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            self.start()
        }
    }

    /// Wipe the venv + sentinel → triggers full re-setup on next start.
    func resetSetup() {
        stop()
        try? FileManager.default.removeItem(at: setupSentinel)
        try? FileManager.default.removeItem(at: venvDir)
        setupLog = ""
        logs     = ""
        phase    = .idle
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 500_000_000)
            self.start()
        }
    }

    // MARK: - Log helpers

    func appendLog(_ text: String) {
        logs += text
        if logs.count > 100_000 { logs = String(logs.suffix(80_000)) }
    }
    func clearLogs() { logs = "" }

    private func appendSetup(_ text: String) { setupLog += text }

    private func processStdout(_ text: String) {
        appendLog(text)
        stdoutBuffer += text
        var lines = stdoutBuffer.components(separatedBy: "\n")
        stdoutBuffer = lines.removeLast()
        for line in lines {
            let t = line.trimmingCharacters(in: .whitespaces)
            if t.hasPrefix("You: ") {
                let stt = String(t.dropFirst(5))
                guard !stt.isEmpty else { continue }
                lastSTT = stt
                scheduleSTTClear()
            }
        }
    }

    private func scheduleSTTClear() {
        sttClearTask?.cancel()
        sttClearTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 8_000_000_000)
            if !Task.isCancelled { self.lastSTT = "" }
        }
    }
}
