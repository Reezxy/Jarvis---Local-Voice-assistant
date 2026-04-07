import Foundation
import Darwin

/// Manages the Python backend subprocess and exposes state to SwiftUI.
@MainActor
final class BackendManager: ObservableObject {

    // MARK: - Published state

    @Published var isReady   = false
    @Published var isFailed  = false
    @Published var logs      = ""
    @Published var lastSTT   = ""          // latest transcribed user text

    // MARK: - Private

    private var process: Process?
    private var stdoutBuffer = ""          // partial-line accumulator
    private var sttClearTask: Task<Void, Never>?

    let projectRoot: String

    // MARK: - Init

    init() {
        projectRoot = Self.findProjectRoot()
    }

    /// Resolve the project root at runtime.
    ///
    /// Priority:
    ///  1. `JarvisProjectRoot` key baked into Info.plist by Xcode (`$(SRCROOT)/..`)
    ///     — works both from Xcode (DerivedData) and from a deployed .app.
    ///  2. Walk up from the app bundle looking for the `.venv311` directory
    ///     — fallback for edge cases.
    private static func findProjectRoot() -> String {
        // 1. Info.plist embed — $(SRCROOT)/.. expanded by Xcode at build time
        if let raw = Bundle.main.infoDictionary?["JarvisProjectRoot"] as? String,
           !raw.isEmpty {
            let resolved = URL(fileURLWithPath: raw).standardized.path
            if FileManager.default.fileExists(atPath: resolved + "/.venv311") {
                return resolved
            }
            // Even if .venv311 check fails, trust the baked-in path
            if resolved != "/" { return resolved }
        }

        // 2. Walk up from app bundle
        var dir = URL(fileURLWithPath: Bundle.main.bundlePath)
        for _ in 0 ..< 12 {
            dir = dir.deletingLastPathComponent()
            if FileManager.default.fileExists(
                atPath: dir.appendingPathComponent(".venv311").path) {
                return dir.path
            }
        }
        // Last resort: parent of .app
        return URL(fileURLWithPath: Bundle.main.bundlePath)
            .deletingLastPathComponent().path
    }

    // MARK: - Lifecycle

    func start() {
        guard process == nil else { return }

        let pythonPath = "\(projectRoot)/.venv311/bin/python"
        let scriptPath = "\(projectRoot)/chatbot_speech_to_speech.py"

        guard FileManager.default.fileExists(atPath: pythonPath) else {
            appendLog("❌  Python not found at \(pythonPath)\n")
            isFailed = true
            return
        }
        guard FileManager.default.fileExists(atPath: scriptPath) else {
            appendLog("❌  Script not found at \(scriptPath)\n")
            isFailed = true
            return
        }

        let proc = Process()
        proc.executableURL       = URL(fileURLWithPath: pythonPath)
        proc.arguments           = [scriptPath]
        proc.currentDirectoryURL = URL(fileURLWithPath: projectRoot)

        // Inherit a clean environment so the venv Python is used correctly
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"   // ensure stdout is unbuffered
        proc.environment = env

        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError  = errPipe

        outPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let str = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor [weak self] in self?.processStdout(str) }
        }
        errPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let str = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor [weak self] in self?.appendLog(str) }
        }

        proc.terminationHandler = { [weak self] p in
            outPipe.fileHandleForReading.readabilityHandler = nil
            errPipe.fileHandleForReading.readabilityHandler = nil
            Task { @MainActor [weak self] in
                self?.process = nil
                self?.isReady = false
                self?.appendLog("\n[Backend exited — code \(p.terminationStatus)]\n")
            }
        }

        do {
            try proc.run()
            process = proc
            appendLog("▶  Launching backend…\nProject root: \(projectRoot)\n\n")
            pollForReady()
        } catch {
            appendLog("❌  Launch failed: \(error)\n")
            isFailed = true
        }
    }

    func stop() {
        guard let proc = process else { return }
        let pid = proc.processIdentifier
        // SIGINT → Python KeyboardInterrupt handler runs cleanly
        Darwin.kill(pid, SIGINT)
        // Force-kill after 4 s if it is still alive
        DispatchQueue.global().asyncAfter(deadline: .now() + 4) {
            if proc.isRunning { Darwin.kill(pid, SIGKILL) }
        }
        process = nil
        isReady = false
    }

    func restart() {
        stop()
        isFailed = false
        isReady  = false
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            self.start()
        }
    }

    func clearLogs() { logs = "" }

    // MARK: - Output parsing

    private func processStdout(_ text: String) {
        appendLog(text)

        stdoutBuffer += text
        var lines = stdoutBuffer.components(separatedBy: "\n")
        stdoutBuffer = lines.removeLast()   // last segment may be incomplete

        for line in lines {
            // Python prints:  You: <transcribed text>
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("You: ") {
                let stt = String(trimmed.dropFirst(5))
                guard !stt.isEmpty else { continue }
                lastSTT = stt
                scheduleSTTClear()
            }
        }
    }

    private func scheduleSTTClear() {
        sttClearTask?.cancel()
        sttClearTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 8_000_000_000)   // 8 s
            if !Task.isCancelled { self.lastSTT = "" }
        }
    }

    func appendLog(_ text: String) {
        logs += text
        if logs.count > 100_000 { logs = String(logs.suffix(80_000)) }
    }

    // MARK: - Port polling

    private func pollForReady() {
        Task { @MainActor in
            // Poll every 0.5 s for up to 10 minutes — model loading can take a long time
            for _ in 0 ..< 1_200 {
                guard self.process != nil else { return }
                if await self.checkPort3000() {
                    self.isReady = true
                    return
                }
                try? await Task.sleep(nanoseconds: 500_000_000)  // 0.5 s
            }
            self.isFailed = true
        }
    }

    private func checkPort3000() async -> Bool {
        guard let url = URL(string: "http://localhost:3000") else { return false }
        var req = URLRequest(url: url, timeoutInterval: 2)
        req.httpMethod = "HEAD"
        do {
            let (_, response) = try await URLSession.shared.data(for: req)
            return (response as? HTTPURLResponse).map { $0.statusCode < 500 } ?? false
        } catch {
            return false
        }
    }
}
