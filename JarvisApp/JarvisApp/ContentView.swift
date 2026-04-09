import AppKit
import SwiftUI
import WebKit

// MARK: - Root view

struct ContentView: View {
    @EnvironmentObject var backend: BackendManager

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            switch backend.phase {
            case .ready:
                ZStack {
                    WebViewWrapper().ignoresSafeArea()
                    VStack {
                        Spacer()
                        HStack {
                            Spacer()
                            STTOverlay(text: backend.lastSTT)
                                .padding(.trailing, 14)
                                .padding(.bottom, 14)
                        }
                    }
                }

            case .setup:
                SetupView()

            case .idle, .starting:
                LoadingView()

            case .failed(let msg):
                ErrorView(message: msg)
            }
        }
        .frame(minWidth: 400, minHeight: 400)
        .onAppear {
            if let delegate = NSApp.delegate as? AppDelegate {
                delegate.backendManager.start()
            }
        }
    }
}

// MARK: - Setup view (first-time install)

struct SetupView: View {
    @EnvironmentObject var backend: BackendManager

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 24) {
                Spacer()

                Text("Jarvis")
                    .font(.system(size: 36, weight: .thin, design: .default))
                    .foregroundColor(.white)

                Text("Setting up for the first time…")
                    .foregroundColor(.white.opacity(0.55))
                    .font(.system(size: 13))

                // Scrolling install log
                ScrollViewReader { proxy in
                    ScrollView(.vertical) {
                        Text(backend.setupLog.isEmpty ? "Starting…" : backend.setupLog)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.green.opacity(0.9))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(12)
                            .id("bottom")
                    }
                    .onChange(of: backend.setupLog) { _ in
                        withAnimation(.none) { proxy.scrollTo("bottom", anchor: .bottom) }
                    }
                }
                .frame(height: 200)
                .background(Color.white.opacity(0.04))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.white.opacity(0.1), lineWidth: 1)
                )
                .cornerRadius(8)

                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle(tint: .white.opacity(0.7)))
                    .scaleEffect(0.9)

                Spacer()
            }
            .padding(.horizontal, 40)
        }
    }
}

// MARK: - WKWebView wrapper

struct WebViewWrapper: NSViewRepresentable {

    func makeNSView(context: Context) -> JarvisWebView {
        let config = WKWebViewConfiguration()
        config.preferences.setValue(true, forKey: "allowFileAccessFromFileURLs")

        let webView = JarvisWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = context.coordinator
        webView.uiDelegate         = context.coordinator
        webView.allowsMagnification = false

        if let url = URL(string: "http://localhost:3000") {
            webView.load(URLRequest(url: url))
        }
        return webView
    }

    func updateNSView(_ nsView: JarvisWebView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate {
        func webView(_ webView: WKWebView,
                     decidePolicyFor action: WKNavigationAction,
                     decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            let host = action.request.url?.host ?? ""
            if host == "localhost" || host == "127.0.0.1" || action.navigationType == .other {
                decisionHandler(.allow)
            } else {
                decisionHandler(.cancel)
            }
        }
    }
}

// MARK: - Custom WKWebView

final class JarvisWebView: WKWebView {
    override var mouseDownCanMoveWindow: Bool { true }
    override func willOpenMenu(_ menu: NSMenu, with event: NSEvent) {
        menu.removeAllItems()
    }
}

// MARK: - STT overlay

struct STTOverlay: View {
    let text: String

    var body: some View {
        if !text.isEmpty {
            Text(text)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundColor(.white.opacity(0.9))
                .lineLimit(3)
                .multilineTextAlignment(.leading)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .frame(maxWidth: 280, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(.ultraThinMaterial.opacity(0.85))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.white.opacity(0.12), lineWidth: 1)
                        )
                )
                .shadow(color: .black.opacity(0.4), radius: 6, x: 0, y: 2)
                .transition(.opacity.combined(with: .scale(scale: 0.96, anchor: .bottomTrailing)))
                .animation(.easeInOut(duration: 0.25), value: text)
        }
    }
}

// MARK: - Loading view

struct LoadingView: View {
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 18) {
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    .scaleEffect(1.4)
                Text("Starting Jarvis…")
                    .foregroundColor(.white.opacity(0.8))
                    .font(.system(size: 15, weight: .light))
            }
        }
    }
}

// MARK: - Error view

struct ErrorView: View {
    @EnvironmentObject var backend: BackendManager
    let message: String

    var isPythonError: Bool { message == "python_not_found" }

    var displayMessage: String {
        if isPythonError {
            return "Python 3 was not found on this Mac.\n\nPlease install Python 3.11 or newer from python.org, then relaunch Jarvis."
        }
        return message
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 18) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 44))
                    .foregroundColor(.red.opacity(0.85))

                Text(isPythonError ? "Python Required" : "Failed to start")
                    .foregroundColor(.white)
                    .font(.system(size: 16, weight: .semibold))

                Text(displayMessage)
                    .foregroundColor(.gray)
                    .font(.system(size: 12))
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 340)

                if isPythonError {
                    Button("Open python.org") {
                        NSWorkspace.shared.open(
                            URL(string: "https://www.python.org/downloads/")!
                        )
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.blue)
                }

                HStack(spacing: 12) {
                    Button("Show Logs") {
                        (NSApp.delegate as? AppDelegate)?.showLogsWindow()
                    }
                    .buttonStyle(.bordered)

                    if !isPythonError {
                        Button("Retry") { backend.restart() }
                            .buttonStyle(.borderedProminent)
                            .tint(.blue)
                    }

                    Button("Reset Setup") { backend.resetSetup() }
                        .buttonStyle(.bordered)
                        .foregroundColor(.red.opacity(0.8))
                }
            }
            .padding(32)
        }
    }
}
