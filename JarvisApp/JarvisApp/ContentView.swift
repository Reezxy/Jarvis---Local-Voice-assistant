import SwiftUI
import WebKit

// MARK: - Root view

struct ContentView: View {
    @EnvironmentObject var backend: BackendManager

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if backend.isReady {
                // Full-window web view
                WebViewWrapper()
                    .ignoresSafeArea()

                // STT overlay — bottom-right corner
                VStack {
                    Spacer()
                    HStack {
                        Spacer()
                        STTOverlay(text: backend.lastSTT)
                            .padding(.trailing, 14)
                            .padding(.bottom, 14)
                    }
                }
            } else if backend.isFailed {
                ErrorView()
            } else {
                LoadingView()
            }
        }
        .frame(minWidth: 400, minHeight: 400)
    }
}

// MARK: - WKWebView wrapper

struct WebViewWrapper: NSViewRepresentable {

    func makeNSView(context: Context) -> JarvisWebView {
        let config = WKWebViewConfiguration()
        // Allow any local network content (ws:// on localhost)
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

    // MARK: Coordinator

    final class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate {

        // Only allow navigation within localhost
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

// MARK: - Custom WKWebView (no context menu, draggable)

final class JarvisWebView: WKWebView {
    /// Let clicks on non-interactive areas drag the window
    override var mouseDownCanMoveWindow: Bool { true }

    /// Remove context menu
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
                    .font(.system(size: 15, weight: .light, design: .default))
            }
        }
    }
}

// MARK: - Error view

struct ErrorView: View {
    @EnvironmentObject var backend: BackendManager

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 18) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 44))
                    .foregroundColor(.red.opacity(0.85))

                Text("Failed to start backend")
                    .foregroundColor(.white)
                    .font(.system(size: 16, weight: .semibold))

                Text("Could not reach http://localhost:3000 after 10 min.\nCheck that .venv311 is set up and press Show Logs.")
                    .foregroundColor(.gray)
                    .font(.system(size: 12))
                    .multilineTextAlignment(.center)

                HStack(spacing: 12) {
                    Button("Show Logs") {
                        if let delegate = NSApp.delegate as? AppDelegate {
                            delegate.showLogsWindow()
                        }
                    }
                    .buttonStyle(.bordered)

                    Button("Retry") {
                        backend.restart()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.blue)
                }
            }
            .padding(32)
        }
    }
}
